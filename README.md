# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を束ねる**マルチモデルゲートウェイ**。

- **`gateway.toml`（モデルカタログ）を書いて 1 プロセス起動するだけ**。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- **モデルは初回リクエスト時に遅延起動**、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード。
- クライアントは公開ポートに繋いで `model` を選ぶだけ。

> **このパッケージはゲートウェイ・サーバー専用**。`local-llm-server`（引数なし）で**ターミナルの TUI
> ダッシュボード**が開き、状態を自動更新表示しつつ起動・停止・再起動を操作できる（`--headless` /
> `--start` / `--stop` / `--status` / `--restart` も）。必要ならトレイ GUI アプリ `local-llm-server-gui`
> も選べる。ゲートウェイ本体は標準ライブラリのみで動き、**`openai` などのコア依存は無い**。
>
> **接続する側（クライアント）は別パッケージ [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-automata-core/tree/main/packages/local-llm-client)**
> に分離した。エージェントはそちらの `LLMClient` / `connect` を使う（または素の `openai` SDK で
> `base_url` を指す）。サーバーを自前で起動する低レベル経路（`ensure_server` / `LocalServer` /
> `RouterServer` 等）は非公開・サポート対象外（後方互換で import は残す）。

## インストール
[uv](https://docs.astral.sh/uv/)を使用する。
```bash
uv add "local-llm-server[mlx]"
```

extras 指定はクォート必須（zsh の glob 展開回避）。内訳:

| extra | 入るもの | 用途 |
|---|---|---|
| `mlx` | `mlx-lm` / `mlx-vlm` | Apple Silicon で実際に推論する |
| `gui` | `pystray` / `pillow` | システムトレイ常駐の状態モニタ（Win/mac/Linux） |

## 使い方

### 1. `gateway.toml`（モデルカタログ）

カレントディレクトリに `gateway.toml` を置く。これがサーバーの唯一の設定。リポジトリ直下に
すぐ使える例を同梱（→ [gateway.toml](gateway.toml)）:

```toml
host = "127.0.0.1"
port = 8799                 # 公開ポート。クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数の上限。超えたら LRU 退避（省略時 無制限）
idle_timeout = 1200         # 20分使われないモデルは自動アンロード（0/省略で無効）
draft_model = "auto"        # MTP の既定（各 [[models]] で上書き・"off" で無効）

[[models]]
model = "mlx-community/Qwen3.6-27B-4bit"
backend = "mlx-vlm"

[[models]]
model = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
backend = "mlx-vlm"
```

MTP（投機的デコード）による高速化 → [docs/mtp.md](docs/mtp.md)。

### 2. 起動・運用（ターミナル）

`gateway.toml` のあるディレクトリで、**引数なしで起動すると TUI ダッシュボード**が開く。
ゲートウェイを裏で常駐させ、状態を自動更新表示しながらキー操作できる（ターミナル版の常駐モニタ。
**リポジトリのコードだけで完結**し、外部にアプリやランチャを一切置かない）。

```bash
uv run local-llm-server            # TUI ダッシュボード（既定）
```

- 上部にゲートウェイ（応答状態・port・起動経過・累計リクエスト）と各モデルの表
  （loaded/idle/unloaded・内部ポート・処理中数・アイドル自動解放までの残り・累計）を**毎秒自動更新**。
- 操作は単キー **`s`停止 / `r`再起動 / `g`起動 / `l`ログ / `q`終了**、`:` で打ち込みコマンド
  （`stop`/`restart`/`start`/`log`/`quit`）。
- `q` で終了してもゲートウェイは常駐し続ける（停止は `s` か下の `--stop`）。

スクリプト用途や TUI を出したくないとき（パイプ/CI/裏起動）は運用フラグを使う。**非対話端末では
自動でフォアグラウンド実行**になる:

```bash
uv run local-llm-server --headless # TUI なしでフォアグラウンド実行（Ctrl-C で停止）
uv run local-llm-server --start    # バックグラウンド常駐起動（端末を離す。Ollama 流）
uv run local-llm-server --status   # 応答可否・PID・提供モデル・ログパス
uv run local-llm-server --stop     # 停止（配下のモデルサーバーも止める）
uv run local-llm-server --restart  # 停止→再起動（gateway.toml 変更の反映に）
```

- いずれも **CWD の `./gateway.toml`** を読む（場所＝設定の単一ルール）。
- ログは `./.local-llm-server/gateway-<port>.log`。
- 配下のモデルは初回リクエストで遅延起動し、`idle_timeout` で自動アンロードされる。
- TUI は標準ライブラリ `curses`（macOS/Linux はそのまま）。Windows のみ `uv add windows-curses`。

### 3.（任意）トレイ GUI アプリ

ターミナルを使わず、デスクトップで状態をひと目で見たい場合の代替。メニューバー（macOS）/通知領域
（Windows）/トレイ（Linux）に常駐し、起動・停止・監視をクリックで行える。CLI と同じ `./gateway.toml`
を読み、同じ運用基盤を共有する。まず一度だけクリック起動アプリを作る。**`make` を使えば 1 コマンド**
（macOS / Linux）:

```bash
make install        # 依存(mlx+gui)を入れて → クリック起動アプリを作成
```

`make` を使わない場合は同等の 2 手順:

```bash
uv add "local-llm-server[gui]"              # pystray / pillow（各 OS のバックエンドも）
uv run local-llm-server-gui --install-app   # gateway.toml のあるディレクトリで1度だけ
```

- **macOS** … `~/Applications/Local LLM Gateway.app`（専用アイコン付きの普通のアプリ。Dock 表示・
  Cmd+Tab 対応）。Finder で **Dock にドラッグすれば常設**。Dock に出さずメニューバーだけにしたい
  ときは `--install-app --menubar-only`。
- **Linux** … `~/.local/share/applications/*.desktop`／**Windows** … デスクトップに `.cmd`。

あとは**アプリをダブルクリック**するだけ。ゲートウェイがバックグラウンドで常駐し、メニューバー
（macOS）/通知領域（Windows）/トレイ（Linux）にアイコンが出る:

- アイコンの**色**で状態（🟢 応答可 / 🟡 起動中 / ⚫ 停止）、**数字**でロード済みモデル数。
- メニューから **起動 / 停止 / 再起動 / ログを開く / 更新**。あわせて状態も表示する:
  ゲートウェイの **PID・起動経過・累計リクエスト数**、各モデルの **バックエンド:内部ポート・
  loaded/idle・処理中数・アイドル自動解放までの残り時間・累計リクエスト数**、運用方針
  （max_resident・idle_timeout）。
- アプリ（トレイ）を閉じてもゲートウェイは常駐し続ける。止めるときはメニューの **停止**。
- 同一バンドル ID なので、起動中に再度クリックしても二重起動しない。

> ゲートウェイ本体は `local-llm-server`（`./gateway.toml` を読むフォアグラウンド実行）で、アプリが
> 端末から切り離して起動・常駐させる（ログは `./.local-llm-server/gateway-<port>.log`）。状態は
> `GET /admin/status` から取得。Linux はトレイ表示にシステムトレイが要る（GNOME は AppIndicator
> 拡張など）。アプリは作成時の **このリポジトリのパスと Python（venv）を固定**するので、クリック
> 時の場所に依存しない。クリック起動をやめるときは作ったランチャを消すだけ。

1 つの公開ポート（例 `http://127.0.0.1:8799/v1`）でカタログのモデルを束ねる。**各モデルは
初回リクエスト時に遅延起動**し、2 回目以降は常駐して即応答。`max_resident` 超過は LRU 退避、
`idle_timeout` で自動アンロード。

### 4. 接続（ `model` で選ぶ）

公開ポートの OpenAI 互換 API に繋ぎ、`model` で使うモデルを選ぶ。**接続用クライアントは別パッケージ
[local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-automata-core/tree/main/packages/local-llm-client)**（エージェント共通の
`LLMClient` / `connect`）。

```bash
uv add local-llm-client
```
```python
from local_llm_client import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8799/v1")
print(llm.respond("ローカルLLMの利点を3つ。"))
```

このパッケージ（`local-llm-server`）を入れなくても、素の `openai` SDK で `base_url` を指すだけでも
接続できる。

## アンインストール

自動起動やシステム改変はしていないので、**コマンド1つ＋リポジトリ削除**で跡形なく消える。

```bash
make uninstall      # アプリ削除＋ゲートウェイ停止＋データ掃除（macOS / Linux）
```

`make` を使わない場合（Windows 含む全 OS）:

```bash
uv run local-llm-server-gui --uninstall-app --purge
#  --purge を外すとログ等は残す。Windows はデスクトップの .cmd を削除。
```

`--purge` が消すもの:

- ランチャ（macOS `.app`／Linux `.desktop`／Windows `.cmd`）と、リポジトリ内のログ
  （`./.local-llm-server`）。
- 念のため macOS がアプリ毎に作りうる場所も掃除（`~/Library/Application Support/local-llm-server`、
  bundle id 配下の Saved State / Preferences / Caches 等）。通常ここには何も作らないが、取りこぼし防止。

最後に**このリポジトリのフォルダを削除**すれば、コードと依存（`.venv`）も消えて完了。

> **モデル本体（重み）には触れない**。mlx / mlx-vlm は HuggingFace の**共有キャッシュ**
> `~/.cache/huggingface` にモデルをダウンロードするが、他ツールと共用のため
> **アンインストールでは一切削除しない**（場所を案内するだけ）。不要になったら自分で
> 中のモデルフォルダを消す。

## ライセンス

Apache-2.0
