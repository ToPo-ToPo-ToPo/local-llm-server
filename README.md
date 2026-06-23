# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を束ねる**マルチモデルゲートウェイ**。

- **`gateway.toml`（モデルカタログ）を書いて 1 プロセス起動するだけ**。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- **モデルは初回リクエスト時に遅延起動**、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード。
- クライアントは公開ポートに繋いで `model` を選ぶだけ。

> **サポートする使い方は 2 つ**:（1）ゲートウェイの運用 — `local-llm-server`（起動/停止/状態）と
> `local-llm-server-gui`（監視 GUI）。ゲートウェイは**このリポジトリで 1 つ起動する**運用が前提。
> （2）クライアント接続 — `LLMClient` で公開ポートに繋ぐ（エージェントごとの再実装を防ぐ共通
> クライアント。素の `openai` SDK で base_url を指してもよい）。`connect()` は起動中ゲートウェイに
> 繋ぐワンライナーで、未起動なら親切なエラー（**サーバーは自前で起動しない**）。
>
> **サーバーを自前で起動する経路**（`ensure_server` の自動起動 / `LocalServer` / `ServerPool` /
> `RouterServer` 等）は**非公開・サポート対象外**（サーバーを立てるのはゲートウェイ 1 箇所だけ、
> という運用にするため。後方互換で import は残すが `__all__` 非公開）。

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

### 2. アプリを作って起動（操作はアプリに一本化）

ゲートウェイの**起動・停止・監視はすべてアプリ**（メニューバー/トレイ常駐）から行う。まず一度だけ
クリック起動アプリを作る:

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
- メニューから **起動 / 停止 / 再起動 / ログを開く / 更新**。各モデルの loaded・処理中数・PID・
  運用方針（max_resident・idle_timeout）も表示。
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

### 3. 接続（ `model` で選ぶ）

公開ポートに繋ぎ、`model` で使うモデルを選ぶ。
```python
from local_llm_server import LLMClient

llm = LLMClient(
  model="mlx-community/Qwen3.6-27B-4bit",
  base_url="http://127.0.0.1:8799/v1"
)
print(llm.respond("ローカルLLMの利点を3つ。"))
```

高度操作 → [docs/connecting.md](docs/connecting.md)。

## ライセンス

Apache-2.0
