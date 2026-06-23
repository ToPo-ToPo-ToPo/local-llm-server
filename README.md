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

### 2. ゲートウェイを起動

`gateway.toml` のあるディレクトリで起動する（管理者の唯一の操作）。**バックグラウンド常駐**
（Ollama 流。ターミナルを占有しない）と**フォアグラウンド**のどちらでも:

```bash
uv run local-llm-server --start   # バックグラウンド常駐（推奨。すぐプロンプトに戻る）
uv run local-llm-server           # フォアグラウンド（Ctrl+C で停止。デバッグ向き）
```

`--start` は端末から切り離した別プロセスで起動し、ログを `./.local-llm-server/gateway-<port>.log`
に書く。停止は `--stop`、設定変更の反映は `--restart`。常駐の起動/停止/監視はトレイ GUI からもできる。

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

### 運用（start / status / stop / restart）

```bash
uv run local-llm-server --start     # バックグラウンド常駐で起動（既に起動済みなら何もしない）
uv run local-llm-server --status    # 稼働確認（カタログ＝全モデル・pid・ログパス）
uv run local-llm-server --stop      # 停止（配下のモデルサーバーも全て止める）
uv run local-llm-server --restart   # 停止して再起動（gateway.toml の変更を反映）
```

`Ctrl+C` / `kill` でも、起動済みのモデルサーバーまで一緒に止まる（孫プロセスは残らない）。
`--start` / `--stop` は **macOS / Linux / Windows** で動く（起動は端末から切り離した別プロセス、
停止はポート→PID 特定を lsof / netstat、終了をプロセスグループ / `taskkill /T` で実施）。

#### トレイ GUI（Windows / macOS / Linux）

ターミナルを開かずにゲートウェイを操作・監視する常駐アプリ。システムトレイ（macOS は
メニューバー、Windows は通知領域、Linux はトレイ）にアイコンを出し、色でゲートウェイ状態
（🟢 応答可 / 🟡 起動中 / ⚫ 停止）を、ロード済みモデル数を数字で表す。クリックで各モデルの
常駐状態（loaded / idle）と処理中リクエスト数、PID・運用方針（max_resident・idle_timeout）を
表示し、メニューから**起動**・**停止**・**再起動**（バックグラウンド常駐）・**ログを開く**・
**再読込**ができる。ウィンドウを占有しないので他の作業の邪魔にならない。

```bash
uv add "local-llm-server[gui]"        # pystray / pillow（各 OS のバックエンドも）
uv run local-llm-server-gui           # gateway.toml のあるディレクトリで
```

状態は GUI 用の読み取り口 `GET /admin/status`（各モデルの loaded / inflight ＋運用方針を
JSON で返す）から取得する。CLI と同じく **カレントディレクトリの `./gateway.toml`** を読む。
Linux はトレイ表示にシステムトレイが要る（GNOME は AppIndicator 拡張など）。

#### クリックして起動できるアプリにする

ターミナルを開かずアイコンのクリックで起動したいときは、ランチャ（macOS は `.app`、Linux は
`.desktop`、Windows は `.cmd`）を 1 度だけ作る:

```bash
uv run local-llm-server-gui --install-app   # gateway.toml のあるディレクトリで
```

- macOS … `~/Applications/Local LLM Gateway.app` を作成（**専用アイコン付きの普通のアプリ**）。
  **ダブルクリックで「ゲートウェイをバックグラウンド起動＋メニューバー常駐」**。Dock に表示され
  Cmd+Tab にも出る。Finder で **Dock へドラッグすれば常設ショートカット**になる（同一バンドル ID
  なので二重起動しない）。Dock に出さずメニューバーだけにしたいときは `--menubar-only` を付ける。
- Linux … `~/.local/share/applications/local-llm-gateway.desktop`／Windows … デスクトップに
  `Local LLM Gateway.cmd`。

ランチャは作成時の **このリポジトリのパスと Python（venv）を固定**して起動する（クリック時に
作業ディレクトリへ `cd` するので、ターミナルの場所に依存しない）。クリック起動を無効化したい
ときは作ったランチャを消すだけ。

## ライセンス

Apache-2.0
