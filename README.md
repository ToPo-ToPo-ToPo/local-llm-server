# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を束ねる**マルチモデルゲートウェイ**。
`gateway.toml` にモデルを並べて 1 プロセス起動するだけで、1 つの公開ポートに複数モデルを配信する。

- **`gateway.toml`（モデルカタログ）を書いて起動するだけ**。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- モデルは**初回リクエスト時に遅延起動**、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード。
- 接続側は公開ポートに繋いで `model` を選ぶだけ（接続クライアントは別パッケージ）。

## インストール

> **これは import するライブラリではなく、別プロセスで動かすサーバー（ゲートウェイ）です。**
> LLM を使うアプリ側には入れません。アプリは別パッケージ
> [local-llm-client](https://pypi.org/project/local-llm-client/)（または素の `openai` SDK）で
> 起動中のゲートウェイのポートに接続します。

[uv](https://docs.astral.sh/uv/) で入れる。**目的に応じて 3 通り**（Apple Silicon は推論バックエンドの
extra `[mlx]` を付ける。`[ ]` は zsh の glob 展開を避けるためクォート）。他 OS は付けず、llama.cpp を
別途用意する → 末尾の注記。

**A. コマンドとして入れる（最も手軽。普通はこれ）**
どこでも使える `local-llm-server` コマンドになる:
```bash
uv tool install "local-llm-server[mlx]"
```

**B. ゲートウェイ用フォルダに入れる（uv プロジェクトとして管理したいとき）**
`gateway.toml` を置く**新規フォルダ**を作り、その依存として入れる:
```bash
mkdir my-gateway && cd my-gateway
uv init
uv add "local-llm-server[mlx]"
```

**C. ソースから動かす（このリポジトリをクローンした場合）**
クローンした**フォルダの中**では `uv add` ではなく `uv sync`（自分自身を依存に足さない）:
```bash
cd local-llm-server          # クローンしたフォルダ
uv sync --extra mlx
```

> **他 OS（Linux / Windows / Intel Mac）＝ llama.cpp** は上記の `[mlx]` / `--extra mlx` を外す。
> 推論には `llama-server` を別途インストールして PATH に通し（OS 別手順は
> [docs/llama-cpp.md](docs/llama-cpp.md)）、`gateway.toml` の `[[models]]` で `backend = "llama-cpp"` を指定する。

## 使い方

### 1. gateway.toml を置く

カレントディレクトリに `gateway.toml` を置く。リポジトリ直下に例を同梱（→ [gateway.toml](gateway.toml)）:

```toml
host = "127.0.0.1"
port = 8799                 # クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数の上限（超過は LRU 退避）
[[models]]
model = "mlx-community/Qwen3.6-27B-4bit"
backend = "mlx-vlm"
```

### 2. 起動

`gateway.toml` のあるフォルダで起動する。**A（tool install）は `local-llm-server`、B・C は `uv run` を付ける**:

```bash
local-llm-server            # A の場合。TUI ダッシュボード（状態を自動更新表示）
uv run local-llm-server     # B / C の場合（プロジェクトの venv 経由）
```

### 3. 接続

公開ポートに繋ぎ、`model` で使うモデルを選ぶ。接続クライアントは別パッケージ
[local-llm-client](https://pypi.org/project/local-llm-client/)

```bash
uv add local-llm-client
```
```python
from local_llm_client import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8799/v1")
print(llm.respond("ローカルLLMの利点を3つ。"))
```

## ドキュメント

- [docs/gateway.md](docs/gateway.md) — `gateway.toml` の全フィールドと振る舞い
- [docs/llama-cpp.md](docs/llama-cpp.md) — llama.cpp（`llama-server`）の OS 別導入・最新モデル追従
- [docs/operation.md](docs/operation.md) — 起動・停止・監視（ターミナル）・アンインストール
- [docs/mtp.md](docs/mtp.md) — MTPによる高速化

## ライセンス

Apache-2.0
