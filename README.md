# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を束ねる**マルチモデルゲートウェイ**。
`gateway.toml` にモデルを並べて 1 プロセス起動するだけで、1 つの公開ポートに複数モデルを配信する。

- **`gateway.toml`（モデルカタログ）を書いて起動するだけ**。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- モデルは**初回リクエスト時に遅延起動**、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード。
- 接続側は公開ポートに繋いで `model` を選ぶだけ（接続クライアントは別パッケージ）。

## インストール

[uv](https://docs.astral.sh/uv/) を使う。
`[backend]` は推論バックエンドを入れる extra。
`[ ]` はシェル（zsh）の glob 展開を避けるためクォートする。

#### 1. Apple Silicon で推論する（mlx-lm / mlx-vlm）
```bash
uv add "local-llm-server[mlx]"
```

#### 2. その他の OS（Linux / Windows / Intel Mac）— llama.cpp
ゲートウェイ本体だけ入れる（バックエンド extra は不要）:
```bash
uv add local-llm-server
```
推論には llama.cpp の `llama-server` を別途インストールして PATH に通す（OS 別の導入手順は
[docs/llama-cpp.md](docs/llama-cpp.md)）。`gateway.toml` の `[[models]]` で `backend = "llama-cpp"` を指定する。

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

```bash
uv run local-llm-server     # TUI ダッシュボード（状態を自動更新表示）
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
- [docs/operation.md](docs/operation.md) — 起動・停止・監視（ターミナル／GUI アプリ）・アンインストール
- [docs/mtp.md](docs/mtp.md) — MTPによる高速化

## ライセンス

Apache-2.0
