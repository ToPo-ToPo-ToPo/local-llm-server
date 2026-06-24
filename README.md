# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を束ねる**マルチモデルゲートウェイ**。
`gateway.toml` にモデルを並べて 1 プロセス起動するだけで、1 つの OpenAI 互換ポートに複数モデルを配信する。

- **`gateway.toml`（モデルカタログ）を書いて起動するだけ**。外部アプリ（Ollama / LM Studio 等）に依存しない。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- モデルは**初回リクエスト時に遅延起動**、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード。
- 接続側は公開ポートに繋いで `model` を選ぶだけ（接続クライアントは別パッケージ）。

## インストール

[uv](https://docs.astral.sh/uv/) を使う（extras 指定はクォート必須）。

```bash
uv add "local-llm-server[mlx]"     # Apple Silicon で推論する（mlx-lm / mlx-vlm）
```

## 使い方

### 1. gateway.toml を置く

カレントディレクトリに `gateway.toml`（モデルカタログ）を置く。リポジトリ直下に例を同梱（→ [gateway.toml](gateway.toml)）:

```toml
host = "127.0.0.1"
port = 8799                 # クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数の上限（超過は LRU 退避）
[[models]]
model = "mlx-community/Qwen3.6-27B-4bit"
backend = "mlx-vlm"
```

全フィールドの説明 → **[docs/gateway.md](docs/gateway.md)**。MTP（高速化）→ **[docs/mtp.md](docs/mtp.md)**。

### 2. 起動

```bash
uv run local-llm-server     # TUI ダッシュボード（状態を自動更新表示）
```

バックグラウンド常駐・停止・監視（`--start` / `--stop` / `--status` / `--headless`）やトレイ GUI アプリ、
アンインストール → **[docs/operation.md](docs/operation.md)**。

### 3. 接続

公開ポートの OpenAI 互換 API に繋ぎ、`model` で使うモデルを選ぶ。接続クライアントは別パッケージ
[local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-automata-core/tree/main/packages/local-llm-client)
（または素の `openai` SDK で `base_url` を指すだけでも可）:

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
- [docs/operation.md](docs/operation.md) — 起動・停止・監視（ターミナル／GUI アプリ）・アンインストール
- [docs/mtp.md](docs/mtp.md) — MTP（投機的デコード）による高速化

## ライセンス

Apache-2.0
