# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を **OpenAI 互換 API** で束ねる
**マルチモデルゲートウェイ**。Ollama と同じイメージで、**`gateway.toml`（モデルカタログ）を
書いて 1 プロセス起動するだけ**。1 つの公開ポートで複数モデルを配信し、リクエストの `model`
で振り分ける。

- **モデルは初回リクエスト時に遅延起動**、`max_resident` 超過で LRU 退避、`idle_timeout`
  で自動アンロード（Ollama の keep-alive 相当）。外部アプリ（Ollama / LM Studio）に依存しない。
- **MTP（投機的デコード）**で本体の出力を変えず ~2倍速（mlx-vlm）。
- クライアントは公開ポートに繋いで `model` を選ぶだけ（クライアントはサーバーを起動しない）。
- ゲートウェイ・MTP 解決は**標準ライブラリのみ**、付属の高レベルクライアントは公式 `openai`
  SDK（コア依存）。推論バックエンドだけ extra で導入。

## インストール（[uv](https://docs.astral.sh/uv/)）

```bash
uv add "local-llm-server[mlx]"
```

extras 指定はクォート必須（zsh の glob 展開回避）。内訳:

| extra | 入るもの | 用途 |
|---|---|---|
| （無し） | コア（標準ライブラリ ＋ `openai`） | ゲートウェイ・MTP・`connect`/`LLMClient` まで全部 |
| `mlx` | `mlx-lm` / `mlx-vlm` | Apple Silicon で実際に推論する |

`connect` / `LLMClient` などライブラリ機能は `uv add "local-llm-server[mlx]"` だけで
すべて使える（client 用の追加 extra は不要）。高レベルクライアントは公式 `openai`
SDK を土台にしており、自動リトライ・型付き応答・ツール呼び出し/構造化出力も使える。

## 使い方

### 1. `gateway.toml`（モデルカタログ）を書く

カレントディレクトリに `gateway.toml` を置く。これがサーバーの唯一の設定。リポジトリ直下に
すぐ使える例を同梱（→ [gateway.toml](gateway.toml)）:

```toml
host = "127.0.0.1"
port = 8799                 # 公開ポート。クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数の上限。超えたら LRU 退避（省略時 無制限）
idle_timeout = 600          # 10分使われないモデルは自動アンロード（0/省略で無効）
draft_model = "auto"        # MTP の既定（各 [[models]] で上書き・"off" で無効）

[[models]]
model = "mlx-community/Qwen3.6-27B-4bit"   # マルチモーダル（テキスト＋画像）
backend = "mlx-vlm"

[[models]]
model = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
backend = "mlx-vlm"
```

### 2. ゲートウェイを起動する

`gateway.toml` のあるディレクトリで起動するだけ（管理者の唯一の操作）:

```bash
uv run local-llm-server
```

1 つの公開ポート（例 `http://127.0.0.1:8799/v1`）でカタログのモデルを束ねる。**各モデルは
初回リクエスト時に遅延起動**し、2 回目以降は常駐して即応答。`max_resident` 超過は LRU 退避、
`idle_timeout` で自動アンロード。

### 3. 接続する（OpenAI 互換 API、`model` で選ぶ）

公開ポートに繋ぎ、`model` で使うモデルを選ぶ。Python（付属の `LLMClient`。`openai` SDK が土台）:

```python
from local_llm_server import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8799/v1")
print(llm.respond("ローカルLLMの利点を3つ。"))
```

ストリーミング・画像、`openai` SDK / `curl` 直叩き、`llm.openai` での高度操作は
→ [docs/connecting.md](docs/connecting.md)。

### 運用（status / stop）

```bash
uv run local-llm-server --status   # 稼働確認（カタログ＝全モデル・pid・ログパス）
uv run local-llm-server --stop     # ゲートウェイ停止（配下のモデルサーバーも全て停止）
```

`Ctrl+C` / `kill` でも、起動済みのモデルサーバーまで一緒に止まる（孫プロセスは残らない）。

ゲートウェイを介さず Python から単一モデルサーバーを起動・利用する `connect()` /
`ensure_server()` / `LocalServer` も公開 → [docs/connecting.md](docs/connecting.md)。

## MTP（投機的デコード）

本体モデルの出力を変えずに ~2倍速にする高速化（Qwen3.6-27B で実測 38→75 tok/s、採択率
93%）。`gateway.toml` の `draft_model = "auto"`（または `connect(draft_model="auto")`）で、本体名
から対応ドラフターを自動選択する（`mlx-vlm` 限定。`"off"` で無効、HF id で明示も可。対応表
`MTP_DRAFTERS` / 解決 `resolve_drafter`）。

## examples

実機で動く完全なサンプル（Apple Silicon）。`uv run` するだけで動く:

```bash
uv run examples/connect_and_generate.py    # 自動起動 + 生成（最短）
uv run examples/generate_with_mtp.py       # LocalServer + openai で MTP 生成 + tok/s 表示
```

詳細は [examples/README.md](examples/README.md)。

## ライセンス

Apache-2.0
