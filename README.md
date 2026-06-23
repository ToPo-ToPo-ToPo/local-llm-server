# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を束ねる**マルチモデルゲートウェイ**。

- **`gateway.toml`（モデルカタログ）を書いて 1 プロセス起動するだけ**。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- **モデルは初回リクエスト時に遅延起動**、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード。
- クライアントは公開ポートに繋いで `model` を選ぶだけ。

## インストール
[uv](https://docs.astral.sh/uv/)を使用する。
```bash
uv add "local-llm-server[mlx]"
```

extras 指定はクォート必須（zsh の glob 展開回避）。内訳:

| extra | 入るもの | 用途 |
|---|---|---|
| `mlx` | `mlx-lm` / `mlx-vlm` | Apple Silicon で実際に推論する |

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

`gateway.toml` のあるディレクトリで起動するだけ（管理者の唯一の操作）:

```bash
uv run local-llm-server
```

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

### 運用（status / stop）

#### 稼働確認（カタログ＝全モデル・pid・ログパス）
```bash
uv run local-llm-server --status
```

#### ゲートウェイ停止（配下のモデルサーバーも全て停止）
```bash
uv run local-llm-server --stop
```

`Ctrl+C` / `kill` でも、起動済みのモデルサーバーまで一緒に止まる（孫プロセスは残らない）。

## ライセンス

Apache-2.0
