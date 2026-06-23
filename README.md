# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を **OpenAI 互換 API** として
立てるサーバー。Ollama と同じイメージで、**サーバーを 1 つ起動して、あとは OpenAI 互換
API を叩くだけ**。テキスト/画像を自動で振り分ける router、本体の出力を変えず高速化する
MTP（投機的デコード）を備える。

- 既定モデル `Qwen3.6-27B-4bit` はマルチモーダル。mac (Apple Silicon) では**何もせず
  画像入力までそのまま動く**（既定バックエンド `mlx-vlm`）。
- コア（サーバー起動・router・MTP 解決）は**標準ライブラリのみ**で動作。

## インストール（[uv](https://docs.astral.sh/uv/)）

```bash
uv add "local-llm-server[mlx]"
```

extras 指定はクォート必須（zsh の glob 展開回避）。内訳:

| extra | 入るもの | 用途 |
|---|---|---|
| （無し） | コアのみ（標準ライブラリ） | 起動・router・MTP 解決・組み込みクライアント |
| `mlx` | `mlx-lm` / `mlx-vlm` | Apple Silicon で実際に推論する |

ライブラリ機能（`connect` / `LLMClient` / `ensure_server` …）はすべて**標準ライブラリ
のみ**で動く。追加の client 用 extra は不要。

## 使い方

### 1. サーバーを起動する

別ターミナルで起動しておく（初回はモデルを自動ダウンロード）。

```bash
uv run local-llm-server                                  # 既定（mac: mlx-vlm + 既定モデル）
uv run local-llm-server --backend mlx                    # テキスト専用で軽く（画像不可）
uv run local-llm-server --draft-model auto               # MTP で高速化（~2倍速）
uv run local-llm-server --backend router                 # テキストLLMとVLMを同時起動し自動振り分け
uv run local-llm-server --backend llama-cpp --model /path/to/model.gguf
```

- `--backend` 省略時は OS 自動判定（mac arm64 → `mlx-vlm`、他 → `llama-cpp`）。
- `--host` / `--port`（既定 `127.0.0.1:8080`）。`--` 以降はバックエンド固有引数。
- 並列: llama.cpp は `--parallel N`、mlx 系は `--instances N`（連番ポート）。

### 2. 接続する（OpenAI 互換 API）

起動後、`http://127.0.0.1:8080/v1` に任意の OpenAI 互換クライアントを向けるだけ
（`api_key` はローカルなので任意）。

`curl`:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role": "user", "content": "俳句を一つ詠んでください。"}]
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Python（組み込みの `LLMClient`。追加依存なし）:

```python
from local_llm_server import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8080/v1")
print(llm.respond("ローカルLLMの利点を3つ。"))                # 非ストリーム → str
for piece in llm.respond("もっと詳しく", stream=True):         # ストリーム → Iterator[str]
    print(piece, end="", flush=True)
```

`openai` など他の OpenAI 互換クライアントでも同じ base_url にそのまま繋がる。

### 3. 停止する

起動した端末で **`Ctrl+C`**。別端末からは `--stop`（起動時と同じ `--port` / `--instances`
/ `--backend router` を指定）:

```bash
uv run local-llm-server --stop
```

`kill $(lsof -ti tcp:8080)` でも止まる。子プロセス（バックエンド・MTP ドラフター）も
一緒に停止する。

## （任意）Python から自動起動する

「サーバーが無ければ自分で起動し、終了時に止める」を 1 呼び出しで済ませたいとき。
上記の手順 1〜3 を内包する利便機能（追加依存なし）。

```python
from local_llm_server import connect

# 既存サーバーがあれば相乗り、無ければ MTP 付きで自動起動 → 繋がった client を返す
llm = connect(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto")
print(llm.respond("こんにちは"))
llm.stop()   # 自動起動した場合のみ停止（相乗りなら無害）
```

サーバーの用意だけ行う `ensure_server()`、サーバー制御の `LocalServer` / `ServerConfig` /
`ServerPool` も公開している（→ [examples/](examples/)）。

## MTP（投機的デコード）

本体モデルの出力を変えずに ~2倍速にする高速化（Qwen3.6-27B で実測 38→75 tok/s、採択率
93%）。起動時に `--draft-model auto`（または `connect(draft_model="auto")`）を渡すと、本体名
から対応ドラフターを自動選択する（`mlx-vlm` 限定。対応表 `MTP_DRAFTERS` / 解決 `resolve_drafter`）。

## examples

実機で動く完全なサンプル（Apple Silicon）。`uv run` するだけで動く:

```bash
uv run examples/connect_and_generate.py    # 自動起動 + 生成（最短）
uv run examples/generate_with_mtp.py       # LocalServer + openai で MTP 生成 + tok/s 表示
```

詳細は [examples/README.md](examples/README.md)。

## ライセンス

Apache-2.0
