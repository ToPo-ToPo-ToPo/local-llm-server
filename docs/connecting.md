# ゲートウェイへの接続

ゲートウェイを起動したら（→ [README](../README.md)）、クライアントは公開ポート
（既定 `http://127.0.0.1:8799/v1`）に繋ぎ、リクエストの `model` で使うモデルを選ぶ。
`model` は `gateway.toml` に登録済みのものを指定する（初回リクエストで遅延ロードされる）。
`api_key` はローカルなので任意（`"not-needed"` 等で可）。

## 付属の `LLMClient`（推奨）

公式 `openai` SDK（コア依存）を土台にした付属クライアント。`respond()` は非ストリームで
生成テキスト（`str`）、`stream=True` で断片の `Iterator[str]` を返す。

```python
from local_llm_server import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8799/v1")

print(llm.respond("ローカルLLMの利点を3つ。"))                 # 非ストリーム → str

for piece in llm.respond("もっと詳しく", stream=True):          # ストリーム → Iterator[str]
    print(piece, end="", flush=True)

llm.respond("これは何？", images=["plot.png"])                  # 画像（マルチモーダル）
```

主な引数: `model` / `base_url` / `api_key` / `temperature` / `max_tokens` / `timeout`。
`respond()` は `system_prompt` / `images` / `stream` のほか、追加の `**kwargs` を
`chat.completions.create` にそのまま渡す。

### 高度な操作（`llm.openai`）

`llm.openai` で土台の openai クライアントに直接アクセスできる。embeddings / tool calling /
構造化出力（`response_format`）/ async など、`respond()` に無い操作はこちらを使う。

```python
emb = llm.openai.embeddings.create(model="...", input="...")
```

## 他の OpenAI 互換クライアント

ゲートウェイは標準的な OpenAI 互換 API なので、`openai` SDK を直接使ったり、他言語の
クライアント、`curl` でもそのまま繋がる。

### `openai` SDK を直接

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8799/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="mlx-community/Qwen3.6-27B-4bit",
    messages=[{"role": "user", "content": "こんにちは"}],
)
print(resp.choices[0].message.content)
```

### `curl`

```bash
curl -s http://127.0.0.1:8799/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role": "user", "content": "俳句を一つ詠んでください。"}]
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## サーバーを使わず Python から直接（ゲートウェイ非経由）

ゲートウェイを立てずに、コードから単一モデルサーバーを起動して使うこともできる。

```python
from local_llm_server import connect

# サーバーが無ければ MTP 付きで自動起動 → 繋がった client を返す（相乗りも可）
llm = connect(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto")
print(llm.respond("こんにちは"))
llm.stop()   # 自動起動した場合のみ停止
```

`ensure_server()`（相乗り/自動起動）、`LocalServer` / `ServerConfig` / `ServerPool`
（サーバー制御）も公開している。
