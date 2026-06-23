# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を **OpenAI 互換 API** として
起動・管理する**拡張サーバーライブラリ**。素の OpenAI 互換サーバーの起動だけでなく、
複数のエージェントで重複しがちな処理（相乗り/自動起動・MTP 高速化・マルチモーダル
生成）を共通機能として備える。

- **LLM 実行** — `LocalServer` / `ServerConfig` / `ServerPool`（プロセス起動・監視・graceful shutdown）
- **ゲートウェイ** — `RouterServer`（テキスト/vision 振り分け）, `ensure_server()`（相乗り or 自動起動）
- **MTP（投機的デコード）** — `resolve_drafter` / `MTP_DRAFTERS`（本体の出力を変えず ~2倍速）
- **高レベルクライアント** — `LLMClient` / `connect()`（任意 extra）

コア（起動・ゲートウェイ・MTP 解決）は**標準ライブラリのみ**で動作し、推論バックエンド
と高レベルクライアントは extras で導入する。

## インストール（[uv](https://docs.astral.sh/uv/)）

uv プロジェクトの依存に追加する（extras 指定はクォート必須＝zsh の glob 展開回避）。

```bash
uv add "local-llm-server[mlx]"            # サーバー + mlx バックエンド
uv add "local-llm-server[mlx,client]"     # + 高レベルクライアント(LLMClient/connect)
```

extras の内訳:

| extra | 入るもの | 用途 |
|---|---|---|
| （無し） | コアのみ（標準ライブラリ） | 起動・ゲートウェイ・MTP 解決だけ使う |
| `mlx` | `mlx-lm` / `mlx-vlm` | Apple Silicon で実際に推論する |
| `client` | `openai` | `LLMClient` / `connect()` で生成まで行う |

## 高レベル API（重複処理の共通化）

複数エージェントが各自書いていた「サーバーを用意して生成する」を 1 呼び出しに:

```python
# uv add "local-llm-server[mlx,client]"
from local_llm_server import connect

# 既存サーバーがあれば相乗り、無ければ MTP 付きで自動起動 → 繋がった client を返す
llm = connect(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto")
print(llm.respond("俳句を一つ詠んでください。"))           # 非ストリーム
for piece in llm.respond("利点は？", stream=True):          # ストリーム
    print(piece, end="", flush=True)
llm.stop()   # connect が自動起動した場合のみ停止（相乗りなら無害）
```

サーバーの用意だけ（クライアントは自前）なら `ensure_server()`:

```python
from local_llm_server import ensure_server

handle = ensure_server(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto",
                       log=print)        # 相乗り判定・モデル整合チェック・自動起動
try:
    base_url = handle.base_url           # ここに OpenAI 互換クライアントを向ける
    for w in handle.warnings:            # 取り違え（モデル不一致）の警告があれば
        ...
finally:
    handle.stop()                        # 自動起動した分だけ停止

## 使い方

```bash
# テキストLLMを起動（既定バックエンドは環境に応じて自動選択）
uv run local-llm-server --backend mlx

# 画像入力対応（vision）
uv run local-llm-server --backend mlx-vlm

# テキストLLMとVLMを同時起動し、リクエスト内容で自動振り分け
uv run local-llm-server --backend router
```

起動後、表示される `base_url`（例 `http://127.0.0.1:8080/v1`）を
OpenAI 互換クライアントに設定する。

## 実LLMで生成する

### 1. CLI で起動 ＋ curl で生成

別ターミナルでサーバーを起動しておく（初回はモデルを自動ダウンロード）。

```bash
uv run local-llm-server --backend mlx
```

OpenAI 互換の `/v1/chat/completions` をそのまま叩ける（`api_key` はローカルなので任意）。

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role": "user", "content": "俳句を一つ詠んでください。"}]
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

### 2. Python から起動 ＋ openai クライアントで生成

サーバーの起動・待機・停止を `LocalServer`（context manager）に任せ、
生成は標準的な `openai` クライアントで行う。

```python
# uv add "local-llm-server[mlx]" openai
import local_llm_server as srv
from openai import OpenAI

config = srv.ServerConfig(backend="mlx", model="mlx-community/Qwen3.6-27B-4bit")

with srv.LocalServer(config) as server:        # サブプロセスで起動
    server.wait_until_ready(timeout=120)        # /v1/models が応答するまで待つ（初回DL含む）

    client = OpenAI(base_url=server.base_url, api_key="not-needed")
    resp = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": "ローカルLLMの利点を3つ、箇条書きで。"}],
    )
    print(resp.choices[0].message.content)
# with を抜けるとサーバーは自動停止する
```

ストリーミングしたい場合は `stream=True` を渡し、`for chunk in resp:` で
`chunk.choices[0].delta.content` を逐次受け取る（OpenAI クライアントと同じ作法）。

### MTP（投機的デコード）で高速化する

本体モデルの出力を変えずに ~2倍速にする MTP（Multi-Token Prediction）を使う
完全な実行サンプルを [`examples/`](examples/) に用意している（`uv run examples/generate_with_mtp.py`）。
詳しくは [examples/README.md](examples/README.md) を参照。

## ライセンス

Apache-2.0
