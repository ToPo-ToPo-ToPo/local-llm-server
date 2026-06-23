# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を **OpenAI 互換 API** として
起動・管理する軽量サーバー。テキストと画像（vision）を自動で振り分ける
**router** モードを備え、任意の OpenAI 互換クライアントからそのまま利用できる。

- コア機能（プロセスの起動・監視・graceful shutdown・router プロキシ）は
  **標準ライブラリのみ**で動作。
- 実際の推論バックエンドは extras で導入（Apple Silicon では `mlx` を自動選択）。

## インストール（[uv](https://docs.astral.sh/uv/)）

用途に応じて使い分ける。extras 指定はクォート必須（zsh の glob 展開回避）。

```bash
# CLI ツールとして常用する（どこからでも local-llm-server コマンドが使える）
uv tool install "local-llm-server[mlx]"

# 自分の uv プロジェクトの依存に追加する（ライブラリとして使う）
uv add "local-llm-server[mlx]"

# インストールせず一度きり試す
uvx --from "local-llm-server[mlx]" local-llm-server --backend mlx
```

`[mlx]` は Apple Silicon 向けの mlx / mlx-vlm バックエンドを同梱する extras。
extras 無し（`local-llm-server`）の場合、コアは標準ライブラリのみで動き、
推論バックエンド（llama.cpp など）は別途用意する。

## 使い方

```bash
# テキストLLMを起動（既定バックエンドは環境に応じて自動選択）
local-llm-server --backend mlx

# 画像入力対応（vision）
local-llm-server --backend mlx-vlm

# テキストLLMとVLMを同時起動し、リクエスト内容で自動振り分け
local-llm-server --backend router
```

> `uv add` でプロジェクト依存に入れた場合は `uv run local-llm-server ...` で起動する。
> `uv tool install` で入れた場合は上記のとおり `local-llm-server` を直接呼べる。

起動後、表示される `base_url`（例 `http://127.0.0.1:8080/v1`）を
OpenAI 互換クライアントに設定する。

## 実LLMで生成する

### 1. CLI で起動 ＋ curl で生成

別ターミナルでサーバーを起動しておく（初回はモデルを自動ダウンロード）。

```bash
local-llm-server --backend mlx
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

## ライセンス

Apache-2.0
