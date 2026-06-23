# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を **OpenAI 互換 API** として
起動・管理する軽量サーバー。テキストと画像（vision）を自動で振り分ける
**router** モードを備え、任意の OpenAI 互換クライアントからそのまま利用できる。

- コア機能（プロセスの起動・監視・graceful shutdown・router プロキシ）は
  **標準ライブラリのみ**で動作。
- 実際の推論バックエンドは extras で導入（Apple Silicon では `mlx` を自動選択）。

## インストール

```bash
pip install local-llm-server          # コア（バックエンドは別途用意）
pip install "local-llm-server[mlx]"   # Apple Silicon 向け mlx / mlx-vlm 同梱
```

## 使い方

```bash
# テキストLLMを起動（既定バックエンドは環境に応じて自動選択）
local-llm-server --backend mlx

# 画像入力対応（vision）
local-llm-server --backend mlx-vlm

# テキストLLMとVLMを同時起動し、リクエスト内容で自動振り分け
local-llm-server --backend router
```

起動後、表示される `base_url`（例 `http://127.0.0.1:8080/v1`）を
OpenAI 互換クライアントに設定する。

```python
import local_llm_server as srv

config = srv.ServerConfig(model="mlx-community/Qwen3.6-27B-4bit", backend="mlx")
with srv.LocalServer(config) as server:
    server.wait_until_ready()
    print(srv.list_models(server.base_url))
```

## ライセンス

Apache-2.0
