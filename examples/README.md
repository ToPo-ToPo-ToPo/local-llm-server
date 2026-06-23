# examples

実LLMを使った動作サンプル。Apple Silicon（mlx-vlm バックエンド）向け。

## generate_with_mtp.py — LLM + MTP で生成する

local-automata が既定で使っていた構成（**Qwen3.6-27B-4bit ＋ MTP ドラフター**による
投機的デコード）をそのまま単体で再現するサンプル。サーバーの起動・待機・停止を
`LocalServer` に任せ、生成は `openai` クライアントで行う。

```bash
uv run examples/generate_with_mtp.py
```

[PEP 723](https://peps.python.org/pep-0723/) のインライン依存を埋め込んであるので、
`uv run` するだけで `local-llm-server[mlx]` と `openai` を含む一時環境が用意される。
初回は本体（`Qwen3.6-27B-4bit`）とドラフター（`Qwen3.6-27B-MTP-4bit`）の2モデルが
自動ダウンロードされる（数GB）。

### MTP（Multi-Token Prediction）とは

本体モデルの出力を**変えずに**推論を高速化する投機的デコード。`draft_model="auto"` を
渡すと本体名から対応ドラフターが自動選択される（`mlx-vlm` バックエンド限定）。
Qwen3.6-27B では実測 ~2倍速（38→75 tok/s、採択率 93%）。

### CLI で同じことをする

Python を介さず、サーバーを直接起動しても同じ構成になる。

```bash
# 本体 + MTP ドラフターで起動（draft-model auto は本体名から自動解決）
uv run local-llm-server --backend mlx-vlm \
  --model mlx-community/Qwen3.6-27B-4bit \
  --draft-model auto

# 別ターミナルから生成
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role": "user", "content": "俳句を一つ詠んでください。"}]
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```
