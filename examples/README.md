# examples

実LLMを使った動作サンプル。Apple Silicon（mlx-vlm バックエンド）向け。
どちらも [PEP 723](https://peps.python.org/pep-0723/) のインライン依存を埋め込んであるので
`uv run` するだけで動く（初回は本体 `Qwen3.6-27B-4bit` と MTP ドラフター
`Qwen3.6-27B-MTP-4bit` を自動ダウンロード、数GB）。

| ファイル | レベル | 内容 |
|---|---|---|
| `connect_and_generate.py` | 高レベル | `connect()` 1 呼び出しで「サーバー用意 → ストリーミング生成」 |
| `generate_with_mtp.py` | 低レベル | `LocalServer` + `openai` を手書きし、tok/s も表示 |

## connect_and_generate.py — 高レベル API で生成する（最短）

```bash
uv run examples/connect_and_generate.py
```

`connect(model=..., draft_model="auto")` が「既存サーバーに相乗り or MTP 付きで自動起動」
してから生成する。最も短い書き方。

## generate_with_mtp.py — LLM + MTP で生成する（低レベル）

サーバーの起動・待機・停止を `LocalServer` に任せ、生成は `openai` クライアントで行う。
速度（tok/s）も表示して MTP の効果を確認できる。

```bash
uv run examples/generate_with_mtp.py
```

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
