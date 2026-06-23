# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を **OpenAI 互換 API** として
起動・管理する**拡張サーバーライブラリ**。素の OpenAI 互換サーバーの起動だけでなく、
複数のエージェントで重複しがちな処理（相乗り/自動起動・MTP 高速化・マルチモーダル
生成）を共通機能として備える。

- **LLM 実行** — `LocalServer` / `ServerConfig` / `ServerPool`（プロセス起動・監視・graceful shutdown）
- **ゲートウェイ** — `ensure_server()`（相乗り or 自動起動）, `RouterServer`（テキスト/vision 振り分け）
- **MTP（投機的デコード）** — `resolve_drafter` / `MTP_DRAFTERS`（本体の出力を変えず ~2倍速）
- **高レベルクライアント** — `LLMClient` / `connect()`（任意 extra）

コア（起動・ゲートウェイ・MTP 解決）は**標準ライブラリのみ**で動作し、推論バックエンド
と高レベルクライアントは extras で導入する。

## インストール（[uv](https://docs.astral.sh/uv/)）

```bash
uv add "local-llm-server[mlx]"            # サーバー + mlx バックエンド
uv add "local-llm-server[mlx,client]"     # + 高レベルクライアント(LLMClient/connect)
```

extras 指定はクォート必須（zsh の glob 展開回避）。内訳:

| extra | 入るもの | 用途 |
|---|---|---|
| （無し） | コアのみ（標準ライブラリ） | 起動・ゲートウェイ・MTP 解決だけ使う |
| `mlx` | `mlx-lm` / `mlx-vlm` | Apple Silicon で実際に推論する |
| `client` | `openai` | `LLMClient` / `connect()` で生成まで行う |

## クイックスタート

`[mlx,client]` を入れていれば、これだけで「サーバー用意 → 生成」が完結する:

```python
from local_llm_server import connect

# 既存サーバーがあれば相乗り、無ければ MTP 付きで自動起動 → 繋がった client を返す
llm = connect(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto")
print(llm.respond("俳句を一つ詠んでください。"))
llm.stop()   # connect が自動起動した場合のみ停止（相乗りなら無害）
```

---

使い方は欲しいレベルに応じて 3 段階。**A: コマンドで起動するだけ** → **B: Python の高レベル
API** → **C: 低レベル API で細かく制御**。

## A. CLI でサーバーを起動する

OpenAI 互換 API を `http://127.0.0.1:8080/v1` に立てるだけなら CLI で済む。

```bash
uv run local-llm-server --backend mlx        # テキスト専用（軽量）
uv run local-llm-server --backend mlx-vlm    # 画像入力対応（vision）
uv run local-llm-server --backend router     # テキストLLMとVLMを同時起動し自動振り分け
uv run local-llm-server --backend mlx-vlm --draft-model auto   # MTP で高速化
```

`--backend` 省略時は OS 自動判定（mac arm64 → `mlx-vlm`、他 → `llama-cpp`）。
起動後、表示される `base_url` を任意の OpenAI 互換クライアントに向ける。`curl` 例:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role": "user", "content": "俳句を一つ詠んでください。"}]
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

主な停止・運用は `--stop` / `--instances` / `--parallel`（→ `local-llm-server --help`）。

## B. 高レベル API（Python・要 `[client]`）

複数エージェントが各自書いていた「サーバー用意 + マルチモーダル生成 + ストリーミング」を
共通化したレイヤー。

### `connect()` — 用意から生成まで一括

```python
from local_llm_server import connect

llm = connect(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto", log=print)
try:
    print(llm.respond("ローカルLLMの利点を3つ。"))                 # 非ストリーム → str
    for piece in llm.respond("もっと詳しく", stream=True):          # ストリーム → Iterator[str]
        print(piece, end="", flush=True)
    llm.respond("これは何？", images=["plot.png"])                  # マルチモーダル
finally:
    llm.stop()
```

### `LLMClient` — サーバーは別途起動済みのとき

```python
from local_llm_server import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8080/v1")
print(llm.respond("こんにちは", system_prompt="簡潔に答えて"))
```

### `ensure_server()` — サーバーの用意だけ（クライアントは自前）

`connect()` の前半部分だけ。相乗り判定・モデル整合チェック・自動起動を行い、ハンドルを返す。

```python
from local_llm_server import ensure_server

handle = ensure_server(model="mlx-community/Qwen3.6-27B-4bit",
                       draft_model="auto", log=print)
try:
    base_url = handle.base_url        # ここに好きな OpenAI 互換クライアントを向ける
    for w in handle.warnings:         # モデル取り違えの警告があれば
        print("注意:", w)
finally:
    handle.stop()                     # 自動起動した分だけ停止（相乗りなら無害）
```

## C. 低レベル API（`LocalServer` を直接制御）

ポート・ライフサイクルを自分で握りたいとき。`ensure_server()` の相乗り判定を挟まず、
常に新しいプロセスを起動する。

```python
import local_llm_server as srv

config = srv.ServerConfig(backend="mlx-vlm", model="mlx-community/Qwen3.6-27B-4bit",
                          draft_model="auto")
with srv.LocalServer(config) as server:    # サブプロセス起動、with 離脱で自動停止
    server.wait_until_ready(timeout=600)
    print(srv.list_models(server.base_url))
    # server.base_url に openai 等で接続して生成
```

複数モデル/インスタンスを束ねる `ServerPool`、振り分けプロキシ `RouterServer` も公開。

## MTP（投機的デコード）

本体モデルの出力を変えずに ~2倍速にする高速化（Qwen3.6-27B で実測 38→75 tok/s、採択率
93%）。`draft_model="auto"` で本体名から対応ドラフターを自動選択する（`mlx-vlm` 限定。
対応表は `MTP_DRAFTERS`、解決は `resolve_drafter`）。

## examples

実機で動く完全なサンプル（Apple Silicon）:

| ファイル | 内容 |
|---|---|
| [`connect_and_generate.py`](examples/connect_and_generate.py) | 高レベル `connect()` で生成（最短） |
| [`generate_with_mtp.py`](examples/generate_with_mtp.py) | `LocalServer` + `openai` で MTP 生成（低レベル） |

```bash
uv run examples/connect_and_generate.py
```

詳細は [examples/README.md](examples/README.md)。

## ライセンス

Apache-2.0
