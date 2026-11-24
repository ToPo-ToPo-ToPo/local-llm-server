# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp** / **vLLM** / **SGLang**）と音声認識
（**whisper** / STT）・音声合成（**mlx-audio** / TTS）・テキスト埋め込み（**embed**）・リランク（**rerank**。RAG）を束ねる**マルチモデルゲートウェイ**。1 プロセス起動するだけで、
1 つの公開ポートに複数モデルを配信する。
モデルの事前登録は不要——クライアントが指定した `model` をその場でロードする。
特徴の全体像は [docs/features.md](docs/features.md)。

## インストール

[uv](https://docs.astral.sh/uv/)（0.13.6 以上）を使う。Python は `.python-version` の版を uv が用意する。

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
uv sync --frozen
uv run --frozen gw enable    # ログイン時に自動で起動するようにする
uv run --frozen gw status
```

`gw enable` のあとは、どのディレクトリからでも `gw ...` で操作できる（`~/.local/bin/gw` にリンクを張る。
PATH に無ければ知らせる）。リポジトリの中で `uv run --frozen gw ...` と打っても同じ。以下は `gw ...` で書く。
環境の作り方・`uv sync` の使い分け・TLS の要件は [docs/operation.md](docs/operation.md)。

## 使い方

ゲートウェイは裏で常駐し、1 つのポート（既定 `http://127.0.0.1:8799/v1`）で OpenAI 互換の API を出す。
使う流れは「モデルを取得する → `model` に名前を書いて送る」の 2 つだけ。

### 1. ゲートウェイが動いているか確かめる

```bash
gw status     # running なら動いている。接続先の URL も出る
```

`gw enable` 済みならログインするたびに自動で起動する。止めていたときは `gw start` で起動する。

### 2. モデルを取得する（`gw pull`）

**ゲートウェイはモデルを自動ではダウンロードしない。** 使う前に一度だけ `gw pull` で取得する。

```bash
gw pull thinkingmachines/Inkling-2-Mini            # mlx で動かすモデル: リポジトリ全体を取得する
gw pull unsloth/Qwen3.6-27B-GGUF:Q4_K_XL           # GGUF: 「:量子化名」で 1 つを選ぶ
```

- 名前は Hugging Face のリポジトリ ID（`組織/リポジトリ`）。モデルのページの上に出ている名前をそのまま使う。
- 取得先は Hugging Face の共有キャッシュ（`~/.cache/huggingface/hub/`）。ほかのツールで取得したものも
  そのまま使える。
- GGUF のリポジトリは量子化ごとにファイルが分かれているので、使う 1 つ（と画像入力用の mmproj）だけを取る。
  `:量子化名` を付けずに複数あると、候補を並べて止まる。
- 途中で止めても、同じ `gw pull` をもう一度打てば続きから取得する。終わると、ゲートウェイが読めるかを
  確かめてから `done` と表示する。
- `gw pull` は取得するだけで、メモリには読み込まない。読み込むのは最初のリクエストのとき。
- 取得していないモデルを送ると、ゲートウェイは「ローカルキャッシュにありません」というエラーを返す。

### 3. `model` に名前を書いて送る

`model` には、取得したときと同じ名前を書く。最初のリクエストでモデルを読み込み、しばらく使わなければ
自動でメモリから外す（`idle_timeout`。既定は 20 分）。

```bash
curl http://127.0.0.1:8799/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "thinkingmachines/Inkling-2-Mini", "messages": [{"role": "user", "content": "こんにちは"}]}'
```

OpenAI の SDK や、OpenAI 互換の API に繋げるアプリは、接続先を `http://127.0.0.1:8799/v1` にする
（API キーは既定では不要。SDK が求めるなら適当な文字列でよい）。

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8799/v1", api_key="local")
r = client.chat.completions.create(
    model="thinkingmachines/Inkling-2-Mini",
    messages=[{"role": "user", "content": "ローカルLLMの利点を3つ。"}],
)
print(r.choices[0].message.content)
```

専用の接続クライアント [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client) も使える。

どの仕組みで動かすかは、名前から自動で決まる。名前に `gguf` を含めば llama.cpp、`mlx` を含めば mlx-vlm、
どちらも無ければ Apple Silicon の Mac では mlx-vlm（それ以外の PC では llama.cpp）。`whisper`（音声認識）や
埋め込み・リランクのモデルも名前で振り分ける（→ [docs/gateway.md](docs/gateway.md)）。

### 4. 様子を見る・管理する

```bash
gw list                      # 使えるモデルの一覧（取得済みのモデルも並ぶ）
gw show <モデル名>            # モデルの中身（量子化・大きさ・画像入力・読み込み状態など）
gw ps                        # いま読み込んでいるモデル
gw log -f                    # ゲートウェイのログを追いかける
gw stop / gw start / gw restart
gw update                    # 新しい版を取り込む（自動では入れ替わらない）
```

コマンドの一覧と細かい動きは [docs/operation.md](docs/operation.md#サブコマンド一覧)。

### 設定

設定は `~/.config/local-llm-server/gateway.toml` の 1 つだけ（初回の `gw start` が自動で作る）。
同時に読み込むモデルの数・使わないときに外すまでの時間・LAN への公開などを変えられる
（→ [docs/gateway.md](docs/gateway.md)）。たいていは変えなくてよい。更新で設定の項目が変わっても、
次の起動で自動で書き換える（元の内容は `gateway.toml.bak` に残す）。

**更新は手動**。新しい版が出るとメニューバーの `gw` に「⬆」が出るが、入れ替わるのは `gw update` を
打ったときだけ。

## モデルごとの注意

ほとんどのモデルは上の流れだけで使える。追加の注意があるものは、それぞれのドキュメントにまとめている。

- Inkling 2 Mini（`thinkingmachines/Inkling-2-Mini`）: 推奨の設定と、自動で入る高速化
  → [docs/llama-cpp.md](docs/llama-cpp.md#inkling-2-minimlx-vlm-と-llamacpp)
- GGUF（llama.cpp）: 量子化の選び方・画像入力・VRAM に収まらない MoE → [docs/llama-cpp.md](docs/llama-cpp.md)
- MTP（先読みで速くする仕組み）に対応するモデル: `gw pull` の最後に、追加で取るものが表示される
  → [docs/mtp.md](docs/mtp.md)
- 音声認識・音声合成・埋め込み・リランク → [docs/gateway.md](docs/gateway.md)

## ドキュメント

- [docs/features.md](docs/features.md) — 特徴と全体像
- [docs/operation.md](docs/operation.md) — 導入の詳細・`gw` のコマンド一覧・自動起動・アンインストール
- [docs/gateway.md](docs/gateway.md) — `gateway.toml` の全項目と振る舞い
- [docs/llama-cpp.md](docs/llama-cpp.md) — llama.cpp（`llama-server`）の自動導入・GGUF の取得と指定・VRAM に収まらない MoE の配置（`--moe-cache-mib`）・Inkling 2 Mini
- [docs/vllm.md](docs/vllm.md) — vLLM / SGLang（高スループット生成）
- [docs/mtp.md](docs/mtp.md) — MTP による高速化
- [experimental/strata_lite](experimental/strata_lite/README.md) — 実験記録: MoE のエキスパートを使用頻度で GPU/RAM に振り分ける試作エンジン（本体とは独立。結論: llama.cpp の `--moe-cache-mib` で十分 → docs/llama-cpp.md）

## ライセンス

Apache-2.0
