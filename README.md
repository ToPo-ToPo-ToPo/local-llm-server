# local-llm-server

複数のローカルモデルを、1 プロセス・1 つのポートの OpenAI 互換 API で配信するゲートウェイ。

- **対応:** チャット（mlx / mlx-vlm / llama.cpp / vLLM / SGLang）、音声認識（whisper）、音声合成（mlx-audio）、
  埋め込み（embed）、リランク（rerank。RAG の 2 段目）。
- **事前登録は不要:** クライアントが指定した `model` を、その場で読み込む。
- 特徴の全体像 → [docs/features.md](docs/features.md)

## インストール

- [uv](https://docs.astral.sh/uv/) 0.13.6 以上が要る。Python は `.python-version` のバージョンを uv が用意する。

1. リポジトリを取得する。

   ```bash
   git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
   ```

2. 取得したフォルダに入る。

   ```bash
   cd local-llm-server
   ```

3. 環境を作る。

   ```bash
   uv sync --frozen
   ```

4. ログイン時に自動で起動するようにする。

   ```bash
   uv run --frozen gw enable
   ```

5. 動いているか確かめる。

   ```bash
   uv run --frozen gw status
   ```

- `gw enable` のあとは、どのディレクトリからでも `gw ...` で操作できる（`~/.local/bin/gw` にリンクを張る。
  PATH に無ければ知らせる）。
- リポジトリの中で `uv run --frozen gw ...` と打っても同じ。以下は `gw ...` で書く。
- 環境の作り方・`uv sync` の使い分け・TLS の要件 → [docs/operation.md](docs/operation.md)

## 使い方

流れは「`gw pull` でモデルを取得する → `model` に名前を書いて送る」だけ。接続先は `http://127.0.0.1:8799/v1`（既定）。

### 1. 動いているか確かめる

```bash
gw status
```

- `running` なら動いている。接続先の URL も出る。
- `gw enable` 済みなら、ログインのたびに自動で起動する。
- 止めていたときは次で起動する。

  ```bash
  gw start
  ```

### 2. モデルを取得する（`gw pull`）

mlx で動かすモデル（リポジトリ全体を取得）:

```bash
gw pull thinkingmachines/Inkling-2-Mini
```

GGUF（「:量子化名」で 1 つを選ぶ）:

```bash
gw pull unsloth/Qwen3.6-27B-GGUF:Q4_K_XL
```

- **自動ではダウンロードしない。** 使う前に一度だけ取得する。取得していないモデルを送ると
  「ローカルキャッシュにありません」というエラーが返る。
- **名前:** Hugging Face のリポジトリ ID（`組織/リポジトリ`）。モデルのページの上に出ている名前をそのまま使う。
- **置き場所:** Hugging Face の共有キャッシュ（`~/.cache/huggingface/hub/`）。ほかのツールで取得したものもそのまま使える。
- **GGUF:** 使う量子化 1 つと、画像入力用の mmproj だけを取る。`:量子化名` を付けずに量子化が複数あると、
  候補を並べて止まる。
- **中断:** 同じ `gw pull` をもう一度打てば、続きから取得する。
- **完了:** ゲートウェイが読めるかを確かめてから `done` と表示する。
- **読み込みはしない:** メモリに読み込むのは、最初のリクエストのとき。

### 3. `model` に名前を書いて送る

curl で送る:

```bash
curl http://127.0.0.1:8799/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "thinkingmachines/Inkling-2-Mini", "messages": [{"role": "user", "content": "こんにちは"}]}'
```

Python（OpenAI の SDK）で送る:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8799/v1", api_key="local")
r = client.chat.completions.create(
    model="thinkingmachines/Inkling-2-Mini",
    messages=[{"role": "user", "content": "ローカルLLMの利点を3つ。"}],
)
print(r.choices[0].message.content)
```

- `model` には、取得したときと同じ名前を書く。
- 最初のリクエストで読み込み、しばらく使わなければ自動でメモリから外す（`idle_timeout`。既定は 20 分）。
- OpenAI の SDK や OpenAI 互換のアプリは、接続先を `http://127.0.0.1:8799/v1` にする。
- API キーは既定では不要。SDK が求めるなら適当な文字列でよい。
- 専用の接続クライアントもある → [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client)
- **動かす仕組みは名前で決まる:**
  - `gguf` を含む → llama.cpp
  - `mlx` を含む → mlx-vlm
  - どちらも無い → Apple Silicon の Mac では mlx-vlm、それ以外の PC では llama.cpp
  - 音声認識（`whisper` など）・埋め込み・リランクも名前で振り分ける → [docs/gateway.md](docs/gateway.md)

### 4. 様子を見る・管理する

使えるモデルの一覧（取得済みのモデルも並ぶ）:

```bash
gw list
```

モデルの中身（量子化・大きさ・画像入力・読み込み状態など。名前は調べたいモデルに変える）:

```bash
gw show ToPo-ToPo/gemma-4-26B-A4B-it-mlx-4bit
```

いま読み込んでいるモデル:

```bash
gw ps
```

ゲートウェイのログを追いかける（Ctrl-C で止める）:

```bash
gw log -f
```

止める:

```bash
gw stop
```

起動する:

```bash
gw start
```

再起動する:

```bash
gw restart
```

新しいバージョンを取り込む:

```bash
gw update
```

- コマンドの一覧と細かい動き → [docs/operation.md](docs/operation.md#サブコマンド一覧)
- **更新は手動:** 新しいバージョンが出るとメニューバーの `gw` に「⬆」が出る。入れ替わるのは `gw update` を打ったときだけ。

### 設定

- 設定は `~/.config/local-llm-server/gateway.toml` の 1 つだけ。初回の `gw start` が自動で作る。
- 同時に読み込むモデルの数・使わないときに外すまでの時間・LAN への公開などを変えられる。たいていは変えなくてよい
  → [docs/gateway.md](docs/gateway.md)
- 更新で設定の項目が変わっても、次の起動で自動で書き換える（元の内容は `gateway.toml.bak` に残す）。

## モデルごとの注意

ほとんどのモデルは上の流れだけで使える。追加の注意があるもの:

- **Inkling 2 Mini**（`thinkingmachines/Inkling-2-Mini`）: 推奨の設定と、自動で入る高速化
  → [docs/llama-cpp.md](docs/llama-cpp.md#inkling-2-minimlx-vlm-と-llamacpp)
- **GGUF（llama.cpp）:** 量子化の選び方・画像入力・VRAM に収まらない MoE → [docs/llama-cpp.md](docs/llama-cpp.md)
- **MTP（先読みで速くする仕組み）に対応するモデル:** `gw pull` の最後に、追加で取るものが表示される
  → [docs/mtp.md](docs/mtp.md)
- **音声認識・音声合成・埋め込み・リランク** → [docs/gateway.md](docs/gateway.md)

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
