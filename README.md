# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）と音声認識（**whisper** / STT）を束ねる
**マルチモデルゲートウェイ**。1 プロセス起動するだけで、1 つの公開ポートに複数モデルを配信する。

- **モデルの事前登録は不要**。クライアントが指定した `model` をその場でロードする。画像入力（mmproj 自動検出）も mlx-vlm の MTP も設定なしで効く。
- **画像・動画入力**。画像はそのまま、**動画（`video_url`）はゲートウェイが ffmpeg で等間隔にフレーム抽出して**モデルへ渡す（llama-cpp / mlx-vlm 共通。ffmpeg は pip 同梱で追加インストール不要 → [動画入力](docs/gateway.md)）。
- **llama.cpp は自動導入**。Linux / Windows / Intel Mac では `llama-server` を OS・GPU 検出のうえ**起動時に自動ダウンロード**（手動導入不要。GPU は Vulkan、ソースビルドも opt-in で可 → [docs/llama-cpp.md](docs/llama-cpp.md)）。
- **vLLM / SGLang も選べる**（Linux/NVIDIA・Windows は WSL2）。`backend = "vllm"` または `"sglang"` で高スループット生成。重量級なので隔離 venv へ**起動時に自動導入**（明示 opt-in。SGLang は RadixAttention でエージェント用途に強い → [docs/vllm.md](docs/vllm.md)）。
- **音声認識（STT）も同じポートで**。`/v1/audio/transcriptions` に音声を投げれば mlx-whisper が遅延起動して文字起こしする。エージェント側に mlx 依存は要らない（→ [音声認識（STT / whisper）](docs/gateway.md#音声認識stt--whisper)）。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- **デーモンは裏で常駐、運用は `gw` の CLI サブコマンド**（Ollama 流）。`gw start` で常駐起動、`gw status`/`gw ps` で稼働確認、`gw stop` で停止。端末を占有しない（→ [起動・運用](docs/operation.md)）。
- **`gw list` が使えるモデルを自動一覧**。カタログに加え HF キャッシュの DL 済みモデルも未ロード候補として並ぶので、どれを指定すればよいか一目で分かる。
- モデルは**初回リクエスト時に遅延起動**、`max_resident`（数）/ `max_memory_fraction`（メモリ量）超過で LRU 退避、`idle_timeout` で自動アンロード。
- エージェントが「使い終わった」と通知すれば、在席が 0 になった瞬間に**待たず即アンロード**してメモリ解放（→ [在席ベースの即時アンロード](docs/gateway.md#在席ベースの即時アンロード)）。
- **別PCからも接続できる**。`host = "0.0.0.0"` で LAN に公開し、`api_key` で認証（→ [別PCから接続する](docs/gateway.md#別pcから接続するネットワーク公開)）。
- **自動更新**。clone 運用でも、常駐デーモンが PyPI 新版を検知して `git pull` で追従し新コードで再起動（作業ツリーがクリーンな時だけ・処理中/在席が空くのを待つ）。手動で今すぐなら `gw update`（→ [自動更新](docs/gateway.md#自動更新pypi-新版に-git-で追従)）。
- 接続側は公開ポートに繋いで `model` を選ぶだけ（接続クライアントは別パッケージ）。

## インストール

[uv](https://docs.astral.sh/uv/) を使う。このリポジトリをクローンして、ソースから動かすのが基本。

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server          # クローンしたフォルダの中で実行（uv add ではなく uv sync）
uv sync
```

以降このフォルダで `gateway.toml` を編集し、`uv run gw` で起動する（→ [使い方](#使い方)）。

> **他 OS（Linux / Windows / Intel Mac）＝ llama.cpp（追加インストール不要）**: `uv sync` は mlx を
> 入れずに済む。`llama-server` はゲートウェイ起動時に**自動でダウンロード・導入される**（OS・CPU
> アーキ・GPU を検出し、GPU なら Vulkan・無ければ CPU を選択。PATH は汚さない）。手動導入や PATH
> 設定は不要で、`uv run gw` して GGUF モデルの ID を投げるだけで動く。挙動の調整・ソースビルド・
> `system`（PATH の llama-server を使う）は `gateway.toml` の `[llama_cpp]` で
> （→ [docs/llama-cpp.md](docs/llama-cpp.md)）。

<details>
<summary>クローンせず PyPI の公開パッケージを使う場合（任意）</summary>

- **コマンドとして入れる**: `uv tool install local-llm-server` → どこでも `gw` で起動。
- **別プロジェクトの依存として入れる**: `gateway.toml` を置く新規フォルダで `uv init` → `uv add local-llm-server` → `uv run gw`。
</details>

## 使い方

### 1. サーバー設定を置く

カレントディレクトリに `gateway.toml` を置く。リポジトリ直下に例を同梱（→ [gateway.toml](gateway.toml)）。
モデルは列挙不要 —— 運用方針（ポートや同時常駐数など）だけ書けばよい:

```toml
host = "127.0.0.1"
port = 8799                 # クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数の上限（超過は LRU 退避）
# モデルは事前登録不要。クライアントが指定した model をその場でロードする。
# parallel や llama.cpp の MTP 等、個別の上書きが要るモデルだけ [[models]] に書く（→ docs/gateway.md）。
```

### 2. サーバー起動

クローンしたフォルダ（`gateway.toml` のある場所）で、デーモンを裏で常駐起動する。
端末は占有しない。稼働確認は `gw status` / `gw ps`、停止は `gw stop`（→ [起動・運用](docs/operation.md)）。

```bash
uv run gw start      # 裏で常駐起動（引数なしの `uv run gw` は start + 状態表示）
uv run gw status     # 稼働/停止・PID・URL・起動経過を表示
uv run gw ps         # ロード中モデルと処理中数
uv run gw list       # 使えるモデル一覧（カタログ＋HF キャッシュ）
```

### 3. このサーバーとの接続　（使用先で実施）

公開ポートに繋ぎ、`model` で使うモデルを選ぶ。接続クライアントは別パッケージ
[local-llm-client](https://pypi.org/project/local-llm-client/)

```bash
uv add local-llm-client
```
```python
from local_llm_client import LLMClient

llm = LLMClient(model="ToPo-ToPo/Qwen3.6-27B-mlx-4bit",
                base_url="http://127.0.0.1:8799/v1")
print(llm.respond("ローカルLLMの利点を3つ。"))
```

## ドキュメント

- [docs/gateway.md](docs/gateway.md) — `gateway.toml` の全フィールドと振る舞い
- [docs/llama-cpp.md](docs/llama-cpp.md) — llama.cpp（`llama-server`）の OS 別導入・最新モデル追従
- [docs/operation.md](docs/operation.md) — 起動・停止・監視（ターミナル）・アンインストール
- [docs/mtp.md](docs/mtp.md) — MTPによる高速化

## ライセンス

Apache-2.0
