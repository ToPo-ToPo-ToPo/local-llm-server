# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）と音声認識（**whisper** / STT）を束ねる
**マルチモデルゲートウェイ**。1 プロセス起動するだけで、1 つの公開ポートに複数モデルを配信する。

- **モデルの事前登録は不要**。クライアントが指定した `model` をその場でロードする。画像入力（mmproj 自動検出）も mlx-vlm の MTP も設定なしで効く。
- **画像・動画入力**。画像はそのまま、**動画（`video_url`）はゲートウェイが ffmpeg で等間隔にフレーム抽出して**モデルへ渡す（llama-cpp / mlx-vlm 共通。ffmpeg は pip 同梱で追加インストール不要 → [動画入力](docs/gateway.md)）。
- **llama.cpp は自動導入**。Linux / Windows / Intel Mac では `llama-server` を OS・GPU 検出のうえ**起動時に自動ダウンロード**（手動導入不要。GPU は Vulkan、ソースビルドも opt-in で可 → [docs/llama-cpp.md](docs/llama-cpp.md)）。
- **vLLM / SGLang も選べる**（Linux/NVIDIA・Windows は WSL2）。`backend = "vllm"` または `"sglang"` で高スループット生成。重量級なので隔離 venv へ**起動時に自動導入**（明示 opt-in。SGLang は RadixAttention でエージェント用途に強い → [docs/vllm.md](docs/vllm.md)）。
- **音声認識（STT）も同じポートで**。`/v1/audio/transcriptions` に音声を投げれば mlx-whisper が遅延起動して文字起こしする。エージェント側に mlx 依存は要らない（→ [音声認識（STT / whisper）](docs/gateway.md#音声認識stt--whisper)）。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- **デーモンは裏で常駐、運用は `gw` の CLI サブコマンド**（Ollama 流）。`gw start` で常駐起動、`gw status`/`gw ps` で稼働確認、`gw stop` で停止。端末を占有しない。`status`/`stop` 等は **`gateway.toml` の無い場所からでも**唯一のデーモンを見つけて叩ける（→ [起動・運用](docs/operation.md)）。
- **`gw list` が使えるモデルを自動一覧**。カタログに加え HF キャッシュの DL 済みモデルも未ロード候補として並ぶので、どれを指定すればよいか一目で分かる。
- モデルは**初回リクエスト時に遅延起動**、`max_resident`（数）/ `max_memory_fraction`（メモリ量）超過で LRU 退避、`idle_timeout` で自動アンロード。
- エージェントが「使い終わった」と通知すれば、在席が 0 になった瞬間に**待たず即アンロード**してメモリ解放（→ [在席ベースの即時アンロード](docs/gateway.md#在席ベースの即時アンロード)）。
- **別PCからも接続できる**。`host = "0.0.0.0"` で LAN に公開し、`api_key` で認証（→ [別PCから接続する](docs/gateway.md#別pcから接続するネットワーク公開)）。
- **自動更新**。clone 運用でも、常駐デーモンが PyPI 新版を検知して `git pull` で追従し新コードで再起動（作業ツリーがクリーンな時だけ・処理中/在席が空くのを待つ）。手動で今すぐなら `gw update`（→ [自動更新](docs/gateway.md#自動更新pypi-新版に-git-で追従)）。
- 接続側は公開ポートに繋いで `model` を選ぶだけ（接続クライアントは別パッケージ）。

## インストール

[uv](https://docs.astral.sh/uv/) を使う。クローンして **`gw` コマンドをインストール**する（Ollama 流に、
一度入れれば**どこからでも `gw`**）。`--editable` なので自動更新（git 追従）もそのまま効く。

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
make install                     # `gw` を PATH に導入（~/.local/bin/gw）。以後どこでも `gw`
```

`make install` は `uv tool install --editable . --reinstall` を実行する（editable は確定なので畳んである）。
再実行すれば入れ直し（依存が変わったときの更新）にもなる。以降は**どのディレクトリからでも**
`gw start` で起動する（設定は初回に自動生成 → [使い方](#使い方)）。`~/.local/bin` が PATH に無いと
言われたら `uv tool update-shell` を一度実行する。

> **他 OS（Linux / Windows / Intel Mac）＝ llama.cpp（追加インストール不要）**: mlx は入らず、
> `llama-server` はゲートウェイ起動時に**自動でダウンロード・導入される**（OS・CPU アーキ・GPU を検出し、
> GPU なら Vulkan・無ければ CPU を選択。PATH は汚さない）。手動導入や PATH 設定は不要で、`gw start`
> して GGUF モデルの ID を投げるだけで動く。挙動の調整・ソースビルド・`system`（PATH の llama-server
> を使う）は `gateway.toml` の `[llama_cpp]` で（→ [docs/llama-cpp.md](docs/llama-cpp.md)）。

<details>
<summary>開発する（テストを回す）場合</summary>

`make dev`（＝`uv sync`）で開発用 venv が作られ、`uv run pytest` でテストできる。
`make install` と併用してよい（ソースは共有され、`gw` は編集が即反映される）。
</details>

## 使い方

### 1. サーバー起動

`gw start` の一発だけ。設定ファイルは **`~/.config/local-llm-server/gateway.toml` の 1 箇所**で、
無ければ初回の `gw start` が自動生成する（クローンの例 [gateway.toml](gateway.toml) を複製）。
デーモンは裏で常駐し、端末は占有しない。**どのディレクトリから打っても**同じ 1 つのデーモンに届く
（→ [起動・運用](docs/operation.md)）。

```bash
gw start      # 裏で常駐起動（初回は設定を自動生成）
gw status     # 稼働/停止・PID・URL・起動経過を表示
gw ps         # ロード中モデルと処理中数
gw list       # 使えるモデル一覧（カタログ＋HF キャッシュ）
gw stop       # 停止
```

### 2. サーバー設定（必要なときだけ）

`~/.config/local-llm-server/gateway.toml` を編集する。モデルは列挙不要 —— クライアントが指定した
`model` をその場でロードする。保存すればポリシー設定は稼働中でも反映される（→ [docs/gateway.md](docs/gateway.md)）。

```toml
host = "127.0.0.1"
port = 8799                 # クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数の上限（超過は LRU 退避）
# モデルは事前登録不要。クライアントが指定した model をその場でロードする。
# parallel や llama.cpp の MTP 等、個別の上書きが要るモデルだけ [[models]] に書く（→ docs/gateway.md）。
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
