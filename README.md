# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp** / **vLLM** / **SGLang**）と音声認識
（**whisper** / STT）・音声合成（**mlx-audio** / TTS）・テキスト埋め込み（**embed**）・リランク（**rerank**。RAG）を束ねる**マルチモデルゲートウェイ**。1 プロセス起動するだけで、
1 つの公開ポートに複数モデルを配信する。
モデルの事前登録は不要——クライアントが指定した `model` をその場でロードする。
特徴の全体像は [docs/features.md](docs/features.md)。

## インストール

[uv](https://docs.astral.sh/uv/) を使う。一度入れれば**どこからでも `gw`**。

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
make install    # 専用環境を作成し、アプリの起動参照と自動起動を登録する
```

Pythonは3.13.18、uvは0.13.6以上を使用する。`scripts/uv`はリポジトリ専用の
ツール・Python・キャッシュを優先し、シェル設定やPC全体のPATHを書き換えない。
`make install`は新しいtool環境を別ディレクトリに作成し、既存環境を保存して
`~/.local/bin/gw`とアプリ専用サービスを切り替える。PATH未設定ならこの絶対パスで実行する。
既存インストールの依存版を引き継ぎ、セキュリティ更新が必要なパッケージだけを検証済み版へ上げる。
更新済み・より新しい版は維持し、任意のバックエンドも消さない。切り替え前に新環境を検証する。
導入の詳細とTLS要件は[運用手順](docs/operation.md#tlsと実行環境)を参照。

モデルの取得には`gw pull`または`./scripts/hf download`を使う。
修正済みhf-xetの動作を確認するまではPython HTTP経路を使用するため、
大容量ファイルの転送速度・帯域・キャッシュ効率に影響する場合がある。

## 使い方

```bash
gw status     # 稼働/停止・PID・URL・起動経過を表示（自動起動済みなら常に稼働）
gw list       # 使えるモデル一覧（カタログ＋HF キャッシュ）
gw stop       # 停止（再開は gw start か次回ログイン）
gw start      # 手動起動（初回は設定を自動生成。自動起動を無効にした場合の入口）
gw update     # 更新を取り込む（新版の確認は自動、適用はこのコマンドだけ）
```

設定は **`~/.config/local-llm-server/gateway.toml` の 1 箇所**（初回の `gw start` が自動生成。
→ [docs/gateway.md](docs/gateway.md)）。更新で設定キーが廃止・改名されても**手で直す必要はない**
——次の起動時に自動で追従する（変更点はログに出し、元の内容は `gateway.toml.bak` に残す）。

**更新は手動**。稼働中のデーモンは新しいリリースを見つけるとメニューバーの `gw` に「⬆」を出すが、
適用するのは `gw update` を実行したときだけで、放置しても勝手に入れ替わらない
（→ [docs/operation.md](docs/operation.md)）。

接続は公開ポート（既定 `http://127.0.0.1:8799/v1`）に繋いで `model` を選ぶだけ。
接続クライアントは別パッケージ [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client):

```python
from local_llm_client import LLMClient

llm = LLMClient(model="ToPo-ToPo/Qwen3.6-27B-mlx-4bit",
                base_url="http://127.0.0.1:8799/v1")
print(llm.respond("ローカルLLMの利点を3つ。"))
```

## ドキュメント

- [docs/features.md](docs/features.md) — 特徴と全体像
- [docs/operation.md](docs/operation.md) — `gw` サブコマンドでの起動・停止・状態確認・アンインストール
- [docs/gateway.md](docs/gateway.md) — `gateway.toml` の全フィールドと振る舞い
- [docs/llama-cpp.md](docs/llama-cpp.md) — llama.cpp（`llama-server`）の自動導入・最新モデル追従・VRAM に収まらない MoE の配置（`--moe-cache-mib`）
- [docs/vllm.md](docs/vllm.md) — vLLM / SGLang（高スループット生成）
- [docs/mtp.md](docs/mtp.md) — MTP による高速化
- [experimental/strata_lite](experimental/strata_lite/README.md) — 実験記録: MoE のエキスパートを使用頻度で GPU/RAM に振り分ける試作エンジン（本体とは独立。結論: llama.cpp の `--moe-cache-mib` で十分 → docs/llama-cpp.md）

## ライセンス

Apache-2.0
