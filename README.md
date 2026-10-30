# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp** / **vLLM** / **SGLang**）と音声認識
（**whisper** / STT）・音声合成（**mlx-audio** / TTS）・テキスト埋め込み（**embed**）・リランク（**rerank**。RAG）を束ねる**マルチモデルゲートウェイ**。1 プロセス起動するだけで、
1 つの公開ポートに複数モデルを配信する。
モデルの事前登録は不要——クライアントが指定した `model` をその場でロードする。
特徴の全体像は [docs/features.md](docs/features.md)。

## インストール

[uv](https://docs.astral.sh/uv/) の標準プロジェクト環境を使う。

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
uv sync --frozen
uv run --frozen gw enable    # 自動起動を使う場合
uv run --frozen gw status
```

Pythonは3.13.18、uvは0.13.6以上を使用する。`.python-version`と`uv.lock`に従い、
uvがリポジトリ直下に実体の`.venv`を作成する。Python実行ファイルなどの内部構造はuv標準のまま。
通常の実行は`uv run --frozen gw ...`。自動起動もこの`.venv`の実行ファイルを直接使う。
`make install`は`uv sync --frozen`と`uv run --frozen gw enable`の補助にすぎず、
独自の世代環境やグローバルコマンドのシンボリックリンクは作成しない。
更新時は同じ`.venv`をロックされた依存に同期する。シェル設定やPC全体のPATHは変更しない。

**利用する場合は`uv sync --frozen`を推奨**する。公開版に同梱された`uv.lock`を更新せず、
その固定内容で環境を同期する指定であり、uv標準のオプション。各実行に対する指定で、永続的な設定ではない。
依存を追加・変更する開発作業では`uv sync`を使い、必要なロック変更も反映する。
設定と`uv.lock`が一致していれば、どちらでも同じ固定内容が使われる。
詳しい使い分けは[運用手順](docs/operation.md#uv-syncの使い分け)を参照。
導入の詳細とTLS要件は[運用手順](docs/operation.md#tlsと実行環境)を参照。

モデルの取得には`uv run --frozen gw pull`または`./scripts/hf download`を使う。
修正済みhf-xetの動作を確認するまではPython HTTP経路を使用するため、
大容量ファイルの転送速度・帯域・キャッシュ効率に影響する場合がある。

## 使い方

```bash
uv run --frozen gw status     # 稼働/停止・PID・URL・起動経過を表示（自動起動済みなら常に稼働）
uv run --frozen gw list       # 使えるモデル一覧（カタログ＋HF キャッシュ）
uv run --frozen gw stop       # 停止（再開は gw start か次回ログイン）
uv run --frozen gw start      # 手動起動（初回は設定を自動生成。自動起動を無効にした場合の入口）
uv run --frozen gw update     # 更新を取り込む（新版の確認は自動、適用はこのコマンドだけ）
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
