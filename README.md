# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）と音声認識（**whisper** / STT）を束ねる
**マルチモデルゲートウェイ**。1 プロセス起動するだけで、1 つの公開ポートに複数モデルを配信する。
モデルの事前登録は不要——クライアントが指定した `model` をその場でロードする。
特徴の全体像は [docs/features.md](docs/features.md)。

## インストール

[uv](https://docs.astral.sh/uv/) を使う。一度入れれば**どこからでも `gw`**。

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
make install            # `gw` の導入・PATH 設定・自動起動の登録まで全部やる
exec $SHELL -l          # 初回のみ: 今のシェルに PATH を反映（新しいターミナルなら不要）
```

導入後は Ollama と同じく**サーバーを意識しなくてよい**——ログイン時に自動起動し、異常終了時は
自動復活する（`gw disable` で従来の手動 `gw start` 運用に戻せる → [docs/operation.md](docs/operation.md)）。

`make install` が「`~/.local/bin` is not on your PATH」と警告するのは、`uv tool install` が
シェル設定を変更しないため。`uv tool update-shell` がその追記を行う（zsh なら `~/.zshrc` では
なく **`~/.zshenv`**）。書き込み先はシェルによって違うので、反映は特定ファイルの `source` では
なく `exec $SHELL -l` か新しいターミナルで行う。アンインストールは `make uninstall`
（→ [docs/operation.md](docs/operation.md)）。

## 使い方

```bash
gw status     # 稼働/停止・PID・URL・起動経過を表示（自動起動済みなら常に稼働）
gw list       # 使えるモデル一覧（カタログ＋HF キャッシュ）
gw stop       # 停止（再開は gw start か次回ログイン）
gw start      # 手動起動（初回は設定を自動生成。自動起動を無効にした場合の入口）
```

設定は **`~/.config/local-llm-server/gateway.toml` の 1 箇所**（初回の `gw start` が自動生成。
→ [docs/gateway.md](docs/gateway.md)）。

接続は公開ポート（既定 `http://127.0.0.1:8799/v1`）に繋いで `model` を選ぶだけ。
接続クライアントは別パッケージ [local-llm-client](https://pypi.org/project/local-llm-client/):

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
- [docs/llama-cpp.md](docs/llama-cpp.md) — llama.cpp（`llama-server`）の自動導入・最新モデル追従
- [docs/vllm.md](docs/vllm.md) — vLLM / SGLang（高スループット生成）
- [docs/mtp.md](docs/mtp.md) — MTP による高速化

## ライセンス

Apache-2.0
