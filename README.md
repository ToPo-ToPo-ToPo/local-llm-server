# local-llm-server

ローカルLLM（**mlx** / **mlx-vlm** / **llama.cpp**）を束ねる**マルチモデルゲートウェイ**。
1 プロセス起動するだけで、1 つの公開ポートに複数モデルを配信する。

- **モデルの事前登録は不要**。クライアントが指定した `model` をその場でロードする（バックエンドは ID から推論）。画像入力（mmproj 自動検出）も mlx-vlm の MTP（対応モデルは自動で ~2倍速）も設定なしで効く。
- 1 つの公開ポートで複数モデルを配信し、リクエストの `model` で振り分ける。
- **TUI ダッシュボードが「使えるモデル」を自動一覧**（DL 済みのチャットモデル。`ollama list` 風）。どれを指定すればよいか一目で分かる。
- モデルは**初回リクエスト時に遅延起動**、`max_resident`（数）/ `max_memory_fraction`（メモリ量）超過で LRU 退避、`idle_timeout` で自動アンロード。
- エージェントが「使い終わった」と通知すれば、在席が 0 になった瞬間に**待たず即アンロード**してメモリ解放（→ [在席ベースの即時アンロード](docs/gateway.md#在席ベースの即時アンロード)）。
- 接続側は公開ポートに繋いで `model` を選ぶだけ（接続クライアントは別パッケージ）。

## インストール

[uv](https://docs.astral.sh/uv/) を使う。**このリポジトリをクローンして、ソースから動かすのが基本。**
Apple Silicon は推論バックエンドの extra `--extra mlx` を付ける（他 OS は付けず llama.cpp を別途用意 → 末尾の注記）。

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server          # クローンしたフォルダの中で実行（uv add ではなく uv sync）
uv sync --extra mlx
```

以降このフォルダで `gateway.toml` を編集し、`uv run local-llm-server` で起動する（→ [使い方](#使い方)）。

> **他 OS（Linux / Windows / Intel Mac）＝ llama.cpp** は `--extra mlx` を外す。推論には `llama-server` を
> 別途インストールして PATH に通し（OS 別手順は [docs/llama-cpp.md](docs/llama-cpp.md)）、`gateway.toml` の
> `[[models]]` で `backend = "llama-cpp"` を指定する。

<details>
<summary>クローンせず PyPI の公開パッケージを使う場合（任意）</summary>

- **コマンドとして入れる**: `uv tool install "local-llm-server[mlx]"` → どこでも `local-llm-server` で起動。
- **別プロジェクトの依存として入れる**: `gateway.toml` を置く新規フォルダで `uv init` → `uv add "local-llm-server[mlx]"` → `uv run local-llm-server`。

（`[ ]` は zsh の glob 展開を避けるためクォート。他 OS は `[mlx]` を外す。）
</details>

## 使い方

### 1. gateway.toml を置く

カレントディレクトリに `gateway.toml` を置く。リポジトリ直下に例を同梱（→ [gateway.toml](gateway.toml)）。
モデルは列挙不要 —— 運用方針（ポートや同時常駐数など）だけ書けばよい:

```toml
host = "127.0.0.1"
port = 8799                 # クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数の上限（超過は LRU 退避）
# モデルは事前登録不要。クライアントが指定した model をその場でロードする。
# parallel や llama.cpp の MTP 等、個別の上書きが要るモデルだけ [[models]] に書く（→ docs/gateway.md）。
```

### 2. 起動

クローンしたフォルダ（`gateway.toml` のある場所）で、**引数なしで起動するのが推奨**。
TUI ダッシュボードが開き、使えるモデル・ロード状況・処理中数が一目で分かる（停止/再起動も画面内で操作）。

```bash
uv run local-llm-server     # ← 推奨。TUI ダッシュボード（状態を自動更新表示）
```

（PyPI の `uv tool install` で入れた場合は `local-llm-server` だけで起動。バックグラウンド常駐は
`--start`、TUI なしは `--headless` → [docs/operation.md](docs/operation.md)。）

### 3. 接続

公開ポートに繋ぎ、`model` で使うモデルを選ぶ。接続クライアントは別パッケージ
[local-llm-client](https://pypi.org/project/local-llm-client/)

```bash
uv add local-llm-client
```
```python
from local_llm_client import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
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
