# gateway.toml リファレンス

サーバーはカレントディレクトリの `./gateway.toml` を 1 つの設定として読む。これがモデルカタログ
（どのモデルを・どのバックエンドで提供するか）と運用方針（同時常駐数・自動アンロード等）を決める。
リポジトリ直下にそのまま使える例を同梱している（[gateway.toml](../gateway.toml)）。

## 全フィールド

```toml
host = "127.0.0.1"          # 公開ホスト（省略時 127.0.0.1）
port = 8799                 # 公開ポート（省略時 8799）。クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数のハード上限。超えたら LRU 退避（省略時 無制限）
load_timeout = 300          # 全枠処理中のとき空くのを待つ最大秒数（超過で 503。省略時 300）
idle_timeout = 1200         # この秒数使われないモデルを自動アンロード（省略時 1200=20分。0 で無効）
internal_base_port = 9001   # 内部モデルサーバーの割当開始ポート（9001, 9002, … と連番）
default_model = "..."       # model 省略リクエスト時のモデル（任意）
draft_model = "auto"        # MTP（speculative decoding）の既定。各 [[models]] で上書き／"off" で無効
dynamic = true              # 未登録モデルを ID 推論で動的ロード（省略時 true。false で事前登録のみ）
disable_thinking = false    # 動的ロード時の既定（思考抑制）。事前登録は各 [[models]] が優先

# [[models]] は任意（dynamic = true なら省略可）。個別オプション（MTP/parallel/mmproj 上書き等）が
# 要るモデルだけ事前登録し、それ以外は動的ロードに任せる、という使い分けができる。
[[models]]
model = "mlx-community/Qwen3.6-27B-4bit"   # HuggingFace のモデル ID
backend = "mlx-vlm"                        # mlx / mlx-vlm / llama-cpp
# draft_model 省略 → 上の既定 "auto" を継承

[[models]]
model = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
backend = "mlx-vlm"
# draft_model = "off"   # このモデルだけ MTP を無効化
```

| キー | 既定 | 説明 |
|---|---|---|
| `host` | `127.0.0.1` | 公開ホスト |
| `port` | `8799` | 公開ポート（クライアントの `base_url` はここ） |
| `max_resident` | 無制限 | 同時に常駐させるモデル数のハード上限。超過は LRU で退避 |
| `load_timeout` | `300` | 全枠が処理中のとき空きを待つ最大秒数（超過で 503） |
| `idle_timeout` | `1200` | この秒数使われないモデルを自動アンロード（`0` で無効） |
| `internal_base_port` | `9001` | 内部モデルサーバーの割当開始ポート |
| `default_model` | なし | `model` 省略リクエスト時に使うモデル |
| `draft_model` | なし | MTP の全体既定（`"auto"` で自動選択／`"off"` で無効）。→ [mtp.md](mtp.md) |
| `dynamic` | `true` | 未登録モデルを ID 推論で動的ロードする。`false` で事前登録のみ（旧挙動） |
| `disable_thinking` | `false` | 動的ロード時の既定。事前登録モデルは各 `[[models]]` の値が優先 |

`[[models]]` は 1 モデル 1 エントリ。`model`（HuggingFace ID）と `backend`（`mlx` / `mlx-vlm` /
`llama-cpp`）が必須。各エントリで `draft_model` を上書きできる。`dynamic = true` なら `[[models]]` は
省略可（全て動的ロード）。

## 振る舞い

- **遅延起動**: 各モデルは**初回リクエスト時に起動**し、2 回目以降は常駐して即応答する。
- **動的ロード（`dynamic = true`）**: `[[models]]` に無いモデルもリクエストされた時点で起動・管理する。
  バックエンドは ID から推論（`gguf`→llama-cpp、`mlx`→mlx-vlm、他→OS 既定。→ [docs/llama-cpp.md](llama-cpp.md)）。
  ロードされると一覧（`/v1/models`・ダッシュボード）に現れ、アンロードされると消える。すでにロード済みの
  モデルが再指定されたら**相乗り**（共有）する。個別オプション（MTP/parallel/mmproj 上書き）は付かない
  ので、必要なモデルだけ `[[models]]` に事前登録する。llama-cpp の repo-id は事前に取得済みである必要が
  あり（未取得は 400）、mlx は HF から自動DLされる。
- **LRU 退避**: 常駐数が `max_resident` を超えると、最も使われていないモデルから停止する。
  全枠が処理中なら空くまで待つ（OOM 回避。`load_timeout` で打ち切り→ 503）。
- **アイドル自動解放**: `idle_timeout` 秒使われないモデルをアンロードしてメモリを返す。
- **1 公開ポートで集約**: 例 `http://127.0.0.1:8799/v1`。クライアントは公開ポートに繋ぎ、
  リクエストの `model` で振り分けられる（クライアントはサーバーを起動しない）。

MTP（speculative decoding）による高速化は [mtp.md](mtp.md) を参照。
