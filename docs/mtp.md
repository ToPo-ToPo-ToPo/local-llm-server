# MTP（speculative decoding）

本体モデルの**出力を変えずに**推論を高速化する Multi-Token Prediction。小さなドラフター
モデルが先読みし、本体がまとめて検証する。Qwen3.6-27B で実測 **~2倍速**（38→75 tok/s、
採択率 93%）。**`mlx-vlm` バックエンドのみ**有効。

## 設定（`gateway.toml`）

`draft_model` で指定する。ゲートウェイ既定と各モデルの個別指定があり、個別 > 既定。

```toml
draft_model = "auto"        # 全モデルの既定（本体名から対応ドラフターを自動選択）

[[models]]
model = "mlx-community/Qwen3.6-27B-4bit"
backend = "mlx-vlm"
# draft_model 省略 → 既定 "auto" を継承（Qwen3.6 の MTP で ~2倍速）

[[models]]
model = "mlx-community/gemma-4-31b-it-4bit"
backend = "mlx-vlm"
draft_model = "mlx-community/gemma-4-31B-it-qat-assistant-bf16"  # HF id で明示

[[models]]
model = "mlx-community/Qwen3.5-27B-4bit"
backend = "mlx"
draft_model = "off"         # このモデルだけ MTP を無効化（mlx は非対応）
```

`draft_model` に取れる値:

| 値 | 意味 |
|---|---|
| `"auto"` | 本体名から対応ドラフターを自動選択（対応表 `MTP_DRAFTERS`）。未対応モデルは起動時にエラー |
| HF id / パス | そのドラフターを明示指定 |
| `"off"` / `"none"` / `""` | 無効化（既定の打ち消しにも使える） |
| 省略 | ゲートウェイ既定 `draft_model` を継承 |

- `backend = "mlx"`（テキスト専用）など mlx-vlm 以外では MTP は無視される。個別に明示した
  場合だけ起動時に「無視される」旨を警告する。
- 初回起動時、本体とドラフターの 2 モデルが自動ダウンロードされる。

## 対応モデル（`MTP_DRAFTERS`）

`"auto"` で解決できる本体→ドラフターの対応表。Qwen3.6 / Gemma 4 系を収録（mlx-community で
検証済み）。一覧は `from local_llm_server import MTP_DRAFTERS` で参照、解決は
`resolve_drafter(model, "auto")` で行う。未収録モデルは `draft_model` に HF id を明示する。
