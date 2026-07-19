# MTP（speculative decoding）

本体モデルの**出力を変えずに**推論を高速化する Multi-Token Prediction。小さなドラフター
モデルが先読みし、本体がまとめて検証する。Qwen3.6-27B で実測 **~2倍速**（38→75 tok/s、
採択率 93%）。**`mlx-vlm` バックエンドのみ**有効。

## 画像入力（vision）と現行 mlx_vlm の注意

**MTP は画像入力を壊しません。** 画像入力の可否は**モデルファミリ**次第です（現行 mlx_vlm 0.6.3 実測）:

- **gemma-4 系**（`gemma-4-31b` / `26B-A4B` 等）… 画像入力は **MTP 有りでも正常**に動く。
- **Qwen3.6-27B（qwen3_5 系）**… 画像入力が **MTP の有無に関わらず壊れている**
  （`mlx_vlm/models/qwen3_5/language.py::get_rope_index` の `attention_mask.tolist()` が
  バッチ用 GPU スレッドと別スレッドの MLX ストリームをまたいで評価し、`RuntimeError: There is
  no Stream(gpu, N) in current thread` になる → クライアントには返らずタイムアウト）。これは
  上流 `mlx_vlm` のバグで、ゲートウェイ側では直せない。

**対処**: `gateway.toml` に `vision_model` を設定し、**画像入りリクエストを gemma-4 系へ振り分ける**。
テキストは元モデル（Qwen3.6 の MTP 高速化そのまま）、画像だけを画像が動くモデルへ流せる。

```toml
vision_model = "ToPo-ToPo/gemma-4-31b-it-mlx-4bit"   # 画像入りリクエストの振り分け先（MTP 有りで可）
```

- 画像を含むリクエストは、元の `model` が何であってもこの `vision_model` へ流れる（テキストは素通り）。
- `vision_model` 自身が 1 モデル常駐するので、テキスト用モデルと合わせて `max_resident` に余裕を（2 以上）。
- gemma-4 の画像入力は MTP 有りで動くので、`vision_model` の MTP を切る必要はない。

## 設定（`gateway.toml`）

**対応表に在る mlx-vlm モデルは設定不要** —— 事前登録せず動的ロードしただけで、本体名から
ドラフターを自動選択して MTP が効く（Qwen3.6-27B なら ~2倍速）。未対応モデルは静かに MTP なしで
ロードされる。`draft_model` を書くのは、既定を変えたい／個別に上書きしたいときだけ。

```toml
# 何も書かなくてよい。mlx-vlm の対応モデルは動的ロード時に MTP が自動で効く。

# ── 既定を変えたいとき（トップレベル）──
# draft_model = "off"       # 動的ロード時の MTP を一律無効化（省略時は mlx-vlm を "auto"）

# ── 個別に上書きしたいとき（事前登録）──
[[models]]
model = "ToPo-ToPo/Qwen3.6-27B-mlx-4bit"
backend = "mlx-vlm"
draft_model = "off"         # このモデルだけ MTP を無効化

[[models]]
model = "ToPo-ToPo/gemma-4-31b-it-mlx-4bit"
backend = "mlx-vlm"
# 対応表に在るので draft_model 省略でも auto で効く。明示するなら Google 公式ドラフター:
draft_model = "google/gemma-4-31B-it-assistant"  # サイズ固有・mlx-vlm で変換不要
```

`draft_model` に取れる値:

| 値 | 意味 |
|---|---|
| 省略 | 動的ロード時は mlx-vlm が `"auto"` 相当（対応表に在れば自動、無ければ MTP なし）。事前登録の `[[models]]` はトップレベル `draft_model` を継承 |
| `"auto"` | 本体名から対応ドラフターを自動選択（対応表 `MTP_DRAFTERS`）。`[[models]]` で明示した場合、未対応モデルは起動時にエラー |
| HF id / パス | そのドラフターを明示指定 |
| `"off"` / `"none"` / `""` | 無効化（既定の打ち消しにも使える） |

- `backend = "mlx"`（テキスト専用）など mlx-vlm 以外では MTP は無視される。`[[models]]` で個別に
  明示した場合だけ起動時に「無視される」旨を警告する。
- MTP が効くとき、初回起動時に本体とドラフターの 2 モデルが自動ダウンロードされる。
- 動的ロードと事前登録（`[[models]]`）で `"auto"` の扱いが少し違う: 動的ロードは未対応モデルでも
  落とさず MTP なしにフォールバックするが、`[[models]]` に `draft_model="auto"` を明示した場合は
  「明示したのに引けない」ので起動時エラーにする（設定ミスを早く気付けるように）。

## ドラフターの事前確認（`gw mtp`）

**ダウンロードする前に**、そのモデルの MTP にどのドラフターが必要か・もう取得済みかを確認できる。
対応表の辞書引きとローカルキャッシュ確認だけで、**モデルの DL は一切しない**。

`gw mtp [model]` を実行すると表示される。

```bash
gw mtp ToPo-ToPo/gemma-4-31b-it-mlx-4bit
gw mtp                                 # 対応モデルを全件一覧（model 省略）
```

```
ToPo-ToPo/gemma-4-31b-it-mlx-4bit
    drafter: google/gemma-4-31B-it-assistant  [available — 未取得]
    hf download google/gemma-4-31B-it-assistant
```

- `[ready]` はドラフター取得済み（そのまま MTP が効く）、`[available]` は未取得（そのまま貼れる
  `hf download` コマンドを添えて表示）。
- 対応表に無いモデルは「MTP 非対応」と表示する（使うなら `draft_model` に
  ドラフターの HF id を明示する）。

## 対応モデル（`MTP_DRAFTERS`）

`"auto"` で解決できる本体→ドラフターの対応表。Qwen3.6 / Gemma 4 系を収録（mlx-community 版と
自作 ToPo-ToPo 版の両方）。ドラフターの選び方:

- **Qwen3.6-27B（ToPo-ToPo 版）**: 同じ Qwen3.6-27B ベースなので mlx-community の MTP ヘッド
  `mlx-community/Qwen3.6-27B-MTP-4bit` を共用する（量子化 4bit/8bit/bf16 とも）。
- **Gemma 4（ToPo-ToPo 版）**: 各 model card 推奨の **Google 公式ドラフター
  `google/gemma-4-<size>-it-assistant`**（mlx-vlm で変換不要・サイズ固有で量子化に依らず共通）。
  `mlx-vlm >= 0.6.3` が必要。ドラフターはサイズ間で互換性が無い（31B / 26B-A4B / E4B / E2B で別）。
  `google/...` は gated（Gemma ライセンス）なので、自動DLには同意済みの HF トークンが要る場合がある。

一覧は `gw mtp`（引数なし）か `from local_llm_server import MTP_DRAFTERS` で参照、解決は
`resolve_drafter(model, "auto")` で行う。未収録モデルは `draft_model` に HF id を明示する。
