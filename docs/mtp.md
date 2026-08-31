# MTP（speculative decoding）

本体モデルの**出力を変えずに**推論を高速化する Multi-Token Prediction。小さなドラフター
モデルが先読みし、本体がまとめて検証する。**`mlx-vlm` バックエンドのみ**有効。

速度の伸びはモデル・量子化・ワークロード（構造化出力ほど採択率が高い）・mlx-vlm のバージョン・
機材で大きく変わる。効果は必ず自分の環境で実測して判断すること。

## 画像入力（vision）

**MTP は画像入力を壊しません。** Qwen3.6-27B・gemma-4 系とも、MTP 有効のまま画像を認識する。

画像入りリクエストを gemma-4 系へ振り分ける `vision_model` 設定は **0.36.3 で削除した**。
画像もテキストも同じモデルが受けるので、画像用のモデルを別に常駐させる必要はない。既存の
`gateway.toml` に残っていれば更新後の起動時に**自動で削除される**（手動なら `gw migrate`、
消える内容だけ見るなら `gw migrate --dry-run` → [operation.md](operation.md)）。

## 設定（`gateway.toml`）

**対応表に在る mlx-vlm モデルは設定不要** —— 事前登録せず動的ロードしただけで、本体名から
ドラフターを自動選択して MTP が効く。未対応モデルは静かに MTP なしで
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
- **未取得でも本体は起動する。** ドラフターが無いときは警告を 1 行出して MTP 無しで起動し、
  本体は通常どおり動く（MTP は高速化の手段でしかないため、ここで止めると「速くならない」で
  済む話が「モデルが使えない」に化ける）。あとから `gw pull <drafter>` で取得し、モデルを
  再ロードすれば有効になる。警告はデーモンログ（`gw log`）に出る:

  ```
  warning: MTP ドラフター 'ToPo-ToPo/Qwen3.6-27B-MTP-bf16' が未取得のため、
  'ToPo-ToPo/Qwen3.6-27B-mlx-4bit' を MTP 無効で起動します（本体は通常どおり動作します）。
  有効化するには `gw pull ToPo-ToPo/Qwen3.6-27B-MTP-bf16`。
  ```

  これは対応表の更新を配ったとき（例: 0.38.5 の Qwen3.6 ドラフター切り替え）に、まだ
  ドラフターを取得していない PC でモデルが起動できなくなるのを防ぐための挙動。

  **「使うと宣言しているのに使えない」ときだけ警告する。** MTP を持たないモデルのほうが
  多数派なので、そこで警告すると動的ロードのたびにノイズが出るだけになる。`"auto"` は
  「対応表に在れば使う」であって宣言ではないので、引けなくても黙って MTP 無しにする。
  警告するのは**使うドラフターが決まっている**（対応表がヒットした／HF id を明示した）のに
  それが手元に無い場合に限る。いずれの経路でも本体の起動は止めない:

  | 状況 | 宣言 | 挙動 |
  |---|---|---|
  | mlx-vlm・対応表がヒット → ドラフター未取得 | あり | **警告** → MTP 無しで起動 |
  | mlx-vlm・`draft_model` に HF id を明示 → 未取得 | あり | **警告** → MTP 無しで起動 |
  | llama-cpp・`-md` のドラフト GGUF が未取得 | あり | **警告** → speculative decoding 無しで起動 |
  | MTP 非対応バックエンドへ `draft_model` を明示 | あり | **警告** → 無視（従来どおり） |
  | mlx-vlm・`"auto"` だが対応表に無い（動的ロード） | なし | 静かに MTP 無し |
  | mlx-vlm・`"auto"` だが対応表に無い（事前登録 `[[models]]`） | なし | 静かに MTP 無し |

  事前登録の `"auto"` は以前まで即エラーだった。トップレベルの `draft_model = "auto"` を
  継承した未収載モデルが 1 つ在るだけで**ゲートウェイ全体の設定読み込みが失敗**し、
  MTP が効かないだけで済む話が全モデルを起動不能にしていた。動的ロードは元から静かに
  MTP 無しへ落としていたので、事前登録だけが厳しすぎた。挙動をそちらに揃えた。
- 対応表に無いモデルは「MTP 非対応」と表示する（使うなら `draft_model` に
  ドラフターの HF id を明示する）。`gw mtp` は gateway.toml を読まない（どのディレクトリからでも
  実行できるようにするため）ので、ここの判定は対応表だけで行う。設定済みかどうかも含めて
  見たいときは `gw list` / `gw ps` の MTP 列を使う——こちらは gateway.toml の `draft_model`
  （明示指定）を対応表より優先して判定する。

## 対応モデル（`MTP_DRAFTERS`）

`"auto"` で解決できる本体→ドラフターの対応表。Qwen3.6 / Qwen3.8 / Gemma 4 系を収録
（mlx-community 版と自作 ToPo-ToPo 版の両方）。ドラフターの選び方:

- **Qwen3.6-27B（ToPo-ToPo 版）**: 公式 bf16 チェックポイント内蔵の `mtp.*` を切り出した
  `ToPo-ToPo/Qwen3.6-27B-MTP-bf16` を量子化 4bit/8bit/bf16 で共用する（Qwen3.8-27B と同じ作り方。
  量子化後のリポからは `mtp` が落ちているので、切り出しは必ず公式 bf16 から行う）。
  `mlx-community/Qwen3.6-27B-MTP-4bit` も同じベースなので使えるが、実測では bf16 の方が
  採択率が高い（コード生成 86.4% / JSON 91.5% / 日本語自由文 48.0%）。
- **Qwen3.8-27B（ToPo-ToPo 版）**: 公式 bf16 チェックポイント内蔵の `mtp.*` を切り出した
  `ToPo-ToPo/Qwen3.8-27B-MTP-bf16` を量子化 4bit/8bit/bf16 で共用する（量子化後のリポからは
  `mtp` が落ちているので、切り出しは必ず公式 bf16 から行う）。
- **Gemma 4（ToPo-ToPo 版）**: 各 model card 推奨の **Google 公式ドラフター
  `google/gemma-4-<size>-it-assistant`**（mlx-vlm で変換不要・サイズ固有で量子化に依らず共通）。
  ドラフターはサイズ間で互換性が無い（31B / 26B-A4B / E4B / E2B で別）。`google/...` は gated
  （Gemma ライセンス）なので、自動DLには同意済みの HF トークンが要る場合がある。
- **Qwen3.8-Flash-Next（qwen4_exp）**: ドラフター `ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16` は
  公開済みで、採択率 94.1%・`--draft-block-size 2` で 1.39 倍を実測（25.95 → 35.99 tok/s）。
  ただし `qwen4_exp_mtp` は **mlx-vlm 0.7.0 以降**が要る（未リリース。本リポジトリのピンは
  0.6.17）ため、まだ対応表には載せていない（`_EXTRA_DRAFTER_REPOS` で一覧から隠すだけ）。
  0.7.0 が出たら対応表へ移す。

一覧は `gw mtp`（引数なし）か `from local_llm_server import MTP_DRAFTERS` で参照、解決は
`resolve_drafter(model, "auto")` で行う。未収録モデルは `draft_model` に HF id を明示する。
