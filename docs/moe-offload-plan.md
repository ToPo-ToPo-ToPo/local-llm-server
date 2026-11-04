# 実装計画: Qwen3.8-Flash-Next をこの Mac で無理なく動かす(エキスパートと n-gram 表の外出し)

2026-11-04。対象は `ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit-e4o8-ngram4`(106 GiB)を M2 Ultra 128 GiB で、
スワップなしに、ゲートウェイ経由(かみかぜマルから)使えるようにすること。

## 1. 確かめた事実

常駐 106.1 GiB の内訳(safetensors のヘッダから集計):

| 部分 | 大きさ | 1 トークンで使う量 |
|---|---|---|
| ルーティングされるエキスパート(48 層 × 512 個、4bit g32) | 70.3 GiB | 48 層 × 10 個 = 480 個(1 個 約 2.9 MiB) |
| n-gram 埋め込み(`ple.ple_embedding.ngram_embedding.shards.*`、4bit) | 29.8 GiB | 表引きで数行 |
| 注意機構・ルーター・埋め込み・出力(8bit) | 5.8 GiB | 全部 |
| 共有エキスパート | 0.2 GiB | 全部 |

2026-11-04 の実機: 固定メモリ 114.7 GiB、残り 13 GiB で OS とアプリがスワップし、生成 3 tok/s(README の 37 tok/s に対し)。

**必要な仕組みは mlx-vlm 0.7.11(ゲートウェイが使っている版)に両方とも入っている。**

- エキスパート: `mlx_vlm.moe_offload`。`repack` が層ごとの積み重ねテンソルを `experts/layer_N.safetensors` に分け、
  `ExpertStore` が mmap(`mx.load` の遅延読み込み)+ バイト予算付き LRU で常駐分を管理する。`mlx_vlm.load()` は
  `offload_index.json` を見つけると自動で `OffloadedSwitchGLU` に差し替える。サーバーの `--expert-cache-gb` で予算、
  `/v1/moe-offload/stats` でヒット率。プリフィルは chunk 必須(`--prefill-step-size`)
- n-gram 表: `mlx_vlm.models.qwen4_exp.ple_storage.prepare_external_ple_model(src, view)`。元のフォルダに
  ハードリンクした「n-gram を除いた見え方」を作り、`config.json` の `text_config.ple_storage` に manifest を書く。
  実行時は `QuantizedMMapNGramEmbedding`(mmap、`cache_rows` の行キャッシュ)が必要な行だけ読む
- ゲートウェイ: `[[models]]` の `extra_args` は全バックエンド共通で末尾に付く。mlx-vlm は `_mlx_vlm_shims` 経由で
  起動するので、必要なら読み込み後のパッチも当てられる。`max_memory_fraction = 0.9`(123.7 GB)。
  `max_recommended_working_set_size` は 122 GiB
- 自作の試作 `experimental/moe_offload`(ブランチ `claude/mac-moe-offload`)は同じ目的で、違いは読み方(並列 `pread`
  6.9 GB/s。mmap のページフォルト読みは実測 0.9 GB/s)と置き方(減衰付き使用回数 `hot`)。mlx_lm の `SwitchGLU` に
  差し込む作りで、Qwen3.8 Next(mlx-vlm の `qwen4_exp`)にはそのままでは付かない

## 2. 方針

部品を増やさない。**mlx-vlm 組み込みの仕組みを使い、ゲートウェイは設定(モデルの場所と `extra_args`)で渡す。**
自作の試作は「測って足りないと分かったときの差し替え(読み方・置き方)」に限る。コードを書くのは段階 4 と 5 だけ。段階 1 から本命のモデルで行う。

## 3. 段階

### 段階 1: n-gram 表の外出し(1 時間、ディスク増なし)

最初から本命のモデルで行う(小さいモデルで先に試しても、本命でしか起きないこと(`qwen4_exp`・n-gram 表・メモリ)は
確かめられない。設定の書き方は `python -m mlx_vlm.server --help` と読み直しで足りる)。

```
python -c "from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model as p; \
  p('<snapshot>', '~/mlx_models/qwen-next-ple', cache_rows=<値>)"
```

- ゲートウェイに `~/mlx_models/qwen-next-ple` を登録して読み込み、**常駐が 106 → 約 76 GiB** になることを `vm_stat` の wired で確かめる
- 速さを測る(短いプロンプトと、かみかぜマル相当の 20k トークンの文脈)。`cache_rows` を 0 / 数万で比べる
- **ここで足りる可能性がある**: 76 GiB + OS とアプリ 13 GiB = 89 GiB、余裕 39 GiB。スワップが消えて 30 tok/s 台なら、
  段階 2 は省いてよい

### 段階 2: エキスパートの外出し(2〜3 時間、ディスク +約 76 GiB)

段階 1 の view に対して `repack`(n-gram を除いた状態で分けるため、順番はこの通り)。

- `extra_args = ["--expert-cache-gb", "40", "--prefill-step-size", "512"]`(予算は 40 / 50 / 60 GiB で比較)
- 測ること: 常駐、生成 tok/s、プリフィル tok/s、`/v1/moe-offload/stats` が `enabled=true` でヒット率と追い出し数、答えが全載せと同じか
- 見込み: 常駐 = 6(resident)+ 40(予算)+ KV + n-gram の行キャッシュ ≒ 50 GiB。生成は層ごとの同期(約 0.7 ms × 48)
  とミス分の読み込み(ミス率 10% で約 0.14 GB/トークン)で **10〜20 tok/s**

### 段階 3: かみかぜマルから(1 時間)

モデルファイル `models/qwen-next.toml` の `model` を offload dir に向け、同じ依頼(熱伝導のライブラリ)を走らせて
122B と比べる(手数・完成度・検算)。思考なしから。

### 段階 4(測って足りないときだけ): 読み方・置き方の差し替え

段階 2 の `stats` で「ミスの読み込み時間が支配的」なら、試作の並列 `pread`(6.9 GB/s)と `hot` 置き方を
`ExpertStore` に移す。入れ方は 2 通りで、先に上流(mlx-vlm)へ出す方を選ぶ:

1. mlx-vlm へ PR(`ExpertStore` に読み方と置き方の選択肢を足す)
2. それまでのつなぎとして `_mlx_vlm_shims` で `ExpertStore` を差し替える(ゲートウェイの中で完結)

### 段階 5: 記録と片付け

- `docs/moe-offload.md` に測定と推奨設定(mtp.md と同じ形)。README の一覧に 1 行
- ゲートウェイのメモリ見積もり(`needs ~114.0GB` の判定)が offload dir を全量で数えるなら、`experts/` を予算分に
  置き換える補正を足す(小さい変更)
- 試作ブランチ `claude/mac-moe-offload` は記録として残す

## 4. 不確かなこと

- mlx-vlm の offload と MTP(draft model)の併用は未確認。最初は draft なしで測る(この量子化では MTP の効果が小さいと README にある)
- `ExpertStore` は「同時に複数の依頼を捌くなら呼び出し側で直列化」と注記がある。ゲートウェイの continuous batching
  との相性は段階 2 で見る(かみかぜマルは 1 本ずつ)
- プリフィルは chunk で全エキスパートを一巡する。20k トークンの文脈で最初の 1 文字までが何秒かは段階 1〜2 で測る
- n-gram の表引きがページフォルトで遅い可能性。`cache_rows` と、必要なら試作の `pread` で対応
- ディスク: 空き 748 GiB。段階 2 で +76 GiB

## 5. 作業場所

`local-llm-server` に新しいブランチ(`feat/moe-offload-qwen-next`)。段階 1〜3 は設定と手順だけなので、
ブランチに入るのは段階 4・5 の変更と docs。

## 6. 結果(段階 1、2026-11-04)

n-gram 表の外出し(`prepare_external_ple_model`、ハードリンク、ディスク増なし)。同じ質問・temperature 0 で比較。

| | 全載せ(元) | n-gram 表を外出し |
|---|---|---|
| GPU に固定されたメモリ | 112〜115 GiB | **84.6 GiB** |
| スワップ | 1.6 GB 使用・増え続ける | 増えない |
| 読み込み | 約 90 秒 | 15 秒(遅延読み込み) |
| 生成(短い入力) | 5 tok/s | **27 tok/s** |
| 生成(35k トークンの文脈) | 3 tok/s(かみかぜマルの実測) | **22 tok/s** |
| プリフィル(35k トークン) | — | 308 tok/s(112 秒) |
| 答え | — | **全載せと 1 字違わず同じ**(91 トークン) |

段階 1 で目的(スワップなし・実用の速さ)に届いたので、**段階 2(エキスパートの外出し)は今は行わない**。
エキスパートを外すと層ごとに選択を待つ固定費が入り、速さは下がる見込みで、今は省メモリより速さが要る。

段階 1 で見つかったこと:

- **開けるファイル数**: 外出しした n-gram 表は断片ごとに mmap を開く(128 本)。launchd が起動するゲートウェイの
  ソフト上限 256 では足りず、読み込みが `Errno 24 Too many open files` で失敗する → ゲートウェイの起動時に
  ソフト上限をハードの範囲で 65536 まで上げる(PR)。この測定は上限を上げた手動起動のモデルサーバーで行った
- **量子化の mode**: このモデルの `config.json` の n-gram 断片の項目に `mode` が無い(上位の `affine` を継ぐ前提)。
  mlx-vlm の manifest 作りは項目ごとの `mode` を要求するので、作るときだけ上位の `mode` で補い、manifest に
  `"mode": "affine"` を書いた。直すならモデルの config に `mode` を足すか、mlx-vlm 側で上位を継ぐ
