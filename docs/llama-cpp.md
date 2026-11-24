# llama.cpp バックエンドの導入

- GGUF（`org/repo` に `.gguf` を含む）のモデルは、llama.cpp の **`llama-server` バイナリ**を呼び出して提供する。
- **`llama-server` はゲートウェイが自動導入する**（下記「自動導入」）。手動インストールや PATH 設定は
  不要（導入方法の選択肢は無い——常に自動導入の一本道。PATH のバイナリは使わない）。
- Python バインディング（`llama-cpp-python`）とは別物なので、uv add では使えない点に注意。
- 導入後は**事前登録なしでよい** —— クライアントが GGUF の repo-id を `model` に指定すれば、バックエンドは
  ID から llama-cpp と推論されて動的ロードされる（画像入力の mmproj も自動検出）。`parallel` や MTP、
  `llama-server` への個別フラグが要るモデルだけ `gateway.toml` の `[[models]]` に書く（→ [docs/gateway.md](gateway.md)）。

## 自動導入（`[llama_cpp]`）

ゲートウェイは、llama-cpp を使う構成なら**起動時に `llama-server` を自動でダウンロード・導入**する
（ggml-org/llama.cpp の公式 Releases から、OS・CPU アーキ・アクセラレータを検出して合うプリビルトを
取得。管理ディレクトリ `~/.cache/local-llm-server/llama.cpp/`（Windows は `%LOCALAPPDATA%`）に置き、
PATH は汚さず絶対パスで起動する）。導入したビルド番号・アクセラレータは `GET /admin/status`
（`llama` フィールド）で確認できる。

```toml
[llama_cpp]              # すべて省略可（＝全自動）
accel = "auto"           # auto: 検出 / cuda / vulkan / metal / cpu を明示
pin = "b9946"            # ビルド番号を固定（省略で最新を取得し、以後は導入済みを使い続ける）
```

- **アクセラレータの自動選択**: macOS は Metal（バイナリ内蔵）。Linux/Windows は GPU を検出できれば
  **Vulkan**（NVIDIA/AMD/Intel 共通・追加ランタイム不要）、無ければ CPU。誤検出時や CUDA を使いたい
  ときは `accel` を明示する（**CUDA は Windows 限定**。別途 cudart が要る。Linux の NVIDIA は Vulkan 推奨）。
- **計算効率の自動チューニング**: GPU なら `-ngl 999`（全層 GPU オフロード）、CPU なら `--threads`
  （物理コア数）を自動付与する（`[[models]]` の `extra_args` で明示すればそちらが優先）。生成スループット
  は `from local_llm_server.server import bench_model` の `bench_model(model, base_url)` で tok/s を実測できる。
- **macOS の注意**: Apple Silicon は既定が mlx-vlm なので、llama-cpp を使うときだけ導入される
  （`[[models]]` に `backend = "llama-cpp"` を登録するか GGUF を要求する構成）。

## VRAM に収まらない MoE モデル（`--n-cpu-moe` / `--moe-cache-mib`）

MoE モデル（Gemma 4 26B-A4B、Qwen3.6-35B-A3B など）は重みの大半がエキスパートで、1 トークンで使うのは
その一部だけ。VRAM に全部載らなくても、**エキスパートを RAM に置き、使われるものを GPU にキャッシュする**
ことで実用速度が出る。llama.cpp（b11896 以降で確認）の 2 つのオプションを `[[models]]` の `extra_args` で渡す:

- `--n-cpu-moe N` … 先頭 N 層のエキスパートを RAM（CPU）に置く。attention・共有 MLP は GPU のまま。
- `--moe-cache-mib M` … RAM に置いたエキスパートのうち、最近よく使われたものを GPU 上の M MiB の
  キャッシュに保持する（SLRU。Strata 流の「よく使うエキスパートを GPU に」が llama.cpp 本体に入っている）。

**実測（Windows 11 / RTX 3080 10GB / RAM 32GB、Gemma 4 26B-A4B QAT Q4_0 GGUF 13.3GB、16K ctx、KV q8_0、
生成 64 トークン）**:

| `extra_args` | 生成 | VRAM |
|---|---|---|
| 未指定（llama.cpp の `--fit` に任せる） | 6 tok/s（ctx が 4096 に縮められる） | 9.9 GB |
| `--n-cpu-moe 20` | 31 tok/s | 7.7 GB |
| `--n-cpu-moe 25 --moe-cache-mib 2048` | 40 tok/s | 7.7 GB |
| **`--n-cpu-moe 30 --moe-cache-mib 4096`**（全層 CPU ＋ 4GiB キャッシュ） | **49 tok/s**（ゲートウェイ経由 45〜63） | 9.0 GB |
| `--n-cpu-moe 20 --moe-cache-mib 512` | 16 tok/s（VRAM 圧迫で逆効果） | 8.2 GB |

目安: **全層を CPU に置き、空いた VRAM をキャッシュに回す**のが最も速かった。キャッシュを大きくするほど
ヒット率が上がる一方、KV キャッシュや計算バッファの分（1〜1.5GB）は残す。`-c` を明示しないと
`--fit` がコンテキストを 4096 まで縮めるので、エージェント用途では `-c` も併せて指定する。

```toml
[[models]]
model = "unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL"
backend = "llama-cpp"
parallel = 1
extra_args = ["-c", "16384", "--n-cpu-moe", "30", "--moe-cache-mib", "4096",
              "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
```

VRAM の大きさやモデルが違えば最適値は変わるので、`llama-server` を直接起動して `--n-cpu-moe` と
`--moe-cache-mib` を数通り試し、`timings.predicted_per_second` を比べるのが確実（上の表はそれで測った）。

## model の書き方（HF repo-id）

`model` は **HF repo-id（`org/repo`）で指定する**（実ファイルパスは非対応）。**DL 済みキャッシュ**から
実 GGUF を解決して `llama-server -m` に渡す（`-hf` の自動DLには依存しない＝トークン不要・401 回避）。
クライアントに見せるモデル ID も repo-id になり読みやすい。

```toml
[[models]]
model = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
backend = "llama-cpp"
```

repo に GGUF が複数ある（量子化違い・MTP ヘッド等）ときは **`org/repo:セレクタ`** でファイル名の一部を
指定して 1 つに絞る（例 `unsloth/gemma-4-26B-A4B-it-qat-GGUF:Q4_K_XL`）。セレクタ無しのときは mmproj と
MTP ヘッドを除いた「本体」を選ぶ（1 つに定まらなければ候補を挙げてエラー）。

> **指定した repo-id がローカルキャッシュに無いとエラーになる**（取得方法は下の「HF からダウンロード」）。
> repo-id 形式でない値（実パス等）もエラー。

> 同一 ID は複数エントリに登録できない。MTP あり/なしを併存させたい等は、**repo を分ける**
> （例: 公式版 `google/...`、MTP 版 `unsloth/...`）と ID が衝突しない。

## モデル（GGUF）を HF からダウンロード

llama-cpp の `model`（repo-id）は、その GGUF が **HF キャッシュに DL 済み**である必要がある
（`-hf` の自動DLには依存しない。未取得だと起動時にエラー）。ふつうは `gw pull` で取得する。
本体 1 種と mmproj（画像入力用）だけを選んで取り、量子化が複数あれば `:量子化名` で絞る
（→ [operation.md](operation.md#モデルの取得と管理gw-pull--gw-list--gw-show)）:

```bash
gw pull unsloth/Qwen3.6-27B-GGUF:Q4_K_XL
```

MTP のヘッドのような別ファイルも要るときや、取るファイルを自分で選びたいときは、同梱の `./scripts/hf`
（`huggingface_hub` の `hf` CLI）でファイルを名指しする:

```bash
# repo の特定ファイルだけ取得（GGUF は巨大なので必要なファイルを名指しするのが確実）
./scripts/hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf gemma-4-26B-it-mmproj.gguf

# MTP（埋め込み）版は本体 GGUF だけでよい
./scripts/hf download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-UD-Q4_K_XL.gguf

# vision を使う本体は mmproj も一緒に取得（同じ snapshot に並ぶ → 自動検出される）
./scripts/hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-UD-Q4_K_XL.gguf mmproj-F16.gguf

# 別ヘッド方式（gemma4 MTP）のドラフトヘッドも同様に取得
./scripts/hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF MTP/gemma-4-26B-A4B-it-F16-MTP.gguf
```

- 取得先は共有キャッシュ `~/.cache/huggingface/hub/`（`HF_HOME` / `HF_HUB_CACHE` で変更可）。本サーバーは
  ここから repo-id を解決する。ファイル名を省いて `hf download <repo>` だと repo 全体（全量子化）を
  落とすので、**使う量子化だけ名指し**するのが無難。

## マルチモーダル（画像入力）

Qwen3.6 のような vision 対応モデルは、本体 GGUF とは別に vision projector（`mmproj-*.gguf`）が要る。
本サーバーは **本体 `model` と同じディレクトリに `*mmproj*.gguf` があれば自動検出して `--mmproj` を
付与する**（手動設定不要）。HF の GGUF リポジトリは慣例で mmproj を本体と同梱するため、通常は
何もしなくても画像入力が有効になる。

- **テキストのみ入力でも mmproj は無害**：画像が来たときだけ使われ、テキスト生成の速度・精度に
  影響しない（実測でも生成速度・出力ともに mmproj 有無で実質差なし）。コストは vision エンコーダを
  保持するぶんの追加 RAM のみ。なので常に有効でよい。
- 自動付与を**無効化**したいときだけ `extra_args = ["--no-mmproj"]` を指定する。
- mmproj を明示指定したいときは `extra_args = ["--mmproj", "/path/to/mmproj.gguf"]`（この場合は
  自動検出より優先）。

検証済みの vision 対応 GGUF（いずれも本体と同じディレクトリに mmproj 同梱 → 自動検出される）:

| モデル | GGUF リポジトリ | mmproj |
|---|---|---|
| Qwen3.6-27B | `lmstudio-community/Qwen3.6-27B-GGUF` | `mmproj-Qwen3.6-27B-BF16.gguf` |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` | `gemma-4-26B-it-mmproj.gguf` |

## speculative decoding（MTP / draft）による高速化

llama.cpp は**speculative decoding**に対応し、ドラフトモデルで本体の生成を先読みして高速化する
（出力は本体が検証するので**ロスレス＝品質は変わらない**）。`[[models]]` の `draft_model` に
ドラフト GGUF のパスを指定すると有効になる（`llama-server -md <path>`）。

MTP には 2 方式ある:

- **別ヘッド方式**（例: Gemma 4）— MTP ヘッドが本体と別 GGUF。`draft_model` にその GGUF を指定する
  （ファイル名に `mtp` を含めば自動で `--spec-type draft-mtp`）。vision(`--mmproj`) と併用できる。
- **埋め込み方式**（例: Qwen3.6）— MTP ヘッドが本体 GGUF に内蔵。`draft_model = "self"`（または `"mtp"`）
  で有効化＝別ドラフト不要（`--spec-type draft-mtp` のみ付く）。**この方式は llama.cpp 側で
  `--mmproj`（vision）と `--parallel>1` が未対応**なので、本サーバーは自動的にそれらを付けない。
  vision も使いたいときは MTP 版とは別に「MTP なし」エントリ（別 repo / 別 quant）を立てる。

その他のドラフト（同系統の小型モデル等）は `draft_model` にそのパス/ID を指定すると `-md` のみで
llama.cpp 既定の `draft-simple` になる。無効化は `draft_model` を省略するか `"off"`。グローバル既定の
`draft_model = "auto"` は mlx-vlm 専用なので llama-cpp では無視される（MTP を使うには明示指定が要る）。

```toml
[[models]]
model = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"            # org/repo を基本に
backend = "llama-cpp"
draft_model = "unsloth/gemma-4-26B-A4B-it-qat-GGUF:F16-MTP"  # MTP ヘッド → draft-mtp 自動
```

伸び幅は機材・量子化・ワークロード・バージョンで変わるので、効果は自分の環境で実測して判断する
（元々高速な MoE では伸びが小さいこともある）。予測しやすい出力（コード・定型文）ほど受理率が
上がり伸びる。MTP ヘッドの GGUF は本体と
同系統のもの（例: `unsloth/gemma-4-26B-A4B-it-qat-GGUF` の `MTP/*.gguf`）を使う。

## Inkling 2 Mini（mlx-vlm と llama.cpp）

`thinkingmachines/Inkling-2-Mini`（中段 12 層を 3 周する looped transformer の MoE）は、
mlx-vlm でも llama.cpp でも動く。どちらも思考（`reasoning_content`）・本文・ツール呼び出しを
正しく分けて返す（実機で確認）。**llama.cpp は b12600 以降**が要る（`inkling2_loop` はそこから。
動作確認は b12631）。導入済みのビルドは自動では上がらないので、古いビルドが入っている PC では
`unknown model architecture: 'inkling2_loop'` で起動に失敗する。そのときは `[llama_cpp]` の
`pin` を新しいビルド番号にする（起動失敗のエラーにもこの直し方が出る）。

### 導入と使い方

ほかのモデルと同じ流れで使える（→ [README](../README.md#使い方)）。

```bash
gw update                                   # Inkling に対応した mlx-vlm を入れる（更新していない PC だけ）
gw pull thinkingmachines/Inkling-2-Mini     # 公式の重みを取得する
```

リクエストの `model` に `thinkingmachines/Inkling-2-Mini` を書く。下の高速化は自動で入る。

- **推奨の設定:** `temperature` 1.0（同梱の generation_config.json。top_p・top_k の指定は無い）。
- **考える深さ:** リクエストの `reasoning_effort`（`none`〜`max`。既定は 0.9＝high）で変える。
  ツールを何手も使う作業は既定のまま、会話やロールプレイは `"low"` が向く。`none` でも考える過程は
  完全には止まらない。考える過程は `reasoning_content`、本文は `content` に分かれて返る。
- **`max_tokens`:** 考える過程の分もトークンを使うので、少なすぎると本文の途中で切れる。
- **`seed`:** mlx-vlm 0.8 は送らないと毎回 0 を使うので、温度を上げても同じ会話には同じ返事が返る。
  毎回違う返事が欲しいときは、リクエストごとに違う `seed` を送る。
- **高速化を止める:** ゲートウェイを起動する環境で `LOCAL_LLM_INKLING2_FAST=off` にして `gw restart`。
- llama.cpp で動かすときは `ggml-org/Inkling-2-Mini-GGUF`（量子化が複数あるので
  `ggml-org/Inkling-2-Mini-GGUF:IQ4_NL` のように選ぶ）。

### mlx-vlm と llama.cpp のどちらを使うか

**Apple Silicon では mlx-vlm を使う。** 公式の重み（専門家は NVFP4/MXFP8、ほかは bf16）をそのまま読み、
下の高速化が自動で当たる。M2 Ultra で比べたところ、読み込み・生成とも llama.cpp
（`ggml-org/Inkling-2-Mini-GGUF` の IQ4_NL）より速かった。IQ4_NL は公式の混合精度より粗い量子化で、
出力の質は同じとは限らない（質の比較はしていない）。同梱の DSpark ドラフターは、高速化と合わせても
コード以外では遅くなる（→ [mtp.md](mtp.md#対応表に入れていないもの-inkling-2-minidspark)）。

### mlx-vlm での高速化（自動）

中段を 3 周するので 1 トークンに 48 回層を通り、その度に小さな演算が数十個ある。M2 Ultra では、
重みを読む時間より、数千個の演算を GPU に送る区切りごとの待ちが生成の時間の大半を占めていた。
ゲートウェイは、モデルの config.json が Inkling 2 系（`model_type` が `inkling2*`）のときだけ、
モデルサーバーに次を渡す（→ `local_llm_server/_inkling2_fast.py`）:

- **GPU コマンドバッファの上限を上げる**（`MLX_MAX_OPS_PER_BUFFER=300`・`MLX_MAX_MB_PER_BUFFER=1000`）。
  計算は変わらず、送る区切りが減る。これより上げても速くならず、メモリだけ増える。
- **演算をまとめる差し替え**（`LOCAL_LLM_INKLING2_FAST=gates,attention,prefill_bound,head,fused_residual,fused_router,fused_gates,step`）。考え方は
  [sybil-solutions/local-ai-registry#461](https://github.com/sybil-solutions/local-ai-registry/pull/461)
  （mlx-lm 向け）と Strata の Metal 版から借りた。
  - `gates`: 前の周の情報を混ぜる門の演算を `mx.compile` でまとめる。結果は完全に同じ。
  - `attention`: 注意機構の sink を、K/V に零の行を継ぎ足す代わりに SDPA の `sinks` で渡す
    （毎トークン K/V とマスクを複製していたのが無くなる）。1 系列・1 トークンのときだけ。鍵が 1 本減って
    和の順序が変わるため、結果はごく僅かにずれる（依頼文を 2 回に分けて読むだけで生じるずれの範囲）。
  - `prefill_bound`: 上限を上げると読み込み中の途中結果が解放されにくく、長い読み込みで最大メモリ使用量が
    大きく増える。読み込み（17 トークン以上）では層を 2 回通るごとに計算を確定させて抑える
    （読み込みは少し遅くなる）。
  - `head`（→ `local_llm_server/_inkling2_step.py`）: 出力層（語彙 201,024 × 2,304）を bf16 のまま読み、
    fp32 で積和する専用の計算にする。mlx-vlm は精度のために fp32 に広げた複製を毎トークン読んでいた。
    値の変換は正確なので、違いは和の順番だけ。
  - `step`（同）: 生成 1 ステップを丸ごと `mx.compile` する。Python で計算のグラフを組む手間と
    細かい演算の起動が減る。mlx-vlm のキャッシュはそのまま使い、配列を渡して書き戻すので、続きの会話や
    読み込み・DSpark の検証の経路と食い違わない。形を変えないため、全体注意の KV は容量を 2 の冪
    （最小 4,096 トークン）まで、スライド窓の KV は窓の大きさまで先に広げる（長い文脈では、容量が
    倍になるたびに 1 回コンパイルし直す）。注意に渡す鍵の本数と並びが変わるため、和の順番が変わり
    結果はビット一致しない（その他の部分は一致を確かめた）。1 系列・1 トークンのときだけで、
    複数系列の同時生成や DSpark の検証は元の実装のまま。素の経路は Python でグラフを組む分、
    CPU が他の仕事で混んでいるときほど遅くなるので、その影響も受けにくくなる。
  - `fused_residual`・`fused_router`・`fused_gates`（→ `local_llm_server/_inkling2_fused.py`）: 1 トークンの計算は
    「前の結果を待ってから次を始める」小さな処理が長く連なり、その 1 段ごとの待ちが時間を決めている
    （中段の 2・3 周目は前の周と門の入力でしかつながらないので、門の入力を定数にすると 3 周が並行に走って
    大きく速くなることから分かった）。そこで、連なりの数段を専用の計算 1〜2 回にまとめる:
    層の入口で前の段を混ぜる計算、専門家の振り分け、前の周の情報を混ぜる門。bf16 に丸める箇所と式は
    mlx-vlm と同じにし、違うのは和の順番だけ（振り分けが選ぶ専門家は同じ）。
  - どの段階でも、下の 51 問の正答と外した問題は変わらなかった。ツールの 2 往復・ストリーミング・画像・
    同時 3 件・長い文脈（約 7.3 万トークン）・続きの会話でのキャッシュの使い回しもそのまま動いた。
- 生成時のメモリは少し増える（`step` が KV を先に広げるため）。`head` で出力層の fp32 の複製は作られなくなる。
- 自分で決めた値があれば、そちらが優先される（ゲートウェイは未設定の分だけ足す）。差し替えだけ
  止めたいときは、ゲートウェイを起動する環境で `LOCAL_LLM_INKLING2_FAST=off` にする。
- mlx-vlm の実装を前提にした差し替えなので、mlx-vlm を上げて前提が崩れたら当たらずに素の実装で
  動く（遅いだけで壊れない）。モデルサーバーのログに `inkling2 fast paths:` で当たったものが出る。

### 答えの質と動作確認（2026-11-23）

ゲートウェイ経由（mlx-vlm・温度 0・推論の既定のまま）で、採点を自動化した自作の 51 問
（数学 16・論理 6・日本語 10・指示の遵守 5・コード 9・ツール呼び出し 5）を解かせた:

| モデル | 正解 | 考える過程 |
|---|---:|---|
| Inkling-2-Mini（高速化あり） | 48/51 | 全問で出る（止められない） |
| Inkling-2-Mini（高速化なし） | 47/51 | 同上（3 問で 8192 トークンの上限まで考え続けた） |
| Qwen3.6-27B（mlx-4bit・MTP） | 49/51 | 出ない（既定） |
| Ornith-1.5-35B-A3B（mlx-4bit） | 44/51 | 出ない（既定） |

- 高速化の有無で正答率は変わらない（出力が完全に一致しない問題もあり、外した問題も違う）。
- Inkling が外したのは日本の知識（面積 2 位の県・「小春日和」の季節）と、ツールを呼ばずに
  「リマインダーを設定しました」と答えた 1 問（高速化なしでは正しく呼んだ）。
- 問題数が少ないので、1〜2 問の差は誤差の範囲。

比べる条件を揃えるため、Qwen と Ornith も考える過程ありで解かせた（リクエストに
`"enable_thinking": true`）:

| モデル | 正解 | 上限まで考え続けた問題 |
|---|---:|---:|
| Inkling-2-Mini（高速化あり・温度 0） | 48/51 | 0 |
| Qwen3.6-27B（温度 0） | 42/51 | 6 |
| Qwen3.6-27B（温度 0.6・top_p 0.95・top_k 20） | 47/51 | 3 |
| Ornith-1.5-35B-A3B（温度 0） | 47/51 | 1 |

- Qwen は考えると 1 問に Inkling の 4〜5 倍のトークンを書くので、終わるまでがずっと長くなる。
- Qwen は温度 0 だと同じ確認の文を繰り返して上限の 8192 トークンに達しやすい。推奨の温度 0.6 で
  減るが、数学の 3 問は温度 0.6 でも上限まで考え続けた。正解数も考えないとき（49/51）より下がった。
- Ornith は考えると 44→47 問に上がり（数学・論理・コードが全問正解）、終わるのも Inkling より早い。
  日本語の知識問題は考えても上がらない（6/10）。

ロールプレイ（キャラ設定をシステムプロンプトに書き、決めた発言で 6〜7 往復。猫の妖精「ミケ」・
江戸の武士・ツンデレの幼なじみの 3 役。温度 0.7。Qwen と Ornith は考える過程なし）:

| | Inkling-2-Mini | Qwen3.6-27B | Ornith-1.5-35B-A3B |
|---|---|---|---|
| 語尾の決まり（「〜にゃ」で終える文） | 50/50 | 42/48 | 28/48 |
| 一人称・呼び方 | 「ミケ」のはずが「私」（6/14 回） | ほぼ守る | 幼なじみが「あたし」を一度も使わない |
| 武士がカタカナ語を使わない | 守る（新幹線を「速き車」と言い換え） | 聞き返しで使う（「コンビニ……？」） | 自分から「スマホ」と言う |
| 「本当は AI でしょ？」・コードを頼む | キャラのまま断る | 同左 | 同左 |
| 1 往復目の名前を最後に覚えている | 覚えている | 覚えている | 覚えている |

- Inkling は決まりを一番よく守るが、文が硬く「同行しようではござる」のような崩れがある。幼なじみ役では
  「…って、言いすぎ？」などの同じ文を何度も繰り返し、話の流れも噛み合わないことがあった。
  考える過程がある分、1 回の返事にかかる時間は 3 モデルで一番長い。
- Qwen は決まりの守り方は Inkling に少し劣るが、返しが一番いきいきしていて自然
  （焦げたクッキーを「キャラメル風味だよ！」と言い張るなど）。
- Ornith は武士の「〜にてござる」を乱用して文法が崩れ、英語も混ざり、長さの指示（2〜4 文）も守らない。
- mlx-vlm 0.8 は `seed` を送らないと毎回 0 を使うので、温度を上げても同じ会話には同じ返事が返る
  （2 回走らせた結果が Inkling と Qwen で完全に一致した）。

Inkling の推奨設定は `temperature 1.0` のみ（同梱の generation_config.json。top_p・top_k の指定は無い）。
考える深さはリクエストの `reasoning_effort`（`none`〜`max`。既定は 0.9＝high）で変えられるが、
`none` でも考える過程は完全には止まらない。推奨設定で同じロールプレイを毎回違う `seed` で 2 回ずつ試した:

| Inkling の設定 | 「〜にゃ」の文 | 一人称が「ミケ」でない返事 |
|---|---:|---:|
| 温度 0.7・top_p 0.9（上の表） | 50/50 | 6/14 |
| 温度 1.0（推奨）・考える深さは既定 | 43/46 | 9/14 |
| 温度 1.0・`reasoning_effort: "low"` | 45/54 | 2/14 |

- 温度 1.0 だと同じ文の繰り返しは消え、幼なじみ役の照れ方も自然になる（「焦げてるのは、うまくできないからよ！」）。
  その代わり日本語の崩れ（「縁遠ない」「でござるろう」、中国語の字「场所」）が増え、
  一人称も「わたくし」「私」にぶれる。1 回は「本当にアジの干物を頬張る体はない」と役を半分降りた。
- `low` にすると考える過程が短くなって返事が早く出る。決まりの守り方や崩れの量は既定の深さと大きく変わらない。
- どの設定でも、日本語のロールプレイの自然さは Qwen3.6-27B（考える過程なし）に届かない。

#### Gemma 4 との比較（エージェント・日本語の文章）

エージェント試験: 偽のツール 17 個を渡し、数手のツール呼び出しが要る 14 問（検索→予約、ファイルを読んで
まとめて保存、返金ポリシーを調べてから判断、503 エラー後の再試行、情報が足りない依頼、2 往復目の予定変更など）を
解かせ、最後の答えとツールの呼び方を自動で採点した。サンプリングは各モデルの推奨（すべて温度 1.0。Gemma は
top_p 0.95・top_k 64、Qwen と Ornith は top_p 0.95・top_k 20）、毎回違う `seed`。Gemma 4 は QAT でない
`ToPo-ToPo/gemma-4-26B-A4B-it-mlx-4bit`（MTP あり）。

| モデル | 正解 | 外した主な問題 |
|---|---:|---|
| Inkling-2-Mini（考える深さは既定） | 42/42 | なし |
| Inkling-2-Mini（`reasoning_effort: "low"`） | 25/28 | 「来週の月曜」を 12 月 7 日にした、数え間違い |
| Gemma 4 26B-A4B（考えない） | 36/42 | エラー後に再試行しない（3 回とも）、気温の差を答えない |
| Gemma 4 26B-A4B（考える） | 25/28 | エラー後に再試行しない |
| Qwen3.6-27B（考えない） | 38/42 | エラー後に再試行しない（3 回とも） |
| Qwen3.6-27B（考える） | 25/28 | ばらばら |
| Ornith-1.5-35B-A3B（考えない） | 36/42 | 「2 万円以下は何便」の数え間違い（3 回とも） |
| Ornith-1.5-35B-A3B（考える） | 27/28 | 予約 1 回 |

（採点は厳しめ。`reasoning_effort: "low"` の予定変更の 1 回と、Ornith のファイル整理の 1 回は、内容としては正しかった。）

- Inkling は「一時的に使えません（503）。再試行してください」と返したツールを、毎回自分で呼び直して最後まで
  やり遂げた。Gemma と Qwen は止まって「もう一度試しますか？」とユーザーに聞き返した。
- Inkling は考える過程が長いので、1 問を終えるまでは 4 モデルで一番長い。Ornith は考えさせると
  Inkling とほぼ同じ正解率で、終わるのはずっと早い。

日本語の文章（説明文 400 字・お詫びメール・記事の 3 文要約・小学生向けの説明。各 2 回）は、モデル名を伏せて読み比べた:

- **Gemma 4:** 4 種類とも一番自然で、変な字も混ざらない。ロールプレイも 3 役とも一番いきいきしていた
  （「すまーとふぉん」をひらがなで聞き返す武士など）。51 問は 49/51。
- **Inkling:** 外国の文字や誤字は一度も出ず、指示された内容はすべて入っている。ただ文は平板で、要約の 1 回は
  「午前9時から午後5時から午後8時まで」と崩れた。文章では Qwen と同程度、ロールプレイでは劣る。
- **Qwen:** 自然だが、中国語の字（「不变」「顶いて」）や誤字（「新入社者」「お待特ください」）がときどき出る。
- **Ornith:** キリル文字（「ぶつкарると」「також」）や英語が混ざり、要約で記事にない因果関係を作った。4 モデルで一番弱い。

ほかに、ツール結果を返す 2 往復目（非ストリーミング・ストリーミング）、ストリーミングでの考える過程と
本文の分離、画像入力、同時 3 件、長い文脈（約 2.4 万・7.3 万トークンの中に埋めた情報の取り出し）、
続きの会話での KV キャッシュの使い回し（使い回しても答えが変わらないこと）が正しく動くことを確かめた。

## 並列スロット（parallel）

`llama-server` は **1 プロセスで複数リクエストを同時処理**できる（`--parallel N`）。本サーバーでは
`parallel` で指定する。`parallel = 1`（既定）だと同じモデルへの同時リクエストは直列化され、`4` に
すれば最大 4 本を並行処理する。llama.cpp は内部でコンテキスト（KVキャッシュ）を N スロットに分割
するので、**増やすほど 1 リクエストあたりのコンテキスト長と計算資源は減る**（同時利用人数に合わせて
決める値）。

- **llama.cpp 専用**。mlx / mlx-vlm は逐次処理でスロットの概念が無いため無視される。
- **埋め込み MTP（`draft_model = "self"`）とは併用不可**。llama.cpp 側が未対応なので自動で外す。
- **事前登録なしで効く**: トップレベルに `parallel = 4` を置くと、動的ロードされる **llama-cpp モデル
  全部**に適用される（mlx 系は無視）。特定モデルだけ変えたいときは `[[models]]` の `parallel` で上書き。

```toml
parallel = 4                # 動的ロードする llama-cpp モデルの既定（mlx 系は無視）

# 特定モデルだけ別の値にしたいときだけ事前登録:
# [[models]]
# model = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"
# backend = "llama-cpp"
# parallel = 8
```

## メモリガード（max_memory_fraction）

複数モデルの同時常駐や `parallel` の引き上げは RAM を食う。`max_memory_fraction` を設定すると、
**常駐モデルの推定占有量の合計が「総RAM × その割合」を超えるロードを拒否**する（OOM 回避）。

```toml
max_memory_fraction = 0.66  # 常駐モデルの推定合計を総RAMの2/3までに抑える（0 < x <= 1。省略で無効）
```

挙動:

- ロード前に新モデルの占有量を見積もり、常駐分との合計が予算を超えるなら、まず **アイドルモデルを
  LRU で退避**して収めようとする。退避してもなお収まらない（=そのモデル単体で予算超過）なら **503**
  を返してロードしない。処理中（inflight>0）のモデルは退避しない（空くまで待ち、`load_timeout` 超過で 503）。
- `max_resident`（常駐**数**の上限）と**併用可**。どちらか一方でも超えれば退避が走る。
- **全バックエンド共通**の設定（mlx / mlx-vlm / llama-cpp すべてに効く）。Apple Silicon は GPU と RAM が
  統合メモリなので、総RAMに対する割合がそのまま効く。

> **見積もりは概算**: 占有量は **重みファイルのサイズ基準**（llama-cpp=GGUF＋mmproj＋ドラフト、
> mlx=HF スナップショット合計）で、KVキャッシュやランタイムバッファは含まない**下限寄り**の値。
> 余裕をもたせたいときは割合を低めにする。占有量を見積もれないモデル（mlx で未DL 等）はガードを
> スキップしてロードする。`psutil`（依存に同梱）で総RAMを取得する。

### `--ctx-size`（`-c`）を必ず検討する：見積もりが大きく外れるケース

`llama-server` は起動時に `-c`/`--ctx-size` を渡さないと、**モデルのネイティブ最大コンテキスト長**
分の KV キャッシュを FP16 で丸ごと事前確保する。長コンテキスト対応モデル（数十万トークン級）だと、
これが重みファイルのサイズを大きく超える実メモリ（RSS/VRAM）を食う——上の「メモリガード」の見積もりは
**重みファイルのサイズだけ**を見ているため、この超過分を検知できない。

256Kコンテキスト対応の 27B・1bit量子化 GGUF を Apple Silicon Metal で実測したところ、`-c` 未指定
（＝ネイティブ最大 262144）では RSS が重みファイルの数倍まで膨れ、`-c 32768` に絞ると重みファイル
と同程度まで下がった。

膨れる／収まるかどうかは `-c` の値ひとつで決まる。**GPU VRAM や
`max_memory_fraction` の予算がタイトなら、必要な会話長に見合った `-c` を `[[models]]` の
`extra_args` で明示すること**:

```toml
[[models]]
model = "org/long-context-model-gguf"
backend = "llama-cpp"
extra_args = ["-c", "32768"]   # 必要な最大コンテキストに合わせて調整（大きい方が要RAM）
```

KV キャッシュ自体をさらに圧縮したい場合は `--cache-type-k` / `--cache-type-v`（例 `q4_0`）で
量子化する手もある（同じ `-c` でも使用量をさらに削れる。対応状況はモデル・llama.cpp のバージョンに依存）。

## 複数モデルの同時利用

`llama-server` 単体は **1 プロセス = 1 モデル**（モデルを束ねる router 機能は無い）。本サーバーは
ゲートウェイが各モデルごとに `llama-server` を**個別プロセスで遅延起動**するため、複数 GGUF を
1 つの公開ポートで同時に提供できる（`model` で振り分け）。同時常駐数は `max_resident`（数）と
`max_memory_fraction`（メモリ量）で制限し、超過分は LRU で退避・`idle_timeout` で自動アンロード
される（→ [docs/gateway.md](gateway.md)）。mlx 系と llama-cpp を混在させてもよい。RAM が許す範囲で
`max_resident` を上げれば本当の同時常駐になる。

## 動作確認

導入されたバイナリの素性（ビルド番号・アクセラレータ・絶対パス）は稼働中に確認できる:

```bash
curl -s http://127.0.0.1:8799/admin/status | python3 -m json.tool | grep -A4 '"llama"'
```

ビルド番号が、使いたいモデルの対応コミットを含むか（= 十分新しいか）の目安になる。
