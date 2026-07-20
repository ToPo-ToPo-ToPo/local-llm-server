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
（mlx と違い `-hf` 自動DLには依存しない。未取得だと起動時にエラー）。`huggingface_hub` の `hf` CLI で取得する:

```bash
# huggingface_hub を入れる（uv なら）
uv pip install -U huggingface_hub        # `hf` コマンドが入る

# repo の特定ファイルだけ取得（GGUF は巨大なので必要なファイルを名指しするのが確実）
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf gemma-4-26B-it-mmproj.gguf

# MTP（埋め込み）版は本体 GGUF だけでよい
hf download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-UD-Q4_K_XL.gguf

# vision を使う本体は mmproj も一緒に取得（同じ snapshot に並ぶ → 自動検出される）
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-UD-Q4_K_XL.gguf mmproj-F16.gguf

# 別ヘッド方式（gemma4 MTP）のドラフトヘッドも同様に取得
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF MTP/gemma-4-26B-A4B-it-F16-MTP.gguf
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
  影響しない（実測でも生成 tok/s・出力ともに mmproj 有無で実質差なし）。コストは vision エンコーダを
  保持する数百 MB〜1GB 程度の追加 RAM のみ。なので常に有効でよい。
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

実測（Gemma 4 26B-A4B QAT q4_0, Apple M3 Ultra）: MTP なし ~108 tok/s → あり ~132 tok/s
（コード生成。ドラフト受理率 ~0.72、平均受理長 3.15）。**MoE で元々高速なモデルでも約1.2倍**。
予測しやすい出力（コード・定型文）ほど受理率が上がり伸びる。MTP ヘッドの GGUF は本体と
同系統のもの（例: `unsloth/gemma-4-26B-A4B-it-qat-GGUF` の `MTP/*.gguf`）を使う。

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

実例（256Kコンテキスト対応の 27B・1bit量子化 GGUF、Apple Silicon Metal 実測）:

| 設定 | 重みファイル | 実測 RSS |
|---|--:|--:|
| `-c` 未指定（既定 = ネイティブ最大 262144） | 3.8 GB | **~24.5 GB** |
| `-c 32768`（32K に制限） | 3.8 GB | **~6.4 GB** |

ファイルサイズの6倍以上に膨れる／収まるかどうかは `-c` の値ひとつで決まる。**GPU VRAM や
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
