# llama.cpp バックエンドの導入

`backend = "llama-cpp"` のモデルは、llama.cpp の **`llama-server` バイナリ**を呼び出して提供する。
ゲートウェイ本体は `llama-server` を **PATH 上から実行するだけ**なので、必要なのは「`llama-server` が
PATH に通っていること」だけ。Python バインディング（`llama-cpp-python` の `python -m llama_cpp.server`）
とは**別物**なので、pip では入らない点に注意。

導入後は `gateway.toml` の `[[models]]` で `backend = "llama-cpp"`、`model` に GGUF を指定する
（→ [docs/gateway.md](gateway.md)）。

## model の書き方（実パス / HF repo-id）

`model` は次のどちらでも書ける:

- **実ファイルパス**: `/path/to/model.gguf`
- **HF repo-id**: `org/repo`（例 `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`）。**DL 済みキャッシュ**から
  実 GGUF を解決して `llama-server -m` に渡す（`-hf` の自動DLには依存しない＝トークン不要・401 回避）。
  クライアントに見せるモデル ID も repo-id になり読みやすい。

repo に GGUF が複数ある（量子化違い・MTP ヘッド等）ときは **`org/repo:セレクタ`** でファイル名の一部を
指定して 1 つに絞る（例 `unsloth/gemma-4-26B-A4B-it-qat-GGUF:Q4_K_XL`）。セレクタ無しのときは mmproj と
MTP ヘッドを除いた「本体」を選ぶ（1 つに定まらなければ候補を挙げてエラー）。

> 同一ファイル（同一 ID）は複数エントリに登録できない。MTP あり/なしを併存させたい等は、**repo を分ける**
> （例: 公式版 `google/...`、MTP 版 `unsloth/...`）と ID が衝突しない。

## マルチモーダル（画像入力）

Qwen3.6 のような vision 対応モデルは、本体 GGUF とは別に **vision projector（`mmproj-*.gguf`）**が要る。
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

> llama.cpp の `-hf` 自動DLはトークン次第で 401 になることがある。確実なのは `huggingface_hub`
> （`hf download <repo> <file>`）で本体と mmproj を同じスナップショットに落とし、その `.gguf` パスを
> `model` に指定する方法。同ディレクトリに mmproj が並ぶので自動検出が効く。

## 投機的デコード（MTP / draft）による高速化

llama.cpp は**投機的デコード**に対応し、ドラフトモデルで本体の生成を先読みして高速化する
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
model = "/models/gemma-4-26B_q4_0-it.gguf"
backend = "llama-cpp"
draft_model = "/models/gemma-4-26B-A4B-it-F16-MTP.gguf"  # MTP ヘッド → draft-mtp 自動
```

実測（Gemma 4 26B-A4B QAT q4_0, Apple M3 Ultra）: MTP なし ~108 tok/s → あり ~132 tok/s
（コード生成。ドラフト受理率 ~0.72、平均受理長 3.15）。**MoE で元々高速なモデルでも約1.2倍**。
予測しやすい出力（コード・定型文）ほど受理率が上がり伸びる。MTP ヘッドの GGUF は本体と
同系統のもの（例: `unsloth/gemma-4-26B-A4B-it-qat-GGUF` の `MTP/*.gguf`）を使う。

## 複数モデルの同時利用

`llama-server` 単体は **1 プロセス = 1 モデル**（モデルを束ねる router 機能は無い）。本サーバーは
ゲートウェイが `[[models]]` ごとに `llama-server` を**個別プロセスで遅延起動**するため、複数 GGUF を
1 つの公開ポートで同時に提供できる（`model` で振り分け）。同時常駐数は `max_resident` で制限し、
超過分は LRU で退避・`idle_timeout` で自動アンロードされる（→ [docs/gateway.md](gateway.md)）。
mlx 系と llama-cpp を混在させてもよい。RAM が許す範囲で `max_resident` を上げれば本当の同時常駐になる。

## OS 別の主導線

| OS | 普段使い（手軽） | 当日公開モデルを最速で追う |
|---|---|---|
| **macOS** | `brew install llama.cpp`（Metal 同梱） | GitHub Releases の arm64(Metal) バイナリに差し替え |
| **Linux** | GitHub Releases バイナリ（CUDA / Vulkan / CPU を選択）or `brew install llama.cpp` | GitHub Releases バイナリ |
| **Windows** | GitHub Releases バイナリ（CUDA / Vulkan / CPU）or `winget install llama.cpp` | GitHub Releases バイナリ |

ポイントは **大半のユーザーはソースビルド不要**ということ。`llama-server` バイナリさえ PATH に
あれば動く。

### 1. Homebrew（macOS / Linux・最も手軽）

```bash
brew install llama.cpp     # llama-server が入る。macOS は Metal 対応込み
brew upgrade llama.cpp     # 更新（当日モデルを使う前に回す習慣を）
```

### 2. 配布済みバイナリ（GitHub Releases・最速かつビルド不要）

[ggml-org/llama.cpp の Releases](https://github.com/ggml-org/llama.cpp/releases) は
**ビルド番号（`b1234` 形式）でほぼ連続的に**公開され、各 OS / アクセラレータ向けの
プリビルト zip が付く。自分の環境に合う版（CUDA / Vulkan / Metal / CPU）を展開し、
`llama-server` を PATH の通った場所に置く。

- 新アーキ対応がマージされると**数時間以内**に対応版バイナリが出る。差し替えるだけで追従できる。
- ビルドの手間も brew のラグも回避できるので、**当日公開モデルを急ぐとき**はこれが最速。

### 3. winget（Windows）

```powershell
winget install llama.cpp
```

### 4. ソースビルド（必要なときだけ）

次のいずれかのときだけ検討する。**配布バイナリと同じバージョンならプログラムの挙動は同一**で、
推論結果・精度・モデル互換性に差は出ない（差が出るのはビルド構成だけ）:

- マージ直後の最先端を、リリースバイナリが切られる前に使いたい
- `-march=native` で CPU 推論を限界まで詰めたい
- 自分のアクセラレータ / OS の組み合わせが配布版に無い

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build                       # アクセラレータ: -DGGML_CUDA=ON / -DGGML_VULKAN=ON 等
cmake --build build --config Release
# build/bin/llama-server を PATH に通す
```

## 動作確認

```bash
which llama-server          # PATH に通っているか
llama-server --version      # version: 9780 (xxxxxxxxx) のように出る
```

`version` のビルド番号が、使いたいモデルの対応コミットを含むか（= 十分新しいか）の目安になる。

## 「その日に公開されたモデル」を動かせるかの考え方

動くかどうかは、llama.cpp の入れ方とは別に **2 つの条件**で決まる:

1. **GGUF ファイルが存在するか**（変換）— 誰かが GGUF に変換・量子化してアップする必要がある。
   人気モデルは作者やコミュニティ（bartowski / unsloth など）が数時間以内に出すことが多い。
   他人の GGUF を使うなら、自分は `llama-server` を更新するだけでよい。
2. **llama.cpp 本体がそのアーキテクチャに対応しているか**（本命）— 既存系統の新版や
   ファインチューンなら少し新しめのバイナリで動く。**全く新しいアーキ**は、対応が
   llama.cpp に**マージされるまでどのビルドでも動かない**（古いバイナリは "unknown architecture"
   等でエラー）。

> 自分で GGUF 変換する場合は `convert_hf_to_gguf.py` も最新が要る → バイナリだけでなく
> ソース側（変換スクリプト）も更新する必要がある。

## バージョンずれの考え方

- **ずれの主因は brew の遅さより「ローカルを更新していないこと」**が多い。使う前に
  `brew upgrade llama.cpp`（またはバイナリ差し替え）を回す習慣で大半は足りる。
- brew formula と Releases の差は通常わずか（数ビルド）。最先端を急ぐ瞬間だけ
  Releases バイナリを使い、普段は brew / winget で十分。
