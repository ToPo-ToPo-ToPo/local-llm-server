# llama.cpp バックエンドの導入

`backend = "llama-cpp"` のモデルは、llama.cpp の **`llama-server` バイナリ**を呼び出して提供する。
ゲートウェイ本体は `llama-server` を **PATH 上から実行するだけ**なので、必要なのは「`llama-server` が
PATH に通っていること」だけ。Python バインディング（`llama-cpp-python` の `python -m llama_cpp.server`）
とは**別物**なので、pip では入らない点に注意。

導入後は `gateway.toml` の `[[models]]` で `backend = "llama-cpp"`、`model` に GGUF を指定する
（→ [docs/gateway.md](gateway.md)）。

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
