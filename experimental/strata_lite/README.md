# strata-lite（実験）

Strata のやり方（よく使う MoE エキスパートを GPU に、残りを RAM に置く）を、Gemma 4 26B A4B 向けに
自作した試作エンジンです。目的は **「GPU 8GB の Windows 機で、llama.cpp の `--n-cpu-moe` より速くなるか」
を自分の機材で確かめること**です。ゲートウェイ本体（`local_llm_server`）とは独立しており、本体の
依存にも検査（`make check`）にも入っていません。

> **状態**: CPU 上の小型モデルでは、計算結果の正しさを自動テスト（32 件）で確認済みです。
> **実際の GPU と実際の Gemma 4 の重みでは、まだ一度も動かしていません**（開発環境に GPU が無く、
> HF にも接続できなかったため）。最初の実行で問題が出る可能性があります。

## 何をするか

| | llama.cpp `--n-cpu-moe` | strata-lite `hot`（既定） |
|---|---|---|
| GPU に載せる単位 | 層ごと（先頭 N 層のエキスパートは全部 CPU） | エキスパートごと（使用回数の多いものから） |
| 載せる対象の決め方 | 固定 | 使いながら入れ替える（使用回数は減衰付き。次回起動用に保存） |
| GPU に無いエキスパート | CPU で計算 | CPU で計算（`--miss cpu`）か、GPU へ転送して計算（`--miss transfer`） |

比較用に `--placement layer`（llama.cpp と同じく、後ろの層から丸ごと GPU に載せる）も入っています。
同じエンジンの中で配置だけを変えて比べられるので、「配置の違い」による差だけを測れます。

置き場所:
- **GPU**: 注意機構・共有 MLP・lm_head（int4）、ルーター（bf16）、KV キャッシュ、エキスパートのスロット
- **RAM**: 全エキスパート（int4、約 13 GB）、単語埋め込み（bf16、約 1.4 GB）

量子化は Q4_0 と同じ規則（32 個ずつ、対称）です。注意機構や KV キャッシュ、生成ループには
transformers の実装をそのまま使い、エキスパート部分だけを差し替えています。

## 必要なもの

- NVIDIA GPU（8GB で動く想定）。**RTX 30 系以降（Ampere 以降）を推奨**します。int4 の高速な行列積
  （tinygemm）は Ampere 以降でしか使えず、それより古い GPU では展開してから掛ける遅い方式になります。
- RAM 32GB 以上
- ディスク: 元の重み（bf16、約 52GB。変換後は消してよい）と、変換後のファイル（約 16〜18GB）
- Python 3.11 以上、[uv](https://docs.astral.sh/uv/)
- Hugging Face のアカウント（Gemma の利用規約への同意と `hf auth login`）

## 導入（Windows / PowerShell）

PyPI にある Windows 版の torch は CPU 専用なので、**CUDA 版の torch を先に入れます**。

```powershell
cd experimental\strata_lite
uv venv
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最後の行で `True` と GPU 名が出れば準備完了です。

## 使い方

### 1. 変換（1 回だけ）

```powershell
hf auth login
python -m strata_lite convert google/gemma-4-26B-A4B-it D:\models\gemma4-26b-strata
```

- 元の重み（bf16）をダウンロードし、エキスパートを int4 にして書き出します。メモリは数 GB で済みます。
- **変換は実際に動かす PC で行ってください。** CPU 用 int4 の並び順は CPU によって変わる可能性があり、
  違う場合はロード時に止まって再変換を促します。
- 終わったら HF のキャッシュ（`%USERPROFILE%\.cache\huggingface`）にある元の重みは消して構いません。

### 2. 見積もり

```powershell
python -m strata_lite plan D:\models\gemma4-26b-strata --vram-gb 8
```

GPU にエキスパートが何個（何 %）載るかを、ロードせずに計算します。8GB なら 3〜4 割程度の見込みです。

### 3. 速度を比べる（本題）

```powershell
python -m strata_lite bench D:\models\gemma4-26b-strata --compare --tokens 128
```

同じプロンプト（日本語・英語・コード）で 3 通りを順に測ります。

```
layer/cpu      … llama.cpp の --n-cpu-moe と同じ置き方
hot  /cpu      … Strata 流（よく使うエキスパートを GPU）
hot  /transfer … Strata 流 ＋ GPU に無いものは転送して GPU で計算
```

各行に、生成速度（tok/s）、プリフィル速度、生成中の GPU ヒット率（選ばれたエキスパートのうち GPU に
あった割合）が出ます。**layer と hot のヒット率の差が、置き方の工夫で得をした分**です。
使用回数は終了時に `strata-profile.pt` として保存され、次回からは最初からよく使うものが GPU に載ります。

llama.cpp と比べる場合は、同じ PC でゲートウェイ経由（`--n-cpu-moe` を調整した GGUF）か
`llama-bench` で生成速度を測ってください。
エンジン全体の速さ（transformers の生成ループと llama.cpp の C++）の差も混ざるので、
**「置き方の差」は bench の layer と hot の比較で、「エンジンの差」は llama.cpp との比較で**読み分けるのが確実です。

### 4. 対話・サーバー

```powershell
python -m strata_lite chat  D:\models\gemma4-26b-strata
python -m strata_lite serve D:\models\gemma4-26b-strata --port 8090
```

`serve` は OpenAI 互換です（`/v1/chat/completions`（ストリーミング可）、`/v1/models`）。
`GET /v1/strata/stats` でヒット率や層ごとの配置を見られます。ゲートウェイへの組み込みはまだしていないので、
クライアントからは `http://127.0.0.1:8090/v1` を直接指定してください。

## 主なオプション

| オプション | 既定 | 意味 |
|---|---|---|
| `--placement` | `hot` | `hot`（使用回数順）/ `layer`（後ろの層から丸ごと。llama.cpp 相当） |
| `--miss` | `cpu` | GPU に無いエキスパートを `cpu` で計算するか、`transfer` で GPU に送るか |
| `--expert-vram-gb` | 自動 | エキスパート用の VRAM。既定は「空き VRAM − `--reserve-gb`」 |
| `--reserve-gb` | 1.5 | KV キャッシュや作業用に残す VRAM。長い会話で溢れるなら増やす |
| `--threads` | 物理コア数 | CPU 側の計算スレッド数 |
| `--gpu-min-tokens` | 4 | 1 つのエキスパートにこの数以上のトークンが来たら、転送して GPU で計算（プリフィル向け） |
| `--rebalance-every` | 16 | 入れ替えを見直す間隔（トークン） |
| `--kernel` | `auto` | `tinygemm`（Ampere 以降）→ 失敗すれば `dequant` に自動で切り替え |
| `--no-pin` | | RAM のエキスパートを固定メモリにしない（固定メモリの確保に失敗する場合） |

## 制約と注意

- **生成ループは transformers（Python）です。** 1 トークンごとの処理の重さは llama.cpp より大きく、
  置き方で勝っても、エンジン全体では負ける可能性があります。
- 1 本ずつしか生成しません（同時リクエストは順番待ち）。画像入力には対応していません（テキストのみ）。
- 量子化は元の bf16 重みからの単純な丸めです。QAT 版の GGUF（`gemma-4-26B-A4B-it-qat-q4_0`）より
  品質がやや落ちる可能性があります。
- 起動時に int4 カーネルを実際の形で自己検査し、合わなければ遅い方式に切り替えます
  （`kernel=dequant` と表示されたら、tinygemm が使えていません）。

## 困ったとき

- **`CUDA out of memory`**: `--expert-vram-gb` を下げるか、`--reserve-gb` を上げてください。
- **`int4 CPU layout ... differs`**: 変換した PC と実行している PC が違います。実行する PC で再変換してください。
- **`pin_memory failed`**: 自動で通常のメモリに切り替わります（転送が少し遅くなります）。

## 開発（テスト）

GPU が無くても、小さなランダム初期化の Gemma 4 で一連の流れ（変換→ロード→参照モデルとの一致、
配置ごとの一致、hot 配置が偏りを学ぶこと、サーバー）を確認できます。

```bash
cd experimental/strata_lite
uv venv && uv pip install torch -e ".[test]"
.venv/bin/python -m pytest
```
