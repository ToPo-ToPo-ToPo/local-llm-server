# vLLM バックエンド追加 — 開発方針と計画

> **注記（0.34.0時点）**: これは策定当時の開発計画・進捗の記録です。文中の TUI ダッシュボードや
> `uv run gw` は **0.34.0 で廃止**され、`make install` による導入と `gw` サブコマンド運用に
> 置き換わりました（現行の使い方は [operation.md](operation.md) を参照）。

対象ブランチ: `feat/vllm-backend`

## ゴール

Linux / Windows 11 で **vLLM をコアにした生成**をゲートウェイのバックエンドとして追加する。
vLLM は NVIDIA GPU 上で PagedAttention・連続バッチングにより**高スループット・多人数同時**に
強い。既存の llama.cpp（広い環境・単人数・GGUF）と役割分担させ、`backend = "vllm"` で選べるようにする。

## 最重要の前提（調べて判明した制約）

vLLM は既存バックエンドと**根本的に性質が違う**。ここを踏まえないと計画を誤る。

### 1. Windows ネイティブは非対応（公式）

- vLLM は **公式に Windows ネイティブ非対応**（ロードマップも無い。2026-07 時点）。理由は
  CUDA/Triton/FlashAttention カーネルの Windows ビルド維持コストが高いため。
- Windows での選択肢は 3 つ:
  1. **WSL2**（公式推奨）: Windows 上の実 Linux カーネル＋NVIDIA GPU パススルー。
  2. **Docker Model Runner**（WSL2 バックエンド。Docker Desktop 4.54+）。
  3. **コミュニティ版フォーク**（SystemPanic/vllm-windows 等のネイティブ wheel。非公式・版が遅れる）。
- **本計画の方針**: Windows は **WSL2 経由**を正とする（＝Windows では「WSL2 内の Linux」で動かす）。
  ネイティブ Windows は追わない（llama.cpp が既にネイティブで動くので、Windows で GPU 生成を
  したいがWSL2を使わない人は llama.cpp を使う、という住み分け）。コミュニティ版フォークは
  非公式・保守リスクが高いので既定にはしない（上級者が `[vllm] install = "custom"` 等で
  自前 wheel を指すのは許容）。

### 2. バイナリでなく重量級 Python パッケージ

- llama.cpp は単一バイナリを DL するだけだったが、vLLM は **pip パッケージ**で、torch + CUDA
  ランタイムを含み**数 GB**。しかも CUDA バージョンに強く依存する。
- **local-llm-server 本体の依存に混ぜてはいけない**（macOS の mlx 環境を壊す・パッケージが巨大化・
  CUDA 版 torch と mlx の衝突）。→ **隔離した専用環境**（`uv venv` で作る別の仮想環境）に導入し、
  そこの `python -m vllm...` を絶対パスで起動する。llama.cpp のプロビジョナと同じ「PATH を汚さない・
  管理ディレクトリに隔離」の思想を、バイナリでなく venv で実現する。

### 3. モデル形式は HF safetensors（GGUF ではない）

- vLLM は基本 **HF safetensors** を配信する（GGUF は実験的サポート）。既存の GGUF 解決
  （resolve_gguf）とは別に、**HF repo-id をそのまま vllm に渡す**（事前 DL 済みを前提、
  もしくは vllm 自身の HF DL に任せる—トークン/オフライン方針は要検討）。
- vision/動画: vLLM も一部モデルで画像入力に対応（OpenAI vision 形式）。ゲートウェイの
  動画フレーム展開はバックエンド非依存なのでそのまま効く。

### 4. GPU 前提（CPU は実用外）

- vLLM の CPU 実行は可能だが実用性能が出ない。**NVIDIA GPU がある Linux/WSL2** が対象。
  GPU 非検出時は「vLLM は GPU が要る。llama.cpp を使うか accel を確認」と明示エラーにする。

## 既存資産との統合点（調査済み）

- `constants.py: BACKENDS` に `"vllm"` を追加。
- `server.py: build_command` に `backend == "vllm"` 分岐を足し、隔離 venv の
  `python -m vllm.entrypoints.openai.api_server --model <repo> --host --port ...` を起動。
- `infer_backend`（ID からの自動推論）には**入れない**。vLLM は safetensors で mlx repo と
  見分けが付かず、かつ GPU 前提で「うっかり選ばれる」と事故る。**明示 opt-in**
  （`[[models]] backend = "vllm"` か、Linux/NVIDIA 環境での `default_backend` 上書き設定）に限る。
- 起動時プロビジョニングは llama.cpp と同じ枠（`provision_*_if_needed`）。vLLM を使う構成の
  ときだけ venv 作成＋`uv pip install vllm` を行う（macOS 等では何もしない）。
- ゲートウェイの動的ロード・LRU・在席即時解放・drain 再起動・メモリガードはそのまま効く
  （vLLM も 1 モデル 1 プロセスの OpenAI サーバとして同じ枠に乗る）。

## 判断が要る点（ユーザー確認）

1. **Windows の扱い**: WSL2 経由を正とする方針でよいか（ネイティブ Windows は追わない）。
2. **vLLM 環境の導入**: 隔離 venv へ `uv pip install vllm` を**自動実行**してよいか
   （数 GB・数分かかる。ネットワーク必須）。それとも「導入コマンドを案内するだけ・自動はしない」か。
3. **既定バックエンド**: Linux+NVIDIA で **vLLM を既定にする**か、それとも llama.cpp を既定のまま
   にして vLLM は明示 opt-in のみにするか（推奨: 後者。vLLM は重く GPU 必須なので、明示選択が安全）。

## フェーズ計画（案）

| Phase | 内容 | リリース |
|---|---|---|
| **1** ✅ | バックエンド骨格: `BACKENDS` 追加・`build_command` の vllm 分岐（隔離 venv の python から OpenAI API サーバ起動）・設定（`[vllm]`）・明示 opt-in | 0.31.0 にまとめて |
| **2** ✅ | 隔離 venv プロビジョナ（`vllm_provisioner.py`）: 管理ディレクトリに venv 作成＋`pip install vllm`・`import vllm` 自己検証・導入済み再利用・GPU/OS ガード・失敗時の明示エラー。起動時配線・`/admin/status`/TUI 表示 | 0.31.0 |
| **3** | 実機検証（Linux/NVIDIA・要ユーザー実機）: 実モデルで生成・画像・スループット（bench）。WSL2 手順は docs 化済み | 0.32.0 |
| **4** ✅(docs) | README/docs のバックエンド選択ガイド（llama.cpp vs vLLM vs mlx）・WSL2 導入導線（docs/vllm.md） | 0.31.0 にまとめて |

Phase 1-2・4(docs) を実装。実 vLLM 導入・GPU 実行は macOS 開発機・CI で検証できないため、
subprocess/venv 作成/pip/import を差し替えて**全経路をユニット検証**（17 テスト）。実 GPU 検証は
Phase 3 でユーザーの Linux/NVIDIA 実機（または WSL2）で行う。

実 vLLM の導入・GPU 実行は macOS の開発機・CI では検証できないため、Phase 1-2 は
subprocess/インストールを差し替え可能にして**全経路をユニット検証**し、実 GPU 検証は
Phase 3 でユーザーの Linux/NVIDIA 実機（または WSL2）で行う。

## スコープ外（明示）

- ネイティブ Windows ビルド（コミュニティ版フォークの公式サポート）。
- CPU での vLLM 実行の最適化（実用外）。
- vLLM の分散推論（テンソル並列・複数 GPU）は初期はスコープ外（`extra_args` で上級者が指定は可）。
