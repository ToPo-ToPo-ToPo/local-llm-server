# vLLM バックエンド

[vLLM](https://vllm.ai) を使った高スループット生成を、ゲートウェイのバックエンドとして
`backend = "vllm"` で選べる。PagedAttention・連続バッチングで**多人数同時・高スループット**に
強い。単人数・広い環境なら llama.cpp で十分なので、vLLM は**明示 opt-in**（自動選択しない）。

## 対象環境

- **Linux + NVIDIA GPU**（本命）。
- **Windows 11 は WSL2 内**で動かす（vLLM は Windows ネイティブ非対応。→ 下記「Windows（WSL2）」）。
- macOS・GPU 無しは非対応（Apple Silicon は `backend = "mlx-vlm"`、GPU 無しは `llama-cpp` を使う）。

## 使い方

`[[models]]` に `backend = "vllm"` で登録するだけ。HF repo-id（safetensors）を指定する。

```toml
[[models]]
model = "Qwen/Qwen3-8B"
backend = "vllm"
# extra_args = ["--max-model-len", "8192", "--tensor-parallel-size", "2"]  # vLLM への個別フラグ（任意）
```

クライアントは公開ポートに `model = "Qwen/Qwen3-8B"` を指定して繋ぐだけ（他バックエンドと同じ）。
画像入力・ゲートウェイの動画フレーム展開・動的ロード・LRU・在席即時解放・drain 再起動も
そのまま効く。

## 導入（extras が既定）

vLLM は torch+CUDA を含む**重量級パッケージ（数 GB）**なので base 依存には入れない。使う人だけ
**extras で明示的に入れる**（既定 `provision = "system"`＝勝手に数 GB を自動DLしない）。**uv だけで完結**する:

```bash
# このリポジトリを clone して uv run gw で動かす運用（既定）:
uv sync --extra vllm          # プロジェクトの venv へ vLLM を入れる（Linux/NVIDIA。以後 uv run gw）
# 一時的に有効化するだけなら: uv run --extra vllm gw

# uv tool として入れている場合:
uv tool install "local-llm-server[vllm]"

# local-llm-server を別の uv プロジェクトの依存にしている場合:
uv add "local-llm-server[vllm]"
```

```toml
[vllm]
provision = "system"   # 既定。現在の環境の vllm を使う（uv sync --extra vllm で入れる）
# provision = "auto"   # 隔離 venv へ起動時に自動導入（下記）
```

- **`provision = "auto"`（隔離 venv・自動導入）**: `~/.cache/local-llm-server/vllm-venv` へ起動時に
  自動導入する（初回のみ・要ネットワーク & GPU）。導入済みは再利用。本体環境を汚さない。
  **vLLM と SGLang を同一マシンで両立**したいときはこちら —— 両者は torch/flashinfer のピンが
  食い違い**同一環境に共存できないことが多い**ので、別々の隔離 venv に入れる必要がある
  （extras で両方を1環境に入れると壊れる）。
- どちらも、導入に失敗（GPU 非検出・pip 失敗・CUDA 不整合）してもゲートウェイは起動を続け、
  vllm モデルの要求時に分かりやすいエラーになる（他バックエンドは動く）。
- 導入した vLLM の素性は `GET /admin/status` の `vllm` フィールドと TUI に出る。

> **まとめ**: 1 つだけ使う → `uv sync --extra vllm` ＋ 既定（system）。vLLM と SGLang を
> 両方使う → 各 backend で `provision = "auto"`（別々の隔離 venv・自動導入）。

## Windows（WSL2）

vLLM は Windows にネイティブ対応しない（公式）。Windows 11 では **WSL2**（Windows 内の実 Linux ＋
NVIDIA GPU パススルー）で動かす:

1. WSL2 と NVIDIA の WSL 用ドライバを入れる（`wsl --install`、GPU ドライバは Windows 側の最新版）。
2. WSL2 の Ubuntu で本サーバを動かす（`uv run gw`）。以後は Linux とまったく同じ。
3. Windows 側のクライアントからは WSL2 の IP:ポートに繋ぐ（`host = "0.0.0.0"` で公開）。

Windows でネイティブに（WSL2 無しで）GPU 生成したい場合は `backend = "llama-cpp"` を使う
（llama.cpp は Windows ネイティブ対応・自動導入される）。

## SGLang（vLLM の対抗）

`backend = "sglang"` でも高スループット生成ができる。導入・対象環境（Linux/NVIDIA・WSL2）・
隔離 venv への自動導入（`[sglang] provision`）は vLLM と同じ。違いは **RadixAttention
（プレフィックスキャッシュ）**で、**共有プレフィックスの多い用途に強い**——同じシステムプロンプトや
ツール定義を毎回送る**エージェント運用**では、共有部分のプレフィルを一度で済ませて後続を
スキップするので、高同時実行での TTFT（初回トークンまでの時間）が下がる。

```toml
[[models]]
model = "Qwen/Qwen3-8B"
backend = "sglang"
```

どちらが速いかは**モデルサイズとプロンプトの共有度に依る**（プロンプトがほぼ毎回ユニークなら
両者ほぼ互角）。実機で `bench [model]` を両方に対して測り、あなたのワークロードで勝った方を
使うのが確実。vLLM の方がハード対応・モデル即日対応・コミュニティ規模で優る「安全な既定」、
SGLang はエージェント的な共有プレフィックスで効く、という関係。

## llama.cpp / mlx との使い分け

| 使いたいこと | 推奨バックエンド |
|---|---|
| 多人数同時・高スループット（Linux/NVIDIA） | **vllm** / **sglang** |
| エージェント（共有システムプロンプト/ツール定義が多い） | **sglang**（RadixAttention） |
| 単人数・広い環境（CPU/各種 GPU）・GGUF・手軽さ | **llama-cpp** |
| Apple Silicon（Mac） | **mlx-vlm** / **mlx** |
