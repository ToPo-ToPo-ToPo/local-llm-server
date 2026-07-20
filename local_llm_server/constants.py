"""エージェント側（agent/）とサーバー側（server/）で共有する中立な定数・ヘルパー。

どちらのサブパッケージにも依存しないため、ここを基点に server/ と agent/ を
完全に分離できる（server/ が agent/ を import する逆向き依存を作らないための層）。
"""
from __future__ import annotations

import os

# サーバー・モデルが明示されないときに使う既定モデル（自作 ToPo-ToPo 版 Qwen3.6-27B）
DEFAULT_MODEL = "ToPo-ToPo/Qwen3.6-27B-mlx-4bit"

# 画像・メディアを処理する vision モデルの既定。
# 既定モデル(Qwen3.6)はマルチモーダルなので、テキストと共通のものを使う。
DEFAULT_VISION_MODEL = DEFAULT_MODEL

# 起動可能なローカルLLMサーバーのバックエンド一覧。
# server/ が実装し、agent/ は agent.toml の backend 検証に使う（共有値）。
# whisper は音声→テキスト（STT）。他はテキスト/画像の生成系。
# vllm / sglang は Linux/NVIDIA（Windows は WSL2 経由）向けの高スループット生成（明示 opt-in）。
# sglang は RadixAttention でプレフィックス共有の多い用途（エージェント）に強い。
BACKENDS = ("mlx", "mlx-vlm", "llama-cpp", "whisper", "vllm", "sglang")


def log_dir() -> str:
    """ログ置き場（`~/.cache/local-llm-server/logs`、Ollama の `~/.ollama/logs` 相当）。

    **cwd 非依存の固定パス**。かつては cwd 相対（`./.local-llm-server`）だったが、起動した
    ディレクトリごとにログが散らばりアンインストールで拾い切れないためやめた。場所は
    provisioner の managed_root と同じ規則（Windows は %LOCALAPPDATA%、他は XDG）で、
    `make uninstall` が消す `~/.cache/local-llm-server` の中に収まる。ディレクトリは
    呼び出し側が必要時に作る。モデル本体は HF/mlx の共有キャッシュ
    （`~/.cache/huggingface`）に置かれ、これには含めない。
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    return os.path.join(base, "local-llm-server", "logs")
