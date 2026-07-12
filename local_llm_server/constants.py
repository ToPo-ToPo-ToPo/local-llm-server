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


def project_cache_dir() -> str:
    """プロジェクト内（カレントディレクトリ）のキャッシュ/ログ用ディレクトリ `./.local-llm-server`。

    ゲートウェイが起動するモデルサーバーのログ等を、ホーム（`~/.cache`）ではなく起動した
    ディレクトリの中に置く。ディレクトリは呼び出し側が必要時に作る。モデル本体は HF/mlx の
    共有キャッシュ（`~/.cache/huggingface`）に置かれ、これには含めない。
    """
    return os.path.join(os.getcwd(), ".local-llm-server")
