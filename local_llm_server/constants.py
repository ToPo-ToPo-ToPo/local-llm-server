"""エージェント側（agent/）とサーバー側（server/）で共有する中立な定数・ヘルパー。

どちらのサブパッケージにも依存しないため、ここを基点に server/ と agent/ を
完全に分離できる（server/ が agent/ を import する逆向き依存を作らないための層）。
"""
from __future__ import annotations

import os

# サーバー・モデルが明示されないときに使う既定モデル
DEFAULT_MODEL = "mlx-community/Qwen3.6-27B-4bit"

# 画像・メディアを処理する vision モデルの既定。
# 既定モデル(Qwen3.6)はマルチモーダルなので、テキストと共通のものを使う。
DEFAULT_VISION_MODEL = DEFAULT_MODEL

# 起動可能なローカルLLMサーバーのバックエンド一覧。
# server/ が実装し、agent/ は agent.toml の backend 検証に使う（共有値）。
BACKENDS = ("mlx", "mlx-vlm", "llama-cpp")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
