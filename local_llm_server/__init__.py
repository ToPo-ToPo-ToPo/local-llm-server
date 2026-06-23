"""local-llm-server — ローカルLLMを OpenAI 互換 API として起動・管理する。

mlx / mlx-vlm / llama.cpp、および画像とテキストを自動振り分けする router を
サブプロセスとして起動・監視する。依存は標準ライブラリのみで、任意の
OpenAI 互換クライアントから利用できる。

公開 API:
  - サーバー管理: LocalServer / ServerConfig / ServerPool / build_command など
  - 起動状態の確認: is_ready / list_models / models_match / parse_host_port
  - 既定値・定数: DEFAULT_MODEL / DEFAULT_VISION_MODEL / DEFAULT_BACKEND / BACKENDS
  - router: RouterServer
"""
from __future__ import annotations

from .constants import (  # noqa: F401
    BACKENDS,
    DEFAULT_MODEL,
    DEFAULT_VISION_MODEL,
    _env_bool,
)
from .server import *  # noqa: F401,F403
from .router import *  # noqa: F401,F403
