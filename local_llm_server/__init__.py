"""local-llm-server — ローカルLLMを OpenAI 互換 API として起動・管理する拡張サーバー。

単なる OpenAI 互換サーバーの起動だけでなく、複数エージェントで重複しがちな処理を
備える:

  - LLM 実行          : LocalServer / ServerConfig / ServerPool（mlx / mlx-vlm / llama.cpp）
  - ゲートウェイ      : RouterServer（テキスト/vision 振り分け）, ensure_server（相乗り/自動起動）
  - MTP（投機的デコード）: resolve_drafter / MTP_DRAFTERS
  - 高レベルクライアント : LLMClient / connect（任意 extra `local-llm-server[client]`）

公開 API は __all__ に列挙したものだけ。client 系（LLMClient / connect / to_image_url /
build_user_content）は openai を必要とするため遅延 import で、参照時に初めて読み込む
（コア import は標準ライブラリのみで完結）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# --- 既定値・定数 -----------------------------------------------------------
from .constants import BACKENDS, DEFAULT_MODEL, DEFAULT_VISION_MODEL

# --- サーバー本体（LLM 実行） ----------------------------------------------
from .server import (
    DEFAULT_BACKEND,
    MTP_DRAFTERS,
    LocalServer,
    ServerConfig,
    ServerPool,
    build_command,
    build_pool_configs,
    default_backend,
    find_pids_on_port,
    install_shutdown_handlers,
    is_ready,
    list_models,
    model_available,
    models_match,
    parallel_supported,
    parse_host_port,
    resolve_drafter,
    running_model,
    stop_pid,
)

# --- ゲートウェイ -----------------------------------------------------------
from .router import RouterServer, needs_vision
from .gateway import (
    DEFAULT_BASE_URL,
    ServerHandle,
    ServerNotRunningError,
    check_model_served,
    ensure_server,
)

if TYPE_CHECKING:  # 型チェック時だけ実体を見せる（実行時は遅延 import）
    from .client import LLMClient, build_user_content, connect, to_image_url

# openai を必要とする client 系は遅延ロード。コア利用者に openai を強制しない。
_CLIENT_EXPORTS = {"LLMClient", "connect", "to_image_url", "build_user_content"}


def __getattr__(name: str):
    if name in _CLIENT_EXPORTS:
        from . import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 定数
    "DEFAULT_MODEL",
    "DEFAULT_VISION_MODEL",
    "DEFAULT_BACKEND",
    "DEFAULT_BASE_URL",
    "BACKENDS",
    # LLM 実行
    "LocalServer",
    "ServerConfig",
    "ServerPool",
    "build_command",
    "build_pool_configs",
    "default_backend",
    "install_shutdown_handlers",
    "find_pids_on_port",
    "stop_pid",
    # 接続状態の確認
    "is_ready",
    "list_models",
    "running_model",
    "model_available",
    "models_match",
    "parallel_supported",
    "parse_host_port",
    # MTP（投機的デコード）
    "resolve_drafter",
    "MTP_DRAFTERS",
    # ゲートウェイ
    "RouterServer",
    "needs_vision",
    "ensure_server",
    "ServerHandle",
    "ServerNotRunningError",
    "check_model_served",
    # 高レベルクライアント（任意 extra: local-llm-server[client]）
    "LLMClient",
    "connect",
    "to_image_url",
    "build_user_content",
]
