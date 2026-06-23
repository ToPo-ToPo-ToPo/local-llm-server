"""local-llm-server — ローカルLLMを OpenAI 互換 API として起動・管理する拡張サーバー。

単なる OpenAI 互換サーバーの起動だけでなく、複数エージェントで重複しがちな処理を
備える:

  - LLM 実行          : LocalServer / ServerConfig / ServerPool（mlx / mlx-vlm / llama.cpp）
  - ゲートウェイ      : RouterServer（テキスト/vision 振り分け）, ensure_server（相乗り/自動起動）
  - MTP（投機的デコード）: resolve_drafter / MTP_DRAFTERS
  - 高レベルクライアント : LLMClient / connect（公式 openai SDK を土台。コア依存）

公開 API は __all__ に列挙したものだけ。サーバー起動・ゲートウェイ・MTP 解決は標準
ライブラリのみ、高レベルクライアントは openai（コア依存）を使う。推論バックエンド本体
だけ extra `local-llm-server[mlx]` で導入する。
"""
from __future__ import annotations

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

# --- 高レベルクライアント（標準ライブラリのみ） ----------------------------
from .client import LLMClient, build_user_content, connect, to_image_url

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
    # 高レベルクライアント（標準ライブラリのみ・追加依存なし）
    "LLMClient",
    "connect",
    "to_image_url",
    "build_user_content",
]
