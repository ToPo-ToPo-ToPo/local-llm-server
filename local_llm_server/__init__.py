"""local-llm-server — gateway.toml で定義するマルチモデルゲートウェイ。

サポートする使い方は次の 2 つ:

  1. ゲートウェイの運用（**このリポジトリで 1 プロセス起動する**）
       - `local-llm-server`      … ./gateway.toml のゲートウェイを起動/停止/状態（cli:main）
       - `local-llm-server-gui`  … 状態を監視する常駐トレイ GUI（gui:main）
  2. クライアントから接続（エージェント共通の高レベルクライアント）
       - `LLMClient` … 公開ポートの OpenAI 互換 API に繋ぐ。respond() / ストリーム / 画像入力 /
         thinking 切替などを共通化し、**エージェントごとの再実装を防ぐ**（素の openai SDK も
         `self.openai` でそのまま使える）。
       - `connect` … 起動中のゲートウェイに繋ぐワンライナー。未起動なら親切なエラー
         （`ServerNotRunningError`）。**サーバーは自前で起動しない**。

公開 API（`__all__`）は上記＝「ゲートウェイの運用」と「クライアント接続」に絞る。

**サーバーを自前で起動する経路**（`ensure_server`（自動起動）/ `LocalServer` /
`ServerPool` / `RouterServer` 等）と低レベル内部実装は**非公開・サポート対象外**にする
（サーバーを立てるのはゲートウェイ 1 箇所だけ、という運用にするため）。後方互換のため
import 自体は残すが `__all__` には載せない。推論バックエンドは extra
`local-llm-server[mlx]`、監視 GUI は `local-llm-server[gui]` で導入する。
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
    daemon_log_path,
    install_shutdown_handlers,
    ignore_shutdown_signals,
    is_ready,
    list_models,
    model_available,
    models_match,
    parallel_supported,
    parse_host_port,
    resolve_drafter,
    running_model,
    server_status,
    stop_pid,
)
from .constants import project_cache_dir

# --- マルチモデルゲートウェイ（gateway.toml で複数モデルを 1 ポートに束ねる） ----
from .daemon import (
    CapacityError,
    GatewayConfig,
    GatewayServer,
    ModelManager,
    load_gateway_config,
    run_gateway,
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

# --- 高レベルクライアント + リクエスト整形ヘルパ ---------------------------
from .client import (
    LLMClient,
    build_user_content,
    connect,
    thinking_extra_body,
    to_image_url,
)

# 公開 API は「ゲートウェイの運用」と「クライアント接続」だけに絞る。サーバーを自前
# 起動する経路（ensure_server（自動起動）/ LocalServer / ServerPool / RouterServer 等）と
# 低レベル内部実装は、後方互換のため import 可能なまま残すが __all__ には載せない
# （＝非公開・サポート対象外。サーバーを立てるのはゲートウェイ 1 箇所だけ）。
__all__ = [
    # ゲートウェイの運用（起動・設定・状態）
    "run_gateway",        # gateway.toml のゲートウェイを起動（cli:main が呼ぶ本体）
    "load_gateway_config",  # ./gateway.toml の読み込み
    "GatewayConfig",      # 読み込んだゲートウェイ設定
    "server_status",      # 稼働状態（応答可否・pid・モデル・ログ）
    # クライアント接続（エージェント共通の高レベルクライアント＋整形ヘルパ）
    "LLMClient",
    "connect",            # 起動中ゲートウェイに繋ぐ（自動起動しない。未起動はエラー）
    "ServerNotRunningError",  # connect の未起動エラー（呼び出し側が捕捉できるよう公開）
    "to_image_url",
    "build_user_content",
    "thinking_extra_body",
    # 接続先の確認・既定値（クライアントから使う読み取り専用ユーティリティ）
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_VISION_MODEL",
    "is_ready",
    "list_models",
    "running_model",
    "model_available",
    "models_match",
]
