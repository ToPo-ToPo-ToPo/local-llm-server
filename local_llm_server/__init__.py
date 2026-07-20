"""local-llm-server — gateway.toml で定義するマルチモデルゲートウェイ（サーバー）。

このパッケージは**ゲートウェイ・サーバー専用**。運用は `gw` の CLI サブコマンドで行う:

  - `gw start/stop/status/ps/list/log/max/mtp/update` … ./gateway.toml のゲートウェイを
              裏で常駐起動し、状態確認・停止などを行う（cli:main）。裏で常駐するゲートウェイ
              本体は `python -m local_llm_server`（__main__）。

**クライアント（接続する側）は別パッケージ `local-llm-client`** に切り出した。エージェントは
そちらの `LLMClient` / `connect` を使う（または素の `openai` SDK で base_url を指す）。本パッケージは
`openai` に依存しない（純粋な HTTP 転送＋プロセス管理。推論バックエンドは extra で導入）。

公開 API（`__all__`）はゲートウェイの運用だけ。`LocalServer` 等の低レベル経路は
デーモンが内部で使うため import は残すが非公開・サポート対象外。推論バックエンドは
extra `local-llm-server[mlx]` で導入する。
"""
from __future__ import annotations

# --- 既定値・定数 -----------------------------------------------------------
from .constants import BACKENDS, DEFAULT_MODEL, DEFAULT_VISION_MODEL, log_dir

# --- サーバー本体（LLM 実行） ----------------------------------------------
from .server import (
    DEFAULT_BACKEND,
    MTP_DRAFTERS,
    LocalServer,
    ServerConfig,
    build_command,
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

# --- マルチモデルゲートウェイ（gateway.toml で複数モデルを 1 ポートに束ねる） ----
from .daemon import (
    CapacityError,
    GatewayConfig,
    GatewayServer,
    ModelManager,
    load_gateway_config,
    run_gateway,
)

# 公開 API は「ゲートウェイの運用」だけに絞る（サーバー専用パッケージ）。クライアント
# （LLMClient / connect）は別パッケージ local-llm-client へ移動した。
__all__ = [
    "run_gateway",        # gateway.toml のゲートウェイを起動（__main__ が呼ぶ本体）
    "load_gateway_config",  # ./gateway.toml の読み込み
    "GatewayConfig",      # 読み込んだゲートウェイ設定
    "server_status",      # 稼働状態（応答可否・pid・モデル・ログ）
]
