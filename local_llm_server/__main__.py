"""`python -m local_llm_server` — ゲートウェイ本体をヘッドレスで実行する内部ワーカー。

TUI（既定の `gw` 起動）が裏で常駐させる実体。`server.start_gateway_background` が
このモジュールを新セッション（POSIX）/ DETACHED_PROCESS（Windows）の別プロセスとして
spawn し、出力はログへ逃がす。ターミナルを持たない前提なので TUI は出さず、CWD の
`./gateway.toml` のゲートウェイをフォアグラウンドで実行する。

kill / ターミナルを閉じる（SIGTERM・SIGHUP）でも下流の finally（gateway の
manager.shutdown）を必ず通し、配下のモデルサーバーを孫プロセスとして残さない。
"""
from __future__ import annotations

import sys
import tomllib

from .daemon import load_gateway_config, run_gateway
from .server import install_shutdown_handlers
from .tui import resolve_config


def main() -> int:
    config_path = resolve_config()
    if config_path is None:
        print("./gateway.toml not found in the current directory.", file=sys.stderr)
        return 2
    try:
        gcfg = load_gateway_config(config_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"Failed to load ./gateway.toml: {exc}", file=sys.stderr)
        return 2
    install_shutdown_handlers()
    # config_path を渡して、gateway.toml の保存でポリシー設定を無停止反映（ホットリロード）。
    return run_gateway(gcfg, config_path=config_path)


if __name__ == "__main__":
    raise SystemExit(main())
