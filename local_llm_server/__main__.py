"""`python -m local_llm_server` — ゲートウェイ本体（常駐デーモン）。**`gw start` 専用の内部入口**。

`gw start`（cli）が裏で常駐させる実体。`server.start_gateway_background` がこのモジュールを
新セッション（POSIX）/ DETACHED_PROCESS（Windows）の別プロセスとして spawn し、出力は
ログへ逃がす。端末を持たず、CWD の `./gateway.toml` のゲートウェイをフォアグラウンドで実行する。
自動更新（PyPI 新版を git で追従）が適用されると、このプロセスを新コードで execv し直す
（execv は環境変数を引き継ぐので、下の spawn ガードは自動更新の再起動でも通る）。

直接の `python -m local_llm_server` は**起動を拒否する**——起動の入口は `gw start` の
1 つだけ（設定・ログ・PID 記録の位置が常に一貫し、出所不明のゲートウェイが立たない）。

kill / ターミナルを閉じる（SIGTERM・SIGHUP）でも下流の finally（gateway の
manager.shutdown）を必ず通し、配下のモデルサーバーを孫プロセスとして残さない。
"""
from __future__ import annotations

import os
import sys
import tomllib

from .cli import resolve_config
from .daemon import load_gateway_config, run_gateway
from .server import install_shutdown_handlers

# `gw start`（start_gateway_background）が spawn 時に付ける内部マーク。これが無い＝
# 手で直接起動しようとしている、なので拒否する（正規ルートは gw start だけ）。
_SPAWN_MARK = "LOCAL_LLM_GW_LAUNCHER"


def main() -> int:
    if os.environ.get(_SPAWN_MARK) != "cli":
        print(
            "このモジュールは gw が内部で使う起動口です。ゲートウェイの起動は `gw start` で行ってください"
            "（設定は ~/.config/local-llm-server/gateway.toml の 1 箇所、状態確認は `gw status`）。",
            file=sys.stderr,
        )
        return 2
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
