"""`local-llm-server` コマンド: ./gateway.toml を読んでマルチモデルゲートウェイを起動する。

カレントディレクトリの `./gateway.toml`（モデルカタログ）を 1 つの真実として、公開ポートに
ゲートウェイを立てる。クライアントは公開ポートへ接続し `model` でモデルを選ぶ。モデルは
初回リクエスト時に遅延起動し、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード
する（→ docs / examples の gateway.toml）。

このコマンドは**ゲートウェイ本体（フォアグラウンド実行）**だけを担う。起動・停止・監視といった
運用操作は**アプリ（トレイ GUI）**に一本化した（`local-llm-server-gui`）。アプリはこのコマンドを
バックグラウンドで起動して常駐させる。
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib

from .daemon import load_gateway_config, run_gateway
from .server import install_shutdown_handlers


def _resolve_config() -> str | None:
    """使う gateway.toml を決める。**カレントディレクトリの `./gateway.toml` のみ**。

    存在すればそのパス、無ければ None（呼び出し側がエラーにする）。場所は CWD 固定で、
    位置引数やホーム等の外部は見ない（「gateway.toml は CWD に置く」という 1 ルール）。
    """
    path = os.path.join(os.getcwd(), "gateway.toml")
    return path if os.path.isfile(path) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-llm-server",
        description=(
            "Run the local LLM gateway defined by ./gateway.toml in the current directory "
            "(model catalog) in the foreground. Operate it (start in the background, stop, "
            "monitor) from the app instead: `local-llm-server-gui` (see --install-app)."
        ),
    )
    parser.parse_args(argv)  # 運用フラグは廃止。--help と未知引数の拒否のためだけに使う。

    config_path = _resolve_config()
    if config_path is None:
        parser.error(
            "./gateway.toml not found in the current directory. Create one here "
            "(see the gateway.toml example in the repo), then run again from that directory."
        )
    try:
        gcfg = load_gateway_config(config_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"Failed to load the gateway config '{config_path}': {exc}")

    # kill / ターミナルを閉じる（SIGTERM・SIGHUP）でも下流の finally（gateway の
    # manager.shutdown）を必ず通し、配下のモデルサーバーを孫プロセスとして残さない。
    install_shutdown_handlers()
    return run_gateway(gcfg)


if __name__ == "__main__":
    raise SystemExit(main())
