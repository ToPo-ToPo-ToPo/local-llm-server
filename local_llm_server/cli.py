"""`local-llm-server` コマンド: ./gateway.toml を読んでマルチモデルゲートウェイを起動する。

カレントディレクトリの `./gateway.toml`（モデルカタログ）を 1 つの真実として、公開ポートに
ゲートウェイを立てる。クライアントは公開ポートへ接続し `model` でモデルを選ぶ。モデルは
初回リクエスト時に遅延起動し、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード
する（→ docs / examples の gateway.toml）。`--status` / `--stop` で運用する。
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib

from .daemon import load_gateway_config, run_gateway
from .server import (
    find_pids_on_port,
    install_shutdown_handlers,
    server_status,
    stop_pid,
)


def _resolve_config() -> str | None:
    """使う gateway.toml を決める。**カレントディレクトリの `./gateway.toml` のみ**。

    存在すればそのパス、無ければ None（呼び出し側がエラーにする）。場所は CWD 固定で、
    位置引数やホーム等の外部は見ない（「gateway.toml は CWD に置く」という 1 ルール）。
    """
    path = os.path.join(os.getcwd(), "gateway.toml")
    return path if os.path.isfile(path) else None


def _stop_servers(ports: list[int]) -> int:
    """指定ポートで動いているプロセスを探して停止する（--stop 用）。

    公開ポート（ゲートウェイ）と内部モデルポートを LISTEN しているプロセスを lsof で
    特定し、プロセスグループごと SIGTERM→SIGKILL で止める。ゲートウェイは終了時に配下の
    モデルサーバーも止めるが、ここで内部ポートも直接止めることで取り残しを防ぐ。
    1つでも止めれば 0、見つからなければ 1 を返す。
    """
    if os.name != "posix":
        print(
            "--stop is supported on macOS / Linux only. On Windows, stop the gateway "
            "from its own window (Ctrl+C) or via Task Manager.",
            file=sys.stderr,
        )
        return 1
    stopped = False
    for port in ports:
        pids = find_pids_on_port(port)
        for pid in pids:
            print(f"Stopping server on port {port} (pid {pid})...", file=sys.stderr)
            if stop_pid(pid):
                stopped = True
    if not stopped:
        ports_str = ", ".join(str(p) for p in ports)
        print(f"No running server found on port(s): {ports_str}.", file=sys.stderr)
        return 1
    print("Stopped.", file=sys.stderr)
    return 0


def _status_servers(host: str, ports: list[int]) -> int:
    """指定ポートで動いているゲートウェイの状態を表示する（--status 用）。

    応答可否・PID・提供モデル（カタログ）・ログパスを並べる。1つでも見つかれば 0、
    皆無なら 1。`--stop` と違い POSIX 以外でも動く（応答可否は HTTP で判定）。
    """
    any_found = False
    for port in ports:
        st = server_status(host, port)
        if st is None:
            print(f"Port {port}: no gateway running.", file=sys.stderr)
            continue
        any_found = True
        state = "ready" if st["ready"] else "not responding (starting?)"
        pids = ", ".join(str(p) for p in st["pids"]) or "unknown"
        print(f"Port {port}: {state}  (pid {pids})  {st['base_url']}", file=sys.stderr)
        for model in st["models"]:
            print(f"    model: {model}", file=sys.stderr)
        if st["log_path"]:
            print(f"    log:   {st['log_path']}", file=sys.stderr)
    return 0 if any_found else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-llm-server",
        description=(
            "Run the local LLM gateway defined by ./gateway.toml in the current directory "
            "(model catalog). Host, port, models and MTP are all configured in that file. Clients "
            "connect to the public port and select a model via `model`; the client never starts a server."
        ),
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the gateway defined by ./gateway.toml (also stops every model server it started) "
        "instead of starting one. macOS / Linux only.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the status of the gateway defined by ./gateway.toml (ready state, pid, served "
        "models, log path) instead of starting one.",
    )
    args = parser.parse_args(argv)

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

    # --stop / --status: 設定の host/port を 1 つの真実として、対象を特定する。
    if args.stop:
        # 公開ポート（ゲートウェイ本体）に加え、配下のモデルサーバーの内部ポートも
        # 直接止める。ゲートウェイの協調シャットダウン（manager.shutdown）が何らかの
        # 理由で完走しなくても、ロード済みモデルをメモリに残さないための保険。
        ports = [gcfg.port] + [m.port for m in gcfg.models]
        return _stop_servers(ports)
    if args.status:
        return _status_servers(gcfg.host, [gcfg.port])

    # kill / ターミナルを閉じる（SIGTERM・SIGHUP）でも下流の finally（gateway の
    # manager.shutdown）を必ず通し、配下のモデルサーバーを孫プロセスとして残さない。
    install_shutdown_handlers()
    return run_gateway(gcfg)


if __name__ == "__main__":
    raise SystemExit(main())
