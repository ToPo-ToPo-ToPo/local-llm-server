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
    gateway_log_path,
    install_shutdown_handlers,
    server_status,
    start_gateway_background,
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

    公開ポート（ゲートウェイ）と内部モデルポートを LISTEN しているプロセスを特定し
    （macOS / Linux は lsof、Windows は netstat）、プロセスツリーごと止める（POSIX は
    SIGTERM→SIGKILL、Windows は taskkill /T /F）。ゲートウェイは終了時に配下のモデル
    サーバーも止めるが、ここで内部ポートも直接止めることで取り残しを防ぐ。
    1つでも止めれば 0、見つからなければ 1 を返す。
    """
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


def _start_background(gcfg) -> int:
    """ゲートウェイをバックグラウンドで常駐起動する（--start 用）。

    ターミナルを占有せずに常駐させる（Ollama 流）。既に起動済みなら何もしない。
    起動して応答可能になれば 0、失敗/タイムアウトは 1。
    """
    base_url = f"http://{gcfg.host}:{gcfg.port}/v1"
    if server_status(gcfg.host, gcfg.port) is not None:
        print(f"Gateway already running on port {gcfg.port} ({base_url}).", file=sys.stderr)
        return 0
    print(f"Starting gateway in the background on port {gcfg.port}...", file=sys.stderr)
    try:
        pid = start_gateway_background(os.getcwd(), gcfg.host, gcfg.port)
    except (RuntimeError, TimeoutError) as exc:
        print(f"Failed to start gateway: {exc}", file=sys.stderr)
        return 1
    print(f"Gateway started (pid {pid}).", file=sys.stderr)
    print(f"  public: {base_url}", file=sys.stderr)
    print(f"  log:    {gateway_log_path(gcfg.port)}", file=sys.stderr)
    print("Stop it with `local-llm-server --stop`.", file=sys.stderr)
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
            print(f"    model log: {st['log_path']}", file=sys.stderr)
        gw_log = gateway_log_path(port)
        if os.path.exists(gw_log):
            print(f"    gateway log: {gw_log}", file=sys.stderr)
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
        "--start",
        action="store_true",
        help="Start the gateway in the background (detached) and return, instead of running it "
        "in the foreground. Use this to keep it resident without holding a terminal.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Stop the running gateway (if any) and start it again in the background. Use after "
        "editing ./gateway.toml to apply the changes.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the gateway defined by ./gateway.toml (also stops every model server it started) "
        "instead of starting one. macOS / Linux / Windows.",
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

    # --start / --restart / --stop / --status: 設定の host/port を 1 つの真実として対象を特定する。
    # 公開ポート＋配下モデルの内部ポートをまとめて止めると、協調シャットダウンが完走しなくても
    # ロード済みモデルをメモリに残さない（取りこぼし防止）。
    all_ports = [gcfg.port] + [m.port for m in gcfg.models]
    if args.stop:
        return _stop_servers(all_ports)
    if args.status:
        return _status_servers(gcfg.host, [gcfg.port])
    if args.restart:
        _stop_servers(all_ports)  # 動いていなければ no-op
        return _start_background(gcfg)
    if args.start:
        return _start_background(gcfg)

    # 既定（引数なし）はフォアグラウンド起動。kill / ターミナルを閉じる（SIGTERM・SIGHUP）でも
    # 下流の finally（gateway の manager.shutdown）を必ず通し、配下のモデルサーバーを孫プロセス
    # として残さない。バックグラウンド常駐は --start（GUI からも起動できる）。
    install_shutdown_handlers()
    return run_gateway(gcfg)


if __name__ == "__main__":
    raise SystemExit(main())
