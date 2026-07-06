"""`local-llm-server` コマンド: ./gateway.toml を読んでマルチモデルゲートウェイを起動する。

カレントディレクトリの `./gateway.toml`（モデルカタログ）を 1 つの真実として、公開ポートに
ゲートウェイを立てる。クライアントは公開ポートへ接続し `model` でモデルを選ぶ。モデルは
初回リクエスト時に遅延起動し、`max_resident` 超過で LRU 退避、`idle_timeout` で自動アンロード
する（→ docs / examples の gateway.toml）。

ターミナルだけで完結する運用フラグを備える:

  * 引数なし   … TUI ダッシュボードを起動（対話端末のとき。ゲートウェイを裏で常駐させ、状態を
                 自動更新表示しつつ s/r/g/l/q と `:` コマンドで操作する → tui.py）
  * --headless … TUI を使わずフォアグラウンドでゲートウェイ実行（パイプ/CI/裏起動向け。
                 端末が非 TTY のときは自動でこちらになる）
  * --start    … バックグラウンド常駐起動（ターミナルを離す。Ollama 流）
  * --stop     … 停止（配下のモデルサーバーも含めて止める）
  * --status   … 状態表示（応答可否・PID・提供モデル・ログパス）
  * --restart  … 停止してからバックグラウンド再起動（gateway.toml 変更の反映に）

CLI と TUI は同じ運用基盤（server.py）を共有する。
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
    local_connect_host,
    pid_looks_like_ours,
    primary_lan_ip,
    server_status,
    start_gateway_background,
    stop_pid,
)


def _interactive_tty() -> bool:
    """TUI を出してよい対話端末か（stdin/stdout が TTY）。

    バックグラウンド起動（出力をログへリダイレクト）やパイプ・CI では False になり、
    呼び出し側はフォアグラウンドのゲートウェイ実行に落とす（curses は端末が要る）。
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


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
            # ポート番号だけで殺すと、たまたま同じポートを使う無関係なプロセス
            # （別プロジェクトの開発サーバー等）を巻き添えにするので、コマンドラインで確認する。
            if not pid_looks_like_ours(pid):
                print(
                    f"Port {port}: pid {pid} does not look like a local-llm-server "
                    "process; leaving it alone.",
                    file=sys.stderr,
                )
                continue
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
    # bind 先が 0.0.0.0 等でも、自分自身への接続はループバックで行う（"0.0.0.0" 宛は不可搬）。
    local_host = local_connect_host(gcfg.host)
    base_url = f"http://{gcfg.host}:{gcfg.port}/v1"
    if server_status(local_host, gcfg.port) is not None:
        print(f"Gateway already running on port {gcfg.port} ({base_url}).", file=sys.stderr)
        return 0
    print(f"Starting gateway in the background on port {gcfg.port}...", file=sys.stderr)
    try:
        pid = start_gateway_background(os.getcwd(), local_host, gcfg.port)
    except (RuntimeError, TimeoutError) as exc:
        print(f"Failed to start gateway: {exc}", file=sys.stderr)
        return 1
    print(f"Gateway started (pid {pid}).", file=sys.stderr)
    print(f"  public: {base_url}", file=sys.stderr)
    if gcfg.host in ("0.0.0.0", "::", "", "*"):
        lan = primary_lan_ip()
        if lan:
            print(f"  reachable from LAN: http://{lan}:{gcfg.port}/v1", file=sys.stderr)
    print(f"  log:    {gateway_log_path(gcfg.port)}", file=sys.stderr)
    print("Stop it with `local-llm-server --stop`.", file=sys.stderr)
    return 0


def mtp_report(model: str | None) -> tuple[str, int]:
    """使う予定のモデルに必要な MTP ドラフターを、ダウンロード前に調べて文面にする。

    CLI の `--check-mtp` と TUI の `mtp` コマンドが共有する本体。対応表（MTP_DRAFTERS）の
    辞書引きとローカルキャッシュ確認だけで、モデルのダウンロードは一切しない
    （`resolve_drafter` / `mtp_status` と同じく非破壊）。gateway.toml も見ないので、
    どのディレクトリからでも実行できる。model を省略（None/空）すると対応表を全件、
    取得状況つきで並べる。

    戻り値: (表示テキスト, 終了コード)。指定モデルが MTP 非対応なら 1、
    それ以外（ready / available / 一覧表示）は 0。
    """
    from .server import MTP_DRAFTERS, mtp_status

    def _describe(target: str) -> str:
        drafter = MTP_DRAFTERS[target]
        # mtp_status は "ready"（ドラフター取得済み）/ "available"（未取得）を返す。DL はしない。
        if mtp_status(target) == "ready":
            return (
                f"{target}\n"
                f"    drafter: {drafter}  [ready — 取得済み。そのまま MTP が効く]"
            )
        return (
            f"{target}\n"
            f"    drafter: {drafter}  [available — 未取得]\n"
            f"    hf download {drafter}"
        )

    if not model:
        lines = ['MTP 対応モデル（mlx-vlm・draft_model="auto" で自動解決）:']
        lines.extend(f"  {_describe(target)}" for target in sorted(MTP_DRAFTERS))
        return "\n".join(lines), 0

    if model not in MTP_DRAFTERS:
        return (
            f"{model}: MTP 非対応（対応表に無い）。使うなら gateway.toml の draft_model に "
            "ドラフターの HF id を明示してください。対応モデル一覧は引数なしの "
            "`--check-mtp`（TUI では `mtp`）で表示。",
            1,
        )
    return _describe(model), 0


def _check_mtp(model: str | None) -> int:
    """`--check-mtp` 用の入口。結果はパイプで拾えるよう stdout に出す。"""
    text, code = mtp_report(model)
    print(text)
    return code


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
            "connect to the public port and select a model via `model`; the client never starts a "
            "server. With no arguments an interactive terminal opens the TUI dashboard (which "
            "keeps the gateway resident in the background); use --headless to run the gateway in "
            "the foreground, or operate it with --start / --stop / --status / --restart."
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the gateway in the foreground without the TUI dashboard (for pipes, CI, or "
        "background launch). A non-interactive terminal selects this automatically.",
    )
    parser.add_argument(
        "--check-mtp",
        metavar="MODEL",
        nargs="?",
        const="",  # フラグはあるが値なし → 対応表を全件表示
        default=None,  # フラグ自体が無い
        help="Show which MTP drafter a model needs (and whether it is already downloaded) without "
        "downloading anything, then exit. Give a model id to check one; omit it to list every "
        "supported model. Does not read gateway.toml, so it works from any directory.",
    )
    args = parser.parse_args(argv)

    # --check-mtp: gateway.toml も HF ダウンロードも要らない純粋な参照。設定ロードより前に処理する。
    if args.check_mtp is not None:
        return _check_mtp(args.check_mtp)

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
        return _status_servers(local_connect_host(gcfg.host), [gcfg.port])
    if args.restart:
        _stop_servers(all_ports)  # 動いていなければ no-op
        return _start_background(gcfg)
    if args.start:
        return _start_background(gcfg)

    # 既定（引数なし）: 対話端末なら TUI ダッシュボード（ゲートウェイを裏で常駐させて監視・操作）。
    # それ以外（--headless / 非 TTY / 裏起動・パイプ・CI）はフォアグラウンドでゲートウェイを実行する。
    if not args.headless and _interactive_tty():
        try:
            from .tui import run_tui
        except ImportError:
            print(
                "TUI is unavailable (curses not found; on Windows: `uv add windows-curses`). "
                "Running headless instead.",
                file=sys.stderr,
            )
        else:
            return run_tui(gcfg)

    # フォアグラウンド実行。kill / ターミナルを閉じる（SIGTERM・SIGHUP）でも下流の finally
    # （gateway の manager.shutdown）を必ず通し、配下のモデルサーバーを孫プロセスとして残さない。
    install_shutdown_handlers()
    return run_gateway(gcfg)


if __name__ == "__main__":
    raise SystemExit(main())
