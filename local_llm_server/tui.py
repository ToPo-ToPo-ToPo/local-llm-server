"""ターミナル常駐の TUI ダッシュボード（`local-llm-server` の既定起動）。

`./gateway.toml` を読み、ゲートウェイを裏で常駐させて `GET /admin/status` を定期ポーリングし、
状態を**全画面で自動更新表示**する。`s`/`r`/`g`/`l`/`q` の単キーと、`:` で開く打ち込みコマンド
（stop/restart/start/log/quit）でゲートウェイを操作できる。トレイ GUI アプリ（pystray）の
ターミナル版で、**リポジトリのコードだけで完結**する（外部にアプリ/ランチャを置かない）。

curses は標準ライブラリ（macOS / Linux）。Windows のみ別途 `windows-curses` が要る。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time

from .server import (
    find_pids_on_port,
    gateway_admin_status,
    gateway_log_path,
    is_ready,
    server_status,
    start_gateway_background,
    stop_pid,
)


def merge_status(gcfg, admin: dict | None) -> dict:
    """gateway.toml のカタログと `/admin/status` のライブ状態を1つのビューに統合する（純粋関数）。

    カタログの全モデルを並べ（未起動も「unloaded」で見せる）、起動中のものはライブ状態
    （loaded/idle/busy・処理中数・累計・アイドル自動解放までの残り）を重ねる。色付けや描画を
    含まないのでそのままテストできる。
    """
    live = {m["model"]: m for m in (admin or {}).get("models", [])}
    idle_timeout = gcfg.idle_timeout
    rows = []
    for c in gcfg.models:
        m = live.get(c.model)
        if not m or not m.get("loaded"):
            state, inflight, requests, remaining = "unloaded", 0, (m or {}).get("requests", 0), None
        else:
            inflight = int(m.get("inflight", 0))
            requests = int(m.get("requests", 0))
            idle_for = m.get("idle_for")
            if inflight > 0:
                state, remaining = "busy", None
            else:
                state = "idle"
                remaining = (
                    max(0.0, idle_timeout - idle_for)
                    if (idle_timeout and idle_for is not None)
                    else None
                )
        rows.append({
            "model": c.model,
            "backend": c.backend,
            "port": c.port,
            "state": state,
            "inflight": inflight,
            "requests": requests,
            "idle_remaining": remaining,
        })
    ready = bool(admin) or is_ready(f"http://{gcfg.host}:{gcfg.port}/v1")
    return {
        "ready": ready,
        "uptime": (admin or {}).get("uptime"),
        "requests": (admin or {}).get("requests", sum(r["requests"] for r in rows)),
        "max_resident": gcfg.max_resident,
        "idle_timeout": idle_timeout,
        "models": rows,
    }


def _fmt_hms(seconds) -> str:
    """秒を H:MM:SS / M:SS に整形する（None は「—」）。"""
    if seconds is None:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _open_log_in_pager(port: int) -> None:
    """ゲートウェイログをページャ（$PAGER / less / more）で開く（ターミナル内で完結）。"""
    path = gateway_log_path(port)
    if not os.path.exists(path):
        return
    pager = os.environ.get("PAGER") or shutil.which("less") or shutil.which("more")
    if not pager:
        return
    args = [pager, "+G", path] if os.path.basename(pager).startswith("less") else [pager, path]
    subprocess.call(args)


# --- curses 描画 -----------------------------------------------------------

def run_tui(gcfg) -> int:
    """TUI を起動する。curses が無ければ呼び出し側が拾えるよう ImportError を素通しする。"""
    import curses

    return curses.wrapper(_main, gcfg)


def _main(stdscr, gcfg) -> int:
    import curses

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)  # getch は最大 200ms 待つ（その間に画面が固まらない）
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)

    host, port = gcfg.host, gcfg.port
    all_ports = [port] + [m.port for m in gcfg.models]

    state = {"admin": None, "msg": "", "busy": None, "last_poll": 0.0}
    lock = threading.Lock()

    def poll() -> None:
        admin = gateway_admin_status(host, port)
        with lock:
            state["admin"] = admin

    def run_action(label: str, fn) -> None:
        """start/stop/restart を別スレッドで実行（UI を固めない）。"""
        with lock:
            if state["busy"]:
                return
            state["busy"], state["msg"] = label, ""

        def worker() -> None:
            try:
                fn()
                done = ""
            except Exception as exc:  # noqa: BLE001 - 失敗はメッセージで見せる
                done = f"{label} failed: {exc}"
            with lock:
                state["busy"], state["msg"] = None, done
            poll()

        threading.Thread(target=worker, daemon=True).start()

    def do_start() -> None:
        run_action("starting gateway…", lambda: start_gateway_background(os.getcwd(), host, port))

    def do_stop() -> None:
        def _stop() -> None:
            for p in all_ports:
                for pid in find_pids_on_port(p):
                    stop_pid(pid)
        run_action("stopping gateway…", _stop)

    def do_restart() -> None:
        def _restart() -> None:
            for p in all_ports:
                for pid in find_pids_on_port(p):
                    stop_pid(pid)
            start_gateway_background(os.getcwd(), host, port)
        run_action("restarting gateway…", _restart)

    # 起動時にゲートウェイを裏で常駐させる（既定起動＝「アプリを開く」感覚）。
    if server_status(host, port) is None:
        do_start()
    poll()

    while True:
        now = time.monotonic()
        if now - state["last_poll"] >= 1.0:
            poll()
            state["last_poll"] = now
        with lock:
            snapshot = dict(state)
        _draw(stdscr, gcfg, merge_status(gcfg, snapshot["admin"]),
              busy=snapshot["busy"], msg=snapshot["msg"])

        ch = stdscr.getch()
        if ch == -1:
            continue
        key = chr(ch) if 0 <= ch < 256 else ""
        if key in ("q", "Q"):
            return 0
        if key == "s":
            do_stop()
        elif key == "r":
            do_restart()
        elif key == "g":
            do_start()
        elif key == "l":
            curses.def_prog_mode()
            curses.endwin()
            _open_log_in_pager(port)
            curses.reset_prog_mode()
            stdscr.refresh()
        elif key == ":":
            cmd = _read_command(stdscr)
            again = _dispatch_command(cmd, do_start, do_stop, do_restart, port, curses, stdscr)
            if again == "quit":
                return 0


def _dispatch_command(cmd, do_start, do_stop, do_restart, port, curses, stdscr):
    cmd = (cmd or "").strip().lower()
    if cmd in ("q", "quit", "exit"):
        return "quit"
    if cmd in ("s", "stop"):
        do_stop()
    elif cmd in ("r", "restart"):
        do_restart()
    elif cmd in ("g", "start"):
        do_start()
    elif cmd in ("l", "log"):
        curses.def_prog_mode()
        curses.endwin()
        _open_log_in_pager(port)
        curses.reset_prog_mode()
        stdscr.refresh()
    return None


def _read_command(stdscr) -> str:
    """画面下端で `:` プロンプトの 1 行入力を読む（Enter 確定 / Esc 取消）。"""
    import curses

    h, w = stdscr.getmaxyx()
    curses.curs_set(1)
    stdscr.nodelay(False)
    buf = ""
    while True:
        stdscr.move(h - 1, 0)
        stdscr.clrtoeol()
        stdscr.addnstr(h - 1, 0, ":" + buf, w - 1)
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (10, 13):  # Enter
            break
        if ch == 27:  # Esc
            buf = ""
            break
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            buf = buf[:-1]
        elif 32 <= ch < 256:
            buf += chr(ch)
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)
    return buf


def _draw(stdscr, gcfg, view: dict, *, busy: str | None, msg: str) -> None:
    import curses

    stdscr.erase()
    h, w = stdscr.getmaxyx()

    def put(y, x, text, attr=0):
        if 0 <= y < h and x < w:
            stdscr.addnstr(y, x, text, max(0, w - x - 1), attr)

    green = curses.color_pair(1)
    yellow = curses.color_pair(2)
    red = curses.color_pair(3)
    cyan = curses.color_pair(4)
    dim = curses.A_DIM
    bold = curses.A_BOLD

    put(0, 0, "local-llm-server", cyan | bold)
    put(0, 17, "· gateway monitor", dim)
    refresh = "starting…" if busy else "⟳ 1s"
    put(0, max(0, w - len(refresh) - 1), refresh, yellow if busy else dim)

    if view["ready"]:
        put(2, 0, "●", green)
        put(2, 2, "ready", bold)
    elif busy:
        put(2, 0, "●", yellow)
        put(2, 2, "starting", bold)
    else:
        put(2, 0, "●", red)
        put(2, 2, "stopped", bold)
    meta = f"  port {gcfg.port}   up {_fmt_hms(view['uptime'])}   reqs {view['requests']:,}"
    put(2, 9, meta, dim)

    put(4, 0, f"{'MODEL':<34} {'STATE':<10} {'PORT':>5} {'INFLT':>6} {'IDLE→UNLOAD':>12} {'REQS':>8}", dim)
    y = 5
    for r in view["models"]:
        if y >= h - 3:
            break
        name = r["model"].split("/")[-1]
        put(y, 0, f"{name:<34.34}")
        st = r["state"]
        color = {"busy": green, "idle": yellow, "unloaded": dim}.get(st, dim)
        dot = {"busy": "●", "idle": "○", "unloaded": "·"}.get(st, "·")
        put(y, 35, f"{dot} {st}", color)
        put(y, 46, f"{r['port']:>5}", cyan)
        put(y, 52, f"{r['inflight']:>6}")
        put(y, 59, f"{_fmt_hms(r['idle_remaining']):>12}")
        put(y, 72, f"{r['requests']:>8,}")
        y += 1

    policy = (
        f"policy  max_resident {view['max_resident'] if view['max_resident'] is not None else '∞'}"
        f"   idle_timeout {int(view['idle_timeout']) if view['idle_timeout'] else 'off'}s"
    )
    put(h - 2, 0, policy, dim)
    if busy or msg:
        put(h - 2, max(0, w - 30), (busy or msg)[:28], yellow if busy else red)

    footer = "[s]top  [r]estart  [g]start  [l]og  [q]uit   : command"
    put(h - 1, 0, footer, cyan)


def main(argv: list[str] | None = None) -> int:
    """`python -m local_llm_server.tui` 用の薄い入口（通常は CLI 既定から呼ばれる）。"""
    from .cli import _resolve_config
    from .daemon import load_gateway_config

    config_path = _resolve_config()
    if config_path is None:
        print("./gateway.toml not found in the current directory.", file=sys.stderr)
        return 2
    return run_tui(load_gateway_config(config_path))


if __name__ == "__main__":
    raise SystemExit(main())
