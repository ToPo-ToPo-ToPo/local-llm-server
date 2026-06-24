"""ターミナル常駐の TUI ダッシュボード（`local-llm-server` の既定起動）。

`./gateway.toml` を読み、ゲートウェイを裏で常駐させて `GET /admin/status` を定期ポーリングし、
状態を**全画面で自動更新表示**する。角丸ボックス罫線と落ち着いた配色で見やすく整え、`s`/`r`/`g`/
`l`/`q` の単キーと、`:` で開く打ち込みコマンド（stop/restart/start/log/quit）で操作できる。
トレイ GUI アプリのターミナル版で、**リポジトリのコードだけで完結**する（外部にアプリ/ランチャを
置かない）。curses は標準ライブラリ（macOS / Linux）。Windows のみ別途 `windows-curses` が要る。
"""
from __future__ import annotations

import locale
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

def _setup_colors():
    """落ち着いた配色を用意する（256色が使えればグレー階調＋暖色アクセント、無ければ 8 色）。"""
    import curses

    if not curses.has_colors():
        z = 0
        return {"accent": z, "green": z, "amber": z, "red": z, "blue": z,
                "text": z, "dim": curses.A_DIM, "border": curses.A_DIM}
    curses.start_color()
    curses.use_default_colors()
    c256 = curses.COLORS >= 256

    def mk(i, fg256, fg8):
        curses.init_pair(i, fg256 if c256 else fg8, -1)
        return curses.color_pair(i)

    soft = 0 if c256 else curses.A_DIM
    return {
        "accent": mk(1, 173, curses.COLOR_YELLOW),
        "green":  mk(2, 78,  curses.COLOR_GREEN),
        "amber":  mk(3, 179, curses.COLOR_YELLOW),
        "red":    mk(4, 167, curses.COLOR_RED),
        "blue":   mk(5, 110, curses.COLOR_CYAN),
        "text":   mk(6, 253, curses.COLOR_WHITE),
        "dim":    mk(7, 244, curses.COLOR_WHITE) | soft,
        "border": mk(8, 239, curses.COLOR_WHITE) | soft,
    }


def run_tui(gcfg) -> int:
    """TUI を起動する。curses が無ければ呼び出し側が拾えるよう ImportError を素通しする。"""
    import curses

    locale.setlocale(locale.LC_ALL, "")  # 罫線などのワイド/UTF-8 文字を curses に通す
    return curses.wrapper(_main, gcfg)


def _main(stdscr, gcfg) -> int:
    import curses

    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.nodelay(True)
    stdscr.timeout(200)  # getch は最大 200ms 待つ（その間に画面が固まらない）
    colors = _setup_colors()

    host, port = gcfg.host, gcfg.port
    all_ports = [port] + [m.port for m in gcfg.models]

    state = {"admin": None, "msg": "", "busy": None, "last_poll": 0.0,
             "cmd_mode": False, "cmd_buf": ""}
    lock = threading.Lock()

    def poll() -> None:
        admin = gateway_admin_status(host, port)
        with lock:
            state["admin"] = admin

    def run_action(label: str, fn) -> None:
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
        run_action("starting…", lambda: start_gateway_background(os.getcwd(), host, port))

    def do_stop() -> None:
        def _stop() -> None:
            for p in all_ports:
                for pid in find_pids_on_port(p):
                    stop_pid(pid)
        run_action("stopping…", _stop)

    def do_restart() -> None:
        def _restart() -> None:
            for p in all_ports:
                for pid in find_pids_on_port(p):
                    stop_pid(pid)
            start_gateway_background(os.getcwd(), host, port)
        run_action("restarting…", _restart)

    def open_log() -> None:
        curses.def_prog_mode()
        curses.endwin()
        _open_log_in_pager(port)
        curses.reset_prog_mode()
        stdscr.refresh()

    def run_cmd(cmd: str) -> bool:
        cmd = (cmd or "").strip().lower()
        if cmd in ("q", "quit", "exit"):
            return True
        {"s": do_stop, "stop": do_stop, "r": do_restart, "restart": do_restart,
         "g": do_start, "start": do_start, "l": open_log, "log": open_log}.get(cmd, lambda: None)()
        return False

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
            snap = dict(state)
        _draw(stdscr, gcfg, merge_status(gcfg, snap["admin"]), colors,
              busy=snap["busy"], msg=snap["msg"],
              cmd_mode=snap["cmd_mode"], cmd_buf=snap["cmd_buf"])

        ch = stdscr.getch()
        if ch == -1:
            continue

        if state["cmd_mode"]:
            if ch in (10, 13):  # Enter
                quit_ = run_cmd(state["cmd_buf"])
                state["cmd_mode"], state["cmd_buf"] = False, ""
                if quit_:
                    return 0
            elif ch == 27:  # Esc
                state["cmd_mode"], state["cmd_buf"] = False, ""
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                state["cmd_buf"] = state["cmd_buf"][:-1]
            elif 32 <= ch < 256:
                state["cmd_buf"] += chr(ch)
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
            open_log()
        elif key == ":":
            state["cmd_mode"], state["cmd_buf"] = True, ""


def _draw(stdscr, gcfg, view, colors, *, busy, msg, cmd_mode, cmd_buf) -> None:
    import curses

    A = colors
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    def put(y, x, s, attr=0):
        if 0 <= y < h and 0 <= x < w:
            try:
                stdscr.addnstr(y, x, s, max(0, w - x - 1), attr)
            except curses.error:
                pass

    def put_r(y, end, s, attr=0):  # end 列で右寄せ
        put(y, end - len(s) + 1, s, attr)

    if w < 48 or h < 14:
        put(0, 0, "window too small — widen the terminal", A["dim"])
        return

    W = min(w - 1, 100)
    cx = 3                       # 内容の左端（│ の内側に 2 マスのパディング）
    reqs_end = W - 4
    unload_end = W - 13
    inflt_end = W - 23
    state_x = cx + 31
    n = max(0, min(len(view["models"]), h - 13))
    rows = view["models"][:n]

    def box(top, height, *, title=False):
        put(top, 0, "╭" + "─" * (W - 2) + "╮", A["border"])
        put(top + height - 1, 0, "╰" + "─" * (W - 2) + "╯", A["border"])
        for i in range(1, height - 1):
            put(top + i, 0, "│", A["border"])
            put(top + i, W - 1, "│", A["border"])

    # --- ダッシュボードボックス ---
    dash_h = n + 9               # 内部 7+n 行 + 上下罫線
    box(0, dash_h)
    put(0, 2, "◆", A["accent"])
    put(0, 3, " local-llm-server ", A["text"])    # 前後の空白で罫線（─）を消してタイトルを浮かせる
    put(0, 21, "· gateway monitor ", A["dim"])
    refresh = "⟳ starting" if busy else "⟳ live"
    put_r(0, W - 2, " " + refresh + " ", A["amber"] if busy else A["dim"])

    # 状態行
    if view["ready"]:
        put(2, cx, "●", A["green"]); put(2, cx + 2, "ready", A["text"])
    elif busy:
        put(2, cx, "●", A["amber"]); put(2, cx + 2, "starting", A["text"])
    else:
        put(2, cx, "●", A["red"]); put(2, cx + 2, "stopped", A["text"])
    sep = "   ·   "
    info = f"{sep}:{gcfg.port}{sep}up {_fmt_hms(view['uptime'])}{sep}{view['requests']:,} reqs"
    put(2, cx + 10, info, A["dim"])

    # モデル表ヘッダ＋区切り
    put(4, cx, "MODEL", A["dim"])
    put(4, state_x, "STATE", A["dim"])
    put_r(4, inflt_end, "INFLT", A["dim"])
    put_r(4, unload_end, "UNLOAD", A["dim"])
    put_r(4, reqs_end, "REQS", A["dim"])
    put(5, cx, "─" * (W - 6), A["border"])

    for i, r in enumerate(rows):
        y = 6 + i
        name = r["model"].split("/")[-1]
        put(y, cx, name[: state_x - cx - 2], A["text"] if r["state"] != "unloaded" else A["dim"])
        st = r["state"]
        dot, color = {
            "busy": ("●", A["green"]),
            "idle": ("○", A["amber"]),
            "unloaded": ("·", A["dim"]),
        }.get(st, ("·", A["dim"]))
        put(y, state_x, dot, color)
        put(y, state_x + 2, st, color)
        put_r(y, inflt_end, str(r["inflight"]), A["text"] if r["inflight"] else A["dim"])
        put_r(y, unload_end, _fmt_hms(r["idle_remaining"]), A["dim"])
        put_r(y, reqs_end, f"{r['requests']:,}", A["dim"])

    policy = (
        f"max_resident {view['max_resident'] if view['max_resident'] is not None else '∞'}"
        f"{sep}idle {int(view['idle_timeout']) if view['idle_timeout'] else 'off'}s"
    )
    put(7 + n, cx, policy, A["dim"])
    if msg:
        put_r(7 + n, reqs_end, msg[:40], A["red"])

    # --- 入力ボックス ---
    y_in = dash_h
    box(y_in, 3)
    put(y_in + 1, 2, "›", A["accent"])
    if cmd_mode:
        put(y_in + 1, 4, cmd_buf, A["text"])
        try:
            curses.curs_set(1)
            stdscr.move(y_in + 1, min(4 + len(cmd_buf), W - 2))
        except curses.error:
            pass
    else:
        put(y_in + 1, 4, "press : for a command  (stop · restart · log · quit)", A["dim"])
        try:
            curses.curs_set(0)
        except curses.error:
            pass

    # --- 操作ヒント ---
    fy = y_in + 3
    x = cx
    for k, label in (("s", "stop"), ("r", "restart"), ("g", "start"), ("l", "log"), ("q", "quit")):
        put(fy, x, k, A["accent"]); put(fy, x + 1, " " + label, A["dim"])
        x += 2 + len(label) + 3


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
