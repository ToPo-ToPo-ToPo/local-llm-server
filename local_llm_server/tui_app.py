"""textual 製の TUI ダッシュボード本体（`local-llm-server` の既定起動が開くアプリ）。

角丸枠・truecolor・本物の入力欄・端末リサイズ追従を備える。状態の取得・統合・操作は
tui.py / server.py の関数を流用し、ここは表示と操作の結線だけを担う。textual は重い import
なので、呼び出し側（tui.run_tui）から遅延 import する（headless では読み込まない）。
"""
from __future__ import annotations

import os

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import DataTable, Footer, Input, Static

from .server import (
    find_pids_on_port,
    gateway_admin_status,
    server_status,
    start_gateway_background,
    stop_pid,
)
from .tui import _fmt_hms, _open_log_in_pager, merge_status

_GREEN = "#7fc99a"
_AMBER = "#e0b46a"
_RED = "#d8645f"
_ACCENT = "#d8a45f"
_DIM = "#83838c"


class GatewayMonitor(App):
    """ゲートウェイ常駐モニタ。1 秒ごとに /admin/status をポーリングして自動更新する。"""

    CSS = """
    Screen { background: #16161a; }
    #dash { border: round #3a3a40; padding: 0 1; height: auto; margin: 1 1 0 1; }
    #title { padding: 0 0 1 0; }
    #status { padding: 0 0 1 0; }
    #policy { padding: 1 0 0 0; color: #83838c; }
    DataTable { height: auto; background: #16161a; }
    DataTable > .datatable--header { color: #83838c; background: #16161a; text-style: none; }
    #cmd { border: round #3a3a40; margin: 1 1 0 1; background: #16161a; }
    Footer { background: #1d1d22; }
    """

    BINDINGS = [
        Binding("s", "stop", "stop"),
        Binding("r", "restart", "restart"),
        Binding("g", "start", "start"),
        Binding("l", "log", "log"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, gcfg):
        super().__init__()
        self.gcfg = gcfg
        self.host = gcfg.host
        self.port = gcfg.port
        self.all_ports = [self.port] + [m.port for m in gcfg.models]
        self.admin = None
        self.busy = ""

    def compose(self) -> ComposeResult:
        with Container(id="dash"):
            yield Static(self._title(), id="title")
            yield Static(id="status")
            yield DataTable(id="models", show_cursor=False, zebra_stripes=False)
            yield Static(id="policy")
        yield Input(placeholder="command — stop · restart · start · log · quit", id="cmd")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.cursor_type = "none"  # 行ハイライトを出さない（読み取り専用の表）
        table.add_columns("MODEL", "STATE", "INFLT", "UNLOAD", "REQS")
        # 起動時にゲートウェイを裏で常駐させる（既定起動＝「アプリを開く」感覚）。
        if server_status(self.host, self.port) is None:
            self.action_start()
        self.set_interval(1.0, self.poll)
        self.poll()

    def _title(self) -> Text:
        t = Text()
        t.append("◆ ", style=_ACCENT)
        t.append("local-llm-server", style="bold")
        t.append("  · gateway monitor", style=_DIM)
        return t

    # --- ポーリング（別スレッドで HTTP、結果は UI スレッドへ） ---
    @work(thread=True, exclusive=True, group="poll")
    def poll(self) -> None:
        admin = gateway_admin_status(self.host, self.port)
        self.call_from_thread(self._apply, admin)

    def _apply(self, admin) -> None:
        self.admin = admin
        self._render()

    def _render(self) -> None:
        view = merge_status(self.gcfg, self.admin)

        st = Text()
        if self.busy:
            st.append("● ", style=_AMBER); st.append(self.busy.strip())
        elif view["ready"]:
            st.append("● ", style=_GREEN); st.append("ready")
        else:
            st.append("● ", style=_RED); st.append("stopped")
        st.append(
            f"    :{self.port}    up {_fmt_hms(view['uptime'])}    {view['requests']:,} reqs",
            style=_DIM,
        )
        self.query_one("#status", Static).update(st)

        table = self.query_one("#models", DataTable)
        table.clear()
        for r in view["models"]:
            name = r["model"].split("/")[-1]
            sym, col = {
                "busy": ("●", _GREEN), "idle": ("○", _AMBER), "unloaded": ("·", _DIM),
            }.get(r["state"], ("·", _DIM))
            state = Text.assemble((sym + " ", col), (r["state"], col))
            inflt = Text(str(r["inflight"]), style="" if r["inflight"] else _DIM)
            table.add_row(
                Text(name, style="" if r["state"] != "unloaded" else _DIM),
                state, inflt,
                Text(_fmt_hms(r["idle_remaining"]), style=_DIM),
                Text(f"{r['requests']:,}", style=_DIM),
            )

        policy = (
            f"max_resident {view['max_resident'] if view['max_resident'] is not None else '∞'}"
            f"    idle {int(view['idle_timeout']) if view['idle_timeout'] else 'off'}s"
        )
        if self.busy:
            policy += f"    · {self.busy.strip()}"
        self.query_one("#policy", Static).update(policy)

    # --- 操作（別スレッドで実行して UI を固めない） ---
    @work(thread=True, group="action")
    def _run(self, label: str, fn) -> None:
        self.busy = label
        self.call_from_thread(self._render)
        try:
            fn()
        finally:
            self.busy = ""
            admin = gateway_admin_status(self.host, self.port)
            self.call_from_thread(self._apply, admin)

    def _kill_ports(self) -> None:
        for p in self.all_ports:
            for pid in find_pids_on_port(p):
                stop_pid(pid)

    def action_start(self) -> None:
        self._run("starting…", lambda: start_gateway_background(os.getcwd(), self.host, self.port))

    def action_stop(self) -> None:
        self._run("stopping…", self._kill_ports)

    def action_restart(self) -> None:
        def _restart():
            self._kill_ports()
            start_gateway_background(os.getcwd(), self.host, self.port)
        self._run("restarting…", _restart)

    def action_log(self) -> None:
        with self.suspend():
            _open_log_in_pager(self.port)

    def action_quit(self) -> None:
        self.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        cmd = (event.value or "").strip().lower()
        event.input.value = ""
        if cmd in ("q", "quit", "exit"):
            self.exit()
            return
        {
            "s": self.action_stop, "stop": self.action_stop,
            "r": self.action_restart, "restart": self.action_restart,
            "g": self.action_start, "start": self.action_start,
            "l": self.action_log, "log": self.action_log,
        }.get(cmd, lambda: None)()
