"""textual 製の TUI ダッシュボード本体（`local-llm-server` の既定起動が開くアプリ）。

角丸枠・truecolor・本物の入力欄・端末リサイズ追従を備える。状態の取得・統合・操作は
tui.py / server.py の関数を流用し、ここは表示と操作の結線だけを担う。textual は重い import
なので、呼び出し側（tui.run_tui）から遅延 import する（headless では読み込まない）。
"""
from __future__ import annotations

import os
import subprocess
import sys

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Static

from .server import (
    find_pids_on_port,
    gateway_admin_status,
    gateway_log_path,
    server_status,
    start_gateway_background,
    stop_pid,
)
from .tui import _fmt_hms, merge_status, read_log_tail

_GREEN = "#7fc99a"
_AMBER = "#e0b46a"
_RED = "#d8645f"
_ACCENT = "#d8a45f"
_DIM = "#83838c"


class LogScreen(ModalScreen):
    """ゲートウェイログをアプリ内のスクロール画面で表示する（外部ページャを使わない）。

    `q` / `Esc` でダッシュボードに戻る。`r` で再読込。端末状態を壊さず、戻り方も明示する。
    """

    CSS = """
    LogScreen { align: center middle; }
    #logbox { width: 92%; height: 90%; border: round #3a3a40; background: #16161a; padding: 0 1; }
    #loghead { padding: 0 0 1 0; color: #83838c; }
    #logbody { height: 1fr; background: #16161a; }
    """

    BINDINGS = [
        Binding("escape", "close", "back"),
        Binding("q", "close", "back"),
        Binding("r", "reload", "reload"),
    ]

    def __init__(self, port: int):
        super().__init__()
        self._port = port

    def compose(self) -> ComposeResult:
        with Container(id="logbox"):
            head = Text()
            head.append("log ", style="bold")
            head.append(gateway_log_path(self._port), style=_DIM)
            head.append("    (q / Esc で戻る · r 更新)", style=_ACCENT)
            yield Static(head, id="loghead")
            with VerticalScroll(id="logbody"):
                yield Static(id="logtext")

    def on_mount(self) -> None:
        self.action_reload()

    def action_reload(self) -> None:
        self.query_one("#logtext", Static).update(read_log_tail(self._port))
        # レイアウト確定後に末尾までスクロール（最新行を見せる）。
        self.call_after_refresh(
            self.query_one("#logbody", VerticalScroll).scroll_end, animate=False
        )

    def action_close(self) -> None:
        self.dismiss()


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
    #hints { background: #1d1d22; padding: 0 1; }
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
        self._row_models: list[str] = []  # 表の表示順モデル ID（行クリック→コピー用）

    def compose(self) -> ComposeResult:
        with Container(id="dash"):
            yield Static(self._title(), id="title")
            yield Static(id="status")
            yield DataTable(id="models", show_cursor=False, zebra_stripes=False)
            yield Static(id="policy")
        yield Input(placeholder="command — stop · restart · start · log · quit", id="cmd")
        # キーの凡例は固定表示にする（Footer はコマンド入力中に隠れてしまうため）。
        yield Static(self._hints(), id="hints")

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.cursor_type = "none"  # 行ハイライトは出さない（選択色を「動作中」と誤認しないため）
        table.add_columns("MODEL", "STATE", "MTP", "INFLT", "AGENTS", "UNLOAD", "REQS")
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

    def _hints(self) -> Text:
        """キー操作の凡例（フォーカスに依らず常に見える固定行。入力中も消えない）。"""
        t = Text()
        for i, (key, label) in enumerate(
            (("s", "stop"), ("r", "restart"), ("g", "start"), ("l", "log"), ("q", "quit"))
        ):
            if i:
                t.append("   ", style=_DIM)
            t.append(f" {key} ", style=f"bold {_ACCENT}")
            t.append(f" {label}", style=_DIM)
        t.append("      行クリックでモデル名コピー", style=_DIM)
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
        # 行クリック→コピー用に、表示順のモデル ID を保持する（行ハイライトは使わない）。
        self._row_models = [r["model"] for r in view["models"]]
        # 表示するモデルが無いとき（起動直後・全アンロード）は表ごと隠す（ヘッダも出さない）。
        # モデルがロードされたら表が現れる。
        table.display = bool(view["models"])
        for r in view["models"]:
            # モデル ID はフル表示（org/repo[:量子化] まで出す）。末尾だけだと org や
            # 量子化が分からず取り違えるため、多少長くても省略しない。
            name = r["model"]
            sym, col = {
                "busy": ("●", _GREEN), "idle": ("○", _AMBER), "unloaded": ("·", _DIM),
            }.get(r["state"], ("·", _DIM))
            state = Text.assemble((sym + " ", col), (r["state"], col))
            # MTP（高速化）の利用可否。ready=緑●、available=淡色（要 hf download）、非対応=「—」。
            mtp_sym, mtp_col, mtp_label = {
                "ready": ("●", _GREEN, "ready"),
                "available": ("○", _DIM, "avail"),
            }.get(r.get("mtp"), ("—", _DIM, ""))
            mtp_cell = Text.assemble((mtp_sym, mtp_col), (" " + mtp_label if mtp_label else "", mtp_col))
            inflt = Text(str(r["inflight"]), style="" if r["inflight"] else _DIM)
            # 在席エージェント数。>0 は緑（解放されない＝使用中）、0 は淡色（即アンロード対象）。
            sess_n = r.get("sessions", 0)
            agents = Text(str(sess_n), style=_GREEN if sess_n else _DIM)
            table.add_row(
                Text(name, style="" if r["state"] != "unloaded" else _DIM),
                state, mtp_cell, inflt, agents,
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
        # 外部ページャを suspend で開くと表示が壊れ戻り方も分かりにくいので、
        # アプリ内のスクロール画面で表示する（q/Esc で戻る）。
        self.push_screen(LogScreen(self.port))

    def action_quit(self) -> None:
        self.exit()

    def on_click(self, event) -> None:
        # 表の行をクリックしたら、その行のモデル名をコピーする（行ハイライトは出さない）。
        # クリック位置のセル meta から行番号を得る（ヘッダ -1・表外は無視）。
        meta = getattr(event, "style", None)
        meta = getattr(meta, "meta", None) or {}
        row = meta.get("row")
        if isinstance(row, int) and 0 <= row < len(self._row_models):
            self._copy_model(self._row_models[row])

    def _copy_model(self, model: str) -> None:
        """モデル名をクリップボードへ。macOS は pbcopy、他は端末の OSC52（textual）。"""
        copied = False
        if sys.platform == "darwin":
            try:
                subprocess.run(["pbcopy"], input=model.encode(), check=True)
                copied = True
            except Exception:  # noqa: BLE001 - pbcopy 不在等は OSC52 にフォールバック
                copied = False
        if not copied:
            try:
                self.copy_to_clipboard(model)  # 端末の OSC52 経由（対応端末のみ）
                copied = True
            except Exception:  # noqa: BLE001
                copied = False
        self.notify(
            f"copied: {model}" if copied else f"クリップボードにコピーできませんでした: {model}",
            timeout=3,
        )

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
