"""textual 製の TUI ダッシュボード本体（`local-llm-server` の既定起動が開くアプリ）。

角丸枠・truecolor・本物の入力欄・端末リサイズ追従を備える。状態の取得・統合・操作は
tui.py / server.py の関数を流用し、ここは表示と操作の結線だけを担う。textual は重い import
なので、呼び出し側（tui.run_tui）から遅延 import する（headless では読み込まない）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

from rich.style import Style
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
    gateway_set_max_resident,
    is_ready,
    local_connect_host,
    pid_looks_like_ours,
    primary_lan_ip,
    server_status,
    start_gateway_background,
    stop_pid,
)
from . import update
from .tui import _fmt_hms, merge_status, read_log_tail

_GREEN = "#7fc99a"
_AMBER = "#e0b46a"
_RED = "#d8645f"
_ACCENT = "#d8a45f"
_DIM = "#83838c"

# PyPI 新版のポーリング間隔（秒）。起動直後に 1 回、以降はこの間隔で確認する。
# 頻繁に叩く必要はない（公開は稀）ので長め。
_UPDATE_CHECK_INTERVAL = 1800.0  # 30 分

# can_apply=False の理由 → バナー用の短い和文。
_UPDATE_REASONS = {
    "dirty": "ローカル変更あり・保留",
    "no-upstream": "追跡ブランチ無し・保留",
    "not-a-git-clone": "git 運用外",
    "offline": "確認できず",
}


def _clipboard_copy(app, text: str) -> bool:
    """テキストをクリップボードへ。macOS は pbcopy、他は端末の OSC52（textual）。

    ダッシュボードの行クリック（モデル名）と MTP 画面のパスクリックで共用する。
    成功したら True。
    """
    if sys.platform == "darwin":
        try:
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            return True
        except Exception:  # noqa: BLE001 - pbcopy 不在等は OSC52 にフォールバック
            pass
    try:
        app.copy_to_clipboard(text)  # 端末の OSC52 経由（対応端末のみ）
        return True
    except Exception:  # noqa: BLE001
        return False


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


class MtpScreen(ModalScreen):
    """MTP ドラフターの要否・取得状況をアプリ内のスクロール画面で表示する（`mtp [model]`）。

    表示内容は tui.mtp_report が組み立てる（ダウンロードはしない）。
    `q` / `Esc` でダッシュボードに戻る。`r` で再読込（`hf download` 後に ready へ変わったか
    その場で確認できる）。
    """

    CSS = """
    MtpScreen { align: center middle; }
    #mtpbox { width: 92%; height: 90%; border: round #3a3a40; background: #16161a; padding: 0 1; }
    #mtphead { padding: 0 0 1 0; color: #83838c; }
    #mtpbody { height: 1fr; background: #16161a; }
    """

    BINDINGS = [
        Binding("escape", "close", "back"),
        Binding("q", "close", "back"),
        Binding("r", "reload", "reload"),
    ]

    def __init__(self, model: str | None):
        super().__init__()
        self._model = model  # None = 対応表を全件表示

    def compose(self) -> ComposeResult:
        with Container(id="mtpbox"):
            head = Text()
            head.append("mtp ", style="bold")
            head.append(self._model or "（対応モデル一覧）", style=_DIM)
            head.append("    (q / Esc で戻る · r 更新 · パスをクリックでコピー)", style=_ACCENT)
            yield Static(head, id="mtphead")
            with VerticalScroll(id="mtpbody"):
                yield Static(id="mtptext")

    def on_mount(self) -> None:
        self.action_reload()

    def action_reload(self) -> None:
        # 辞書引き＋ローカルキャッシュ確認だけの軽い処理（HTTP/DL なし）なので UI スレッドでよい
        # （read_log_tail と同じ扱い）。
        from .tui import mtp_report

        text, _code = mtp_report(self._model)
        self.query_one("#mtptext", Static).update(self._linkify(text))

    def _linkify(self, report: str) -> Text:
        """レポート中のドラフター HF id と `hf download` 行を、クリックでコピー可能にする。

        文面そのものは CLI（cli.mtp_report）と共通のまま、行フォーマットを頼りに該当スパンへ
        @click メタ（→ action_copy_path）を貼る。ドラフター id はそのまま、`hf download` 行は
        コマンドごとコピーして端末に貼れるようにする。HF id / コマンドに引用符は現れないので
        アクション引数のクォート衝突はない。
        """

        def _link(txt: str) -> Style:
            return Style(color=_ACCENT, underline=True, meta={"@click": f"screen.copy_path('{txt}')"})

        out = Text()
        for i, line in enumerate(report.splitlines()):
            if i:
                out.append("\n")
            stripped = line.strip()
            head, sep, rest = line.partition("drafter: ")
            if sep and rest:
                # "    drafter: <id>  [ready …]" — id は直後の 2 連スペースまで。
                drafter, sep2, tail = rest.partition("  ")
                out.append(head + sep)
                out.append(drafter, style=_link(drafter))
                out.append(sep2 + tail, style=_DIM)
            elif stripped.startswith("hf download "):
                # ダウンロードコマンドは行ごとコピー（そのまま端末へ貼れる）。
                out.append(line[: len(line) - len(stripped)])
                out.append(stripped, style=_link(stripped))
            else:
                out.append(line)
        return out

    def action_copy_path(self, path: str) -> None:
        copied = _clipboard_copy(self.app, path)
        self.app.notify(
            f"copied: {path}" if copied else f"クリップボードにコピーできませんでした: {path}",
            timeout=3,
        )

    def action_close(self) -> None:
        self.dismiss()


# --- 起動スプラッシュ（Claude Code CLI 風のロゴ画面）--------------------------------
# 5x5 の固定升目フォント。文字ごとに幅が変わる本物の figlet と違い、全グリフが同じ
# 升目なので文字間のズレが起きない（手打ちの ASCII アートで一番事故りやすい部分）。
_GLYPH_H = 5
_GLYPH_W = 5
_LETTERS: dict[str, list[str]] = {
    "L": ["█    ", "█    ", "█    ", "█    ", "█████"],
    "O": [" ███ ", "█   █", "█   █", "█   █", " ███ "],
    "C": [" ████", "█    ", "█    ", "█    ", " ████"],
    "A": [" ███ ", "█   █", "█████", "█   █", "█   █"],
    "-": ["     ", "     ", "█████", "     ", "     "],
    "M": ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
    "S": [" ████", "█    ", " ███ ", "    █", "████ "],
    "E": ["█████", "█    ", "████ ", "█    ", "█████"],
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "V": ["█   █", "█   █", "█   █", " █ █ ", "  █  "],
}


def _big_word(word: str, gap: int = 1) -> list[str]:
    """word の各文字を _LETTERS の升目でつなげ、5 行のアスキーアートにする。"""
    glyphs = [_LETTERS[ch] for ch in word]
    return [(" " * gap).join(g[row] for g in glyphs) for row in range(_GLYPH_H)]


def ascii_logo_big() -> str:
    """'LOCAL-LLM' と 'SERVER' を升目フォントで積んだ大きいロゴ（幅 53 桁）。"""
    top = _big_word("LOCAL-LLM")
    bottom = _big_word("SERVER")
    width = len(top[0])
    bottom = [line.center(width) for line in bottom]
    return "\n".join([*top, "", *bottom])


def ascii_logo_small() -> str:
    """幅の狭い端末向けの 1 行フォールバック（大きいロゴが収まらないとき）。"""
    return "LOCAL-LLM-SERVER"


# 大きいロゴが収まる最小の端末サイズ。満たさなければ小さいフォールバックを出す。
_BIG_LOGO_MIN_WIDTH = 60
_BIG_LOGO_MIN_HEIGHT = 18
# 自動で閉じるまでの秒数（キー入力があれば即閉じる）。テストはこの秒数を短く差し替える。
SPLASH_AUTO_DISMISS_SECONDS = 1.6


class SplashScreen(ModalScreen):
    """起動直後に一瞬だけ出すロゴ画面（Claude Code CLI 風）。

    裏のダッシュボードは通常どおり起動処理（ゲートウェイ起動・ポーリング開始）を進めて
    おり、このスプラッシュは見た目だけの演出——キー入力、または一定時間で自動的に閉じて
    ダッシュボードに移る（起動そのものを遅らせない）。
    """

    CSS = """
    SplashScreen { align: center middle; background: #16161a; }
    #splash { width: auto; height: auto; }
    #splash_art { color: #d8a45f; text-style: bold; width: auto; }
    #splash_sub { text-align: center; padding: 1 0 0 0; }
    #splash_hint { color: #83838c; text-align: center; padding: 1 0 0 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._done = False

    def compose(self) -> ComposeResult:
        size = self.app.size
        big_enough = (
            size.width >= _BIG_LOGO_MIN_WIDTH and size.height >= _BIG_LOGO_MIN_HEIGHT
        )
        art = ascii_logo_big() if big_enough else ascii_logo_small()
        with Container(id="splash"):
            yield Static(art, id="splash_art")
            yield Static(id="splash_sub")
            yield Static("press any key to continue", id="splash_hint")

    def on_mount(self) -> None:
        sub = Text()
        sub.append("gateway monitor", style=_DIM)
        ver = update.installed_version()
        if ver:
            sub.append(f"  ·  v{ver}", style=_ACCENT)
        self.query_one("#splash_sub", Static).update(sub)
        self.set_timer(SPLASH_AUTO_DISMISS_SECONDS, self._dismiss_once)

    def on_key(self, _event) -> None:
        self._dismiss_once()

    def _dismiss_once(self) -> None:
        # タイマーとキー入力の両方から呼ばれ得るため、二重 dismiss を防ぐ。
        if not self._done:
            self._done = True
            self.dismiss()


class GatewayMonitor(App):
    """ゲートウェイ常駐モニタ。1 秒ごとに /admin/status をポーリングして自動更新する。"""

    CSS = """
    Screen { background: #16161a; }
    #dash { border: round #3a3a40; padding: 0 1; height: 1fr; margin: 1 1 0 1; }
    #title { padding: 0 0 1 0; }
    #status { padding: 0 0 1 0; }
    #policy { padding: 1 0 0 0; color: #83838c; }
    DataTable { height: auto; background: #16161a; }
    DataTable > .datatable--header { color: #83838c; background: #16161a; text-style: none; }
    /* コマンド欄は上・凡例は下に固定（dock）。中央の #dash が伸びても両者は隠れない。 */
    #cmd { dock: top; border: round #3a3a40; margin: 1 1 0 1; background: #16161a; }
    #hints { dock: bottom; background: #1d1d22; padding: 0 1; }
    """

    BINDINGS = [
        Binding("s", "stop", "stop"),
        Binding("r", "restart", "restart"),
        Binding("g", "start", "start"),
        Binding("m", "prefill_max", "max"),
        Binding("l", "log", "log"),
        Binding("u", "update", "update"),
        Binding("q", "quit", "quit"),
    ]

    # 起動時にコマンド入力欄へ自動フォーカスしない。フォーカスされた Input は印字キーを
    # 消費するため、上の単キーショートカット（s/r/g/m/l/q）が一切効かなくなってしまう。
    # コマンドを打つときは入力欄をクリック（または m キーでプリフィル）する。
    AUTO_FOCUS = None

    def __init__(self, gcfg, *, show_splash: bool | None = None):
        super().__init__()
        self.gcfg = gcfg
        # None = 自動（headless=テスト実行時は出さない、実端末では出す）。テストが明示的に
        # True/False を渡してスプラッシュ自体の挙動も検証できるようにする。
        self._show_splash = show_splash
        self.bind_host = gcfg.host                       # 公開 bind 先（表示用）
        # 自分自身のゲートウェイへの接続はループバックで（bind が 0.0.0.0 等でも "0.0.0.0" 宛は不可搬）。
        self.host = local_connect_host(gcfg.host)
        self.port = gcfg.port
        self.all_ports = [self.port] + [m.port for m in gcfg.models]
        # ネットワーク公開（0.0.0.0 等）のとき、リモートのクライアントが指す LAN URL を1度だけ解決。
        self.reachable_url = None
        if gcfg.host in ("0.0.0.0", "::", "", "*"):
            lan = primary_lan_ip()
            if lan:
                self.reachable_url = f"http://{lan}:{self.port}/v1"
        self.admin = None
        self._gw_ready = False  # ポーリングワーカーが判定した稼働状態（UI スレッドで HTTP しない）
        self.busy = ""
        self._row_models: list[str] = []  # 表の表示順モデル ID（行クリック→コピー用）
        # --- 自動更新（PyPI 新版を git pull で追従） ---
        self._update = None            # update.UpdateStatus（新版検知時のみ）
        self._update_applying = False  # 適用中（多重起動を防ぐ）
        self._update_done = False      # 適用完了→再起動待ち
        self.restart_after_exit = False  # run_tui が見て、新コードで TUI を再 exec する合図

    def compose(self) -> ComposeResult:
        # コマンド欄を最上部に固定する（dock: top）。モデル一覧が増えて #dash が縦に伸びても、
        # 入力欄が画面外へ押し出されない。中央の一覧は #dash 内でスクロールする。
        yield Input(
            placeholder="command — stop · restart · start · max <n> · mtp [model] · log · update · quit",
            id="cmd",
        )
        with VerticalScroll(id="dash"):
            yield Static(self._title(), id="title")
            yield Static(id="status")
            yield DataTable(id="models", show_cursor=False, zebra_stripes=False)
            yield Static(id="policy")
        # キーの凡例は最下部に固定する（Footer はコマンド入力中に隠れてしまうため）。
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
        # PyPI 新版の検知（起動直後に 1 回＋定期）。auto_update 無効なら一切走らせない。
        if getattr(self.gcfg, "auto_update", True):
            self.set_interval(_UPDATE_CHECK_INTERVAL, self.check_update)
            self.check_update()
        # 起動ロゴ（見た目だけ。裏のゲートウェイ起動・ポーリングは上ですでに始まっている）。
        # 既定は「実端末では出す・pytest の run_test（headless）では出さない」の自動判定。
        want_splash = self._show_splash
        if want_splash is None:
            want_splash = not self.is_headless
        if want_splash:
            self.push_screen(SplashScreen())

    def _title(self) -> Text:
        # 製品名＋自分のバージョンを Claude Code CLI 風に出す。自動更新でソースが入れ替わるため、
        # 「今どの版が動いているか」を一目で分かるようにする（version は importlib.metadata から）。
        ver = update.installed_version()
        t = Text()
        t.append("◆ ", style=_ACCENT)
        t.append("local-llm-server", style="bold")
        t.append(f"  v{ver}" if ver else "  (dev)", style=_ACCENT)
        t.append("  · gateway monitor", style=_DIM)
        return t

    def _hints(self) -> Text:
        """キー操作の凡例（フォーカスに依らず常に見える固定行。入力中も消えない）。"""
        t = Text()
        for i, (key, label) in enumerate(
            (("s", "stop"), ("r", "restart"), ("g", "start"),
             ("m", "max"), ("l", "log"), ("u", "update"), ("q", "quit"))
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
        # /admin/status が取れなくても、応答だけはある（起動直後等）ならフォールバックで確認。
        # HTTP はすべてこのワーカースレッドで行う（UI スレッドを固めない）。
        ready = admin is not None or is_ready(f"http://{self.host}:{self.port}/v1")
        self.call_from_thread(self._apply, admin, ready)

    def _apply(self, admin, ready: bool | None = None) -> None:
        self.admin = admin
        self._gw_ready = bool(admin) if ready is None else ready
        self._render()
        # ゲートウェイがアイドルに戻った瞬間に、保留中の更新があれば適用する。
        self._maybe_apply_update()

    def _render(self) -> None:
        view = merge_status(self.gcfg, self.admin, ready=self._gw_ready)

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
        # 自動更新バナー: 適用できる場合は「適用予定/適用中」、保留（ローカル変更等）は理由つき。
        up = self._update
        if up is not None and up.available:
            if self._update_applying:
                st.append(f"    ⬆ {up.latest} に更新中…", style=_ACCENT)
            elif up.can_apply:
                st.append(f"    ⬆ {up.latest} に自動更新（アイドル時）", style=_ACCENT)
            else:
                st.append(
                    f"    ⬆ {up.latest} 利用可（{_UPDATE_REASONS.get(up.reason, up.reason)}・u で適用）",
                    style=_AMBER,
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
            # 同一モデルが複数インスタンスで並列稼働しているときは「×N」を添える（並列度の目安）。
            if r.get("instances", 0) > 1:
                state.append(f" ×{r['instances']}", style=_ACCENT)
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
        # ネットワーク公開時は、リモートのクライアントが指す LAN URL を添える（クリックでコピー可）。
        if self.reachable_url:
            policy += f"    LAN {self.reachable_url}"
        if self.busy:
            policy += f"    · {self.busy.strip()}"
        # 起動元情報: いつ・どこから・どの経路（tui/headless）で立ったゲートウェイに attach して
        # いるかを常時表示する。裏でヘッドレス起動されたサーバーでも出所が一目で分かる。
        admin = self.admin or {}
        if admin.get("started_at"):
            cwd = admin.get("cwd", "")
            home = os.path.expanduser("~")
            if cwd.startswith(home):
                cwd = "~" + cwd[len(home):]
            policy += (
                f"\nstarted {admin['started_at']} · {admin.get('launcher', '?')}"
                f" · pid {admin.get('pid', '?')} · {cwd}"
            )
        self.query_one("#policy", Static).update(policy)

    # --- 操作（別スレッドで実行して UI を固めない） ---
    @work(thread=True, group="action")
    def _run(self, label: str, fn) -> None:
        self.busy = label
        self.call_from_thread(self._render)
        try:
            fn()
        except (RuntimeError, TimeoutError, OSError) as exc:
            # 起動失敗（ポート占有・タイムアウト等）はアプリを落とさず通知する
            # （@work のワーカー例外は既定でアプリ全体をクラッシュさせるため必ず捕まえる）。
            self.call_from_thread(
                self.notify, f"{label.strip('…')} に失敗しました: {exc}",
                severity="error", timeout=8,
            )
        finally:
            self.busy = ""
            admin = gateway_admin_status(self.host, self.port)
            ready = admin is not None or is_ready(f"http://{self.host}:{self.port}/v1")
            self.call_from_thread(self._apply, admin, ready)

    def _kill_ports(self) -> None:
        # このパッケージ由来に見えるプロセスだけ止める（同じポート番号をたまたま使っている
        # 無関係なプロセスを巻き添えにしない）。stop_pid は 1 件あたり最長 ~10s 待つので、
        # 並列に止めて合計時間を抑える（q 終了時は同期実行のため体感に直結する）。
        pids = {pid for p in self.all_ports for pid in find_pids_on_port(p)}
        pids = [pid for pid in pids if pid_looks_like_ours(pid)]
        threads = [threading.Thread(target=stop_pid, args=(pid,)) for pid in pids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

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

    # --- 自動更新（PyPI 新版を git pull で追従）------------------------------
    @work(thread=True, exclusive=True, group="update-check")
    def check_update(self) -> None:
        """PyPI 最新版を調べる（別スレッド。HTTP と git はここで行う）。"""
        st = update.check(timeout=3.0)
        self.call_from_thread(self._on_update_status, st)

    def _on_update_status(self, st) -> None:
        # 新版があるときだけ保持する（無ければバナーを消す）。適用済み待ちのときは触らない。
        if self._update_done:
            return
        self._update = st if st.available else None
        self._render()
        self._maybe_apply_update()

    def _gateway_idle(self) -> bool:
        """再起動して安全か（処理中リクエストも在席エージェントも無い）。

        ゲートウェイ未起動（admin なし）も「安全」とみなす。実行中の推論/文字起こしや、
        在席中のエージェントがある間は再起動を避け、空いた瞬間に適用する。
        """
        admin = self.admin
        if not admin:
            return True
        models = admin.get("models", [])
        inflight = sum(int(m.get("inflight", 0)) for m in models)
        sessions = sum(int(m.get("sessions", 0)) for m in models)
        return inflight == 0 and sessions == 0

    def _maybe_apply_update(self) -> None:
        """保留中の更新を、条件が揃っていれば自動適用する（アイドル時のみ）。"""
        st = self._update
        if st is None or not st.available or not st.can_apply:
            return
        if self._update_applying or self._update_done:
            return
        if not self._gateway_idle():
            return  # 処理中/在席あり → 完了を待つ（次のポーリングで再判定）
        self._update_applying = True
        self._render()
        self._apply_update_worker()

    def action_update(self) -> None:
        """`u` キー: 手動で更新を適用する（アイドルを待たず、その場で試みる）。"""
        st = self._update
        if st is None or not st.available:
            self.notify("最新です（更新はありません）", timeout=3)
            return
        if not st.can_apply:
            self.notify(
                f"自動更新できません: {_UPDATE_REASONS.get(st.reason, st.reason)}",
                severity="warning", timeout=8,
            )
            return
        if self._update_applying or self._update_done:
            return
        self._update_applying = True
        self._render()
        self._apply_update_worker()

    @work(thread=True, group="update-apply")
    def _apply_update_worker(self) -> None:
        """git pull（＋uv sync）で更新し、成功したら新コードで TUI を再起動する。"""
        self.busy = "updating…"
        self.call_from_thread(self._render)
        ok, msg = update.apply_update()
        if ok:
            # 旧ゲートウェイを止めてから抜ける（再 exec した新 TUI が新コードで起動し直す）。
            self._kill_ports()
            self._update_done = True
            self.restart_after_exit = True
            self.call_from_thread(self.exit)
            return
        # 失敗（作業ツリーが汚れた・ff 不可・pull 失敗）: 落とさず通知し、手動 u で再試行できる。
        self._update_applying = False
        self.busy = ""
        self.call_from_thread(
            self.notify, f"自動更新を見送りました: {msg}", severity="warning", timeout=10
        )
        self.call_from_thread(self._render)

    def action_prefill_max(self) -> None:
        # `m` キー: コマンド欄に "max " を入れてフォーカスし、数値だけ打てば送信できるようにする。
        inp = self.query_one("#cmd", Input)
        inp.value = "max "
        inp.focus()
        inp.cursor_position = len(inp.value)

    def _set_max_resident(self, arg: str) -> None:
        """`max <n>` / `max off` を解釈して稼働中のゲートウェイに反映する（再起動不要・busy 継続）。

        n は 1 以上の整数。off / none / unlimited / 0 / ∞ は無制限。稼働中のモデルは止めず、
        超過分はサーバー側でアイドルから順に非同期退避される（→ POST /admin/config）。
        """
        arg = arg.strip().lower()
        if arg in ("off", "none", "unlimited", "inf", "∞", "0"):
            value: int | None = None
        else:
            try:
                value = int(arg)
            except ValueError:
                self.notify(
                    f"max_resident には数値か off を指定してください: '{arg}'",
                    severity="error", timeout=4,
                )
                return
            if value < 1:
                self.notify(
                    "max_resident は 1 以上、または off（無制限）です",
                    severity="error", timeout=4,
                )
                return
        label = "∞" if value is None else str(value)

        def _apply() -> None:
            res = gateway_set_max_resident(value, self.host, self.port)
            msg, sev = (
                (f"max_resident を {label} に変更しました", "information")
                if res is not None
                else ("max_resident の変更に失敗しました（ゲートウェイ未起動？）", "error")
            )
            self.call_from_thread(self.notify, msg, severity=sev, timeout=3)

        self._run(f"max_resident → {label}…", _apply)

    def action_quit(self) -> None:
        # q（終了）ではゲートウェイ・デーモンも停止する。ダッシュボードを閉じたら裏の常駐も
        # 完全に終了させることで、次回 `uv run gw` 起動時に必ず最新コードで立ち上がる。
        # 停止はロード済みモデルの解放待ちで時間がかかるため、UI スレッドで同期実行すると
        # 画面が固まる。別スレッドに逃がして "closing…" を出し、完了したら終了する。
        self._quit_worker()

    @work(thread=True, group="quit")
    def _quit_worker(self) -> None:
        self.busy = "closing…"
        self.call_from_thread(self._render)
        try:
            self._kill_ports()
        finally:
            # 停止が終わってから閉じる（ポートが解放され、次回起動が旧プロセスに相乗りしない）。
            self.call_from_thread(self.exit)

    def on_click(self, event) -> None:
        # 表の行をクリックしたら、その行のモデル名をコピーする（行ハイライトは出さない）。
        # クリック位置のセル meta から行番号を得る（ヘッダ -1・表外は無視）。
        meta = getattr(event, "style", None)
        meta = getattr(meta, "meta", None) or {}
        row = meta.get("row")
        if isinstance(row, int) and 0 <= row < len(self._row_models):
            self._copy_model(self._row_models[row])

    def _copy_model(self, model: str) -> None:
        copied = _clipboard_copy(self, model)
        self.notify(
            f"copied: {model}" if copied else f"クリップボードにコピーできませんでした: {model}",
            timeout=3,
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = (event.value or "").strip()
        event.input.value = ""
        cmd = raw.lower()
        if cmd in ("q", "quit", "exit"):
            self.action_quit()  # キー `q` と同じく、デーモンも停止してから終了する
            return
        # `max <n>` / `m <n>`: 引数を伴うので先頭トークンで分岐する（例: "max 2", "max off"）。
        head, _, tail = raw.partition(" ")
        if head.lower() in ("max", "m") and tail.strip():
            self._set_max_resident(tail)
            return
        # `mtp [model]`: 必要な MTP ドラフターと取得状況を表示する（tui.mtp_report）。
        # モデル ID は大文字小文字を区別するので raw から取る（cmd に落とさない）。
        if head.lower() == "mtp":
            self.push_screen(MtpScreen(tail.strip() or None))
            return
        handler = {
            "s": self.action_stop, "stop": self.action_stop,
            "r": self.action_restart, "restart": self.action_restart,
            "g": self.action_start, "start": self.action_start,
            "l": self.action_log, "log": self.action_log,
            "u": self.action_update, "update": self.action_update,
        }.get(cmd)
        if handler is None:
            if raw:  # 空 Enter は無視。タイプミス等は黙殺せず知らせる
                self.notify(f"不明なコマンドです: '{raw}'", severity="warning", timeout=4)
            return
        handler()
