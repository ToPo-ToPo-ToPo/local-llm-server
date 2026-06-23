"""システムトレイ常駐 GUI: ゲートウェイの状態をひと目で確認・操作する。

CLI の `--status` / `--stop` は都度コマンドを打つ必要があり、状況の把握がしにくい。
この GUI はデスクトップのシステムトレイ（macOS はメニューバー、Windows は通知領域、
Linux はトレイ）にアイコンを常駐させ、ウィンドウを占有せずに

  * ゲートウェイの応答可否（ready / 起動中 / 停止）をアイコンの色で
  * 各モデルの常駐状態（loaded）と処理中リクエスト数（inflight）
  * 公開ポート / PID / 運用方針（max_resident・idle_timeout）

を一定間隔で更新表示する。メニューからゲートウェイの停止・ログを開く・再読込もできる。

`pystray` を使うため **Windows / macOS / Linux で動く**（Web アプリのように画面を
占有しない）。設定は CLI と同じく **カレントディレクトリの `./gateway.toml`** を読む。

起動:

    local-llm-server-gui            # ./gateway.toml のあるディレクトリで
    python -m local_llm_server.gui

依存（`pystray` / `pillow`）は GUI extra。未導入なら案内して終了する:

    pip install 'local-llm-server[gui]'

Linux はトレイ表示にシステムトレイ（GNOME は AppIndicator 拡張など）が必要。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field

from .constants import project_cache_dir
from .daemon import load_gateway_config
from .server import (
    find_pids_on_port,
    gateway_admin_status,
    server_status,
    stop_pid,
)

REFRESH_SECONDS = 4.0

# 状態 → アイコン色（RGB）。テキストは付けられない環境もあるので色で表す。
_COLORS = {
    "ready": (46, 204, 64),     # 緑: 応答可能
    "starting": (255, 220, 0),  # 黄: LISTEN しているが未応答（起動中）
    "down": (170, 170, 170),    # 灰: 停止
    "error": (255, 65, 54),     # 赤: 設定エラー等
}


@dataclass
class View:
    """画面に出す状態の集約（プレゼンテーション用、純粋データ）。"""

    state: str = "down"           # ready / starting / down / error
    loaded: int = 0               # ロード済みモデル数（アイコンの数字）
    summary: str = ""             # ゲートウェイ行
    policy: str = ""              # max_resident / idle-unload 行
    model_lines: list[str] = field(default_factory=list)
    stop_enabled: bool = False


def _fmt_seconds(sec) -> str:
    """idle/load timeout 表示用（off / 1200s / 20m 風）。"""
    if not sec:
        return "off"
    sec = float(sec)
    if sec >= 60 and sec % 60 == 0:
        return f"{int(sec // 60)}m"
    return f"{sec:g}s"


def build_view(
    host: str,
    port: int,
    model_ids: list[str],
    config_error: str | None = None,
) -> View:
    """現在の状態を取得して View にまとめる（pystray 非依存・テスト可能）。

    まず GET /admin/status（常駐モデルのライブ状態）を試し、取れなければ server_status に
    フォールバックする。何も応答せず LISTEN も無ければ「停止」とみなす。
    """
    if config_error is not None:
        return View(state="error", summary="gateway.toml error (see Reload)",
                    policy=config_error[:100])

    admin = gateway_admin_status(host, port)
    st = server_status(host, port)

    if st is None:
        return View(state="down", summary=f"Gateway stopped  (:{port})",
                    model_lines=[f"{m}   —" for m in model_ids], stop_enabled=False)

    ready = bool(st["ready"])
    live: dict[str, dict] = {}
    if admin and isinstance(admin.get("models"), list):
        for m in admin["models"]:
            if isinstance(m, dict) and m.get("model"):
                live[m["model"]] = m
    loaded = sum(1 for m in live.values() if m.get("loaded"))

    pids = ", ".join(str(p) for p in st["pids"]) or "?"
    summary = f"Gateway {'ready' if ready else 'starting…'}  (:{port}, pid {pids})"

    lines: list[str] = []
    for model in model_ids:
        info = live.get(model)
        if info is None:
            lines.append(f"{model}   {'idle' if ready else '—'}")
            continue
        mark = "loaded" if info.get("loaded") else "idle"
        inflight = info.get("inflight") or 0
        suffix = f" ({inflight} in-flight)" if inflight else ""
        lines.append(f"{model}   {mark}{suffix}")

    policy = ""
    if admin:
        cap = admin.get("max_resident")
        cap_s = "∞" if cap is None else str(cap)
        policy = (
            f"resident {loaded}/{cap_s}   idle-unload {_fmt_seconds(admin.get('idle_timeout'))}"
        )

    return View(state="ready" if ready else "starting", loaded=loaded,
                summary=summary, policy=policy, model_lines=lines, stop_enabled=True)


def _make_image(state: str, count: int):
    """状態色の丸アイコンを生成する（ロード済み数があれば数字を重ねる）。"""
    from PIL import Image, ImageDraw  # noqa: PLC0415 - GUI extra

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = _COLORS.get(state, _COLORS["down"])
    d.ellipse([6, 6, size - 6, size - 6], fill=color + (255,))
    if state == "ready" and count:
        text = str(count) if count < 10 else "9+"
        # 既定フォントで中央寄せ（環境差を避けるため textbbox で測る）。
        try:
            bbox = d.textbbox((0, 0), text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            d.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
                   text, fill=(255, 255, 255, 255))
        except Exception:  # noqa: BLE001 - 数字が出せなくても致命的でない
            pass
    return img


def _open_path(path: str) -> None:
    """OS のファイルマネージャでパスを開く（mac/Windows/Linux）。"""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _require_deps():
    """pystray を import する。無ければ案内して終了（GUI extra 未導入）。"""
    try:
        import pystray  # noqa: PLC0415
        from PIL import Image  # noqa: F401,PLC0415
    except ImportError:
        sys.stderr.write(
            "The tray GUI needs the 'pystray' and 'pillow' packages.\n"
            "Install them with:  pip install 'local-llm-server[gui]'\n"
        )
        raise SystemExit(1)
    return pystray


class TrayApp:
    """ゲートウェイ監視トレイアプリ本体。"""

    def __init__(self, config_path: str) -> None:
        self._pystray = _require_deps()
        self.config_path = config_path
        self._config_error: str | None = None
        self.host = "127.0.0.1"
        self.port = 8799
        self.model_ids: list[str] = []
        self.model_ports: list[int] = []
        self._view = View()
        self._stop_event = threading.Event()

        self._load_config()
        self._view = self._collect()
        self.icon = self._pystray.Icon(
            "local-llm-server",
            icon=_make_image(self._view.state, self._view.loaded),
            title=self._view.summary,
            menu=self._build_menu(),
        )

    # ---- 設定（gateway.toml） ----------------------------------------------
    def _load_config(self) -> None:
        """./gateway.toml を読み、ホスト/ポート/モデルを取り込む（失敗は記録して継続）。"""
        try:
            cfg = load_gateway_config(self.config_path)
        except Exception as exc:  # noqa: BLE001 - 設定エラーは画面に出して動き続ける
            self._config_error = str(exc)
            return
        self._config_error = None
        self.host = cfg.host
        self.port = cfg.port
        self.model_ids = [m.model for m in cfg.models]
        self.model_ports = [m.port for m in cfg.models]

    # ---- メニュー構築 -------------------------------------------------------
    def _build_menu(self):
        """現在のモデル数に合わせてメニューを組む（行テキストは表示時に動的評価）。"""
        ps = self._pystray
        item, menu, sep = ps.MenuItem, ps.Menu, ps.Menu.SEPARATOR

        # 表示専用行（enabled=False でグレー表示・押せない）。テキストは callable。
        rows = [
            item(lambda _i: self._view.summary, None, enabled=False),
            sep,
        ]
        for k in range(len(self.model_ids)):
            rows.append(item(
                lambda _i, k=k: (self._view.model_lines[k]
                                 if k < len(self._view.model_lines) else ""),
                None, enabled=False,
            ))
        rows += [
            sep,
            item(lambda _i: self._view.policy, None, enabled=False,
                 visible=lambda _i: bool(self._view.policy)),
            sep,
            item("Stop gateway", self._on_stop,
                 enabled=lambda _i: self._view.stop_enabled),
            item("Open log folder", self._on_open_logs),
            item("Refresh now", self._on_refresh),
            item("Reload gateway.toml", self._on_reload),
            sep,
            item("Quit", self._on_quit),
        ]
        return menu(*rows)

    # ---- 状態取得・反映 -----------------------------------------------------
    def _collect(self) -> View:
        return build_view(self.host, self.port, self.model_ids, self._config_error)

    def _refresh(self) -> None:
        """状態を取り直してアイコン色・ツールチップ・メニューを更新する。"""
        self._view = self._collect()
        try:
            self.icon.icon = _make_image(self._view.state, self._view.loaded)
            self.icon.title = self._view.summary
            self.icon.update_menu()
        except Exception:  # noqa: BLE001 - 更新失敗で poller を落とさない
            pass

    def _poll_loop(self, icon) -> None:
        icon.visible = True
        self._refresh()
        while not self._stop_event.wait(REFRESH_SECONDS):
            self._refresh()

    # ---- 操作 ---------------------------------------------------------------
    def _on_stop(self, _icon, _item) -> None:
        """公開ポートと内部モデルポートのプロセスを停止する（CLI --stop 相当）。"""
        ports = [self.port] + list(self.model_ports)
        for port in ports:
            for pid in find_pids_on_port(port):
                stop_pid(pid)
        self._refresh()

    def _on_open_logs(self, _icon, _item) -> None:
        """ログの保存ディレクトリ（./.local-llm-server/）を開く。"""
        log_dir = project_cache_dir()
        if os.path.isdir(log_dir):
            _open_path(log_dir)

    def _on_refresh(self, _icon, _item) -> None:
        self._refresh()

    def _on_reload(self, _icon, _item) -> None:
        """gateway.toml を読み直し、モデル行が変わればメニューを作り直す。"""
        prev = list(self.model_ids)
        self._load_config()
        if self.model_ids != prev:
            self.icon.menu = self._build_menu()
        self._refresh()

    def _on_quit(self, _icon, _item) -> None:
        self._stop_event.set()
        self.icon.stop()

    def run(self) -> None:
        # setup=... はアイコンが可視になった直後に呼ばれる。そこで更新ループを回す。
        self.icon.run(setup=self._poll_loop)


def _resolve_config() -> str | None:
    """CLI と同じく、カレントディレクトリの ./gateway.toml のみを使う。"""
    path = os.path.join(os.getcwd(), "gateway.toml")
    return path if os.path.isfile(path) else None


def main(argv: list[str] | None = None) -> int:
    config_path = _resolve_config()
    if config_path is None:
        sys.stderr.write(
            "./gateway.toml not found in the current directory. Run the GUI from the "
            "directory that holds your gateway.toml.\n"
        )
        return 1
    TrayApp(config_path).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
