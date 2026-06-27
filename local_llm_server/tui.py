"""ターミナル常駐の TUI ダッシュボード（`local-llm-server` の既定起動）。

`./gateway.toml` を読み、ゲートウェイを裏で常駐させて `GET /admin/status` を定期ポーリングし、
状態を自動更新表示する。実体は textual 製アプリ（→ tui_app.py、角丸枠・truecolor・本物の入力欄）。
このモジュールは表示に依らない**データ層**（カタログ×ライブ状態の統合・時間整形・ログ表示）と、
textual を遅延 import する起動口だけを持つ（merge_status 等は端末なしでテストできる）。
"""
from __future__ import annotations

import sys

from .server import gateway_log_path, is_ready


def merge_status(gcfg, admin: dict | None) -> dict:
    """gateway.toml のカタログと `/admin/status` のライブ状態を1つのビューに統合する（純粋関数）。

    カタログの全モデルを並べ（未起動も「unloaded」で見せる）、起動中のものはライブ状態
    （loaded/idle/busy・処理中数・累計・アイドル自動解放までの残り）を重ねる。色付けや描画を
    含まないのでそのままテストできる。
    """
    live = {m["model"]: m for m in (admin or {}).get("models", [])}
    idle_timeout = gcfg.idle_timeout

    def _row(model, backend, port, m):
        """ライブ状態 m（None=未ロード）から表示用の 1 行を作る。"""
        if not m or not m.get("loaded"):
            return {
                "model": model, "backend": backend, "port": port,
                "state": "unloaded", "inflight": 0,
                "requests": (m or {}).get("requests", 0), "idle_remaining": None,
                "sessions": (m or {}).get("sessions", 0),
            }
        inflight = int(m.get("inflight", 0))
        idle_for = m.get("idle_for")
        if inflight > 0:
            state, remaining = "busy", None
        else:
            state = "idle"
            remaining = (
                max(0.0, idle_timeout - idle_for)
                if (idle_timeout and idle_for is not None) else None
            )
        return {
            "model": model, "backend": backend, "port": port,
            "state": state, "inflight": inflight,
            "requests": int(m.get("requests", 0)), "idle_remaining": remaining,
            "sessions": int(m.get("sessions", 0)),  # 在席エージェント数（0 で即アンロード対象）
        }

    rows = []
    listed = set()
    # 事前登録モデル（未ロードでも unloaded で見せる）。
    for c in gcfg.models:
        listed.add(c.model)
        rows.append(_row(c.model, c.backend, c.port, live.get(c.model)))
    # 動的ロードされたモデル（事前登録に無い、現在管理中のものを追加表示）。
    for model, m in live.items():
        if model not in listed:
            listed.add(model)
            rows.append(_row(model, m.get("backend", "?"), m.get("port"), m))
    # キャッシュにある DL 済みモデル（まだロードしていない候補。LM Studio 風に「選べる一覧」）。
    for d in (admin or {}).get("available", []):
        mid = d.get("id")
        if mid and mid not in listed:
            listed.add(mid)
            rows.append(_row(mid, d.get("backend", "?"), None, None))
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


def read_log_tail(port: int, max_lines: int = 1000) -> str:
    """ゲートウェイログの末尾（最大 max_lines 行）を返す（TUI 内のログ画面で表示用）。

    外部ページャ（less 等）には頼らず、textual アプリ内のスクロール画面に出すための純データ。
    ログがまだ無い／空のときは案内文を返す。
    """
    path = gateway_log_path(port)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return f"(ログはまだありません: {path})"
    if not lines:
        return f"(ログは空です: {path})"
    return "".join(lines[-max_lines:])


def run_tui(gcfg) -> int:
    """TUI を起動する。textual は重いので遅延 import（headless では読み込まない）。

    textual 未導入なら ImportError を素通しし、呼び出し側（cli）が headless 実行へ落とす。
    """
    from .tui_app import GatewayMonitor

    GatewayMonitor(gcfg).run()
    return 0


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
