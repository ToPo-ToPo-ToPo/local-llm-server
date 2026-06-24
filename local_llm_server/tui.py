"""ターミナル常駐の TUI ダッシュボード（`local-llm-server` の既定起動）。

`./gateway.toml` を読み、ゲートウェイを裏で常駐させて `GET /admin/status` を定期ポーリングし、
状態を自動更新表示する。実体は textual 製アプリ（→ tui_app.py、角丸枠・truecolor・本物の入力欄）。
このモジュールは表示に依らない**データ層**（カタログ×ライブ状態の統合・時間整形・ログ表示）と、
textual を遅延 import する起動口だけを持つ（merge_status 等は端末なしでテストできる）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
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
