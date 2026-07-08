"""ターミナル常駐の TUI ダッシュボード（`local-llm-server` の既定起動）。

`./gateway.toml` を読み、ゲートウェイを裏で常駐させて `GET /admin/status` を定期ポーリングし、
状態を自動更新表示する。実体は textual 製アプリ（→ tui_app.py、角丸枠・truecolor・本物の入力欄）。
このモジュールは表示に依らない**データ層**（カタログ×ライブ状態の統合・時間整形・ログ表示）と、
textual を遅延 import する起動口だけを持つ（merge_status 等は端末なしでテストできる）。
"""
from __future__ import annotations

import os
import sys
import tomllib

from .daemon import load_gateway_config
from .server import MTP_DRAFTERS, gateway_log_path, mtp_status


def resolve_config() -> str | None:
    """使う gateway.toml を決める。**カレントディレクトリの `./gateway.toml` のみ**。

    存在すればそのパス、無ければ None（呼び出し側がエラーにする）。場所は CWD 固定で、
    位置引数やホーム等の外部は見ない（「gateway.toml は CWD に置く」という 1 ルール）。
    """
    path = os.path.join(os.getcwd(), "gateway.toml")
    return path if os.path.isfile(path) else None


def mtp_report(model: str | None) -> tuple[str, int]:
    """使う予定のモデルに必要な MTP ドラフターを、ダウンロード前に調べて文面にする。

    TUI の `mtp` コマンド（MtpScreen）が使う本体。対応表（MTP_DRAFTERS）の辞書引きと
    ローカルキャッシュ確認だけで、モデルのダウンロードは一切しない（`mtp_status` と同じく
    非破壊）。gateway.toml も見ないので、どのディレクトリからでも実行できる。model を省略
    （None/空）すると対応表を全件、取得状況つきで並べる。

    戻り値: (表示テキスト, 終了コード)。指定モデルが MTP 非対応なら 1、
    それ以外（ready / available / 一覧表示）は 0。
    """

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
            "`mtp` コマンドで表示。",
            1,
        )
    return _describe(model), 0


def merge_status(gcfg, admin: dict | None, ready: bool | None = None) -> dict:
    """gateway.toml のカタログと `/admin/status` のライブ状態を1つのビューに統合する（純粋関数）。

    カタログの全モデルを並べ（未起動も「unloaded」で見せる）、起動中のものはライブ状態
    （loaded/idle/busy・処理中数・累計・アイドル自動解放までの残り）を重ねる。色付けや描画を
    含まないのでそのままテストできる。
    """
    live = {m["model"]: m for m in (admin or {}).get("models", [])}
    idle_timeout = gcfg.idle_timeout

    def _row(model, backend, port, m):
        """ライブ状態 m（None=未ロード）から表示用の 1 行を作る。"""
        # MTP（高速化）の利用可否は本体名から判定する（ドラフターがキャッシュ済みなら "ready"）。
        mtp = mtp_status(model)
        if not m or not m.get("loaded"):
            return {
                "model": model, "backend": backend, "port": port,
                "state": "unloaded", "inflight": 0, "instances": 0,
                "requests": (m or {}).get("requests", 0), "idle_remaining": None,
                "sessions": (m or {}).get("sessions", 0), "mtp": mtp,
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
            # 起動中インスタンス数（負荷ベースの複製で >1 になる。並列度の目安）。
            "instances": int(m.get("instances", 1)),
            "requests": int(m.get("requests", 0)), "idle_remaining": remaining,
            "sessions": int(m.get("sessions", 0)),  # 在席エージェント数（0 で即アンロード対象）
            "mtp": mtp,
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
    # ready は呼び出し側（ポーリングのワーカースレッド）が判定して渡す。ここで HTTP を
    # 叩くと UI スレッドが固まる（_render は毎秒呼ばれる）ため、純粋関数のまま保つ。
    # 省略時は「/admin/status が取れた＝稼働中」で判定する。
    if ready is None:
        ready = bool(admin)
    # max_resident は実行中に変更できる（POST /admin/config）。ライブ値（admin）があれば
    # それを優先し、無ければ gateway.toml の起動時値にフォールバックする。admin では None が
    # 「無制限」を意味するので、キーが在ればその値（None 含む）をそのまま使う。
    live_max = (admin or {}).get("max_resident", gcfg.max_resident) if admin else gcfg.max_resident
    return {
        "ready": ready,
        "uptime": (admin or {}).get("uptime"),
        "requests": (admin or {}).get("requests", sum(r["requests"] for r in rows)),
        "max_resident": live_max,
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


def read_log_tail(port: int, max_lines: int = 1000, max_bytes: int = 512 * 1024) -> str:
    """ゲートウェイログの末尾（最大 max_lines 行）を返す（TUI 内のログ画面で表示用）。

    外部ページャ（less 等）には頼らず、textual アプリ内のスクロール画面に出すための純データ。
    ログはローテーションされず肥大化しうるので、末尾 max_bytes だけ読む（全読みすると
    巨大ログで UI スレッドが固まる）。ログがまだ無い／空のときは案内文を返す。
    """
    path = gateway_log_path(port)
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)  # 末尾へ
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return f"(ログはまだありません: {path})"
    if not data:
        return f"(ログは空です: {path})"
    lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
    if size > max_bytes and lines:
        lines = lines[1:]  # 途中から読んだ先頭の欠け行は捨てる
    return "".join(lines[-max_lines:])


def run_tui(gcfg) -> int:
    """TUI を起動する。textual は重いので遅延 import（headless ワーカーでは読み込まない）。

    自動更新（PyPI 新版を git pull で追従）が適用されると、アプリは
    `restart_after_exit` を立てて終了する。その場合は新コードで TUI を再 exec する
    （textual の run() を抜けて端末を復帰させてから exec するので画面が乱れない）。
    """
    from .tui_app import GatewayMonitor

    app = GatewayMonitor(gcfg)
    app.run()
    if getattr(app, "restart_after_exit", False):
        from . import update

        update.reexec_tui()  # 戻らない（現プロセスを新 TUI で置き換える）
    return 0


def main(argv: list[str] | None = None) -> int:
    """`gw` コマンド: カレントディレクトリの `./gateway.toml` で TUI ダッシュボードを起動する。

    引数は取らない（停止/再起動/起動・max 変更・MTP 確認・ログ表示は、すべて TUI 内の
    単キー `s/r/g/m/l/q` と入力欄コマンドで行う → tui_app.py）。ダッシュボードはゲートウェイ
    本体を裏で常駐させる（`python -m local_llm_server` = __main__ を別プロセスで起動）。
    """
    config_path = resolve_config()
    if config_path is None:
        print(
            "./gateway.toml not found in the current directory. Create one here "
            "(see the gateway.toml example in the repo), then run again from that directory.",
            file=sys.stderr,
        )
        return 2
    try:
        gcfg = load_gateway_config(config_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"Failed to load ./gateway.toml: {exc}", file=sys.stderr)
        return 2
    return run_tui(gcfg)


if __name__ == "__main__":
    raise SystemExit(main())
