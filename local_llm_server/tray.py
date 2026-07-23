"""メニューバー常駐アイコン（macOS 専用・デーモンの随伴プロセス）。

Ollama がメニューバーのアイコンで「動いている」を伝えるのと同じく、ゲートウェイの
稼働中だけメニューバーにアイコンを表示する。デーモン（_run_gateway_locked）が子プロセス
として起動し、トレイ専用パイプ（--fd）を継承する——**アイコンの存在＝デーモンの生存**。
デーモンがどんな死に方をしてもパイプ EOF で自分から消えるので、「死んでいるのに
アイコンだけ残る」嘘をつかない。gateway.toml の `tray = false` で無効化できる。

運用は Ollama と同じ「静的アイコン・定期処理なし」:
  - アイコンは同梱のモノクロ図形（assets/tray-icon.png。テンプレート画像なので
    メニューバーのライト/ダークに自動追従）。ロード数などのライブ表示はしない
  - 状態（接続先 URL・ロード中モデル）は**メニューを開いた瞬間にだけ**取得する
    （常駐中のポーリングは無い——CPU を使うのはパイプからイベントが届いたときだけ）。
    開く動作は取得を**待たない**: 前回の内容で即座に開き、裏スレッドの取得が終わったら
    開いたまま項目を差し替える（menuWillOpen: → _refresh_async → applyStatus:）
  - できる操作は ログを開く / ゲートウェイを停止（gw stop 相当）の 2 つだけ。
    「アイコンだけ隠す」は置かない——**アイコンの有無＝デーモンの生死**という対応を
    例外なく保つ（隠せると「動いているのに出ていない」状態が生まれ、対応が崩れる）

更新も Ollama と同じ体験（更新マーク → ワンクリックで閉じて更新して再起動）:
  - デーモンの update watcher が新版を検知すると、専用パイプに `update-ready <ver>`
    （取得済み・再起動待ち）/ `update-available <ver>`（未取得。auto_update=false や
    dirty tree）を書く。トレイはそれを受けてアイコンの隣に「⬆」を出す
    ——**プッシュ通知なのでここでもポーリングは増えない**
  - メニューには**常に**更新項目がある: 新版検知済みなら「今すぐ更新して再起動」、
    そうでなければ「更新を確認」。どちらも同じ POST /admin/update を叩き、デーモンが
    確認→（あれば）取得して自分を execv で新コードに置き換える。旧トレイはパイプ EOF で
    消え、新デーモンが素のアイコンのトレイを出し直す——「一度閉じて更新して再起動」。
    最新だった・失敗した場合は macOS 通知で知らせる（メニューは閉じているため）

GUI は **pyobjc（AppKit）で直接**実装する。rumps は使わない——rumps 0.4 は 10.10 で
非推奨になった `NSStatusItem.setTitle_/setImage_` に依存しており、最新 macOS
（Darwin 25 系）ではステータスアイテムが一切表示されない（実測）。現行 API の
`NSStatusItem.button()` を使う。macOS 以外・pyobjc 不在では何もせず終了する
（デーモンは起動失敗を無視する——アイコンは飾りで、ゲートウェイの本体機能ではない）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

from .server import (
    gateway_admin_status,
    gateway_log_path,
    read_gateway_runtime,
    stop_pid,
)

# メニューバーのアイコン（静的。Ollama と同じく状態でチカチカさせない）。
# テンプレート画像（黒＋透過）なので、ライト/ダークメニューバーに OS が自動で合わせる。
_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "tray-icon.png")
# アイコンが読めないときの予備の題字と、更新検知時にアイコンの隣へ出す印。
_FALLBACK_TITLE = "gw"
_UPDATE_MARK = "⬆"

# メニューを開いた瞬間の取得なので短めに切る（ハングした上流でメニューを固めない）。
_STATUS_TIMEOUT_S = 1.0
# 「今すぐ更新」は未取得なら git pull + 依存同期が走る（数十秒〜数分）ので長めに待つ。
_UPDATE_TIMEOUT_S = 600.0


def format_rows(admin: dict | None, host: str, port: int) -> list[str]:
    """メニューの情報行（クリック不可の表示専用）を作る（純粋関数・テスト可能）。"""
    rows = [f"http://{host}:{port}/v1"]
    models = (admin or {}).get("models", [])
    loaded = [m for m in models if m.get("loaded")]
    if not loaded:
        rows.append("モデル未ロード（初回リクエストでロード）")
    for m in loaded:
        busy = int(m.get("inflight", 0))
        state = f"処理中 {busy}" if busy else "待機"
        rows.append(f"{m.get('model', '?')} — {state}")
    return rows


def parse_update_event(line: str) -> tuple[str, str] | None:
    """デーモンからの通知行を解釈する（純粋関数・テスト可能）。

    `update-ready <ver>`（取得済み・再起動だけ）/ `update-available <ver>`（未取得）の
    2 種だけを認識し、他は None（将来の行を黙って読み飛ばす前方互換）。
    """
    parts = line.strip().split()
    if len(parts) == 2 and parts[0] in ("update-ready", "update-available"):
        return parts[0], parts[1]
    return None


def merge_update_info(info: dict, admin: dict | None) -> dict:
    """パイプ通知（info）と /admin/status の update 欄を統合する（純粋関数・テスト可能）。

    どちらか一方しか届いていなくても更新マークを出せるようにする（通知はプッシュ、
    admin はメニューを開いた瞬間の確認、の 2 経路）。fetched（取得済み）が最優先。
    """
    upd = (admin or {}).get("update") or {}
    merged = dict(info)
    if upd.get("fetched"):
        merged["kind"] = "update-ready"
        merged["latest"] = upd.get("latest") or merged.get("latest")
    elif upd.get("available") and not merged.get("kind"):
        merged["kind"] = "update-available"
        merged["latest"] = upd.get("latest") or merged.get("latest")
    return merged


def _watch_pipe(fd: int, on_event) -> None:
    """トレイ専用パイプを読む: 行 = 更新通知、EOF = デーモンの死（アイコンごと消える）。"""
    buf = b""
    while True:
        try:
            data = os.read(fd, 4096)
        except InterruptedError:
            continue
        except OSError:
            break
        if not data:
            break
        buf += data
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            event = parse_update_event(raw.decode("utf-8", errors="replace"))
            if event is not None:
                on_event(event)
    os._exit(0)


def _stop_gateway() -> None:
    """ゲートウェイを停止する（gw stop 相当。自分もパイプ EOF で道連れに消える）。"""
    rec = read_gateway_runtime()
    pid = rec.get("pid") if rec else None
    if isinstance(pid, int):
        threading.Thread(target=stop_pid, args=(pid,), daemon=True).start()


def _notify(title: str, message: str) -> None:
    """macOS 通知を出す（best-effort）。権限が無い/ヘッドレスでも黙って無視する。

    メニューはクリックで閉じてしまうため、更新結果（最新だった・失敗した）の
    視覚フィードバックはメニュー外＝通知で返す。更新が実際に走る場合はアイコンが
    消えて出直すこと自体がフィードバックになる。
    """
    try:
        subprocess.Popen(
            ["osascript", "-e",
             f"display notification {json.dumps(message)} with title {json.dumps(title)}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _post_update_now(host: str, port: int) -> None:
    """POST /admin/update: 更新の確認と適用（Ollama の Restart to update 相当）。

    デーモンが（新版があれば git pull + 依存入れ直しをして）自分を新コードで再起動する。
    最新だった・適用できなかった場合は通知で知らせる（メニューは閉じているため）。
    """
    req = urllib.request.Request(
        f"http://{host}:{port}/admin/update",
        data=b"{}", headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_UPDATE_TIMEOUT_S) as resp:
            data = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            reason = json.loads(exc.read() or b"{}").get("error", "")
        except (ValueError, OSError):
            reason = ""
        _notify("gw の更新", reason or f"更新できませんでした（HTTP {exc.code}）")
        return
    except OSError:
        _notify("gw の更新", "ゲートウェイに接続できませんでした")
        return
    status = data.get("status")
    if status == "up-to-date":
        cur = data.get("current")
        _notify("gw の更新", f"最新です（v{cur}）" if cur else "最新です")
    elif status == "restarting":
        latest = data.get("latest")
        _notify("gw の更新",
                f"更新して再起動します（v{latest}）" if latest else "更新して再起動します")


def run_app(host: str, port: int, fd: int | None) -> int:
    try:
        import AppKit
        from Foundation import NSObject
        from PyObjCTools import AppHelper
    except ImportError:
        print("tray: pyobjc が無いためアイコンは出しません"
              "（依存が古い導入のままです。クローンで `make install` を再実行してください）",
              file=sys.stderr)
        return 0

    nsapp = AppKit.NSApplication.sharedApplication()
    # Dock にもアプリスイッチャーにも出さない、メニューバーだけの常駐（Ollama と同じ）。
    nsapp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    update_info: dict = {"kind": None, "latest": None}

    # ステータスアイテム（現行 API: button() 経由で画像とタイトルを設定する）。
    # NSStatusBar はアイテムを retain しない——この関数フレームが生きている限り
    # ローカル変数が強参照になる（runEventLoop がここでブロックし続ける）。
    status = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(
        AppKit.NSVariableStatusItemLength
    )
    button = status.button()
    icon = (
        AppKit.NSImage.alloc().initWithContentsOfFile_(_ICON_PATH)
        if os.path.isfile(_ICON_PATH) else None
    )
    if icon is not None:
        icon.setSize_((18, 18))
        icon.setTemplate_(True)  # ライト/ダークメニューバーに OS が自動で合わせる
        button.setImage_(icon)
        button.setImagePosition_(AppKit.NSImageLeft)  # 更新マーク（title）を隣に出せる配置
    else:  # アセット欠落でもアイコン無しにはしない（文字で出す）
        button.setTitle_(_FALLBACK_TITLE)

    # 状態キャッシュ: メニューは前回の内容で**即座に開き**、裏で取りに行って開いたまま
    # 差し替える（NSMenu は表示中でも項目を変更できる）。開く動作が取得を待たないので
    # ラグゼロ、かつ「開いた瞬間にだけ取得」（定期処理ゼロ）はそのまま。
    fetch_state: dict = {"admin": None, "fetched_once": False, "inflight": False}

    class _TrayDelegate(NSObject):
        """メニューの delegate 兼、メニュー項目のターゲット（ObjC セレクタの受け口）。"""

        def menuWillOpen_(self, menu) -> None:  # noqa: N815 - ObjC セレクタ命名
            _rebuild_menu(menu)   # キャッシュから即描画（ブロックしない）
            _refresh_async(menu)  # 裏で取得 → applyStatus: で差し替え

        def applyStatus_(self, menu) -> None:  # noqa: N815
            _rebuild_menu(menu)

        def openLog_(self, _sender) -> None:  # noqa: N815
            subprocess.Popen(["open", gateway_log_path(port)])

        def stopGateway_(self, _sender) -> None:  # noqa: N815
            _stop_gateway()

        def updateNow_(self, _sender) -> None:  # noqa: N815
            threading.Thread(target=_post_update_now, args=(host, port),
                             daemon=True).start()

        def showUpdateMark_(self, _arg) -> None:  # noqa: N815
            button.setTitle_(_UPDATE_MARK)  # アイコンの隣に ⬆ を添える

    delegate = _TrayDelegate.alloc().init()

    def _refresh_async(menu) -> None:
        """裏スレッドで状態を取得し、メインスレッドでメニューを差し替える。

        連打で取得を積まないよう in-flight は 1 本だけ。UI 操作（差し替え）は
        AppKit の掟どおり performSelectorOnMainThread でメインスレッドに戻す。
        """
        if fetch_state["inflight"]:
            return
        fetch_state["inflight"] = True

        def _work() -> None:
            try:
                admin = gateway_admin_status(host, port, timeout=_STATUS_TIMEOUT_S)
                fetch_state["admin"] = admin
                # 成功した取得だけを「一度取得できた」と数える。デーモン起動直後は
                # トレイが先に出ていてゲートウェイはまだ準備中（自動導入など）のことが
                # あり、失敗を既成事実にすると「未ロード」と嘘の断言をしてしまう。
                if admin is not None:
                    fetch_state["fetched_once"] = True
            finally:
                fetch_state["inflight"] = False
            delegate.performSelectorOnMainThread_withObject_waitUntilDone_(
                "applyStatus:", menu, False
            )

        threading.Thread(target=_work, daemon=True).start()

    def _add_info(menu, text: str) -> None:
        # action 無し＝自動で無効（グレー表示）。情報行として使う。
        menu.addItem_(
            AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(text, None, "")
        )

    def _add_action(menu, text: str, selector: str) -> None:
        item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            text, selector, ""
        )
        item.setTarget_(delegate)
        menu.addItem_(item)

    def _rebuild_menu(menu) -> None:
        """キャッシュ済みの状態からメニューを組み直す（ネットワークに触れない・即時）。"""
        admin = fetch_state["admin"]
        merged = merge_update_info(update_info, admin)
        update_info.update(merged)
        menu.removeAllItems()
        if admin is None and not fetch_state["fetched_once"]:
            # 一度も取得できていない（初回オープン直後など）。未ロードと断言せず
            # 「取得中」を出す——裏の取得が終わればこのまま差し替わる。
            _add_info(menu, f"http://{host}:{port}/v1")
            _add_info(menu, "状態を取得中…")
        else:
            for row in format_rows(admin, host, port):
                _add_info(menu, row)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        # 更新項目は**常に**出す（いつでもアイコンから更新できる）。新版が検知済みなら
        # 「今すぐ更新して再起動」、そうでなければ「更新を確認」——どちらも同じ
        # /admin/update を叩き、デーモンが確認→（あれば）適用→再起動する。
        if merged.get("kind"):
            button.setTitle_(_UPDATE_MARK)  # ここはメインスレッドなので直接更新してよい
            latest = merged.get("latest")
            label = (f"今すぐ更新して再起動（v{latest}）" if latest
                     else "今すぐ更新して再起動")
            _add_action(menu, label, "updateNow:")
        else:
            _add_action(menu, "更新を確認", "updateNow:")
        _add_action(menu, "ログを開く", "openLog:")
        _add_action(menu, "ゲートウェイを停止", "stopGateway:")

    # メニューを開く瞬間に状態を取りに行く（Ollama 流: 常駐中の定期処理を持たない）。
    menu = AppKit.NSMenu.alloc().init()
    menu.setDelegate_(delegate)
    _rebuild_menu(menu)   # 初期メニュー（delegate 不発時でも空メニューにはしない保険）
    _refresh_async(menu)  # 起動直後に 1 回だけ温めておく（初クリックからキャッシュが効く）
    status.setMenu_(menu)

    # パイプ読みスレッドからの更新マーク表示はメインスレッドへ橋渡しする（AppKit の掟）。
    def _on_pipe_event(event: tuple[str, str]) -> None:
        kind, latest = event
        update_info["kind"], update_info["latest"] = kind, latest
        delegate.performSelectorOnMainThread_withObject_waitUntilDone_(
            "showUpdateMark:", None, False
        )

    if fd is not None:
        threading.Thread(target=_watch_pipe, args=(fd, _on_pipe_event),
                         daemon=True).start()

    AppHelper.runEventLoop()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="local_llm_server.tray", add_help=False)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--fd", type=int, default=None)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if sys.platform != "darwin":
        return 0
    return run_app(args.host, args.port, args.fd)


if __name__ == "__main__":
    raise SystemExit(main())
