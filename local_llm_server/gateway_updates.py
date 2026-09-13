"""Release update status and the background update watcher."""

from __future__ import annotations

import sys
import threading
import time


def refresh_update_state(state: dict) -> None:
    """update.check() を 1 回だけ実行して update_state を更新する（**適用はしない**）。

    「更新の有無」を最新化する純粋な確認。オンデマンド（トレイのメニューを開いたとき）に
    リスタート無しで呼ぶための小片。取得や再起動は一切しない——適用は
    ターミナルの `gw update` だけが行う。ネットワーク I/O は失敗しても握りつぶす。
    """
    from . import update

    try:
        st = update.check(timeout=3.0)
    except Exception:  # noqa: BLE001 - 確認失敗（オフライン等）は状態を変えず黙って戻る
        return
    state["available"] = bool(st.available)
    state["current"] = st.current
    state["latest"] = st.latest
    state["reason"] = st.reason
    # 「取ってくるものは無いが、走っているコードが古い」＝再起動だけで新版になる状態。
    state["restart_required"] = bool(st.restart_required)
    state["running"] = update.running_source_version()


def maybe_refresh_update_state(
    srv, *, wait: bool = False, throttle: float = 30.0, refresh=refresh_update_state
) -> None:
    """スロットル付きで、バックグラウンドに 1 本だけオンデマンド確認を走らせる。

    /admin/status GET のたびに呼ばれる（トレイがメニューを開くたび）。前回から
    _UPDATE_ONDEMAND_THROTTLE 秒未満・状態が無いときは何もしない。通常はレスポンスを
    ブロックしない。wait=True（トレイの裏スレッドからの明示指定）では、同じレスポンスへ
    最新状態を載せるため確認完了を待つ。既に別の確認が走っている場合もその完了を待つ。
    """
    state = getattr(srv, "update_state", None)
    if state is None:
        return
    now = time.monotonic()
    if getattr(srv, "_update_check_inflight", False):
        done = getattr(srv, "_update_check_done", None)
        if wait and done is not None:
            # update.check の timeout (3 秒) より少し長く待つ。失敗時も finally で必ず set される。
            done.wait(3.5)
        return
    if now - getattr(srv, "_last_update_check", 0.0) < throttle:
        return
    srv._last_update_check = now
    srv._update_check_inflight = True
    done = threading.Event()
    srv._update_check_done = done

    def _work() -> None:
        try:
            refresh(state)
        finally:
            srv._update_check_inflight = False
            done.set()

    if wait:
        _work()
    else:
        threading.Thread(target=_work, daemon=True).start()


def update_watcher(
    stop: threading.Event,
    *,
    state: dict | None = None,
    notify=None,
    warmup_interval: float = 60.0,
    check_interval: float = 3600.0,
) -> None:
    """新しいリリースタグを確認し、利用可能ならトレイへ通知する。

    このスレッドは確認専用で、ソース取得・依存同期・再起動は一切行わない。更新を適用できる
    経路は、ユーザーがターミナルで明示的に実行する ``gw update`` だけに限定する。
    検知状態は ``state``（``/admin/status`` に掲載）へ書き、``notify`` には同じ種類と版を重複させず
    ``update-available <ver>`` または ``update-ready <ver>`` を送る。ネットワーク障害は
    稼働中の推論へ影響させない。
    """
    from . import update

    notified: str | None = None  # この種類＋版は通知済み（毎時間再通知しない）
    first = True

    def _tell(kind: str, latest: str | None) -> None:
        nonlocal notified
        if notify is None or notified == f"{kind}:{latest}":
            return
        notified = f"{kind}:{latest}"
        try:
            notify(f"{kind} {latest}")
        except Exception:  # noqa: BLE001 - 通知はおまけ（トレイ不在等で失敗しても続行）
            pass

    while not stop.wait(warmup_interval if first else check_interval):
        first = False
        try:
            st = update.check(timeout=3.0)
        except Exception as exc:  # noqa: BLE001 - 監視スレッドは落とさない
            print(f"Update check failed ({exc}).", file=sys.stderr)
            continue
        if state is not None:
            state.update(
                {
                    "available": bool(st.available),
                    "current": st.current,
                    "latest": st.latest,
                    "reason": st.reason,
                    "restart_required": bool(st.restart_required),
                    "running": update.running_source_version(),
                }
            )
        if st.available:
            _tell("update-available", st.latest)
        elif st.restart_required:
            # 手動 git pull 等でソースだけが先に新しくなった場合も、
            # 「再起動すれば反映できる」ことをアイコンで通知する。
            _tell("update-ready", st.current or st.latest)
