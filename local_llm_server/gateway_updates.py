"""Release update status and the background update watcher."""

from __future__ import annotations

import sys
import threading
import time
from typing import Protocol


class RestartableGateway(Protocol):
    def quiesce_for_restart(self) -> bool: ...


def refresh_update_state(state: dict) -> None:
    """update.check() を 1 回だけ実行して update_state を更新する（**適用はしない**）。

    「更新の有無」を最新化する純粋な確認。オンデマンド（トレイのメニューを開いたとき）に
    リスタート無しで呼ぶための小片。取得や再起動は一切しない——それは _update_watcher と
    手動更新（/admin/update）の役目。ネットワーク I/O は失敗しても握りつぶす。
    fetched（取得済み・再起動待ち）フラグは watcher が立てたものを消さない（触らない）。
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
    manager: object,
    server: RestartableGateway,
    stop: threading.Event,
    restart_requested: threading.Event,
    *,
    auto_apply: bool = True,
    state: dict | None = None,
    notify=None,
    warmup_interval: float = 60.0,
    check_interval: float = 3600.0,
    drain_poll_interval: float = 30.0,
) -> None:
    """新しいリリースタグを検知し、（auto_apply なら）作業ツリーがクリーンな時に追従する常駐スレッド。

    旧 TUI が担っていた自動更新（clone 運用でリリースタグへ追従）をデーモン本体へ移したもの。
    安全側の 2 段構え —— ①**取得は稼働中に先に済ませる**（`git pull`＋`uv sync`。プロセスには
    触れず、この間も通常どおりリクエストを受ける）②**再起動は drain が通ったときだけ**行う。
    `manager.begin_drain()` が「処理中 0・在席 0」の確認と新規受付停止を**原子的に**行うので、
    確認と再起動の隙に生成が滑り込んで強制終了される余地が無い。busy なら何も止めずに保留し、
    空いた瞬間に再起動する。ネットワーク I/O・git は失敗しても握りつぶす（稼働は妨げない）。

    未検知のあいだは 1 時間おき、取得済みで再起動待ちのあいだは 30 秒おきに drain を再試行する。

    **チェック自体は auto_apply=false でも行う**（適用はしない）——Ollama と同じく
    「更新がある」ことをトレイの更新マークで見せるため。検知状態は `state`
    （server.update_state。/admin/status に載る）へ書き、`notify`（トレイへの通知線）に
    `update-available <ver>` / `update-ready <ver>` を 1 版につき 1 回だけ流す。
    """
    from . import update

    fetched = False  # ソースは新版へ追従済みで、あとは drain が通れば再起動するだけ
    notified: str | None = None  # この版は通知済み（毎時間チカチカ再通知しない）
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

    while not stop.wait(
        warmup_interval
        if first
        else (drain_poll_interval if fetched else check_interval)
    ):
        first = False
        if not fetched:
            try:
                st = update.check(timeout=3.0)
            except Exception as exc:  # noqa: BLE001 - 監視スレッドは落とさない
                print(f"Auto-update: check failed ({exc}).", file=sys.stderr)
                continue
            if state is not None:
                state.update(
                    {
                        "available": bool(st.available),
                        "current": st.current,
                        "latest": st.latest,
                        "reason": st.reason,
                    }
                )
            if not st.available:
                continue  # オフライン・最新
            if not (auto_apply and st.can_apply):
                # 自動適用しない（auto_update=false）／できない（dirty で WIP を守る等）。
                # 更新マークだけ出して、適用はユーザーの「今すぐ更新」（/admin/update）に任せる。
                _tell("update-available", st.latest)
                continue
            # 取得は稼働中に先に済ませる（プロセスには触れない。ここでは再起動しない）。
            try:
                ok, msg = update.apply_update()
            except Exception as exc:  # noqa: BLE001
                print(f"Auto-update: fetch skipped ({exc}).", file=sys.stderr)
                continue
            if not ok:
                print(f"Auto-update: not applied ({msg}).", file=sys.stderr)
                continue
            fetched = True
            if state is not None:
                state["fetched"] = True
            _tell("update-ready", st.latest)
            print(
                f"Auto-update: fetched ({msg}); will restart on new code when idle.",
                file=sys.stderr,
            )
        # ソース追従済み。accept を止めて処理中の接続が掃けた（quiesce 成功）ときだけ再起動する。
        # Listen ソケットは開いたままなので、この後に来た接続は 503 にも接続拒否にもならず
        # accept キューで待ち、execv 後の新イメージがソケットごと引き継いで処理する
        # （＝再起動の窓に投げられたリクエストを 1 つも落とさない）。
        # 在席セッションは見ない: 在席は「解放を早める」だけの存在で、更新を塞ぐ権限を
        # 持たせない（release を送れず落ちたエージェントの置き去りが残っても更新は進む）。
        if server.quiesce_for_restart():
            print(
                "Auto-update: idle; restarting the gateway on new code...",
                file=sys.stderr,
            )
            restart_requested.set()
            return
        # busy（受信中・生成中の接続あり）→ 何も止めずに保留（次周期で再試行）。
