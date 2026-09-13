"""通知専用の /admin/status と、廃止した更新エンドポイントのテスト。"""
from __future__ import annotations

import http.client
import json
import threading
import time
import types

from local_llm_server import daemon as gw
from local_llm_server.daemon import ModelManager


def _req(port, method, path, payload=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    body = json.dumps(payload or {})
    conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    conn.close()
    return resp.status, data


def _start_bare_gateway():
    mgr = ModelManager([])
    server = gw.GatewayServer(("127.0.0.1", 0), mgr, catalog=[])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, mgr


def test_admin_status_includes_update_state():
    server, mgr = _start_bare_gateway()
    try:
        server.update_state = {"available": True, "latest": "9.9.9"}
        status, obj = _req(server.server_address[1], "GET", "/admin/status")
        assert status == 200
        assert obj["update"]["available"] is True
        assert obj["update"]["latest"] == "9.9.9"
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_admin_update_endpoint_is_removed(monkeypatch):
    """HTTP 経由では更新できず、適用は `gw update` の一本だけ。"""
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(
        upd_mod,
        "apply_update",
        lambda: (_ for _ in ()).throw(AssertionError("must not apply over HTTP")),
    )
    server, mgr = _start_bare_gateway()
    try:
        status, obj = _req(server.server_address[1], "POST", "/admin/update")
        assert status == 404
        assert "gw update" in obj["error"]
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_refresh_update_state_updates_without_apply(monkeypatch):
    """refresh_update_state は check() の結果を state に反映するが、適用はしない。"""
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=True, current="0.36.1", latest="0.37.0", reason="ok",
        restart_required=False))
    monkeypatch.setattr(upd_mod, "running_source_version", lambda: "0.36.1")
    # apply_update が呼ばれたら失敗させる（確認だけのはず）。
    monkeypatch.setattr(upd_mod, "apply_update",
                        lambda: (_ for _ in ()).throw(AssertionError("must not apply")))
    state = {"available": False, "current": None, "latest": None, "reason": None}
    gw.refresh_update_state(state)
    assert state["available"] is True
    assert state["current"] == "0.36.1" and state["latest"] == "0.37.0"
    # 「走っているコードの版」と「要再起動」も公開する（editable 運用の穴の検知用）。
    assert state["running"] == "0.36.1" and state["restart_required"] is False


def test_refresh_update_state_exposes_restart_required(monkeypatch):
    """pull 済みでプロセスだけ古いとき、state に restart_required が立つ。"""
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=False, current="0.37.1", latest="0.37.1", reason="ok",
        restart_required=True))
    monkeypatch.setattr(upd_mod, "running_source_version", lambda: "0.37.0")
    state = {}
    gw.refresh_update_state(state)
    assert state["available"] is False          # 取ってくるものは無い
    assert state["restart_required"] is True    # でも再起動は要る
    assert state["running"] == "0.37.0" and state["current"] == "0.37.1"


def test_admin_status_triggers_ondemand_check(monkeypatch):
    """/admin/status GET が（リスタート無しで）オンデマンド確認を1本走らせる。"""
    calls = []
    monkeypatch.setattr(gw, "refresh_update_state", lambda state: calls.append(state))
    server, mgr = _start_bare_gateway()
    try:
        server.update_state = {"available": False, "current": "0.36.1",
                               "latest": None, "reason": None}
        server._last_update_check = 0.0
        server._update_check_inflight = False
        _req(server.server_address[1], "GET", "/admin/status")
        # 背景スレッドの完了を少し待つ
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not calls:
            time.sleep(0.02)
        assert calls, "オンデマンド確認が走らなかった"
        # 直後の 2 回目はスロットルで走らない（リモートを叩きすぎない）。
        calls.clear()
        _req(server.server_address[1], "GET", "/admin/status")
        time.sleep(0.2)
        assert calls == []
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_admin_status_can_wait_for_fresh_update_state(monkeypatch):
    """トレイ指定では確認結果を同じ応答へ載せ、二度の操作を不要にする。"""
    from local_llm_server.server import gateway_admin_status

    def _refresh(state):
        state.update({"available": True, "current": "0.38.12", "latest": "0.39.0",
                      "reason": "ok", "restart_required": False})

    monkeypatch.setattr(gw, "refresh_update_state", _refresh)
    server, mgr = _start_bare_gateway()
    try:
        server.update_state = {"available": False, "current": "0.38.12",
                               "latest": None, "reason": None}
        server._last_update_check = 0.0
        server._update_check_inflight = False
        obj = gateway_admin_status(
            "127.0.0.1", server.server_address[1], timeout=1.0,
            refresh_updates=True,
        )
        assert obj is not None
        assert obj["update"]["available"] is True
        assert obj["update"]["latest"] == "0.39.0"
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_watcher_notifies_when_not_applying(monkeypatch):
    """常駐ウォッチャーは検知・通知だけを行い、適用はしない。

    同じ版の再通知はしない（毎時間マークをチカチカさせない）。
    """
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=True, can_apply=True, current="1.0", latest="2.0", reason="ok",
        restart_required=False))
    monkeypatch.setattr(
        upd_mod,
        "apply_update",
        lambda: (_ for _ in ()).throw(AssertionError("watcher must not apply")),
    )
    monkeypatch.setattr(upd_mod, "running_source_version", lambda: "1.0")
    monkeypatch.setattr(gw, "_UPDATE_WARMUP_INTERVAL", 0.01)
    monkeypatch.setattr(gw, "_UPDATE_CHECK_INTERVAL", 0.01)
    notes: list[str] = []
    stop = threading.Event()
    state: dict = {}
    t = threading.Thread(
        target=gw._update_watcher,
        args=(stop,),
        kwargs={"state": state, "notify": notes.append},
        daemon=True,
    )
    t.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not notes:
        time.sleep(0.02)
    time.sleep(0.1)  # 追加の周期を数回回して重複通知が無いことを見る
    stop.set()
    t.join(timeout=3.0)
    assert notes == ["update-available 2.0"]  # 1 回だけ（重複なし）
    assert state["available"] is True and state["latest"] == "2.0"


def test_watcher_notifies_when_restart_will_activate_pulled_source(monkeypatch):
    """別経路で pull 済みなら、再起動可能なこともアイコンへ通知する。"""
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=False, can_apply=True, current="2.0", latest="2.0", reason="ok",
        restart_required=True))
    monkeypatch.setattr(upd_mod, "running_source_version", lambda: "1.0")
    monkeypatch.setattr(gw, "_UPDATE_WARMUP_INTERVAL", 0.01)
    monkeypatch.setattr(gw, "_UPDATE_CHECK_INTERVAL", 0.01)
    notes: list[str] = []
    stop = threading.Event()
    t = threading.Thread(
        target=gw._update_watcher,
        args=(stop,),
        kwargs={"state": {}, "notify": notes.append},
        daemon=True,
    )
    t.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not notes:
        time.sleep(0.02)
    stop.set()
    t.join(timeout=3.0)
    assert notes == ["update-ready 2.0"]
