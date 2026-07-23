"""更新エンドポイント（POST /admin/update）と /admin/status の update 欄のテスト。

トレイの「今すぐ更新して再起動」（Ollama の Restart to update 相当）の裏口。
実際の git/PyPI には触れない——update.check / apply_update をスタブする。
"""
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
        server.update_state = {"available": True, "latest": "9.9.9", "fetched": False}
        status, obj = _req(server.server_address[1], "GET", "/admin/status")
        assert status == 200
        assert obj["update"]["available"] is True
        assert obj["update"]["latest"] == "9.9.9"
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_update_now_when_up_to_date(monkeypatch):
    """新版が無ければ up-to-date を返して何もしない（再起動しない）。"""
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=False, can_apply=False, current="1.0", latest="1.0", reason="ok"))
    server, mgr = _start_bare_gateway()
    try:
        restarted = threading.Event()
        server.request_restart = restarted.set
        server.update_state = {"fetched": False}
        status, obj = _req(server.server_address[1], "POST", "/admin/update")
        assert status == 200 and obj["status"] == "up-to-date"
        assert not restarted.wait(1.0), "must not restart when up to date"
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_update_now_restarts_when_already_fetched():
    """自動更新が取得済み（fetched）なら、適用はせず再起動だけを要求する。"""
    server, mgr = _start_bare_gateway()
    try:
        restarted = threading.Event()
        server.request_restart = restarted.set
        server.update_state = {"fetched": True, "latest": "9.9.9"}
        status, obj = _req(server.server_address[1], "POST", "/admin/update")
        assert status == 200 and obj["status"] == "restarting"
        assert obj["latest"] == "9.9.9"
        assert restarted.wait(3.0), "restart must be requested after the response"
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_update_now_applies_then_restarts(monkeypatch):
    """未取得なら check → apply してから再起動を要求する（応答は restarting）。"""
    from local_llm_server import update as upd_mod

    applied = []
    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=True, can_apply=True, current="1.0", latest="2.0", reason="ok"))
    monkeypatch.setattr(upd_mod, "apply_update",
                        lambda: (applied.append(True) or (True, "pulled")))
    server, mgr = _start_bare_gateway()
    try:
        restarted = threading.Event()
        server.request_restart = restarted.set
        server.update_state = {"fetched": False, "latest": None}
        status, obj = _req(server.server_address[1], "POST", "/admin/update")
        assert status == 200 and obj["status"] == "restarting"
        assert applied == [True]
        assert server.update_state["fetched"] is True
        assert restarted.wait(3.0)
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_update_now_refuses_dirty_tree(monkeypatch):
    """適用できない（dirty tree 等）ときは 409 で理由を返し、再起動しない。"""
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=True, can_apply=False, current="1.0", latest="2.0",
        reason="working tree is dirty"))
    server, mgr = _start_bare_gateway()
    try:
        restarted = threading.Event()
        server.request_restart = restarted.set
        server.update_state = {"fetched": False}
        status, obj = _req(server.server_address[1], "POST", "/admin/update")
        assert status == 409
        assert not restarted.wait(0.8)
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()


def test_watcher_notifies_when_not_applying(monkeypatch):
    """auto_apply=false（auto_update=false 相当）でも検知・通知はする（適用はしない）。

    同じ版の再通知はしない（毎時間マークをチカチカさせない）。
    """
    from local_llm_server import update as upd_mod

    monkeypatch.setattr(upd_mod, "check", lambda timeout=3.0: types.SimpleNamespace(
        available=True, can_apply=True, current="1.0", latest="2.0", reason="ok"))
    monkeypatch.setattr(gw, "_UPDATE_WARMUP_INTERVAL", 0.01)
    monkeypatch.setattr(gw, "_UPDATE_CHECK_INTERVAL", 0.01)
    notes: list[str] = []
    stop = threading.Event()
    state: dict = {}
    mgr = ModelManager([])
    t = threading.Thread(
        target=gw._update_watcher,
        args=(mgr, stop, threading.Event()),
        kwargs={"auto_apply": False, "state": state, "notify": notes.append},
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
    mgr.shutdown()
