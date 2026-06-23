"""ensure_server / check_model_served のテスト（実サーバーは起動しない）。"""
from __future__ import annotations

import pytest

from local_llm_server import gateway
from local_llm_server.gateway import (
    ServerHandle,
    ServerNotRunningError,
    check_model_served,
    ensure_server,
)

MODEL = "mlx-community/Qwen3.6-27B-4bit"


class FakeServer:
    """LocalServer の差し替え。起動せず、構成と呼び出しを記録するだけ。"""

    last_config = None

    def __init__(self, config):
        FakeServer.last_config = config
        self.config = config
        self.log_path = "/tmp/fake.log"
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def wait_until_ready(self, timeout=120.0):
        pass

    def stop(self):
        self.stopped = True


# --- 相乗り（既存サーバーあり） -------------------------------------------
def test_ride_along_returns_handle_without_server(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: True)
    monkeypatch.setattr(gateway, "list_models", lambda url, *a, **k: [MODEL])

    handle = ensure_server(model=MODEL)
    assert isinstance(handle, ServerHandle)
    assert handle.rode_along is True
    assert handle.started is False
    assert handle.warnings == []
    handle.stop()  # 相乗り時の stop は無害


def test_ride_along_warns_on_single_model_mismatch(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: True)
    monkeypatch.setattr(gateway, "list_models", lambda url, *a, **k: ["some/other-model"])

    handle = ensure_server(model=MODEL)
    assert handle.rode_along is True
    assert len(handle.warnings) == 1
    assert "other-model" in handle.warnings[0]


def test_ride_along_warns_when_multimodel_catalog_missing(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: True)
    monkeypatch.setattr(gateway, "list_models", lambda url, *a, **k: ["a", "b", "c"])

    handle = ensure_server(model=MODEL)
    assert len(handle.warnings) == 1
    assert "3 models" in handle.warnings[0]


# --- 自動起動しない ---------------------------------------------------------
def test_not_ready_without_autostart_raises(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: False)
    with pytest.raises(ServerNotRunningError):
        ensure_server(model=MODEL, auto_start=False)


# --- 自動起動 ---------------------------------------------------------------
def test_autostart_starts_server_and_handle_stops_it(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: False)
    monkeypatch.setattr(gateway, "LocalServer", FakeServer)

    handle = ensure_server(model=MODEL, backend="mlx-vlm", register_atexit=False)
    assert handle.rode_along is False
    assert handle.started is True
    assert handle.server.started is True
    handle.stop()
    assert handle.server is None  # stop 後は手放す


def test_autostart_requires_model(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: False)
    monkeypatch.setattr(gateway, "LocalServer", FakeServer)
    with pytest.raises(ValueError):
        ensure_server(model=None)


def test_autostart_auto_draft_disabled_for_unsupported_model(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: False)
    monkeypatch.setattr(gateway, "LocalServer", FakeServer)

    ensure_server(
        model="some/unsupported-model",
        backend="mlx-vlm",
        draft_model="auto",
        register_atexit=False,
    )
    # MTP_DRAFTERS に無いモデルは draft_model="auto" を無効化して起動する。
    assert FakeServer.last_config.draft_model is None


def test_autostart_auto_draft_kept_for_supported_model(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: False)
    monkeypatch.setattr(gateway, "LocalServer", FakeServer)

    ensure_server(model=MODEL, backend="mlx-vlm", draft_model="auto", register_atexit=False)
    # 対応モデルは "auto" のまま LocalServer に渡す（build_command 側で解決）。
    assert FakeServer.last_config.draft_model == "auto"


def test_context_manager_stops_started_server(monkeypatch):
    monkeypatch.setattr(gateway, "is_ready", lambda url, **k: False)
    monkeypatch.setattr(gateway, "LocalServer", FakeServer)

    with ensure_server(model=MODEL, register_atexit=False) as handle:
        srv = handle.server
        assert srv.started is True
    assert srv.stopped is True


# --- check_model_served 単体 ------------------------------------------------
def test_check_model_served_no_model_no_warning(monkeypatch):
    assert check_model_served("http://x/v1", None) == []


def test_check_model_served_served_no_warning(monkeypatch):
    monkeypatch.setattr(gateway, "list_models", lambda url, *a, **k: [MODEL])
    assert check_model_served("http://x/v1", MODEL) == []


# --- GET /admin/status（GUI 監視用エンドポイント） --------------------------
def test_admin_status_reports_live_model_state():
    """/admin/status が常駐状態（loaded/inflight）＋運用方針を返す。"""
    import threading

    from local_llm_server.daemon import GatewayServer, ModelManager
    from local_llm_server.server import ServerConfig, gateway_admin_status

    cfgs = [
        ServerConfig(backend="mlx-vlm", model="org/A", host="127.0.0.1", port=9001),
        ServerConfig(backend="mlx", model="org/B", host="127.0.0.1", port=9002),
    ]
    mgr = ModelManager(cfgs, max_resident=1, load_timeout=300)
    # org/A をロード済み・処理中 2 件に見立てる（実サーバーは起動しない）。
    a = mgr._models["org/A"]
    a.server = object()
    a.ready = True
    a.inflight = 2
    srv = GatewayServer(
        ("127.0.0.1", 0), mgr, catalog=["org/A", "org/B"],
        default_model=None, max_resident=1, idle_timeout=1200, load_timeout=300,
    )
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        data = gateway_admin_status("127.0.0.1", port)
    finally:
        srv.shutdown()
        srv.server_close()

    assert data is not None
    assert data["object"] == "gateway.status"
    assert data["max_resident"] == 1
    assert data["idle_timeout"] == 1200
    assert data["uptime"] >= 0                 # 起動経過（秒）
    assert data["requests"] == 0               # acquire を通していないので 0
    by_model = {m["model"]: m for m in data["models"]}
    assert by_model["org/A"]["loaded"] is True
    assert by_model["org/A"]["inflight"] == 2
    assert by_model["org/A"]["requests"] == 0
    assert "idle_for" in by_model["org/A"]     # 処理中なので None
    assert by_model["org/B"]["loaded"] is False


def test_gateway_admin_status_none_when_down():
    """応答が無ければ None（GUI は server_status にフォールバックできる）。"""
    from local_llm_server.server import gateway_admin_status

    # 使われていないであろうポート。urlopen が失敗して None。
    assert gateway_admin_status("127.0.0.1", 6, timeout=0.5) is None
