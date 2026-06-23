"""トレイ GUI のプレゼンテーション層と、クロスプラットフォームなポート停止のテスト。

pystray / pillow が無くても import できる範囲（build_view 等）を中心に検証する。
画像生成は pillow があるときだけ確認する。
"""
from __future__ import annotations

import pytest

from local_llm_server import gui, server


# --- build_view（状態 → 表示）------------------------------------------------
def test_build_view_error_state():
    v = gui.build_view("127.0.0.1", 8799, ["m/A"], config_error="boom")
    assert v.state == "error"
    assert "boom" in v.policy
    assert v.stop_enabled is False


def test_build_view_down_when_no_process(monkeypatch):
    monkeypatch.setattr(gui, "gateway_admin_status", lambda *a, **k: None)
    monkeypatch.setattr(gui, "server_status", lambda *a, **k: None)
    v = gui.build_view("127.0.0.1", 8799, ["m/A", "m/B"])
    assert v.state == "down"
    assert v.stop_enabled is False
    assert v.model_lines == ["m/A   —", "m/B   —"]


def test_build_view_ready_with_live_models(monkeypatch):
    monkeypatch.setattr(gui, "server_status", lambda *a, **k: {
        "ready": True, "pids": [4242], "log_path": None,
    })
    monkeypatch.setattr(gui, "gateway_admin_status", lambda *a, **k: {
        "max_resident": 1, "idle_timeout": 1200,
        "models": [
            {"model": "m/A", "loaded": True, "inflight": 3},
            {"model": "m/B", "loaded": False, "inflight": 0},
        ],
    })
    v = gui.build_view("127.0.0.1", 8799, ["m/A", "m/B"])
    assert v.state == "ready"
    assert v.loaded == 1
    assert v.stop_enabled is True
    assert "pid 4242" in v.summary
    assert v.model_lines[0] == "m/A   loaded (3 in-flight)"
    assert v.model_lines[1] == "m/B   idle"
    assert v.policy == "resident 1/1   idle-unload 20m"


def test_build_view_ready_without_admin_endpoint(monkeypatch):
    """旧ゲートウェイ（/admin/status 無し）でも server_status にフォールバックする。"""
    monkeypatch.setattr(gui, "gateway_admin_status", lambda *a, **k: None)
    monkeypatch.setattr(gui, "server_status", lambda *a, **k: {
        "ready": True, "pids": [7], "log_path": None,
    })
    v = gui.build_view("127.0.0.1", 8799, ["m/A"])
    assert v.state == "ready"
    assert v.loaded == 0
    assert v.model_lines == ["m/A   idle"]
    assert v.policy == ""


def test_build_view_starting_when_listening_not_ready(monkeypatch):
    monkeypatch.setattr(gui, "gateway_admin_status", lambda *a, **k: None)
    monkeypatch.setattr(gui, "server_status", lambda *a, **k: {
        "ready": False, "pids": [9], "log_path": None,
    })
    v = gui.build_view("127.0.0.1", 8799, ["m/A"])
    assert v.state == "starting"
    assert v.model_lines == ["m/A   —"]


def test_fmt_seconds():
    assert gui._fmt_seconds(0) == "off"
    assert gui._fmt_seconds(None) == "off"
    assert gui._fmt_seconds(1200) == "20m"
    assert gui._fmt_seconds(90) == "90s"


def test_make_image_returns_icon():
    pytest.importorskip("PIL")
    img = gui._make_image("ready", 2)
    assert img.size == (64, 64)
    assert img.mode == "RGBA"


# --- クロスプラットフォームなポート→PID 特定 --------------------------------
def test_find_pids_netstat_parses_listening_lines(monkeypatch):
    sample = (
        "\r\n"
        "Active Connections\r\n"
        "\r\n"
        "  Proto  Local Address          Foreign Address        State           PID\r\n"
        "  TCP    0.0.0.0:8799           0.0.0.0:0              LISTENING       1234\r\n"
        "  TCP    127.0.0.1:9001         0.0.0.0:0              LISTENING       5678\r\n"
        "  TCP    0.0.0.0:443            0.0.0.0:0              LISTENING       999\r\n"
        "  TCP    127.0.0.1:8799         203.0.113.5:51000     ESTABLISHED     4321\r\n"
        "  TCP    [::]:8799              [::]:0                LISTENING       1234\r\n"
    )

    class _R:
        stdout = sample

    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: _R())
    pids = server._find_pids_netstat(8799)
    assert pids == [1234]          # ESTABLISHED 行と別ポートは除外、重複もまとめる


def test_find_pids_on_port_dispatches_to_windows(monkeypatch):
    monkeypatch.setattr(server, "_POSIX", False)
    monkeypatch.setattr(server.os, "name", "nt")
    monkeypatch.setattr(server, "_find_pids_netstat", lambda port: [42])
    assert server.find_pids_on_port(1234) == [42]


def test_stop_pid_windows_uses_taskkill(monkeypatch):
    calls = {}

    class _R:
        returncode = 0

    def fake_run(cmd, *a, **k):
        calls["cmd"] = cmd
        return _R()

    monkeypatch.setattr(server.os, "name", "nt")
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    assert server.stop_pid(4321) is True
    assert calls["cmd"][:2] == ["taskkill", "/PID"]
    assert "/T" in calls["cmd"] and "/F" in calls["cmd"]
