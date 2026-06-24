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
        "max_resident": 1, "idle_timeout": 1200, "uptime": 750, "requests": 5,
        "models": [
            {"model": "m/A", "backend": "mlx-vlm", "port": 9001,
             "loaded": True, "inflight": 3, "requests": 4, "idle_for": None},
            {"model": "m/B", "backend": "mlx", "port": 9002,
             "loaded": True, "inflight": 0, "requests": 1, "idle_for": 300},
        ],
    })
    v = gui.build_view("127.0.0.1", 8799, ["m/A", "m/B"])
    assert v.state == "ready" and v.loaded == 2
    assert "pid 4242" in v.summary
    assert v.metrics == "up 12m   ·   5 requests"           # 起動経過＋累計リクエスト
    # 処理中のモデル: バックエンド:内部ポート + in-flight + 累計
    assert "mlx-vlm:9001" in v.model_lines[0]
    assert "3 in-flight" in v.model_lines[0] and "4 req" in v.model_lines[0]
    # アイドルのロード済み: 自動アンロードまでの残り（1200-300=900s=15m）
    assert "unload in 15m" in v.model_lines[1]
    assert v.policy == "resident 2/1   idle-unload 20m"


def test_fmt_dur():
    assert gui._fmt_dur(45) == "45s"
    assert gui._fmt_dur(750) == "12m"
    assert gui._fmt_dur(3780) == "1h03m"
    assert gui._fmt_dur(-5) == "0s"


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


def test_build_view_down_allows_start(monkeypatch):
    monkeypatch.setattr(gui, "gateway_admin_status", lambda *a, **k: None)
    monkeypatch.setattr(gui, "server_status", lambda *a, **k: None)
    v = gui.build_view("127.0.0.1", 8799, ["m/A"])
    assert v.start_enabled is True and v.stop_enabled is False


def test_build_view_ready_disables_start(monkeypatch):
    monkeypatch.setattr(gui, "gateway_admin_status", lambda *a, **k: None)
    monkeypatch.setattr(gui, "server_status", lambda *a, **k: {
        "ready": True, "pids": [1], "log_path": None,
    })
    v = gui.build_view("127.0.0.1", 8799, ["m/A"])
    assert v.start_enabled is False and v.stop_enabled is True


# --- バックグラウンド常駐起動（Ollama 流） ----------------------------------
def test_start_gateway_background_skips_when_already_running(monkeypatch):
    monkeypatch.setattr(server, "find_pids_on_port", lambda port: [4242])
    # 既に LISTEN している → 多重起動せず既存 PID を返す。
    assert server.start_gateway_background("/tmp", "127.0.0.1", 8799) == 4242


def test_start_gateway_background_spawns_detached(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "find_pids_on_port", lambda port: [])
    monkeypatch.setattr(server, "gateway_log_path", lambda port: str(tmp_path / "gw.log"))
    readies = iter([False, True])  # 起動前→起動後
    monkeypatch.setattr(server, "is_ready", lambda url, *a, **k: next(readies))
    captured = {}

    class _Proc:
        pid = 9999

        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return _Proc()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    pid = server.start_gateway_background(str(tmp_path), "127.0.0.1", 8799, start_timeout=5)
    assert pid == 9999
    # --headless 必須: 裏起動は非 TTY なので TUI を出さずゲートウェイ本体を回す。
    assert captured["cmd"][1:] == ["-m", "local_llm_server.cli", "--headless"]
    assert captured["kw"]["cwd"] == str(tmp_path)
    import os as _os
    if _os.name == "nt":
        assert "creationflags" in captured["kw"]                   # Windows: DETACHED_PROCESS
    else:
        assert captured["kw"]["start_new_session"] is True         # POSIX: setsid で切り離し


# --- クリックして起動できるランチャ（アプリ）の生成 ------------------------
def test_install_macos_app_writes_bundle(tmp_path, monkeypatch):
    import os
    import stat

    monkeypatch.setattr(gui.sys, "platform", "darwin")
    dest = str(tmp_path / "App.app")
    out = gui.install_launcher("/proj dir", "/venv/bin/python", start_gateway=True, dest=dest)
    assert out == dest
    plist = (tmp_path / "App.app" / "Contents" / "Info.plist").read_text()
    assert gui.APP_NAME in plist
    assert "LSUIElement" not in plist                               # 既定は普通の Dock アプリ
    launch = (tmp_path / "App.app" / "Contents" / "MacOS" / "launch").read_text()
    assert "cd '/proj dir'" in launch                               # 作業ディレクトリ固定
    assert "-m local_llm_server.gui --start-gateway" in launch      # 1クリックで常駐
    mode = os.stat(tmp_path / "App.app" / "Contents" / "MacOS" / "launch").st_mode
    assert mode & stat.S_IXUSR                                       # 実行ビット


def test_install_macos_app_menubar_only(tmp_path):
    dest = str(tmp_path / "App.app")
    gui._install_macos_app("/proj", "/py", start_gateway=True, menubar_only=True, dest=dest)
    plist = (tmp_path / "App.app" / "Contents" / "Info.plist").read_text()
    assert "LSUIElement" in plist                                   # agent（メニューバー専用）


def test_install_macos_app_icon_referenced_when_built(tmp_path):
    pytest.importorskip("PIL")
    dest = str(tmp_path / "App.app")
    gui._install_macos_app("/proj", "/py", start_gateway=True, dest=dest)
    icns = tmp_path / "App.app" / "Contents" / "Resources" / "icon.icns"
    plist = (tmp_path / "App.app" / "Contents" / "Info.plist").read_text()
    # アイコン生成に成功していれば plist が参照する（環境差で失敗しても本体は壊れない）。
    if icns.exists():
        assert "CFBundleIconFile" in plist
    else:
        assert "CFBundleIconFile" not in plist


def test_install_macos_app_without_start_gateway(tmp_path):
    dest = str(tmp_path / "App.app")
    gui._install_macos_app("/proj", "/py", start_gateway=False, dest=dest)
    launch = (tmp_path / "App.app" / "Contents" / "MacOS" / "launch").read_text()
    assert "--start-gateway" not in launch


def test_uninstall_launcher_removes_bundle(tmp_path):
    dest = tmp_path / "App.app"
    (dest / "Contents").mkdir(parents=True)
    (dest / "Contents" / "Info.plist").write_text("x")
    removed = gui.uninstall_launcher(dest=str(dest))
    assert removed == [str(dest)]
    assert not dest.exists()


def test_uninstall_launcher_noop_when_absent(tmp_path):
    assert gui.uninstall_launcher(dest=str(tmp_path / "nope.app")) == []


def test_app_data_paths_darwin(monkeypatch):
    monkeypatch.setattr(gui.sys, "platform", "darwin")
    paths = gui._app_data_paths()
    assert any(p.endswith("Application Support/local-llm-server") for p in paths)
    assert any("com.local-llm-server.gui" in p for p in paths)   # bundle id 配下も掃除


def test_app_data_paths_empty_off_darwin(monkeypatch):
    monkeypatch.setattr(gui.sys, "platform", "linux")
    assert gui._app_data_paths() == []


def test_purge_never_targets_model_cache(monkeypatch):
    # --purge が消すのはランチャ・リポジトリ内ログ・macOS per-app のみ。共有モデル
    # キャッシュ（~/.cache/huggingface）は他ツールと共用なので絶対に対象に入れない。
    monkeypatch.setattr(gui.sys, "platform", "darwin")
    targets = [gui.project_cache_dir(), *gui._app_data_paths()]
    assert not any("huggingface" in t or "/.cache/" in t for t in targets)


def test_install_linux_desktop_writes_entry(tmp_path):
    dest = str(tmp_path / "app.desktop")
    gui._install_linux_desktop("/proj", "/py", start_gateway=True, dest=dest)
    body = (tmp_path / "app.desktop").read_text()
    assert body.startswith("[Desktop Entry]")
    assert "Terminal=false" in body and "local_llm_server.gui --start-gateway" in body


def test_install_windows_cmd_writes_script(tmp_path):
    dest = str(tmp_path / "app.cmd")
    gui._install_windows_cmd(r"C:\proj", "C:\\py\\python.exe", start_gateway=True, dest=dest)
    body = (tmp_path / "app.cmd").read_text()
    assert 'cd /d "C:\\proj"' in body and "-m local_llm_server.gui --start-gateway" in body


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
