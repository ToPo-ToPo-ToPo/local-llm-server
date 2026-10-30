"""OS サービス登録（Phase 0c: service.py と gw serve/enable/disable）のテスト。

launchctl / systemctl は叩かない——定義ファイルの内容（純粋関数）と CLI 配線だけを
検証する。実登録は環境依存なので、受け入れ基準（再ログイン後の稼働・kill -9 で子が
残らない）は手動確認手順（docs/operation.md）に委ねる。
"""
from __future__ import annotations

import plistlib

import pytest

from local_llm_server import service


def test_launchd_plist_restarts_only_on_failure():
    """plist: 異常終了だけ復活（SuccessfulExit=false）・ログイン時起動・スロットリング付き。"""
    text = service.render_launchd_plist("/Users/x/.local/bin/gw")
    data = plistlib.loads(text.encode("utf-8"))
    assert data["Label"] == service.LAUNCHD_LABEL
    assert data["ProgramArguments"] == ["/Users/x/.local/bin/gw", "serve", "--managed"]
    assert data["RunAtLoad"] is True
    # gw stop（正常終了 0）では復活しない——手動停止の意思を上書きしないための要。
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] >= 10  # クラッシュループの頭打ち
    assert data["StandardOutPath"] == data["StandardErrorPath"]


def test_systemd_unit_restarts_only_on_failure():
    """unit: Restart=on-failure（正常 stop では復活しない）と起動回数の頭打ち。"""
    text = service.render_systemd_unit("/home/x/.local/bin/gw")
    assert "ExecStart=/home/x/.local/bin/gw serve --managed" in text
    assert "Restart=on-failure" in text
    assert "StartLimitBurst=" in text
    assert "WantedBy=default.target" in text


def test_cli_wires_serve_enable_disable():
    """gw のサブコマンドとして serve / enable / disable が登録されている。"""
    from local_llm_server.cli import _COMMANDS, build_parser

    for name in ("serve", "enable", "disable"):
        assert name in _COMMANDS
    args = build_parser().parse_args(["serve", "--managed"])
    assert args.cmd == "serve" and args.managed is True
    args = build_parser().parse_args(["serve"])
    assert args.managed is False


def test_managed_serve_exits_zero_when_already_running(monkeypatch, tmp_path):
    """--managed の serve は「既に稼働中」（rc 3）を成功（0）として退く。

    手動 `gw start` のデーモンが居るとき、launchd/systemd が失敗と誤認して
    再起動ループに入らないための仕様。手動 serve（--managed なし）は 3 のまま。
    """
    import types

    from local_llm_server import cli as cli_mod
    from local_llm_server import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "run_gateway", lambda cfg, config_path=None: 3)
    monkeypatch.setattr(cli_mod, "install_shutdown_handlers", lambda: None)
    gcfg = types.SimpleNamespace(_config_dir=str(tmp_path))
    managed = types.SimpleNamespace(managed=True)
    manual = types.SimpleNamespace(managed=False)
    assert cli_mod.cmd_serve(gcfg, managed) == 0
    assert cli_mod.cmd_serve(gcfg, manual) == 3



def test_uv_project_entry_is_registered_directly(monkeypatch, tmp_path):
    entry = tmp_path / ".venv/bin/gw"
    entry.parent.mkdir(parents=True);entry.write_text("project gw");entry.chmod(0o755)
    monkeypatch.setattr(service.sys, "argv", [str(entry), "enable"])
    monkeypatch.setattr(service.shutil, "which", lambda name: "/legacy/bin/gw")
    assert service.gw_executable() == str(entry)
    arguments = plistlib.loads(service.render_launchd_plist(service.gw_executable()).encode())["ProgramArguments"]
    assert arguments[0] == str(entry)


def test_linked_launcher_registers_the_project_entry(monkeypatch, tmp_path):
    """~/.local/bin/gw のリンクから gw enable しても、登録するのはリンク先の .venv/bin/gw。"""
    entry = tmp_path / ".venv/bin/gw"
    entry.parent.mkdir(parents=True);entry.write_text("project gw");entry.chmod(0o755)
    link = tmp_path / "bin/gw";link.parent.mkdir();link.symlink_to(entry)
    monkeypatch.setattr(service.sys, "argv", [str(link), "enable"])
    assert service.gw_executable() == str(entry.resolve())


def test_install_launcher_links_and_replaces_only_links(monkeypatch, tmp_path):
    """リンクは作成・張り替え・据え置き。リンクでないファイルは置き換えない。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    link = tmp_path / ".local/bin/gw"
    assert service.launcher_path() == str(link)
    assert service.install_launcher("/repo/.venv/bin/gw") == "created"
    assert str(link.readlink()) == "/repo/.venv/bin/gw"
    assert service.install_launcher("/repo/.venv/bin/gw") == "unchanged"
    assert service.install_launcher("/other/.venv/bin/gw") == "updated"
    assert str(link.readlink()) == "/other/.venv/bin/gw"
    assert [p.name for p in link.parent.iterdir()] == ["gw"]  # 一時リンクを残さない

    link.unlink();link.write_text("user script")
    with pytest.raises(RuntimeError):
        service.install_launcher("/repo/.venv/bin/gw")
    assert link.read_text() == "user script"


def test_launcher_on_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:~/.local/bin")
    assert service.launcher_on_path() is True
    monkeypatch.setenv("PATH", "/usr/bin")
    assert service.launcher_on_path() is False


def test_enable_places_launcher_and_keeps_going_on_failure(monkeypatch, tmp_path, capsys):
    """gw enable は登録後に ~/.local/bin/gw を置く。置けなくても enable は成功のまま。"""
    import types

    from local_llm_server import cli as cli_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / ".local/bin"))
    entry = tmp_path / "repo/.venv/bin/gw"
    entry.parent.mkdir(parents=True);entry.write_text("project gw");entry.chmod(0o755)
    monkeypatch.setattr(service.sys, "argv", [str(entry), "enable"])
    monkeypatch.setattr(service, "service_kind", lambda: "launchd")
    monkeypatch.setattr(service, "enable", lambda: None)
    monkeypatch.setattr(cli_mod, "_endpoint", lambda gcfg: ("127.0.0.1", 8799, [8799]))
    monkeypatch.setattr(cli_mod, "_collect_gateway_pids", lambda *a: [])
    monkeypatch.setattr(cli_mod, "is_ready", lambda url: True)
    monkeypatch.setattr(cli_mod, "gateway_admin_status", lambda host, port: None)
    monkeypatch.setattr(cli_mod, "render_status", lambda *a: "gateway: running")
    args = types.SimpleNamespace()

    assert cli_mod.cmd_enable(None, args) == 0
    link = tmp_path / ".local/bin/gw"
    assert str(link.readlink()) == str(entry)
    out = capsys.readouterr()
    assert str(link) in out.out and "PATH" not in out.err

    link.unlink();link.write_text("user script")
    assert cli_mod.cmd_enable(None, args) == 0
    assert link.read_text() == "user script"
    assert "置き換えません" in capsys.readouterr().err


def _register(monkeypatch, kind="launchd"):
    """定義ファイルを置いて「gw enable 済み」にする（パスは conftest でテスト専用）。"""
    import os

    monkeypatch.setattr(service, "service_kind", lambda: kind)
    path = service.launchd_plist_path() if kind == "launchd" else service.systemd_unit_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("registered")


def test_is_registered_follows_the_definition_file(monkeypatch):
    monkeypatch.setattr(service, "service_kind", lambda: "launchd")
    assert service.is_registered() is False
    _register(monkeypatch)
    assert service.is_registered() is True
    monkeypatch.setattr(service, "service_kind", lambda: None)
    assert service.is_registered() is False


@pytest.mark.parametrize("kind, expected", [
    ("launchd", ["launchctl", "kickstart", "-k"]),
    ("systemd", ["systemctl", "--user", "restart", service.SYSTEMD_UNIT]),
])
def test_start_asks_the_service_manager(monkeypatch, kind, expected):
    import subprocess

    calls = []
    monkeypatch.setattr(service, "service_kind", lambda: kind)
    monkeypatch.setattr(service, "_run", lambda cmd: calls.append(cmd)
                        or subprocess.CompletedProcess(cmd, 0, "", ""))
    service.start()
    assert calls[0][:len(expected)] == expected
    if kind == "launchd":
        assert calls[0][-1].endswith("/" + service.LAUNCHD_LABEL)

    monkeypatch.setattr(service, "_run", lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "no job"))
    with pytest.raises(RuntimeError, match="no job"):
        service.start()


def _start_env(monkeypatch, tmp_path, *, admin=None):
    """cmd_start を外部に触れずに回すための差し替え一式。起きた事を events に記録する。"""
    import types

    from local_llm_server import cli as cli_mod

    events = []
    state = {"up": admin is not None}
    monkeypatch.setattr(cli_mod, "_endpoint", lambda gcfg: ("127.0.0.1", 8799, [8799]))
    monkeypatch.setattr(cli_mod, "gateway_admin_status",
                        lambda host, port: {"pid": 7} if state["up"] else None)
    monkeypatch.setattr(cli_mod, "is_ready", lambda url, **k: state["up"])
    monkeypatch.setattr(cli_mod, "render_status", lambda *a: "gateway: running")
    monkeypatch.setattr(cli_mod, "start_gateway_background",
                        lambda *a, **k: events.append("background") or state.update(up=True) or 8)

    def fake_start():
        events.append("service")
        state["up"] = True
    monkeypatch.setattr(service, "start", fake_start)
    gcfg = types.SimpleNamespace(_config_dir=str(tmp_path))
    return cli_mod, gcfg, events, state


def test_gw_start_uses_the_service_when_registered(monkeypatch, tmp_path):
    """gw enable 済みなら、gw start（stop 後の再開・restart・update の再起動）もサービス経由。"""
    _register(monkeypatch)
    cli_mod, gcfg, events, _ = _start_env(monkeypatch, tmp_path)
    assert cli_mod.cmd_start(gcfg, None) == 0
    assert events == ["service"]


def test_gw_start_reuses_a_running_gateway_without_restarting(monkeypatch, tmp_path):
    _register(monkeypatch)
    cli_mod, gcfg, events, _ = _start_env(monkeypatch, tmp_path, admin={"pid": 7})
    assert cli_mod.cmd_start(gcfg, None) == 0
    assert events == []


def test_gw_start_without_registration_starts_in_background(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "service_kind", lambda: "launchd")
    cli_mod, gcfg, events, _ = _start_env(monkeypatch, tmp_path)
    assert cli_mod.cmd_start(gcfg, None) == 0
    assert events == ["background"]


def test_gw_start_falls_back_when_the_service_cannot_start(monkeypatch, tmp_path, capsys):
    _register(monkeypatch)
    cli_mod, gcfg, events, _ = _start_env(monkeypatch, tmp_path)

    def broken():
        raise RuntimeError("launchctl kickstart に失敗しました")
    monkeypatch.setattr(service, "start", broken)
    assert cli_mod.cmd_start(gcfg, None) == 0
    assert events == ["background"]
    assert "gw enable" in capsys.readouterr().err


def test_gw_restart_stops_then_starts_through_the_service(monkeypatch, tmp_path):
    """restart（update の再起動も同じ順）: 全プロセスを止めてから、サービスとして起動し直す。"""
    _register(monkeypatch)
    cli_mod, gcfg, events, state = _start_env(monkeypatch, tmp_path, admin={"pid": 7})
    monkeypatch.setattr(cli_mod, "_collect_gateway_pids", lambda *a: [7])
    monkeypatch.setattr(cli_mod, "_stop_pids",
                        lambda pids: events.append(("stop", pids)) or state.update(up=False))
    assert cli_mod.cmd_restart(gcfg, None) == 0
    assert events == [("stop", [7]), "service"]
