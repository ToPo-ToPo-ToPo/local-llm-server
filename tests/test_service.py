"""OS サービス登録（Phase 0c: service.py と gw serve/enable/disable）のテスト。

launchctl / systemctl は叩かない——定義ファイルの内容（純粋関数）と CLI 配線だけを
検証する。実登録は環境依存なので、受け入れ基準（再ログイン後の稼働・kill -9 で子が
残らない）は手動確認手順（docs/operation.md）に委ねる。
"""
from __future__ import annotations

import plistlib

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
