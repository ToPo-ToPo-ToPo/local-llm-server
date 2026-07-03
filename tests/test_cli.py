"""`local-llm-server` CLI のターミナル運用フラグ（--start/--stop/--status/--restart）。

運用の実体（プロセス探索・停止・状態取得）は server.py 側でテスト済みなので、ここでは
CLI が各フラグを正しい処理へ振り分けることと、既定（引数なし）がフォアグラウンド起動に
落ちることを、server.py の関数を差し替えて検証する。
"""
import pytest

from local_llm_server import cli


_MIN_GATEWAY = 'port = 8799\n[[models]]\nmodel = "org/A"\nbackend = "mlx"\n'


@pytest.fixture
def in_gateway_dir(tmp_path, monkeypatch):
    """`./gateway.toml` のあるディレクトリを CWD にする。"""
    (tmp_path / "gateway.toml").write_text(_MIN_GATEWAY, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_missing_gateway_toml_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # gateway.toml の無いディレクトリ
    with pytest.raises(SystemExit):
        cli.main(["--status"])


def test_default_runs_foreground_when_not_tty(in_gateway_dir, monkeypatch):
    # 非対話端末（パイプ/CI/裏起動）では TUI を出さずフォアグラウンド実行に落ちる。
    monkeypatch.setattr(cli, "_interactive_tty", lambda: False)
    called = {}
    monkeypatch.setattr(cli, "install_shutdown_handlers", lambda: called.update(sig=True))
    monkeypatch.setattr(cli, "run_gateway", lambda cfg: called.update(port=cfg.port) or 0)
    assert cli.main([]) == 0
    assert called == {"sig": True, "port": 8799}


def test_default_opens_tui_when_interactive(in_gateway_dir, monkeypatch):
    from local_llm_server import tui
    monkeypatch.setattr(cli, "_interactive_tty", lambda: True)
    seen = {}
    monkeypatch.setattr(tui, "run_tui", lambda cfg: seen.update(port=cfg.port) or 0)
    monkeypatch.setattr(cli, "run_gateway", lambda cfg: pytest.fail("must not run foreground"))
    assert cli.main([]) == 0
    assert seen["port"] == 8799


def test_headless_forces_foreground_even_on_tty(in_gateway_dir, monkeypatch):
    from local_llm_server import tui
    monkeypatch.setattr(cli, "_interactive_tty", lambda: True)  # 端末でも
    monkeypatch.setattr(tui, "run_tui", lambda cfg: pytest.fail("must not open TUI"))
    ran = {}
    monkeypatch.setattr(cli, "install_shutdown_handlers", lambda: None)
    monkeypatch.setattr(cli, "run_gateway", lambda cfg: ran.update(ok=True) or 0)
    assert cli.main(["--headless"]) == 0
    assert ran["ok"]


def test_status_dispatches(in_gateway_dir, monkeypatch):
    seen = {}

    def fake_status(host, port):
        seen["q"] = (host, port)
        return None  # 未起動扱い → _status_servers は 1 を返す

    monkeypatch.setattr(cli, "server_status", fake_status)
    monkeypatch.setattr(cli, "run_gateway", lambda cfg: pytest.fail("must not start"))
    assert cli.main(["--status"]) == 1
    assert seen["q"] == ("127.0.0.1", 8799)


def test_stop_dispatches(in_gateway_dir, monkeypatch):
    stopped = []
    monkeypatch.setattr(cli, "find_pids_on_port", lambda port: [1000 + port])
    # --stop はコマンドラインで「うちのプロセス」か確認してから止める（無関係を巻き添えにしない）。
    monkeypatch.setattr(cli, "pid_looks_like_ours", lambda pid: True)
    monkeypatch.setattr(cli, "stop_pid", lambda pid, **_: stopped.append(pid) or True)
    monkeypatch.setattr(cli, "run_gateway", lambda cfg: pytest.fail("must not start"))
    assert cli.main(["--stop"]) == 0
    # 公開ポート＋内部モデルポートの両方を止めにいく
    assert stopped  # 1つ以上停止した


def test_stop_skips_unrelated_processes(in_gateway_dir, monkeypatch):
    # ポートを LISTEN していても、うちのプロセスに見えなければ止めない。
    monkeypatch.setattr(cli, "find_pids_on_port", lambda port: [1000 + port])
    monkeypatch.setattr(cli, "pid_looks_like_ours", lambda pid: False)
    monkeypatch.setattr(cli, "stop_pid",
                        lambda pid, **_: pytest.fail("must not kill unrelated processes"))
    monkeypatch.setattr(cli, "run_gateway", lambda cfg: pytest.fail("must not start"))
    assert cli.main(["--stop"]) == 1  # 止めたものは無い


def test_start_dispatches_background(in_gateway_dir, monkeypatch):
    started = {}
    monkeypatch.setattr(cli, "server_status", lambda host, port: None)  # 未起動
    monkeypatch.setattr(cli, "start_gateway_background",
                        lambda cwd, host, port: started.setdefault("at", (host, port)) or 4242)
    monkeypatch.setattr(cli, "run_gateway", lambda cfg: pytest.fail("must not run foreground"))
    assert cli.main(["--start"]) == 0
    assert started["at"] == ("127.0.0.1", 8799)
