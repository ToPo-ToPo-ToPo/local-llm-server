"""起動口の検証: `gw`（tui.main）と裏の常駐ワーカー（__main__.main）。

CLI の運用フラグは廃止し、運用はすべて TUI 内で行う。ここでは 2 つの入口が
`./gateway.toml` を正しく解決し、それぞれ TUI 起動 / ヘッドレスのゲートウェイ実行へ
振り分けることを、重い依存（textual / run_gateway）を差し替えて検証する。
"""
import pytest

from local_llm_server import tui
from local_llm_server import __main__ as worker


_MIN_GATEWAY = 'port = 8799\n[[models]]\nmodel = "org/A"\nbackend = "mlx"\n'


@pytest.fixture
def in_gateway_dir(tmp_path, monkeypatch):
    """`./gateway.toml` のあるディレクトリを CWD にする。"""
    (tmp_path / "gateway.toml").write_text(_MIN_GATEWAY, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_resolve_config_finds_cwd_toml(in_gateway_dir):
    assert tui.resolve_config() == str(in_gateway_dir / "gateway.toml")


def test_resolve_config_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # gateway.toml の無いディレクトリ
    assert tui.resolve_config() is None


def test_gw_opens_tui(in_gateway_dir, monkeypatch):
    seen = {}
    monkeypatch.setattr(tui, "run_tui", lambda cfg: seen.update(port=cfg.port) or 0)
    assert tui.main([]) == 0
    assert seen["port"] == 8799


def test_gw_errors_without_gateway_toml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tui, "run_tui", lambda cfg: pytest.fail("must not open TUI"))
    assert tui.main([]) == 2
    assert "gateway.toml" in capsys.readouterr().err


def test_worker_runs_gateway_headless(in_gateway_dir, monkeypatch):
    # __main__ は TUI を出さず、シャットダウンハンドラを張ってゲートウェイ本体を回す。
    called = {}
    monkeypatch.setattr(worker, "install_shutdown_handlers", lambda: called.update(sig=True))
    monkeypatch.setattr(
        worker, "run_gateway",
        lambda cfg, config_path=None: called.update(port=cfg.port, cfg_path=config_path) or 0)
    assert worker.main() == 0
    # config_path を run_gateway に渡している（ホットリロード監視を有効化するため）。
    assert called["sig"] is True and called["port"] == 8799
    assert called["cfg_path"] and called["cfg_path"].endswith("gateway.toml")


def test_worker_errors_without_gateway_toml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker, "run_gateway", lambda cfg: pytest.fail("must not start"))
    assert worker.main() == 2
    assert "gateway.toml" in capsys.readouterr().err
