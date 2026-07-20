"""起動口の検証: `gw`（cli.main）と裏の常駐ワーカー（__main__.main）。

運用は `gw` の CLI サブコマンドで行う（起動は `gw start` の 1 つだけ。引数なしはコマンド一覧）。
ここではデーモン側の `./gateway.toml` 解決（gw start が設定ディレクトリを cwd に spawn する前提）と、
ヘッドレス実行への振り分けを、重い依存（start_gateway_background / run_gateway）を差し替えて検証する。
"""
import pytest

from local_llm_server import cli
from local_llm_server import __main__ as worker


_MIN_GATEWAY = 'port = 8799\n[[models]]\nmodel = "org/A"\nbackend = "mlx"\n'


@pytest.fixture
def in_gateway_dir(tmp_path, monkeypatch):
    """`./gateway.toml` のあるディレクトリを CWD にする。"""
    (tmp_path / "gateway.toml").write_text(_MIN_GATEWAY, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_resolve_config_finds_cwd_toml(in_gateway_dir):
    assert cli.resolve_config() == str(in_gateway_dir / "gateway.toml")


def test_resolve_config_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # gateway.toml の無いディレクトリ
    assert cli.resolve_config() is None


def test_bare_gw_shows_help(tmp_path, monkeypatch, capsys):
    # 引数なし `gw` はコマンド一覧を表示する（Ollama 流。起動の入口は `gw start` の 1 つだけ）。
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "start_gateway_background",
                        lambda *a, **k: pytest.fail("bare gw must not start the daemon"))
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "start" in out and "stop" in out and "status" in out


def test_worker_runs_gateway_when_spawned_by_gw_start(in_gateway_dir, monkeypatch):
    # gw start の spawn マーク付きなら、シャットダウンハンドラを張ってゲートウェイ本体を回す。
    monkeypatch.setenv("LOCAL_LLM_GW_LAUNCHER", "cli")
    called = {}
    monkeypatch.setattr(worker, "install_shutdown_handlers", lambda: called.update(sig=True))
    monkeypatch.setattr(
        worker, "run_gateway",
        lambda cfg, config_path=None: called.update(port=cfg.port, cfg_path=config_path) or 0)
    assert worker.main() == 0
    # config_path を run_gateway に渡している（ホットリロード監視を有効化するため）。
    assert called["sig"] is True and called["port"] == 8799
    assert called["cfg_path"] and called["cfg_path"].endswith("gateway.toml")


def test_worker_refuses_direct_invocation(in_gateway_dir, monkeypatch, capsys):
    # spawn マーク無しの直接 `python -m local_llm_server` は拒否する（入口は gw start の 1 本）。
    monkeypatch.delenv("LOCAL_LLM_GW_LAUNCHER", raising=False)
    monkeypatch.setattr(worker, "run_gateway",
                        lambda *a, **k: pytest.fail("must not start"))
    assert worker.main() == 2
    assert "gw start" in capsys.readouterr().err


def test_worker_errors_without_gateway_toml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_LLM_GW_LAUNCHER", "cli")
    monkeypatch.setattr(worker, "run_gateway", lambda cfg: pytest.fail("must not start"))
    assert worker.main() == 2
    assert "gateway.toml" in capsys.readouterr().err
