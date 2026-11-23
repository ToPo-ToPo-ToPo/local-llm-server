"""`gw claude`（ゲートウェイ経由で Claude Code を起動する）の検証。claude 本体は起動しない。"""
import pytest

from local_llm_server import claude_code, cli

MODEL = "org/some-model-mlx-4bit"


# --- 純ロジック -------------------------------------------------------------
def test_split_model_arg_takes_model_and_passes_the_rest():
    assert claude_code.split_model_arg(["--model", MODEL, "-p", "hi"]) == (MODEL, ["-p", "hi"])
    assert claude_code.split_model_arg([f"--model={MODEL}", "--resume"]) == (MODEL, ["--resume"])
    assert claude_code.split_model_arg(["-p", "hi"]) == (None, ["-p", "hi"])
    # `--` 以降は claude のもの（--model が入っていても取らない）
    assert claude_code.split_model_arg(["--", "--model", "x"]) == (None, ["--", "--model", "x"])


@pytest.mark.parametrize("argv", [["--model"], ["--model="], ["--model", "  "]])
def test_split_model_arg_rejects_empty_model(argv):
    with pytest.raises(ValueError):
        claude_code.split_model_arg(argv)


def test_build_env_points_claude_code_at_the_gateway(tmp_path):
    base = {"PATH": "/bin", "ANTHROPIC_API_KEY": "sk-real", "ANTHROPIC_MODEL": "claude-x"}
    env = claude_code.build_env(base, base_url="http://127.0.0.1:8799", api_key=None,
                                model=MODEL, cdir=str(tmp_path))
    assert env["PATH"] == "/bin"
    assert "ANTHROPIC_API_KEY" not in env        # 本物のキーをゲートウェイへ送らない
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8799"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "local"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path)
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
        assert env[key] == MODEL                 # 裏の処理でも別のモデルを読み込ませない
    assert base["ANTHROPIC_API_KEY"] == "sk-real"  # 元の環境は変えない

    keyed = claude_code.build_env({}, base_url="u", api_key="secret", model=MODEL, cdir="d")
    assert keyed["ANTHROPIC_AUTH_TOKEN"] == "secret"


def test_saved_model_roundtrip(tmp_path):
    cdir = tmp_path / "claude-code"
    assert claude_code.read_saved_model(str(cdir)) is None
    claude_code.save_model(str(cdir), MODEL)
    assert claude_code.read_saved_model(str(cdir)) == MODEL


def test_launch_without_claude_installed(monkeypatch, capsys):
    monkeypatch.setattr(claude_code.shutil, "which", lambda name, path=None: None)
    assert claude_code.launch([], {"PATH": ""}) == 127
    assert "claude" in capsys.readouterr().err


# --- gw claude のディスパッチ ------------------------------------------------
@pytest.fixture
def gw(tmp_path, monkeypatch):
    """tmp の gateway.toml・稼働中のゲートウェイ・claude の起動（記録するだけ）を用意する。"""
    cfg = tmp_path / "gateway.toml"
    cfg.write_text("port = 8799\n", encoding="utf-8")
    monkeypatch.setattr(cli, "user_config_path", lambda: str(cfg))
    monkeypatch.setattr(cli, "read_gateway_runtime", lambda: None)
    monkeypatch.setattr(cli, "is_ready", lambda url, **k: True)
    launched = []
    monkeypatch.setattr(claude_code, "launch", lambda argv, env: launched.append((argv, env)) or 0)
    return cfg, launched


def test_gw_claude_launches_with_model_and_passes_args(gw, tmp_path):
    _, launched = gw
    # argparse を通さないので claude の -p / --resume もそのまま渡る
    assert cli.main(["claude", "--model", MODEL, "-p", "hi", "--resume"]) == 0
    argv, env = launched[0]
    assert argv == ["-p", "hi", "--resume"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8799"
    assert env["ANTHROPIC_MODEL"] == MODEL
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude-code")


def test_gw_claude_remembers_the_last_model(gw):
    _, launched = gw
    assert cli.main(["claude", "--model", MODEL]) == 0
    assert cli.main(["claude"]) == 0
    assert launched[1][1]["ANTHROPIC_MODEL"] == MODEL


def test_gw_claude_needs_a_model(gw, capsys):
    _, launched = gw
    assert cli.main(["claude"]) == 2
    assert "--model" in capsys.readouterr().err
    assert launched == []


def test_gw_claude_falls_back_to_default_model(gw):
    cfg, launched = gw
    cfg.write_text(f'port = 8799\ndefault_model = "{MODEL}"\n', encoding="utf-8")
    assert cli.main(["claude"]) == 0
    assert launched[0][1]["ANTHROPIC_MODEL"] == MODEL


def test_gw_claude_uses_api_key_from_config_even_with_runtime_record(gw, monkeypatch):
    cfg, launched = gw
    cfg.write_text('port = 8799\napi_key = "secret"\n', encoding="utf-8")
    # 稼働中デーモンの記録には api_key が無い → 設定ファイルから読む
    monkeypatch.setattr(cli, "read_gateway_runtime",
                        lambda: {"host": "0.0.0.0", "port": 8795, "pid": 1})
    assert cli.main(["claude", "--model", MODEL]) == 0
    env = launched[0][1]
    assert env["ANTHROPIC_AUTH_TOKEN"] == "secret"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8795"  # 全 IF 公開でも自分へは loopback


def test_gw_claude_refuses_when_gateway_is_down(gw, monkeypatch, capsys):
    _, launched = gw
    monkeypatch.setattr(cli, "is_ready", lambda url, **k: False)
    assert cli.main(["claude", "--model", MODEL]) == 1
    assert "gw start" in capsys.readouterr().err
    assert launched == []


def test_gw_help_lists_claude(capsys):
    assert cli.main(["help"]) == 0
    assert "claude" in capsys.readouterr().out
