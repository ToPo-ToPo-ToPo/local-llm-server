"""モデル定義を同梱する repo（PLaMo 3 Translate など）の扱いと、起動失敗の診断。

mlx-lm 用のコード（ModelArgs）を同梱する repo を mlx-vlm に渡すと
``module 'custom_model' has no attribute 'ModelConfig'`` で落ち、ゲートウェイは
``server exited early (code 3)`` しか返さなかった。
"""

import json
import sys

import pytest

from local_llm_server import daemon as gw
from local_llm_server import model_catalog, server
from local_llm_server.backend_core import ServerConfig

_MLX_LM_CODE = "class ModelArgs(BaseModelArgs):\n    pass\n\nclass Model(nn.Module):\n    pass\n"
_MLX_VLM_CODE = "class ModelConfig(BaseModelConfig):\n    pass\n\nclass Model(nn.Module):\n    pass\n"


def _repo(cache, repo, *, code=None, model_file="modeling_custom.py"):
    """HF キャッシュの形（models--org--name/snapshots/<rev>/）で repo を作る。"""
    org, name = repo.split("/")
    snap = cache / f"models--{org}--{name}" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    cfg = {"architectures": ["Plamo3ForCausalLM"], "model_type": "plamo3"}
    if code is not None:
        cfg["model_file"] = model_file
        (snap / model_file).write_text(code, encoding="utf-8")
    (snap / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"\0" * 16)
    return snap


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    return root


def test_bundled_mlx_code_tells_mlx_lm_from_mlx_vlm(cache):
    assert model_catalog.bundled_mlx_code(str(_repo(cache, "a/lm", code=_MLX_LM_CODE))) == "mlx-lm"
    assert model_catalog.bundled_mlx_code(str(_repo(cache, "a/vlm", code=_MLX_VLM_CODE))) == "mlx-vlm"
    assert model_catalog.bundled_mlx_code(str(_repo(cache, "a/plain"))) is None


def test_bundled_mlx_code_stays_inside_snapshot(cache, tmp_path):
    # model_file が ../ でスナップショットの外を指していても読まない。
    (tmp_path / "outside.py").write_text(_MLX_LM_CODE, encoding="utf-8")
    snap = _repo(cache, "a/escape", code=_MLX_LM_CODE, model_file="../../../../../outside.py")
    assert model_catalog.bundled_mlx_code(str(snap)) is None


def test_resolve_backend_picks_mlx_for_mlx_lm_code(cache):
    _repo(cache, "pfnet/plamo-3-translate-8b-MLX-4bit", code=_MLX_LM_CODE)
    _repo(cache, "org/Other-MLX-4bit")
    assert model_catalog.resolve_backend("pfnet/plamo-3-translate-8b-MLX-4bit") == "mlx"
    assert model_catalog.resolve_backend("org/Other-MLX-4bit") == "mlx-vlm"
    # 未取得なら ID からの推論のまま（ここでは例外にしない。起動時の ensure_cached が案内する）
    assert model_catalog.resolve_backend("org/Missing-MLX-4bit") == "mlx-vlm"
    assert model_catalog.resolve_backend("unsloth/Foo-GGUF:Q4") == "llama-cpp"


def test_dynamic_register_uses_mlx_for_bundled_mlx_lm_code(cache):
    _repo(cache, "pfnet/plamo-3-translate-8b-MLX-4bit", code=_MLX_LM_CODE)
    mgr = gw.ModelManager([], dynamic=True, internal_base_port=9001, public_port=8799)
    mm = mgr._register_dynamic_locked("pfnet/plamo-3-translate-8b-MLX-4bit")
    assert mm.config.backend == "mlx"
    assert mm.config.draft_model is None  # MTP は mlx-vlm 専用
    assert mgr.backend_for("pfnet/plamo-3-translate-8b-MLX-4bit") == "mlx"


def test_build_mlx_adds_trust_remote_code_only_for_bundled_code(cache):
    _repo(cache, "a/lm", code=_MLX_LM_CODE)
    _repo(cache, "a/plain")
    cmd = server.build_command(ServerConfig(backend="mlx", model="a/lm", port=9001))
    assert cmd[:3] == [sys.executable, "-m", "mlx_lm.server"]
    assert cmd.count("--trust-remote-code") == 1
    # 利用者が extra_args で付けていれば二重にしない
    cmd = server.build_command(
        ServerConfig(backend="mlx", model="a/lm", port=9001, extra_args=["--trust-remote-code"]))
    assert cmd.count("--trust-remote-code") == 1
    plain = server.build_command(ServerConfig(backend="mlx", model="a/plain", port=9001))
    assert "--trust-remote-code" not in plain


def test_build_mlx_vlm_refuses_mlx_lm_code_before_launch(cache):
    _repo(cache, "a/lm", code=_MLX_LM_CODE)
    with pytest.raises(ValueError, match='backend = "mlx"'):
        server.build_command(ServerConfig(backend="mlx-vlm", model="a/lm", port=9001))


_PLAMO_LOG = """\
2026-11-16 15:22:51,548 - INFO - Loading model: pfnet/plamo-3-translate-8b-MLX-4bit
2026-11-16 15:22:51,563 - ERROR - Error loading model pfnet/plamo-3-translate-8b-MLX-4bit: \
module 'custom_model' has no attribute 'ModelConfig'
Traceback (most recent call last):
  File "mlx_vlm/utils.py", line 478, in load_model
    model_config = model_class.ModelConfig.from_dict(config)
AttributeError: module 'custom_model' has no attribute 'ModelConfig'
fastapi.exceptions.HTTPException: 400: Failed to load model: module 'custom_model' has no attribute 'ModelConfig'

ERROR:    Application startup failed. Exiting.
"""


def test_describe_early_exit_names_cause_and_log(tmp_path):
    msg = server.describe_early_exit(3, _PLAMO_LOG, backend="mlx-vlm", log_path="/x/server-9005.log")
    first = msg.splitlines()[0]
    assert first.startswith("server exited early (code 3): ")
    assert "mlx-lm 用" in first
    error_line = next(ln for ln in msg.splitlines() if ln.startswith("error: "))
    assert "AttributeError: module 'custom_model' has no attribute 'ModelConfig'" in error_line
    assert "Traceback" not in error_line
    assert "backend: mlx-vlm" in msg
    assert "log: /x/server-9005.log" in msg
    assert "Application startup failed" in msg  # ログの末尾


def test_describe_early_exit_signal_and_unknown():
    assert "メモリ不足" in server.describe_early_exit(-9, "", backend="mlx", log_path="l")
    assert "killed by signal 9" in server.describe_early_exit(-9, "", backend="mlx", log_path="l")
    msg = server.describe_early_exit(3, "", backend="mlx", log_path="l")
    assert "モデルのロード" in msg and "--- log tail ---" not in msg


def test_describe_early_exit_old_llama_cpp_names_pin():
    """古い llama.cpp が新しいアーキテクチャの GGUF で落ちたら、pin の上げ方を示す（b12211 の実ログ）。"""
    log = (
        "0.00.200.780 E llama_model_load: error loading model: unknown model architecture: 'inkling2_loop'\n"
        "0.00.200.973 E llama_model_load_from_file_impl: failed to load model\n"
    )
    msg = server.describe_early_exit(1, log, backend="llama-cpp", log_path="l")
    assert "[llama_cpp] pin" in msg and "inkling2_loop" in msg


def test_early_exit_reads_only_this_start(tmp_path, monkeypatch):
    # 同じポートのログは追記される。前回の起動のエラーを今回の原因として出さない。
    log = tmp_path / "server-9001.log"
    log.write_text("OLD: Insufficient Memory\n", encoding="utf-8")
    srv = server.LocalServer(ServerConfig(backend="mlx", model="a/b", port=9001), str(log))
    monkeypatch.setattr(server, "build_command", lambda cfg: [
        sys.executable, "-c",
        "import sys; print('ValueError: Model type plamo3 not supported.'); sys.exit(3)"])
    monkeypatch.setattr(server, "register_worker", lambda *a, **k: None)
    monkeypatch.setattr(server, "unregister_worker", lambda *a, **k: None)
    srv.start()
    try:
        with pytest.raises(RuntimeError) as info:
            srv.wait_until_ready(timeout=10, interval=0.05)
    finally:
        srv.stop(grace=0)
    msg = str(info.value)
    assert msg.startswith("server exited early (code 3): バックエンドがこのモデルのアーキテクチャに未対応")
    assert "ValueError: Model type plamo3 not supported." in msg
    assert "Insufficient Memory" not in msg
