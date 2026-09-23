"""mlx-audio（音声合成 TTS）隔離 venv プロビジョナのテスト。

実 pip install は使わず、venv 作成/導入/検証（create_venv/run/importable を差し替え）と
Apple Silicon ガードをユニット検証する。本体の環境に混ぜない（依存がぶつかる）ことが要点。
"""
from __future__ import annotations

import os

import pytest

from local_llm_server import _venv_backend
from local_llm_server import mlx_audio_provisioner as mp


def _ok(returncode=0, stderr=b""):
    class _P:
        pass
    p = _P()
    p.returncode, p.stdout, p.stderr = returncode, b"", stderr
    return p


@pytest.fixture
def apple_silicon(monkeypatch):
    monkeypatch.setattr(mp.provisioner, "detect_os", lambda: "macos")
    monkeypatch.setattr(_venv_backend, "_is_arm64", lambda: True)


def test_refuses_other_than_apple_silicon(monkeypatch):
    monkeypatch.setattr(mp.provisioner, "detect_os", lambda: "linux")
    with pytest.raises(mp.MlxAudioUnavailable):
        mp.ensure_mlx_audio()
    monkeypatch.setattr(mp.provisioner, "detect_os", lambda: "macos")
    monkeypatch.setattr(_venv_backend, "_is_arm64", lambda: False)   # Intel Mac
    with pytest.raises(mp.MlxAudioUnavailable):
        mp.ensure_mlx_audio()


def test_macos_is_allowed_without_nvidia_gpu(apple_silicon, monkeypatch):
    """vLLM/SGLang と違い、GPU（NVIDIA）判定も macOS 拒否もしない。"""
    monkeypatch.setattr(_venv_backend, "gpu_available", lambda: False)
    py = mp.ensure_mlx_audio(importable=lambda p, run: True)
    assert py == _venv_backend.sys.executable


def test_auto_creates_venv_and_installs_server_extra(tmp_path, monkeypatch, apple_silicon):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    created, installed = {}, []
    state = {"installed": False}

    def fake_create(venv_dir):
        created["dir"] = venv_dir
        os.makedirs(os.path.dirname(mp.venv_python(venv_dir)), exist_ok=True)

    def marking_run(cmd, capture_output=False, timeout=None):
        if "install" in cmd:
            installed.append(cmd)
            state["installed"] = True
        return _ok(0)

    py = mp.ensure_mlx_audio(create_venv=fake_create, run=marking_run,
                             importable=lambda p, run: state["installed"] and p != mp._venv_backend.sys.executable)
    assert created["dir"].endswith("mlx-audio-venv")
    # サーバーを立てるのに server extra が要る（tts だけだと uvicorn が無くて起動しない）
    assert any(mp.MLX_AUDIO_PACKAGE in c for c in installed)
    assert "server" in mp.MLX_AUDIO_PACKAGE and "tts" in mp.MLX_AUDIO_PACKAGE
    assert py == mp.venv_python(created["dir"])


def test_reuses_existing_venv(tmp_path, monkeypatch, apple_silicon):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    venv_dir = os.path.normpath(mp.mlx_audio_venv_dir())
    py = mp.venv_python(venv_dir)
    os.makedirs(os.path.dirname(py), exist_ok=True)
    open(py, "w").close()

    def boom(*a, **k):
        raise AssertionError("must not create venv when already installed")

    got = mp.ensure_mlx_audio(create_venv=boom, run=lambda *a, **k: _ok(0),
                              importable=lambda p, run: p != mp._venv_backend.sys.executable)
    assert got == py


def test_pip_failure_raises(tmp_path, monkeypatch, apple_silicon):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def fake_create(venv_dir):
        os.makedirs(os.path.dirname(mp.venv_python(venv_dir)), exist_ok=True)

    with pytest.raises(mp.MlxAudioUnavailable, match="pip install"):
        mp.ensure_mlx_audio(create_venv=fake_create,
                            run=lambda cmd, **k: _ok(1, b"no wheel") if "install" in cmd else _ok(0),
                            importable=lambda p, run: False)


def test_real_venv_creation_uses_symlinks(tmp_path, monkeypatch, apple_silicon):
    """venv.create の既定（コピー）だと、macOS では uv 管理の python 本体が起動できず pip の導入が
    落ちる（実測）。`python -m venv` と同じくリンクで作る。"""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    seen = {}

    def fake_venv_create(vdir, **kw):
        seen.update(kw)
        os.makedirs(os.path.dirname(mp.venv_python(vdir)), exist_ok=True)

    monkeypatch.setattr(_venv_backend.venv, "create", fake_venv_create)
    state = {"installed": False}
    mp.ensure_mlx_audio(run=lambda cmd, **k: state.__setitem__("installed", True) or _ok(0),
                        importable=lambda p, run: state["installed"] and p != _venv_backend.sys.executable)
    assert seen.get("with_pip") is True
    assert seen.get("symlinks") is (os.name != "nt")
