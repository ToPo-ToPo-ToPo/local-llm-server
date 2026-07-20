"""SGLang 隔離 venv プロビジョナのテスト（vllm_provisioner と同構造）。

実 pip install・実 GPU は使わず、venv 作成/導入/検証（create_venv/run/importable を
差し替え）と GPU/OS ガードをユニット検証する。
"""
from __future__ import annotations

import os

import pytest

from local_llm_server import sglang_provisioner as sp


def _ok(returncode=0, stdout=b"", stderr=b""):
    class _P:
        pass
    p = _P()
    p.returncode, p.stdout, p.stderr = returncode, stdout, stderr
    return p


def test_requires_gpu(monkeypatch):
    monkeypatch.setattr(sp, "gpu_available", lambda: False)
    monkeypatch.setattr(sp.provisioner, "detect_os", lambda: "linux")
    with pytest.raises(sp.SglangUnavailable):
        sp.ensure_sglang()


def test_refuses_macos(monkeypatch):
    monkeypatch.setattr(sp, "gpu_available", lambda: True)
    monkeypatch.setattr(sp.provisioner, "detect_os", lambda: "macos")
    with pytest.raises(sp.SglangUnavailable):
        sp.ensure_sglang()


def test_current_env_preferred_when_importable(monkeypatch):
    monkeypatch.setattr(sp, "gpu_available", lambda: True)
    monkeypatch.setattr(sp.provisioner, "detect_os", lambda: "linux")
    py = sp.ensure_sglang(importable=lambda p, run: True)
    assert py == sp.sys.executable


def test_auto_creates_venv_and_installs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sp.provisioner, "detect_os", lambda: "linux")
    monkeypatch.setattr(sp, "gpu_available", lambda: True)
    created = {}
    installed = []
    state = {"installed": False}

    def fake_create(venv_dir):
        created["dir"] = venv_dir
        os.makedirs(os.path.dirname(sp.venv_python(venv_dir)), exist_ok=True)

    def marking_run(cmd, capture_output=False, timeout=None):
        if "install" in cmd and "sglang" in cmd:
            state["installed"] = True
            installed.append(cmd)
        return _ok(0)

    py = sp.ensure_sglang(create_venv=fake_create, run=marking_run,
                          importable=lambda p, run: state["installed"])
    assert created["dir"].endswith("sglang-venv")
    assert any("sglang" in c for c in installed)
    assert py == sp.venv_python(created["dir"])


def test_auto_reuses_existing_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sp.provisioner, "detect_os", lambda: "linux")
    monkeypatch.setattr(sp, "gpu_available", lambda: True)
    venv_dir = os.path.normpath(sp.sglang_venv_dir())
    py = sp.venv_python(venv_dir)
    os.makedirs(os.path.dirname(py), exist_ok=True)
    open(py, "w").close()

    def boom(*a, **k):
        raise AssertionError("must not create venv when already installed")

    got = sp.ensure_sglang(create_venv=boom, run=lambda *a, **k: _ok(0),
                           importable=lambda p, run: p != sp.sys.executable)
    assert got == py


def test_pip_install_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sp.provisioner, "detect_os", lambda: "linux")
    monkeypatch.setattr(sp, "gpu_available", lambda: True)

    def fail_run(cmd, capture_output=False, timeout=None):
        if "sglang" in cmd:
            return _ok(returncode=1, stderr=b"no CUDA wheel")
        return _ok(0)

    with pytest.raises(sp.SglangUnavailable):
        sp.ensure_sglang(create_venv=lambda d: None,
                         run=fail_run, importable=lambda p, run: False)
