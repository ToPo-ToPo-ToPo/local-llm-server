"""vLLM 隔離 venv プロビジョナのテスト。

実 pip install・実 GPU は使わず、venv 作成/導入/検証（create_venv/run/importable を
差し替え）と GPU/OS ガードをユニット検証する。
"""
from __future__ import annotations

import os

import pytest

from local_llm_server import vllm_provisioner as vp


def _ok(returncode=0, stdout=b"", stderr=b""):
    class _P:
        pass
    p = _P()
    p.returncode, p.stdout, p.stderr = returncode, stdout, stderr
    return p


def test_venv_python_path_per_os():
    assert vp.venv_python("/x/venv").endswith(os.path.join("bin", "python")) or \
        vp.venv_python("/x/venv").endswith(os.path.join("Scripts", "python.exe"))


def test_requires_gpu(monkeypatch):
    monkeypatch.setattr(vp, "gpu_available", lambda: False)
    monkeypatch.setattr(vp.provisioner, "detect_os", lambda: "linux")
    with pytest.raises(vp.VllmUnavailable):
        vp.ensure_vllm()


def test_refuses_macos(monkeypatch):
    monkeypatch.setattr(vp, "gpu_available", lambda: True)
    monkeypatch.setattr(vp.provisioner, "detect_os", lambda: "macos")
    with pytest.raises(vp.VllmUnavailable):
        vp.ensure_vllm()


def test_current_env_preferred_when_importable(monkeypatch):
    monkeypatch.setattr(vp, "gpu_available", lambda: True)
    monkeypatch.setattr(vp.provisioner, "detect_os", lambda: "linux")
    py = vp.ensure_vllm(importable=lambda p, run: True)
    assert py == vp.sys.executable


def test_auto_creates_venv_and_installs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(vp.provisioner, "detect_os", lambda: "linux")
    monkeypatch.setattr(vp, "gpu_available", lambda: True)
    created = {}
    installed = []

    def fake_create(venv_dir):
        created["dir"] = venv_dir
        os.makedirs(os.path.dirname(vp.venv_python(venv_dir)), exist_ok=True)

    def fake_run(cmd, capture_output=False, timeout=None):
        if "install" in cmd:
            installed.append(cmd)
        return _ok(0)

    # importable: 導入前 False → 導入後 True。
    state = {"installed": False}

    def fake_importable(py, run):
        return state["installed"]

    def marking_run(cmd, capture_output=False, timeout=None):
        if "install" in cmd and "vllm" in cmd:
            state["installed"] = True
            installed.append(cmd)
        return _ok(0)

    py = vp.ensure_vllm(create_venv=fake_create, run=marking_run,
                        importable=fake_importable)
    assert created["dir"].endswith("vllm-venv")
    assert any("vllm" in c for c in installed)
    assert py == vp.venv_python(created["dir"])


def test_auto_reuses_existing_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(vp.provisioner, "detect_os", lambda: "linux")
    monkeypatch.setattr(vp, "gpu_available", lambda: True)
    # venv の python が既に在り import 可 → 何も作らない。
    venv_dir = os.path.normpath(vp.vllm_venv_dir())
    py = vp.venv_python(venv_dir)
    os.makedirs(os.path.dirname(py), exist_ok=True)
    open(py, "w").close()

    def boom(*a, **k):
        raise AssertionError("must not create venv when already installed")

    got = vp.ensure_vllm(create_venv=boom, run=lambda *a, **k: _ok(0),
                         importable=lambda p, run: p != vp.sys.executable)
    assert got == py


def test_pip_install_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(vp.provisioner, "detect_os", lambda: "linux")
    monkeypatch.setattr(vp, "gpu_available", lambda: True)

    def fail_run(cmd, capture_output=False, timeout=None):
        if "vllm" in cmd:
            return _ok(returncode=1, stderr=b"no CUDA wheel")
        return _ok(0)

    with pytest.raises(vp.VllmUnavailable):
        vp.ensure_vllm(create_venv=lambda d: None,
                       run=fail_run, importable=lambda p, run: False)
