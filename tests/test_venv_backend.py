"""隔離 venv バックエンドの共通ヘルパ（vLLM/SGLang が共有）の契約テスト。"""
from __future__ import annotations

import os

import pytest

from local_llm_server import _venv_backend as vb


class _Exc(RuntimeError):
    pass


def _ok(returncode=0, stderr=b""):
    class _P:
        pass
    p = _P()
    p.returncode, p.stdout, p.stderr = returncode, b"", stderr
    return p


def _base(**over):
    kw = dict(package="pkg", import_name="pkg", venv_dir="/x/pkg-venv",
              human_name="PKG", unavailable=_Exc, gpu_check=lambda: True)
    kw.update(over)
    return kw


def test_gpu_guard_uses_passed_gpu_check(monkeypatch):
    monkeypatch.setattr(vb.provisioner, "detect_os", lambda: "linux")
    with pytest.raises(_Exc):
        vb.ensure_backend(**_base(gpu_check=lambda: False))


def test_macos_refused(monkeypatch):
    monkeypatch.setattr(vb.provisioner, "detect_os", lambda: "macos")
    with pytest.raises(_Exc):
        vb.ensure_backend(**_base())


def test_system_mode(monkeypatch):
    monkeypatch.setattr(vb.provisioner, "detect_os", lambda: "linux")
    py = vb.ensure_backend(**_base(provision="system", importable=lambda p, run: True))
    assert py == vb.sys.executable
    with pytest.raises(_Exc):
        vb.ensure_backend(**_base(provision="system", importable=lambda p, run: False))


def test_auto_install_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(vb.provisioner, "detect_os", lambda: "linux")
    vdir = str(tmp_path / "pkg-venv")
    state = {"installed": False}

    def create(d):
        py = vb.venv_python(d)
        os.makedirs(os.path.dirname(py), exist_ok=True)
        open(py, "w").close()  # 実 venv 同様に python 実ファイルを作る（再利用判定に必要）

    def run(cmd, capture_output=False, timeout=None):
        if "install" in cmd and "pkg" in cmd:
            state["installed"] = True
        return _ok(0)

    py = vb.ensure_backend(**_base(venv_dir=vdir, create_venv=create, run=run,
                                   importable=lambda p, r: state["installed"]))
    assert py == vb.venv_python(os.path.normpath(vdir))

    # 2 回目: 導入済み → create を呼ばない。
    def boom(d):
        raise AssertionError("must not recreate venv")

    again = vb.ensure_backend(**_base(venv_dir=vdir, create_venv=boom,
                                      run=lambda *a, **k: _ok(0),
                                      importable=lambda p, r: True))
    assert again == py


def test_pip_failure_raises_given_exc(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(vb.provisioner, "detect_os", lambda: "linux")

    def run(cmd, capture_output=False, timeout=None):
        return _ok(returncode=1, stderr=b"boom") if "pkg" in cmd else _ok(0)

    with pytest.raises(_Exc):
        vb.ensure_backend(**_base(venv_dir=str(tmp_path / "v"),
                                  create_venv=lambda d: None, run=run,
                                  importable=lambda p, r: False))
