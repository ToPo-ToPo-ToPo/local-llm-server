"""Security boundaries and transport policy, independent of the user's trust store."""
import importlib.metadata
from pathlib import Path
import ssl
import subprocess
import sys
import tomllib
import types

import pytest
from packaging.specifiers import SpecifierSet

from local_llm_server import tls_policy, update


@pytest.mark.parametrize("before,after", [
    ((3,11,17),(3,11,18)), ((3,12,15),(3,12,16)),
    ((3,13,17),(3,13,18)), ((3,14,9),(3,14,10)), ((3,15,1),(3,15,2)),
])
def test_security_release_boundaries_match_package_metadata(before, after):
    root = Path(__file__).resolve().parents[1]
    for project in (root, root / "experimental/strata_lite"):
        metadata = tomllib.loads((project/"pyproject.toml").read_text())
        requirement = SpecifierSet(metadata["project"]["requires-python"])
        assert ".".join(map(str,before)) not in requirement
        assert ".".join(map(str,after)) in requirement
    assert not tls_policy.python_is_secure(before)
    assert tls_policy.python_is_secure(after)


def test_xet_disabled_even_if_hub_was_imported(monkeypatch):
    constants = types.SimpleNamespace(HF_HUB_DISABLE_XET=False)
    monkeypatch.setitem(sys.modules,"huggingface_hub.constants",constants)
    monkeypatch.setenv("HF_HUB_DISABLE_XET","0")
    tls_policy.disable_xet()
    assert constants.HF_HUB_DISABLE_XET is True
    assert __import__("os").environ["HF_HUB_DISABLE_XET"] == "1"


def test_git_child_cannot_inherit_disabled_tls(monkeypatch):
    monkeypatch.setenv("GIT_SSL_NO_VERIFY","true")
    seen = {}
    monkeypatch.setattr(update.subprocess,"run",lambda *a,**kw:seen.update(kw))
    update._git(Path("."),"status")
    assert "GIT_SSL_NO_VERIFY" not in seen["env"]
    assert Path(seen["env"]["GIT_SSL_CAPATH"]).is_dir()
    assert Path(seen["env"]["GIT_SSL_CAINFO"]).is_file()
    # The parent process and shared Git settings are not modified.
    assert __import__("os").environ["GIT_SSL_NO_VERIFY"] == "true"


def test_ca_lookup_rejects_outdated_bundle(monkeypatch):
    monkeypatch.setattr(importlib.metadata,"version",lambda _:"2026.7.22")
    with pytest.raises(RuntimeError,match="certifi"):
        tls_policy.ca_file()


def test_environment_check_runs_in_selected_interpreter():
    assert tls_policy.runtime_compatible(sys.executable,subprocess.run,require_pip=True)
    assert not tls_policy.runtime_compatible("/does-not-exist/python",subprocess.run)


@pytest.mark.parametrize("old_importable", [True, False])
def test_legacy_backend_is_preserved_and_new_path_used(tmp_path,monkeypatch,old_importable):
    from local_llm_server import _venv_backend as vb
    monkeypatch.setattr(vb.provisioner,"detect_os",lambda:"linux")
    old=tmp_path/"backend"
    oldpy=Path(vb.venv_python(str(old)))
    oldpy.parent.mkdir(parents=True);oldpy.write_text("old environment")
    monkeypatch.setattr(tls_policy,"runtime_compatible",lambda py,run,**kw:py!=str(oldpy))
    installed=set()
    def create(path):
        py=Path(vb.venv_python(path));py.parent.mkdir(parents=True);py.touch();installed.add(str(py))
    def importable(py,run):return (old_importable and py==str(oldpy)) or py in installed
    def run(*a,**kw):return types.SimpleNamespace(returncode=0,stderr=b"")
    selected=vb.ensure_backend(package="pkg",import_name="pkg",venv_dir=str(old),
        human_name="test",unavailable=RuntimeError,gpu_check=lambda:True,
        importable=importable,create_venv=create,run=run)
    assert selected!=str(oldpy)
    assert oldpy.read_text()=="old environment"
    assert "-tls-" in selected


def test_ca_context_keeps_hostname_and_chain_verification():
    context=ssl.create_default_context(cafile=tls_policy.ca_file())
    assert context.check_hostname
    assert context.verify_mode==ssl.CERT_REQUIRED


def test_macos_backend_matches_available_python_wheels(monkeypatch):
    from local_llm_server import backend_core
    monkeypatch.setattr(backend_core.sys, "platform", "darwin")
    monkeypatch.setattr(backend_core.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backend_core.sys, "version_info", (3, 14, 10))
    assert backend_core.default_backend() == "mlx-vlm"
    monkeypatch.setattr(backend_core.sys, "version_info", (3, 15, 2))
    assert backend_core.default_backend() == "llama-cpp"
