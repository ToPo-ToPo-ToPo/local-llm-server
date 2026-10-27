"""Dependency floors, preservation of old environments and launcher discovery."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from packaging.requirements import Requirement

from local_llm_server import _venv_backend, tls_policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,version", [
    ("filelock", "4.2.0"), ("pillow", "12.3.0"),
    ("setuptools", "84.0.0"), ("urllib3", "2.7.0"),
    ("pillow", "13.0.0rc1"),
])
def test_backend_refuses_vulnerable_or_prerelease_dependency(name, version):
    setup = (
        "import importlib.metadata as m,sys;sys.platform='darwin';"
        "original=m.version;"
        f"m.version=lambda n: {version!r} if n=={name!r} else original(n);"
    )
    result = subprocess.run([sys.executable, "-O", "-c", setup + tls_policy.RUNTIME_CHECK],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert name in result.stderr


def test_backend_accepts_fixed_dependencies():
    setup = (
        "import importlib.metadata as m,sys;sys.platform='darwin';"
        "original=m.version;"
        "versions={'filelock':'4.2.4','pillow':'13.0.0','setuptools':'84.2.1','urllib3':'2.8.0'};"
        "m.version=lambda n: versions[n] if n in versions else original(n);"
    )
    result = subprocess.run([sys.executable, "-O", "-c", setup + tls_policy.RUNTIME_CHECK],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_isolated_install_constraints_enforce_floors_without_forcing_linux_setuptools(tmp_path):
    path = Path(_venv_backend.write_constraints(str(tmp_path)))
    requirements = [Requirement(line) for line in path.read_text().splitlines()]
    for platform in ("darwin", "linux", "win32"):
        active = {r.name: r for r in requirements
                  if r.marker is None or r.marker.evaluate({"sys_platform": platform})}
        assert "84.0.0" not in active["setuptools"].specifier if platform == "darwin" else "setuptools" not in active
        for name, old in (("filelock", "4.2.0"), ("pillow", "12.3.0"), ("urllib3", "2.7.0")):
            assert old not in active[name].specifier
        assert "1.0" not in active["litellm"].specifier


def test_public_package_requirements_enforce_security_floors():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = {Requirement(s).name: Requirement(s) for s in project["project"]["dependencies"]}
    for name, old in (("filelock", "4.2.0"), ("pillow", "12.3.0"),
                      ("setuptools", "84.0.0"), ("urllib3", "2.7.0")):
        assert old not in requirements[name].specifier


def test_installer_finds_python_beside_long_path_shell_launcher(tmp_path):
    spec = importlib.util.spec_from_file_location("install_runtime", ROOT / "scripts/install_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bin_dir = tmp_path / ("long-environment-path-" * 8) / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.touch()
    launcher = bin_dir / "gw"
    launcher.write_text("#!/bin/sh\n'''exec' '" + str(python) + "' \"$0\" \"$@\"\n")
    symlink = tmp_path / "gw"
    symlink.symlink_to(launcher)
    assert module.launcher_python(symlink) == python
    python.unlink()
    assert module.launcher_python(symlink) is None
