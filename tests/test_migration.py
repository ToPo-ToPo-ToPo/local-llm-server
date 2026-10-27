"""Safe old-environment migration and failure isolation."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import types

import pytest

from local_llm_server import cli, daemon, migration_policy as policy, tls_policy, update

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {"certifi": "2026.7.22", "pip": "26.2.1", "filelock": "3.32.5",
          "Pillow": "12.3.0", "setuptools": "84.0.0", "urllib3": "2.7.0",
          "msgpack": "1.2.2", "custom-backend": "1.4.0", "torch": "2.13.0+cu128"}


def test_updates_only_security_dependencies_and_keeps_optional_backend():
    pins, changes = policy.plan_migration(LEGACY, platform="darwin")
    assert set(changes) == set(policy.SECURITY_RELEASES)
    assert pins["custom-backend"] == LEGACY["custom-backend"]
    assert pins["torch"] == LEGACY["torch"]
    assert LEGACY["filelock"] == "3.32.5"
    assert "Pillow" not in pins and pins["pillow"] == "13.0.0"


def test_keeps_newer_versions_and_does_not_add_optional_packages():
    before = {"certifi": "2027.1.1", "pip": "27.0.0", "filelock": "4.4.0", "torch": "2.14.1"}
    pins, changes = policy.plan_migration(before, platform="darwin")
    assert pins == before and changes == {}
    assert "pillow" not in pins and "msgpack" not in pins


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_does_not_break_cuda_setuptools_cap(platform):
    pins, changes = policy.plan_migration({"setuptools": "80.10.2"}, platform=platform)
    assert pins["setuptools"] == "80.10.2" and "setuptools" not in changes


@pytest.mark.parametrize("version,below", [
    ("13.0", False), ("13.0.0rc1", True), ("13.0.0.dev1", True),
    ("13.0.0.post1", False), ("13.0.0.post1.dev1", False),
    ("13.1.0rc1", False), ("1!12.0.0", False), ("13.0.0+vendor.1", False),
])
def test_handles_normalized_versions_without_downgrading(version, below):
    assert policy.below_floor(version, (13, 0, 0)) is below


def test_rejects_ambiguous_or_unsafe_metadata():
    with pytest.raises(ValueError):
        policy.plan_migration({"a_b": "1.0", "a-b": "2.0"})
    with pytest.raises(ValueError):
        policy.plan_migration({"bad\npackage": "1.0"})
    with pytest.raises(ValueError):
        policy.plan_migration({"filelock": "4.2.4\n-r elsewhere"})


def test_refuses_to_replace_direct_url_dependency_with_an_index_release(monkeypatch):
    dist = types.SimpleNamespace(metadata={"Name": "custom-backend"}, version="1.0",
                                 read_text=lambda _: json.dumps({"dir_info": {"editable": True}}))
    monkeypatch.setattr(policy.importlib.metadata, "distributions", lambda: [dist])
    with pytest.raises(ValueError, match="custom-backend.*direct URL"):
        policy.installed_versions()


@pytest.mark.parametrize("name,version", [("pip", "26.2.2"), ("certifi", "2026.10.19")])
def test_accepts_locally_staged_bootstrap_wheels(monkeypatch, name, version):
    source = {"url": f"file:///temporary/{name}-{version}-py3-none-any.whl", "archive_info": {}}
    dist = types.SimpleNamespace(metadata={"Name": name}, version=version,
                                 read_text=lambda _: json.dumps(source))
    monkeypatch.setattr(policy.importlib.metadata, "distributions", lambda: [dist])
    versions = policy.installed_versions()
    pins, changes = policy.plan_migration(versions)
    assert pins[name] == version and name not in changes


def test_security_migration_floors_match_runtime_checks():
    for name, minimum in tls_policy.DEPENDENCY_FLOORS.items():
        assert policy.SECURITY_RELEASES[name][0] == minimum
    assert policy.SECURITY_RELEASES["certifi"][0] == tls_policy.CERTIFI_MIN
    assert policy.SECURITY_RELEASES["pip"][0] == tls_policy.PIP_MIN


@pytest.fixture
def installer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("migration_installer", ROOT/"scripts/install_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path/"project"
    (project/"local_llm_server").mkdir(parents=True)
    shutil.copy2(ROOT/"local_llm_server/migration_policy.py", project/"local_llm_server/migration_policy.py")
    home = tmp_path/"home"
    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setattr(module.Path, "home", lambda: home)
    old = tmp_path/"old/bin"
    old.mkdir(parents=True)
    (old/"python").write_text("old Python")
    (old/"gw").write_text("old launcher")
    launcher = home/".local/bin/gw"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(old/"gw")
    pins, _ = policy.plan_migration(LEGACY)
    monkeypatch.setattr(module, "read_versions", lambda py: LEGACY.copy() if py == old/"python" else pins.copy())
    return module, project, old, launcher


def test_installer_dry_run_does_not_create_environment_or_switch_launcher(installer, monkeypatch, capsys):
    module, project, old, launcher = installer
    monkeypatch.setattr(module, "run", lambda *a, **kw: pytest.fail("dry run must not install"))
    module.main(["--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["security_changes"]["filelock"]["after"] == "4.2.4"
    assert not (project/".runtime").exists()
    assert launcher.resolve() == old/"gw"


@pytest.mark.parametrize("failure", [None, "install", "pins", "check", "enable"])
def test_installer_preserves_old_runtime_and_verifies_before_activation(installer, monkeypatch, failure):
    module, project, old, launcher = installer
    calls = []
    original_reader = module.read_versions
    if failure == "pins":
        def read(py):
            versions = original_reader(py)
            if py != old/"python":
                versions["custom-backend"] = "2.0.0"
            return versions
        monkeypatch.setattr(module, "read_versions", read)
    def run(command, **kwargs):
        calls.append(command)
        if "tool" in command and "install" in command:
            if failure == "install":
                raise subprocess.CalledProcessError(1, command)
            prefix = Path(kwargs["env"]["UV_TOOL_DIR"])/"local-llm-server/bin"
            prefix.mkdir(parents=True)
            (prefix/"gw").write_text("new launcher")
            (prefix/"python").write_text("new Python")
            assert "--with-requirements" in command
            requirements = Path(command[command.index("--with-requirements")+1]).read_text()
            assert "custom-backend==1.4.0" in requirements
            assert "filelock==4.2.4" in requirements
        if failure == "check" and "check" in command:
            raise subprocess.CalledProcessError(1, command)
        if failure == "enable" and "enable" in command:
            raise subprocess.CalledProcessError(1, command)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module, "run", run)
    if failure:
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            module.main(["--activate"])
    else:
        module.main(["--activate"])
    records = list((project/".runtime/installs").glob("*/install.json"))
    record = json.loads(records[0].read_text())
    assert (old/"python").read_text() == "old Python"
    assert (old/"gw").read_text() == "old launcher"
    if failure:
        assert launcher.resolve() == old/"gw"
        assert record["status"] == "failed" and not record["activated"]
    else:
        assert launcher.resolve() == Path(record["gw"])
        assert record["status"] == "activated"


def test_tool_refresh_requires_staged_migration_even_if_fingerprint_matches(tmp_path, monkeypatch):
    env = tmp_path/"tool"
    env.mkdir()
    (tmp_path/"pyproject.toml").write_text("[project]")
    (tmp_path/"uv.lock").write_text("lock")
    (env/update._DEPS_FINGERPRINT_NAME).write_text(update._deps_fingerprint(tmp_path))
    monkeypatch.setattr(update, "tool_env_root", lambda: env)
    monkeypatch.setattr(policy, "installed_versions", lambda: LEGACY.copy())
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **kw: pytest.fail("must not alter old environment"))
    ok, message = update.refresh_tool_env(tmp_path)
    assert not ok and "make install" in message


@pytest.mark.parametrize("available", [True, False])
def test_cli_does_not_stop_running_gateway_after_failed_dependency_migration(monkeypatch, available, tmp_path):
    state = types.SimpleNamespace(available=available, current="0.38.25", latest="0.38.25",
                                  can_apply=True, reason="ok")
    monkeypatch.setattr(update, "check", lambda: state)
    monkeypatch.setattr(update, "apply_update", lambda: (True, "source updated"))
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "refresh_tool_env", lambda root: (False, "migration needed"))
    monkeypatch.setattr(cli, "_endpoint", lambda cfg: ("127.0.0.1", 8799, [8799]))
    monkeypatch.setattr(cli, "gateway_admin_status", lambda *a: {"update": {"restart_required": True}})
    monkeypatch.setattr(cli, "_stop_pids", lambda *a: pytest.fail("running gateway must remain alive"))
    monkeypatch.setattr(cli, "cmd_start", lambda *a: pytest.fail("must not restart"))
    assert cli.cmd_update(object(), object()) == 1


def test_daemon_does_not_reexec_after_failed_dependency_migration(monkeypatch):
    events = []
    class Lock:
        def acquire(self): return self
        def release(self): events.append("released")
    monkeypatch.setattr(daemon, "GatewayLock", Lock)
    monkeypatch.setattr(daemon, "reap_orphan_workers", lambda: [])
    monkeypatch.setattr(daemon, "_run_gateway_locked", lambda *a, **kw: daemon._RESTART_CODE)
    monkeypatch.setattr(update, "refresh_tool_env", lambda root: (False, "migration needed"))
    monkeypatch.setattr(update, "reexec_daemon", lambda: pytest.fail("must not reexec"))
    assert daemon.run_gateway(object()) == 1
    assert events == ["released"]
