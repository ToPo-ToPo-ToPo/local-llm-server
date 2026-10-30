"""Retire only unused generations after a successful activation."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from local_llm_server import runtime_retention as retention

ROOT = Path(__file__).resolve().parents[1]


def make_install(root, day, *, activated=True):
    directory = root / f".runtime/installs/202610{day:02d}-120000-000000"
    prefix = directory / "uv/tools/local-llm-server"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("home = shared Python")
    for name in ("python", "gw"):
        (prefix / "bin" / name).write_text(name)
    record = {"status": "activated" if activated else "verified", "activated": activated,
              "prefix": str(prefix), "python": str(prefix / "bin/python"),
              "gw": str(prefix / "bin/gw")}
    (directory / "install.json").write_text(json.dumps(record))
    return directory, prefix


@pytest.fixture
def generations(tmp_path, monkeypatch):
    root = tmp_path / "project"
    older, _ = make_install(root, 27)
    old, previous = make_install(root, 28)
    current_dir, current = make_install(root, 29)
    launcher = tmp_path / "gw"
    launcher.symlink_to(current / "bin/gw")
    monkeypatch.setattr(retention, "_references", lambda root: [])
    return root, older, old, current_dir, current, previous, launcher


def test_keeps_active_and_actual_predecessor_only(generations):
    root, older, old, current_dir, current, previous, launcher = generations
    models = root / "models/weights"
    models.parent.mkdir(); models.write_text("model")
    shared = root / ".runtime/python/python"
    shared.parent.mkdir(); shared.write_text("shared Python")
    audio = root / ".runtime/mlx-audio-venv/bin/python"
    audio.parent.mkdir(parents=True); audio.write_text("another role")
    # Internal package symlinks must not remove their external target.
    (older / "uv/tools/local-llm-server/model-link").symlink_to(models)
    result = retention.prune_installations(root, current, str(previous / "bin/gw"), launcher)
    assert result["removed"] == [str(older)]
    assert old.exists() and current_dir.exists()
    assert models.read_text() == "model" and shared.exists() and audio.exists()


@pytest.mark.parametrize("case", ["staged", "malformed", "foreign", "symlink", "custom-files", "referenced"])
def test_does_not_delete_unowned_or_used_environments(generations, monkeypatch, tmp_path, case):
    root, older, old, current_dir, current, previous, launcher = generations
    record = older / "install.json"
    if case == "staged":
        data = json.loads(record.read_text());data.update(status="verified", activated=False)
        record.write_text(json.dumps(data))
    elif case == "malformed":
        record.write_text("broken")
    elif case == "foreign":
        data = json.loads(record.read_text());data["prefix"] = str(tmp_path)
        record.write_text(json.dumps(data))
    elif case == "symlink":
        saved = tmp_path / "external"
        older.rename(saved);older.symlink_to(saved, target_is_directory=True)
    elif case == "custom-files":
        (older / "user-model").write_text("keep")
    else:
        monkeypatch.setattr(retention, "_references", lambda root: [str(older / "bin/python")])
    result = retention.prune_installations(root, current, str(previous / "bin/gw"), launcher)
    assert not result["removed"] and older.exists()


def test_incomplete_process_inspection_never_deletes(generations, monkeypatch):
    root, older, old, current_dir, current, previous, launcher = generations
    def unavailable(root):raise RuntimeError("cannot inspect")
    monkeypatch.setattr(retention, "_references", unavailable)
    with pytest.raises(RuntimeError, match="cannot inspect"):
        retention.prune_installations(root, current, str(previous / "bin/gw"), launcher)
    assert all(p.exists() for p in (older, old, current_dir))


def test_stops_if_launcher_changes(generations):
    root, older, old, current_dir, current, previous, launcher = generations
    launcher.unlink();launcher.symlink_to(previous / "bin/gw")
    with pytest.raises(RuntimeError, match="Launcher changed"):
        retention.prune_installations(root, current, str(previous / "bin/gw"), launcher)
    assert older.exists()


def test_missing_predecessor_retains_latest_successful_generation(generations):
    root, older, old, current_dir, current, previous, launcher = generations
    result = retention.prune_installations(root, current, None, launcher)
    assert result["removed"] == [str(older)] and old.exists()


def test_serializes_installers(tmp_path):
    with retention.installation_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another runtime installation"):
            with retention.installation_lock(tmp_path):
                pytest.fail("second installer must not enter")


def test_health_failure_retains_all_generations(installer, monkeypatch):
    module, project, old, launcher = installer
    def run(command, **kwargs):
        if "tool" in command and "install" in command:
            prefix = Path(kwargs["env"]["UV_TOOL_DIR"]) / "local-llm-server/bin"
            prefix.mkdir(parents=True)
            for name in ("gw", "python"):(prefix / name).write_text(name)
        return subprocess.CompletedProcess(command, 0, "", "")
    def fail(python):raise RuntimeError("health failed")
    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(module, "verify_activation", fail)
    monkeypatch.setattr(module, "prune_old_installations", lambda *a: pytest.fail("must not prune"))
    with pytest.raises(RuntimeError, match="health failed"):
        module.main(["--activate"])
    assert (old / "gw").exists()
    record = json.loads(next((project / ".runtime/installs").glob("*/install.json")).read_text())
    assert record["status"] == "failed" and "cleanup" not in record


@pytest.fixture
def installer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("retention_installer", ROOT / "scripts/install_runtime.py")
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    project = tmp_path / "source";(project / "local_llm_server").mkdir(parents=True)
    for name in ("migration_policy.py", "runtime_retention.py"):
        shutil.copy2(ROOT / "local_llm_server" / name, project / "local_llm_server" / name)
    home = tmp_path / "home";monkeypatch.setattr(module.Path, "home", lambda: home)
    monkeypatch.setattr(module, "ROOT", project)
    old = tmp_path / "old/bin";old.mkdir(parents=True)
    for name in ("python", "gw"):(old / name).write_text(name)
    launcher = home / ".local/bin/gw";launcher.parent.mkdir(parents=True);launcher.symlink_to(old / "gw")
    monkeypatch.setattr(module, "read_versions", lambda _: {"pip": "26.2.2", "certifi": "2026.10.23"})
    return module, project, old, launcher


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cleanup_runs_only_after_activation_and_health(installer, monkeypatch, cleanup_fails):
    module, project, old, launcher = installer
    events = []
    def run(command, **kwargs):
        if "tool" in command and "install" in command:
            prefix = Path(kwargs["env"]["UV_TOOL_DIR"]) / "local-llm-server/bin"
            prefix.mkdir(parents=True)
            for name in ("gw", "python"):(prefix / name).write_text(name)
        if "enable" in command:events.append("enabled")
        return subprocess.CompletedProcess(command, 0, "", "")
    def healthy(python):
        assert events == ["enabled"]
        events.append("healthy")
        return {"pid": 123}
    def cleanup(python, prefix, previous, target):
        assert events == ["enabled", "healthy"]
        assert previous == str(old / "gw")
        events.append("cleanup")
        if cleanup_fails:raise RuntimeError("inspection unavailable")
        return {"removed": []}
    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(module, "verify_activation", healthy)
    monkeypatch.setattr(module, "prune_old_installations", cleanup)
    module.main(["--activate"])
    record = json.loads(next((project / ".runtime/installs").glob("*/install.json")).read_text())
    assert record["status"] == "activated" and record["activated"]
    assert events == ["enabled", "healthy", "cleanup"]
    assert ("error" in record["cleanup"]) == cleanup_fails


def test_relative_predecessor_is_resolved_from_launcher(generations):
    import os
    root, older, old, current_dir, current, previous, launcher = generations
    relative = os.path.relpath(previous / "bin/gw", launcher.parent)
    result = retention.prune_installations(root, current, relative, launcher)
    assert result["removed"] == [str(older)] and old.exists()


@pytest.mark.parametrize("denied", [False, True])
def test_reference_scan_skips_privileged_processes_but_requires_user_visibility(tmp_path, monkeypatch, denied):
    import os
    import psutil
    from types import SimpleNamespace
    def unreadable():raise psutil.AccessDenied(123)
    protected = SimpleNamespace(info={"uids": SimpleNamespace(effective=os.getuid()+1)},
                                cmdline=unreadable, cwd=unreadable)
    own = SimpleNamespace(info={"uids": SimpleNamespace(effective=os.getuid())},
                          cmdline=unreadable if denied else lambda: ["python", "app"],
                          cwd=lambda: str(tmp_path))
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [protected, own])
    monkeypatch.setattr(retention.shutil, "which", lambda name: "/usr/sbin/lsof")
    monkeypatch.setattr(retention.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, "n/open/file\n", ""))
    if denied:
        with pytest.raises(psutil.AccessDenied):retention._references(tmp_path)
    else:
        assert "/open/file" in retention._references(tmp_path)
