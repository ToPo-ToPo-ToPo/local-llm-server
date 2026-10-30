"""Conservative retention of project-owned runtime installations."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


@contextmanager
def installation_lock(root: Path):
    """Serialize installers, including activation and retirement."""
    import fcntl

    directory = root / ".runtime"
    directory.mkdir(exist_ok=True)
    if directory.resolve() != directory:
        raise RuntimeError("Refusing a symlinked runtime directory")
    fd = os.open(directory / "installer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another runtime installation is in progress") from error
        yield
    finally:
        os.close(fd)


def _owned_install(root: Path, directory: Path) -> dict | None:
    """Recognize only successful, unmodified installer layouts."""
    base = root / ".runtime/installs"
    if (base.resolve() != base or directory.parent != base
            or directory.is_symlink() or not directory.is_dir()
            or not re.fullmatch(r"\d{8}-\d{6}-\d{6}", directory.name)):
        return None
    prefix = directory / "uv/tools/local-llm-server"
    if any(p.is_symlink() for p in (directory / "uv", prefix.parent, prefix)):
        return None
    try:
        if not {p.name for p in directory.iterdir()} <= {"uv", "bin", "constraints.txt", "install.json"}:
            return None
        if any(p.is_dir() and p.name != "local-llm-server" for p in prefix.parent.iterdir()):
            return None
        record = json.loads((directory / "install.json").read_text())
        if not isinstance(record, dict):
            return None
        if (record.get("status") != "activated" or record.get("activated") is not True
                or record.get("prefix") != str(prefix)
                or record.get("python") != str(prefix / "bin/python")
                or record.get("gw") != str(prefix / "bin/gw")
                or not (prefix / "pyvenv.cfg").is_file()):
            return None
        return record
    except (OSError, ValueError):
        return None


def _references(root: Path) -> list[str]:
    """Observe processes, open/mapped files, and conventional launch links.

    Incomplete inspection raises: callers keep environments instead of guessing.
    """
    import psutil

    refs = []
    for process in psutil.process_iter(["uids"]):
        uids = process.info.get("uids")
        if uids is None or uids.effective != os.getuid():
            continue
        try:
            refs.extend(process.cmdline())
            refs.append(process.cwd())
        except psutil.NoSuchProcess:
            continue
    command = shutil.which("lsof")
    if command is None and Path("/usr/sbin/lsof").is_file():
        command = "/usr/sbin/lsof"
    if command is None:
        raise RuntimeError("lsof unavailable; keeping old runtimes")
    result = subprocess.run([command, "-nP", "-a", "-u", str(os.getuid()), "-Fn"],
                            capture_output=True, text=True, timeout=15, check=False)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError("Open-file inspection incomplete; keeping old runtimes")
    refs.extend(line[1:] for line in result.stdout.splitlines() if line.startswith("n"))
    for directory in (Path.home() / ".local/bin", Path("/usr/local/bin"), root):
        if directory.exists():
            refs.extend(str(p.resolve()) for p in directory.iterdir() if p.is_symlink())
    return refs


def prune_installations(root: Path, current: Path, previous: str | None,
                        launcher: Path) -> dict:
    """Keep the activated runtime and its actual predecessor, not another role.

    Caller must hold installation_lock and have verified the new gateway's health.
    Staged/failed/foreign layouts and referenced environments are never removed.
    """
    result: dict = {"removed": [], "kept": [], "skipped": {}}
    current_dir = current.parents[2]
    if _owned_install(root, current_dir) is None:
        raise RuntimeError("Unrecognized active installation; cleanup refused")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Safe directory removal unavailable; cleanup refused")
    previous_path = ((launcher.parent / previous).resolve() if previous else None)
    candidates = sorted(current_dir.parent.iterdir(), reverse=True)
    successful = [p for p in candidates if _owned_install(root, p) is not None]
    if previous_path is None or not previous_path.is_file():
        # A missing launcher must not cause the last usable predecessor to vanish.
        previous_dir = next((p for p in successful if p.name < current_dir.name), None)
        if previous_dir is not None:
            previous_path = previous_dir / "uv/tools/local-llm-server/bin/gw"
    refs = _references(root)
    for directory in candidates:
        if directory == current_dir or (previous_path is not None and previous_path.is_relative_to(directory)):
            result["kept"].append(str(directory))
            continue
        if directory not in successful or directory.name >= current_dir.name:
            result["skipped"][str(directory)] = "not an older successful managed runtime"
            continue
        if any(str(directory) in ref for ref in refs):
            result["skipped"][str(directory)] = "referenced by a process, open file, or launcher"
            continue
        if not launcher.is_symlink() or launcher.resolve() != current / "bin/gw":
            raise RuntimeError("Launcher changed; cleanup stopped")
        if _owned_install(root, directory) is None:
            result["skipped"][str(directory)] = "installation changed during cleanup"
            continue
        try:
            shutil.rmtree(directory)
            result["removed"].append(str(directory))
        except OSError as error:
            result["skipped"][str(directory)] = str(error)
    return result
