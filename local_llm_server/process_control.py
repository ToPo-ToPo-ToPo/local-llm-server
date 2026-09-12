"""Cross-platform process inspection and termination primitives."""

from __future__ import annotations

import os
import signal
import subprocess
import time


POSIX = os.name == "posix"


def _process_group_is_alive(pgid: int | None) -> bool:
    if not POSIX or pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # killpg(..., 0) also reports a group containing only zombies as present.
    # Such processes cannot execute or hold ports/memory and will be reaped by
    # their parent; treating them as live makes every normal child stop wait for
    # the full timeout before its parent calls wait().
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "status"]):
            try:
                if os.getpgid(proc.pid) != pgid:
                    continue
                if proc.info.get("status") != psutil.STATUS_ZOMBIE:
                    return True
            except (ProcessLookupError, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (PermissionError, psutil.AccessDenied):
                return True
        return False
    except Exception:  # noqa: BLE001 - conservative fallback without psutil
        return True


def _wait_process_group_gone(pgid: int | None, timeout: float) -> bool:
    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        if not _process_group_is_alive(pgid):
            return True
        time.sleep(0.05)
    return not _process_group_is_alive(pgid)


def install_shutdown_handlers() -> None:
    """Translate graceful shutdown signals to ``KeyboardInterrupt``."""

    def _raise_keyboard_interrupt(signum, frame):  # noqa: ANN001,ARG001
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _raise_keyboard_interrupt)
        except (ValueError, OSError):
            pass


def signal_process_tree(proc: subprocess.Popen, *, kill: bool) -> None:
    """Signal a process and, on POSIX, its isolated process group."""
    if proc.poll() is not None:
        return
    if POSIX:
        sig = signal.SIGKILL if kill else signal.SIGTERM
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if kill:
        proc.kill()
    else:
        proc.terminate()


def stop_process_tree(
    proc: subprocess.Popen,
    *,
    grace: float = 10.0,
    kill_timeout: float = 5.0,
) -> bool:
    """Stop and reap a directly spawned process tree.

    Returning ``True`` means that the direct child has actually exited, not merely
    that a signal was delivered.  Callers must retain their ownership record when
    this returns ``False`` so a later reconciliation pass can retry the cleanup.
    """
    pgid: int | None = None
    if POSIX:
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if proc.poll() is not None and not _process_group_is_alive(pgid):
        # ``poll`` reaps a child that has already exited.
        return True

    if os.name == "nt":
        # Windows has no signal-based graceful tree shutdown equivalent here.
        # taskkill /T /F is required even when the direct launcher exits quickly,
        # otherwise its backend children can survive detached.
        tree_stopped = _stop_pid_windows(proc.pid)
        try:
            if proc.poll() is None:
                proc.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            return False
        return tree_stopped and proc.poll() is not None

    if grace > 0:
        try:
            signal_process_tree(proc, kill=False)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.wait(timeout=grace)
            if not _process_group_is_alive(pgid):
                return True
        except subprocess.TimeoutExpired:
            pass

    try:
        if pgid is not None:
            # The direct child may have exited after SIGTERM while a descendant
            # ignored it.  Keep the captured group id so that descendant is still
            # force-killed even though proc.poll() is no longer None.
            os.killpg(pgid, signal.SIGKILL)
        else:
            signal_process_tree(proc, kill=True)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        if proc.poll() is None:
            proc.wait(timeout=kill_timeout)
        return _wait_process_group_gone(pgid, kill_timeout)
    except subprocess.TimeoutExpired:
        return proc.poll() is not None and _wait_process_group_gone(pgid, 0.0)


def find_pids_on_port(port: int) -> list[int]:
    """Return PIDs listening on ``port`` on macOS, Linux, or Windows."""
    if POSIX:
        return _find_pids_lsof(port)
    if os.name == "nt":
        return _find_pids_netstat(port)
    return []


def _find_pids_lsof(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for token in result.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            pass
    return pids


def _find_pids_netstat(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local, state, pid = parts[1], parts[3], parts[4]
        if state.upper() != "LISTENING" or local.rsplit(":", 1)[-1] != str(port):
            continue
        try:
            parsed = int(pid)
        except ValueError:
            continue
        if parsed and parsed not in pids:
            pids.append(parsed)
    return pids


def pid_looks_like_ours(pid: int) -> bool:
    """Conservatively identify a process launched by this package."""
    try:
        import psutil

        argv = psutil.Process(pid).cmdline()
    except Exception:  # noqa: BLE001
        return False
    if not argv:
        return False
    if os.path.basename(argv[0]) == "llama-server":
        return True
    modules = {argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "-m"}
    return bool(
        modules.intersection(
            {
                "local_llm_server",
                "local_llm_server.tether",
                "local_llm_server.stt_server",
                "mlx_lm.server",
                "mlx_vlm.server",
                "vllm.entrypoints.openai.api_server",
                "sglang.launch_server",
            }
        )
    )


def pid_looks_like_gateway(pid: int) -> bool:
    """Conservatively identify the package's canonical gateway command."""
    try:
        import psutil

        argv = psutil.Process(pid).cmdline()
    except Exception:  # noqa: BLE001
        return False
    module_entry = any(
        argv[i:]
        in (["-m", "local_llm_server"], ["-m", "local_llm_server", "__daemon__"])
        for i in range(len(argv))
    )
    # launchd/systemd use the installed console entry point instead of ``-m``.
    # Match the complete remaining argv so an unrelated process that merely
    # mentions these words is never treated as ours.
    managed_service = any(
        os.path.basename(argv[i]) in {"gw", "local-llm-server"}
        and argv[i + 1 :] == ["serve", "--managed"]
        for i in range(len(argv))
    )
    return module_entry or managed_service


def process_fingerprint(pid: int) -> dict | None:
    """Return start time and argv so PID reuse can be detected."""
    try:
        import psutil

        proc = psutil.Process(pid)
        return {"create_time": proc.create_time(), "cmdline": proc.cmdline()}
    except Exception:  # noqa: BLE001
        return None


def pid_matches_record(pid: int, record: dict) -> bool:
    """Whether a live PID still matches a persisted ownership record."""
    expected_time = record.get("create_time")
    expected_cmd = record.get("cmdline")
    if not isinstance(expected_time, (int, float)) or not isinstance(
        expected_cmd, list
    ):
        return False
    current = process_fingerprint(pid)
    return bool(
        current is not None
        and abs(float(current["create_time"]) - float(expected_time)) < 0.01
        and current["cmdline"] == expected_cmd
    )


def pid_is_alive(pid: int) -> bool:
    """Check liveness without using destructive ``os.kill(pid, 0)`` on Windows."""
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except psutil.AccessDenied:
            return True
    except Exception:  # noqa: BLE001
        return True


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    """Stop a process tree on POSIX or Windows."""
    if os.name == "nt":
        return _stop_pid_windows(pid)
    if not POSIX:
        return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False

    def _alive() -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (not _alive() or not pid_is_alive(pid)) and not _process_group_is_alive(
            pgid
        ):
            return True
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    # A successfully delivered SIGKILL is not the same as confirmed death.  Keep
    # the ledger entry if the process remains alive so the next startup can retry.
    kill_deadline = time.monotonic() + min(max(timeout, 0.2), 5.0)
    while time.monotonic() < kill_deadline:
        if (not _alive() or not pid_is_alive(pid)) and not _process_group_is_alive(
            pgid
        ):
            return True
        time.sleep(0.1)
    return (not _alive() or not pid_is_alive(pid)) and not _process_group_is_alive(
        pgid
    )


def _stop_pid_windows(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    if result.returncode != 0 and pid_is_alive(pid):
        return False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not pid_is_alive(pid):
            return True
        time.sleep(0.1)
    return not pid_is_alive(pid)
