"""起動時の孤児掃除（Phase 0a: ワーカー台帳と reap_orphan_workers）のテスト。

デーモンが kill -9 等で死んだあと、次回起動時に前回のワーカーを台帳から見つけて
回収できること・無関係なプロセスには手を出さないことを検証する。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from local_llm_server import gateway_runtime as runtime_mod
from local_llm_server import process_control
from local_llm_server.gateway_runtime import (
    _load_workers_unlocked,
    reap_orphan_workers,
    register_worker,
    unregister_worker,
)


@pytest.fixture
def workers_file(tmp_path, monkeypatch):
    """台帳の置き場をテスト専用に隔離する（実マシンの台帳を汚さない）。"""
    path = str(tmp_path / "workers.json")
    monkeypatch.setattr(runtime_mod, "workers_state_path", lambda: path)
    return path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_register_unregister_roundtrip(workers_file):
    """register で台帳に載り、unregister で消える（同一 PID の再登録は上書き）。"""
    register_worker(111, 9001, "m/a")
    register_worker(222, 9002, "m/b")
    register_worker(111, 9003, "m/a2")  # 再登録は置き換え
    entries = _load_workers_unlocked()
    assert {(e["pid"], e["port"]) for e in entries} == {(111, 9003), (222, 9002)}
    unregister_worker(111)
    assert [e["pid"] for e in _load_workers_unlocked()] == [222]


def test_reap_kills_our_orphan_and_clears_ledger(workers_file, monkeypatch):
    """台帳の「自分由来」プロセスを止め、台帳を空にする（startup reconciliation）。"""
    if os.name == "nt":
        pytest.skip("POSIX-only process test")
    # 引数に目印（local_llm_server）を含む生存プロセス＝孤児ワーカーのふり。
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)", "local_llm_server"],
        start_new_session=True,
    )
    try:
        register_worker(proc.pid, 9101, "orphan/model")
        monkeypatch.setattr(
            process_control, "pid_looks_like_ours", lambda pid: pid == proc.pid
        )
        monkeypatch.setattr(
            runtime_mod, "pid_looks_like_ours", process_control.pid_looks_like_ours
        )
        reaped = reap_orphan_workers()
        assert proc.pid in reaped
        proc.wait(timeout=10)  # 止められている（wait が返る）
        assert _load_workers_unlocked() == []  # 台帳は空に戻る
    finally:
        if proc.poll() is None:
            proc.kill()


def test_reap_spares_unrelated_process(workers_file):
    """目印の無い（無関係な）プロセスは、台帳に載っていても殺さない。"""
    if os.name == "nt":
        pytest.skip("POSIX-only process test")
    # sys.executable は使わない——このリポジトリの venv パス自体に "local-llm-server" が
    # 含まれ、「自分由来」の目印に誤マッチしてしまう（実運用の無関係プロセス相当は /bin/sleep）。
    proc = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        register_worker(proc.pid, 9102, "someone/else")
        reaped = reap_orphan_workers()
        assert reaped == []
        assert _alive(proc.pid), "unrelated process must not be killed"
        # 台帳は掃除後に空へ戻る（stale 記録を持ち越さない）。
        assert _load_workers_unlocked() == []
    finally:
        proc.kill()


def test_reap_ignores_dead_pids(workers_file):
    """既に死んでいる PID の記録は静かに捨てる（誤射も例外もなし）。"""
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    register_worker(dead.pid, 9103, "gone/model")
    assert reap_orphan_workers() == []
    assert _load_workers_unlocked() == []


def test_reap_preserves_owned_worker_when_stop_fails(workers_file, monkeypatch):
    """停止失敗時に台帳を消すと、次回起動で二度と回収できなくなる。"""
    runtime_mod._save_workers_unlocked(
        [
            {
                "pid": 444,
                "port": 9104,
                "model": "stuck/model",
                "create_time": 1.0,
                "cmdline": ["llama-server"],
            },
        ]
    )
    monkeypatch.setattr(runtime_mod, "pid_matches_record", lambda *_a: True)
    monkeypatch.setattr(runtime_mod, "pid_looks_like_ours", lambda *_a: True)
    monkeypatch.setattr(runtime_mod, "stop_pid", lambda *_a, **_k: False)
    monkeypatch.setattr(runtime_mod, "pid_is_alive", lambda *_a: True)
    assert reap_orphan_workers() == []
    assert [e["pid"] for e in _load_workers_unlocked()] == [444]


def test_process_marker_requires_executable_or_module_position(monkeypatch):
    import psutil

    class _Proc:
        def __init__(self, argv):
            self._argv = argv

        def cmdline(self):
            return self._argv

    argv = ["/usr/bin/python", "unrelated.py", "llama-server", "local-llm-server"]
    monkeypatch.setattr(psutil, "Process", lambda _pid: _Proc(argv))
    assert process_control.pid_looks_like_ours(1) is False
    assert process_control.pid_looks_like_gateway(1) is False

    argv[:] = ["/usr/bin/python", "-m", "local_llm_server", "__daemon__"]
    assert process_control.pid_looks_like_ours(1) is True
    assert process_control.pid_looks_like_gateway(1) is True

    argv[:] = ["/workspace/.venv/bin/python3", "-m", "local_llm_server"]
    assert process_control.pid_looks_like_gateway(1) is True

    argv[:] = ["/usr/bin/python", "-m", "local_llm_server", "unrelated"]
    assert process_control.pid_looks_like_gateway(1) is False

    argv[:] = [
        "/managed/venv/bin/python3",
        "/usr/local/bin/gw",
        "serve",
        "--managed",
    ]
    assert process_control.pid_looks_like_gateway(1) is True

    argv[:] = [
        "/usr/bin/python",
        "unrelated.py",
        "/usr/local/bin/gw",
        "serve",
        "--managed",
        "extra",
    ]
    assert process_control.pid_looks_like_gateway(1) is False

    argv[:] = ["/opt/llama-server", "--port", "9001"]
    assert process_control.pid_looks_like_ours(1) is True
    assert process_control.pid_looks_like_gateway(1) is False


def test_stop_process_tree_reports_unconfirmed_death(monkeypatch):
    class _Proc:
        pid = 8123

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("fake", timeout)

    signals = []
    monkeypatch.setattr(
        process_control,
        "signal_process_tree",
        lambda _proc, *, kill: signals.append(kill),
    )
    assert (
        process_control.stop_process_tree(_Proc(), grace=0.1, kill_timeout=0.1)
        is False
    )
    assert signals == [False, True]


def test_stop_process_tree_confirms_exit_after_forced_kill(monkeypatch):
    class _Proc:
        pid = 8124

        def __init__(self):
            self.waits = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fake", timeout)
            return -9

    signals = []
    monkeypatch.setattr(
        process_control,
        "signal_process_tree",
        lambda _proc, *, kill: signals.append(kill),
    )
    assert process_control.stop_process_tree(_Proc(), grace=0.1) is True
    assert signals == [False, True]


def test_stop_process_tree_uses_taskkill_tree_on_windows(monkeypatch):
    class _Proc:
        pid = 8125

        def __init__(self):
            self.exited = False

        def poll(self):
            return 0 if self.exited else None

        def wait(self, timeout=None):
            self.exited = True
            return 0

    proc = _Proc()
    called = []
    monkeypatch.setattr(process_control.os, "name", "nt")
    monkeypatch.setattr(
        process_control,
        "_stop_pid_windows",
        lambda pid: called.append(pid) or True,
    )
    assert process_control.stop_process_tree(proc, grace=10.0) is True
    assert called == [8125]
