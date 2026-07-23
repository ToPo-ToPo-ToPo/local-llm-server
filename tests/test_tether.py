"""繋留ラッパー（Phase 0b: local_llm_server.tether）のテスト。

「デーモンが死ねばワーカーも死ぬ」を、パイプの書き込み端を閉じる＝デーモンの死の
シミュレーションで検証する。POSIX 専用（Windows は 0a の起動時掃除が受け皿）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="tether is POSIX-only")


def _spawn_tethered(read_fd: int, child_argv: list[str]) -> subprocess.Popen:
    """LocalServer.start と同じ形で tether ラッパーを起動する。"""
    return subprocess.Popen(
        [sys.executable, "-m", "local_llm_server.tether",
         "--fd", str(read_fd), "--", *child_argv],
        pass_fds=(read_fd,),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _group_gone(pgid: int) -> bool:
    """プロセスグループにもう誰もいないか。"""
    try:
        os.killpg(pgid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def test_child_dies_when_pipe_closes():
    """書き込み端が閉じられる（＝デーモン死）と、実サーバーごとグループが消える。"""
    r, w = os.pipe()
    proc = _spawn_tethered(r, [sys.executable, "-c", "import time; time.sleep(300)"])
    os.close(r)  # 親側の読み取り端は不要（ラッパーが継承済み）
    try:
        time.sleep(0.5)  # ラッパーが子を起動して EOF 監視に入るまで少し待つ
        assert proc.poll() is None, "wrapper must stay alive while the daemon lives"
        os.close(w)  # ← デーモンの死（kill -9 相当）。OS がパイプを閉じる
        proc.wait(timeout=15)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if _group_gone(proc.pid):
                break
            time.sleep(0.1)
        assert _group_gone(proc.pid), "worker group must be gone after daemon death"
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), 15)


def test_wrapper_exits_with_child_code():
    """実サーバーが自然終了したら、ラッパーは同じ終了コードで exit する。

    ゲートウェイの健全性チェック（is_alive / wait）はラッパーの生死を見るので、
    「実サーバーが死んだのにラッパーが生きている」と検知が壊れる。
    """
    r, w = os.pipe()
    proc = _spawn_tethered(r, [sys.executable, "-c", "import sys; sys.exit(7)"])
    os.close(r)
    try:
        rc = proc.wait(timeout=15)
        assert rc == 7
    finally:
        os.close(w)


def test_wrapper_group_survives_until_child_exits():
    """書き込み端が開いている限り（＝デーモン稼働中）、勝手に落ちない。"""
    r, w = os.pipe()
    proc = _spawn_tethered(r, [sys.executable, "-c", "import time; time.sleep(2)"])
    os.close(r)
    try:
        time.sleep(0.8)
        assert proc.poll() is None
        rc = proc.wait(timeout=15)  # 子の自然終了（sleep 2 明け）を看取って exit 0
        assert rc == 0
    finally:
        os.close(w)
