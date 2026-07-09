"""ゲートウェイの単一起動ロック（GatewayLock）のテスト。

「1 マシンにつき 1 ゲートウェイ」を保証する OS レベルの排他ロックを検証する。
別ディレクトリ・別ポートから起動しても 2 個目を拒否できることが要点。
"""
from __future__ import annotations

import os

import pytest

from local_llm_server.server import (
    GatewayAlreadyRunning,
    GatewayLock,
    _read_lock_pid,
)


def test_second_acquire_is_refused(tmp_path):
    """同じロックファイルを 2 回取ろうとすると 2 回目は拒否される（PID 付き）。"""
    lock_path = str(tmp_path / "gw.lock")
    first = GatewayLock(lock_path).acquire()
    try:
        with pytest.raises(GatewayAlreadyRunning) as exc:
            GatewayLock(lock_path).acquire()
        # 保持者（このプロセス）の PID とロックパスがエラーに載る。
        assert exc.value.pid == os.getpid()
        assert exc.value.path == lock_path
    finally:
        first.release()


def test_release_allows_reacquire(tmp_path):
    """解放後は同じロックを取り直せる（正常終了→再起動が通る）。"""
    lock_path = str(tmp_path / "gw.lock")
    GatewayLock(lock_path).acquire().release()
    # 取り直せること（例外が出ない）。
    second = GatewayLock(lock_path).acquire()
    second.release()


def test_lock_records_holder_pid(tmp_path):
    """ロック取得中はファイルに保持者 PID が記録される。"""
    lock_path = str(tmp_path / "gw.lock")
    lock = GatewayLock(lock_path).acquire()
    try:
        assert _read_lock_pid(lock_path) == os.getpid()
    finally:
        lock.release()


def test_context_manager(tmp_path):
    """with 文で取得・解放できる。"""
    lock_path = str(tmp_path / "gw.lock")
    with GatewayLock(lock_path):
        with pytest.raises(GatewayAlreadyRunning):
            GatewayLock(lock_path).acquire()
    # 抜けたら取り直せる。
    GatewayLock(lock_path).acquire().release()
