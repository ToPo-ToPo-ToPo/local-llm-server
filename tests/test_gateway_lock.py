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


# --- ランタイム記録（gw status / gw stop を任意ディレクトリから叩くための接続先） ---
def test_runtime_record_roundtrip_and_stale(tmp_path, monkeypatch):
    """記録を書けば読め、正常終了で消える。死んだ PID の記録は stale として None。"""
    from local_llm_server import server

    monkeypatch.setattr(server.tempfile, "gettempdir", lambda: str(tmp_path))
    # 生きている PID（自分）で書けば読める。
    server.write_gateway_runtime("127.0.0.1", 8799, os.getpid(), str(tmp_path), "now")
    rec = server.read_gateway_runtime()
    assert rec and rec["host"] == "127.0.0.1" and rec["port"] == 8799
    # 使われていない（＝死んだ）PID の記録は None（クラッシュで残った記録を掴まない）。
    dead = _find_free_pid()
    server.write_gateway_runtime("127.0.0.1", 8799, dead, str(tmp_path), "now")
    assert server.read_gateway_runtime() is None
    # 正常終了時の clear で消える。
    server.write_gateway_runtime("127.0.0.1", 8799, os.getpid(), str(tmp_path), "now")
    server.clear_gateway_runtime()
    assert server.read_gateway_runtime() is None


def _find_free_pid() -> int:
    """存在しない PID を1つ返す（生存確認テスト用。cross-platform に psutil で判定）。"""
    import psutil

    for pid in range(999_000, 1, -1):
        if not psutil.pid_exists(pid):
            return pid
    return 999_999
