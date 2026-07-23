"""テスト共通フィクスチャ。

ワーカー台帳（workers_state_path）はマシン共通の固定パスなので、テストが
LocalServer.start/stop を呼ぶと実運用の台帳に書いてしまう。全テストで
テスト専用パスに差し替えて隔離する（実マシンで稼働中のゲートウェイを守る）。
"""
from __future__ import annotations

import pytest

from local_llm_server import server as srv_mod


@pytest.fixture(autouse=True)
def _isolate_workers_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(
        srv_mod, "workers_state_path",
        lambda: str(tmp_path / "workers-ledger.json"),
    )
