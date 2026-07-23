"""メニューバーアイコン（tray.py）のテスト。

GUI（rumps）は起動しない——メニュー行の整形（純粋関数）、静的アイコンの方針
（Ollama 流: 常駐中の定期処理なし）、デーモン側の spawn 条件を検証する。
"""
from __future__ import annotations

import sys

from local_llm_server import tray as tray_mod
from local_llm_server.tray import format_rows, merge_update_info, parse_update_event


def test_icon_is_static_with_no_periodic_work():
    """アイコンは静的なテンプレート画像で、定期ポーリングを持たない。

    Ollama と同じ運用: 状態はメニューを開いた瞬間（menuWillOpen:）にだけ取得する。
    GUI は AppKit 直接（rumps は非推奨 API 依存で最新 macOS では表示されないため不使用）。
    """
    import inspect
    import os

    assert os.path.isfile(tray_mod._ICON_PATH), "同梱アイコンが無い（assets/tray-icon.png）"
    src = inspect.getsource(tray_mod)
    assert "import rumps" not in src  # 最新 macOS で壊れている rumps に戻したら落ちる
    assert "Timer" not in src  # 定期処理を再導入したらこのテストが落ちる
    assert "menuWillOpen_" in src
    assert "setTemplate_(True)" in src  # ライト/ダーク自動追従のテンプレート画像
    assert "status.button()" in src or ".button()" in src  # 現行 API（button 経由）を使う


def test_rows_show_url_and_loaded_models():
    rows = format_rows(
        {"models": [
            {"model": "a/b", "loaded": True, "inflight": 2},
            {"model": "c/d", "loaded": True, "inflight": 0},
            {"model": "e/f", "loaded": False},
        ]},
        "127.0.0.1", 8799,
    )
    assert rows[0] == "http://127.0.0.1:8799/v1"
    assert "a/b — 処理中 2" in rows
    assert "c/d — 待機" in rows
    assert all("e/f" not in r for r in rows)  # 未ロードは並べない


def test_rows_when_nothing_loaded():
    rows = format_rows(None, "127.0.0.1", 8799)
    assert rows[0] == "http://127.0.0.1:8799/v1"
    assert any("未ロード" in r for r in rows)


def test_parse_update_event():
    """デーモンからの通知行の解釈。未知の行は None（前方互換で読み飛ばす）。"""
    assert parse_update_event("update-ready 0.36.0\n") == ("update-ready", "0.36.0")
    assert parse_update_event("update-available 0.36.0") == ("update-available", "0.36.0")
    assert parse_update_event("something-else 1.0") is None
    assert parse_update_event("") is None
    assert parse_update_event("update-ready") is None  # 版なしは不正


def test_merge_update_info_prefers_fetched():
    """admin の fetched（取得済み）はパイプ通知より優先。available は kind 未設定時のみ補完。"""
    # パイプ通知だけ
    m = merge_update_info({"kind": "update-available", "latest": "0.36.0"}, None)
    assert m["kind"] == "update-available"
    # admin が fetched なら update-ready に昇格
    m = merge_update_info({"kind": "update-available", "latest": None},
                          {"update": {"fetched": True, "latest": "0.36.0"}})
    assert m == {"kind": "update-ready", "latest": "0.36.0"}
    # admin が available でも、通知済みの kind は上書きしない
    m = merge_update_info({"kind": "update-ready", "latest": "0.36.0"},
                          {"update": {"available": True, "latest": "0.36.0"}})
    assert m["kind"] == "update-ready"
    # どちらも無ければマークなし
    assert merge_update_info({"kind": None, "latest": None}, {"update": {}})["kind"] is None


def test_daemon_spawns_tray_only_when_enabled(monkeypatch):
    """spawn 条件: macOS かつ tray=true。false なら起動しない。"""
    import types

    from local_llm_server import daemon as daemon_mod

    calls = []
    monkeypatch.setattr(daemon_mod.subprocess, "Popen",
                        lambda cmd, **kw: calls.append((cmd, kw)) or None)
    cfg_on = types.SimpleNamespace(tray=True, host="127.0.0.1", port=8799)
    cfg_off = types.SimpleNamespace(tray=False, host="127.0.0.1", port=8799)

    daemon_mod._maybe_spawn_tray(cfg_off)
    assert calls == []
    daemon_mod._maybe_spawn_tray(cfg_on)
    if sys.platform == "darwin":
        assert len(calls) == 1
        cmd, _kw = calls[0]
        assert "local_llm_server.tray" in cmd
        assert "--port" in cmd and "8799" in cmd
    else:
        assert calls == []  # macOS 以外では出さない


def test_tray_config_default_and_override(tmp_path):
    """gateway.toml の tray は既定 true、false で無効化できる。"""
    from local_llm_server.daemon import load_gateway_config

    p = tmp_path / "gateway.toml"
    p.write_text('host = "127.0.0.1"\nport = 8799\n', encoding="utf-8")
    assert load_gateway_config(str(p)).tray is True
    p.write_text('host = "127.0.0.1"\nport = 8799\ntray = false\n', encoding="utf-8")
    assert load_gateway_config(str(p)).tray is False
