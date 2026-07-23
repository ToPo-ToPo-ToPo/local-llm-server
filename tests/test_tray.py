"""メニューバーアイコン（tray.py）のテスト。

GUI（rumps）は起動しない——メニュー行の整形（純粋関数）、静的アイコンの方針
（Ollama 流: 常駐中の定期処理なし）、デーモン側の spawn 条件を検証する。
"""
from __future__ import annotations

import json
import sys

from local_llm_server import tray as tray_mod
from local_llm_server.tray import (
    format_rows,
    merge_update_info,
    parse_update_event,
    update_menu_item,
)


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
    # メニューは即座に開く: menuWillOpen で同期取得（ブロック）に戻したら落ちる。
    # 取得は裏スレッド（_refresh_cache）でキャッシュ更新のみ。
    assert "_refresh_cache" in src
    import re
    open_body = re.search(r"def menuWillOpen_.*?(?=\n        def )", src, re.S).group(0)
    assert "gateway_admin_status" not in open_body  # 開く動作は取得を待たない
    # 開いているメニューは触らない: 開いたメニューを作り直すとクリックを飲み込む（実測）。
    # 非同期からメニューを差し替える applyStatus: 経路が復活したら落ちる。
    assert "applyStatus" not in src
    assert "performSelectorOnMainThread" in src  # showUpdateMark（ボタン題字）だけには使う
    # 自動有効化を切る（既定 True だとアクション項目まで無効化されクリック無反応・実測）。
    # 情報行は明示 disabled。これらが欠けたらクリックが死ぬので回帰ガードにする。
    assert "setAutoenablesItems_(False)" in src
    assert "setEnabled_(False)" in src
    # いつでもアイコンから更新できる: 新版未検知でも「更新を確認」を常設する。
    assert "更新を確認" in src
    # 更新を押したら即フィードバック: アイコン横に「更新中…」を出す（通知に依らない）。
    assert tray_mod._UPDATING_TITLE == "更新中…"
    assert "setTitle_(_UPDATING_TITLE)" in src and "resetTitle_" in src


def test_menu_action_items_are_enabled_and_wired():
    """実コードと同じ構成で、アクション項目が enabled かつ target/action 配線済みで、
    情報行は disabled になることを検証する（autoenables=False の管理が正しいか）。

    自動有効化（既定 True）はアクション項目まで無効化してクリックを殺すため、
    ここで「アクション項目 enabled・情報行 disabled」を実 AppKit で確かめる。"""
    try:
        import AppKit
        from Foundation import NSObject
    except ImportError:
        import pytest
        pytest.skip("pyobjc 不在（macOS 以外）")

    class _D(NSObject):
        def openLog_(self, s): pass
        def updateNow_(self, s): pass
        def stopGateway_(self, s): pass

    d = _D.alloc().init()
    menu = AppKit.NSMenu.alloc().init()
    menu.setAutoenablesItems_(False)
    info = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("info", None, "")
    info.setEnabled_(False)
    menu.addItem_(info)
    for title, sel in (("更新を確認", "updateNow:"), ("ログを開く", "openLog:"),
                       ("ゲートウェイを停止", "stopGateway:")):
        it = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, sel, "")
        it.setTarget_(d)
        menu.addItem_(it)
    menu.update()
    assert menu.itemAtIndex_(0).isEnabled() is False  # 情報行はグレー
    for i in (1, 2, 3):  # アクション項目はすべて有効・配線済み
        it = menu.itemAtIndex_(i)
        assert it.isEnabled() is True, f"{it.title()} が無効（クリック不可）"
        assert it.target() is not None and it.action() is not None


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


def test_update_menu_item_states():
    """更新項目の出し分け: 新版あり=クリック可 / 最新=非クリック / 未確認=クリック可。"""
    # 新版あり（パイプ通知）→ 「今すぐ更新して再起動（vX）」・クリック可
    label, clickable = update_menu_item({"kind": "update-ready", "latest": "0.37.0"}, None)
    assert clickable is True and "0.37.0" in label and "今すぐ更新" in label
    # 最新（PyPI 版まで確認できて available=False）→ 「最新です（vX）」・**選べない**
    label, clickable = update_menu_item(
        {"kind": None}, {"update": {"available": False, "latest": "0.36.1",
                                    "current": "0.36.1"}})
    assert clickable is False and label == "最新です（v0.36.1）"
    # 未確認/オフライン（latest を引けていない）→ 断言せず「更新を確認」・クリック可
    label, clickable = update_menu_item({"kind": None}, {"update": {"available": False,
                                                                    "latest": None}})
    assert clickable is True and label == "更新を確認"
    # admin 情報がまだ無い（初回オープン前）→ 「更新を確認」
    label, clickable = update_menu_item({"kind": None}, None)
    assert clickable is True and label == "更新を確認"


def test_daemon_spawns_tray_before_provisioning():
    """アイコンはデーモン起動の最初に出す——llama.cpp 等の自動導入（初回は数分）を
    待たせない。spawn を導入処理の後ろへ動かしたらこのテストが落ちる。"""
    import inspect

    from local_llm_server import daemon as daemon_mod

    src = inspect.getsource(daemon_mod._run_gateway_locked)
    assert src.index("_maybe_spawn_tray") < src.index("provision_llama_if_needed")


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


# --- アイコンからの更新（_post_update_now の結果ハンドリング） -----------------
def _fake_resp(payload: dict):
    import io

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
    return _R(json.dumps(payload).encode())


def test_update_now_notifies_up_to_date(monkeypatch):
    """最新だったら通知で知らせる（メニューは閉じているため）。再起動はしない。"""
    notes = []
    monkeypatch.setattr(tray_mod, "_notify", lambda t, m: notes.append((t, m)))
    monkeypatch.setattr(tray_mod.urllib.request, "urlopen",
                        lambda req, timeout=0: _fake_resp(
                            {"status": "up-to-date", "current": "0.35.1"}))
    tray_mod._post_update_now("127.0.0.1", 8799)
    assert notes and "最新" in notes[0][1] and "0.35.1" in notes[0][1]


def test_update_now_notifies_restarting(monkeypatch):
    notes = []
    monkeypatch.setattr(tray_mod, "_notify", lambda t, m: notes.append((t, m)))
    monkeypatch.setattr(tray_mod.urllib.request, "urlopen",
                        lambda req, timeout=0: _fake_resp(
                            {"status": "restarting", "latest": "0.36.0"}))
    tray_mod._post_update_now("127.0.0.1", 8799)
    assert notes and "0.36.0" in notes[0][1]


def test_update_now_notifies_http_error(monkeypatch):
    """409（dirty tree 等）はエラー本文を通知に載せる。"""
    import io

    notes = []
    monkeypatch.setattr(tray_mod, "_notify", lambda t, m: notes.append((t, m)))

    def _raise(req, timeout=0):
        raise tray_mod.urllib.error.HTTPError(
            "u", 409, "Conflict", {}, io.BytesIO(json.dumps(
                {"error": "作業ツリーに未コミットの変更があります"}).encode()))
    monkeypatch.setattr(tray_mod.urllib.request, "urlopen", _raise)
    tray_mod._post_update_now("127.0.0.1", 8799)
    assert notes and "未コミット" in notes[0][1]


def test_update_now_notifies_connection_failure(monkeypatch):
    notes = []
    monkeypatch.setattr(tray_mod, "_notify", lambda t, m: notes.append((t, m)))

    def _raise(req, timeout=0):
        raise OSError("connection refused")
    monkeypatch.setattr(tray_mod.urllib.request, "urlopen", _raise)
    tray_mod._post_update_now("127.0.0.1", 8799)
    assert notes and "接続" in notes[0][1]
