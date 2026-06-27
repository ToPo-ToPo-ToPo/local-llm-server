"""TUI ダッシュボードの純ロジック（catalog × ライブ状態のマージ・時間整形・ログ末尾）と、
アプリ内ログ画面の開閉（textual pilot）を検証する。
"""
import asyncio

from local_llm_server import tui
from local_llm_server.daemon import load_gateway_config


def test_read_log_tail(tmp_path, monkeypatch):
    log = tmp_path / "gw.log"
    log.write_text("\n".join(f"line {i}" for i in range(1, 2001)) + "\n", encoding="utf-8")
    monkeypatch.setattr(tui, "gateway_log_path", lambda port: str(log))
    out = tui.read_log_tail(123, max_lines=10)
    lines = out.strip().splitlines()
    assert lines[-1] == "line 2000" and len(lines) == 10
    # ファイルが無いときは案内文
    monkeypatch.setattr(tui, "gateway_log_path", lambda port: str(tmp_path / "nope.log"))
    assert tui.read_log_tail(123).startswith("(ログはまだ")


def test_key_hints_stay_visible_while_typing_command(tmp_path, monkeypatch):
    # コマンド入力中（Input にフォーカス）でも、stop/start 等のキー凡例が消えないこと。
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})  # 自動起動させない
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: None)

    async def scenario():
        app = tui_app.GatewayMonitor(gcfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            hints = app.query_one("#hints")
            text = hints.render().plain
            assert all(w in text for w in ("stop", "restart", "start", "log", "quit"))
            # 入力欄にフォーカスして打鍵 → 凡例は表示されたまま
            app.query_one("#cmd").focus()
            await pilot.pause()
            await pilot.press("s", "t", "o", "p")
            await pilot.pause()
            assert app.query_one("#cmd").value == "stop"
            assert app.query_one("#hints").display is True

    asyncio.run(scenario())


def _write_cfg(tmp_path, body):
    p = tmp_path / "gateway.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_row_selection_copies_model_name(tmp_path, monkeypatch):
    # 表の行を選ぶ（クリック/Enter）と、その行のモデル名がコピーされること。
    from textual.widgets import DataTable
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: None)
    admin = {
        "uptime": 1.0, "requests": 0,
        "models": [{"model": "mlx-community/Foo-4bit", "backend": "mlx-vlm",
                    "loaded": True, "inflight": 0, "requests": 0, "idle_for": 1.0}],
        "available": [{"id": "unsloth/Bar-GGUF", "backend": "llama-cpp"}],
    }

    async def scenario():
        app = tui_app.GatewayMonitor(gcfg)
        copied = []
        async with app.run_test() as pilot:
            app._copy_model = lambda m: copied.append(m)
            app._apply(admin)
            await pilot.pause()
            table = app.query_one("#models", DataTable)
            assert table.row_count == 2
            table.focus()
            table.move_cursor(row=1)            # 2 行目 = unsloth/Bar-GGUF（未ロード候補）
            table.action_select_cursor()        # Enter 相当
            await pilot.pause()
        assert copied == ["unsloth/Bar-GGUF"]

    asyncio.run(scenario())


def test_copy_model_uses_pbcopy_on_macos(tmp_path, monkeypatch):
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(tui_app.sys, "platform", "darwin")
    recorded = {}
    monkeypatch.setattr(
        tui_app.subprocess, "run",
        lambda cmd, input=None, check=False: recorded.update(cmd=cmd, input=input),
    )

    async def scenario():
        app = tui_app.GatewayMonitor(gcfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._copy_model("org/My-Model:Q4")
            await pilot.pause()
        assert recorded["cmd"] == ["pbcopy"]
        assert recorded["input"] == b"org/My-Model:Q4"

    asyncio.run(scenario())


def test_log_screen_opens_and_closes(tmp_path, monkeypatch):
    # 外部ページャでなくアプリ内画面でログを出し、q / Esc でダッシュボードに戻れること。
    from textual.app import App
    from local_llm_server import tui_app

    log = tmp_path / "gw.log"
    log.write_text("hello log\n", encoding="utf-8")
    monkeypatch.setattr(tui, "gateway_log_path", lambda port: str(log))
    monkeypatch.setattr(tui_app, "gateway_log_path", lambda port: str(log))

    class Host(App):
        def on_mount(self):
            self.push_screen(tui_app.LogScreen(8799))

    async def scenario():
        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert type(app.screen).__name__ == "LogScreen"
            await pilot.press("q")
            await pilot.pause()
            assert type(app.screen).__name__ != "LogScreen"   # q で戻る
            app.push_screen(tui_app.LogScreen(8799))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert type(app.screen).__name__ != "LogScreen"   # Esc でも戻る

    asyncio.run(scenario())


def _gcfg(tmp_path, body):
    p = tmp_path / "gateway.toml"
    p.write_text(body, encoding="utf-8")
    return load_gateway_config(str(p))


_TWO_MODELS = (
    'port = 8799\nidle_timeout = 1200\n'
    '[[models]]\nmodel = "org/A"\nbackend = "mlx"\n'
    '[[models]]\nmodel = "org/B"\nbackend = "mlx"\n'
)


def test_merge_includes_dynamic_loaded_models(tmp_path):
    # 事前登録に無い動的ロードモデルも、ライブ状態にあれば表示に追加される
    gcfg = _gcfg(tmp_path, _TWO_MODELS)
    admin = {
        "uptime": 5.0, "requests": 3,
        "models": [
            {"model": "org/A", "backend": "mlx", "port": 9001,
             "loaded": True, "inflight": 1, "requests": 2, "idle_for": None},
            {"model": "dyn/NEW-GGUF", "backend": "llama-cpp", "port": 9003,
             "loaded": True, "inflight": 0, "requests": 1, "idle_for": 3.0},
        ],
    }
    view = tui.merge_status(gcfg, admin)
    rows = {r["model"]: r for r in view["models"]}
    # 事前登録 2 つ＋動的 1 つ
    assert set(rows) == {"org/A", "org/B", "dyn/NEW-GGUF"}
    assert rows["dyn/NEW-GGUF"]["backend"] == "llama-cpp"
    assert rows["dyn/NEW-GGUF"]["state"] == "idle"
    assert rows["org/A"]["state"] == "busy"


def test_merge_lists_cached_available_models_unloaded(tmp_path):
    # /admin/status の "available"（DL 済みだが未ロード）も unloaded 候補として一覧に出る。
    gcfg = _gcfg(tmp_path, _TWO_MODELS)
    admin = {
        "uptime": 5.0, "requests": 0, "models": [],
        "available": [
            {"id": "org/A", "backend": "mlx"},                 # 事前登録と重複 → 二重に出さない
            {"id": "mlx-community/Qwen3.6-27B-4bit", "backend": "mlx-vlm"},
        ],
    }
    rows = {r["model"]: r for r in tui.merge_status(gcfg, admin)["models"]}
    assert set(rows) == {"org/A", "org/B", "mlx-community/Qwen3.6-27B-4bit"}
    assert rows["mlx-community/Qwen3.6-27B-4bit"]["state"] == "unloaded"
    assert rows["mlx-community/Qwen3.6-27B-4bit"]["backend"] == "mlx-vlm"


def test_merge_marks_unlisted_models_unloaded(tmp_path):
    gcfg = _gcfg(tmp_path, _TWO_MODELS)
    admin = {
        "uptime": 10.0, "requests": 7,
        "models": [{"model": "org/A", "backend": "mlx", "port": 9001,
                    "loaded": True, "inflight": 0, "requests": 7, "idle_for": 200.0}],
    }
    view = tui.merge_status(gcfg, admin)
    rows = {r["model"]: r for r in view["models"]}
    # カタログ順で全モデルが並ぶ
    assert [r["model"] for r in view["models"]] == ["org/A", "org/B"]
    # ロード済み・処理中なし → idle。idle_remaining = idle_timeout - idle_for
    assert rows["org/A"]["state"] == "idle"
    assert rows["org/A"]["idle_remaining"] == 1000.0
    # admin に出ないモデルは unloaded
    assert rows["org/B"]["state"] == "unloaded"
    assert rows["org/B"]["idle_remaining"] is None


def test_merge_busy_when_inflight(tmp_path):
    gcfg = _gcfg(tmp_path, _TWO_MODELS)
    admin = {"uptime": 1.0, "requests": 3, "models": [
        {"model": "org/A", "backend": "mlx", "port": 9001,
         "loaded": True, "inflight": 2, "requests": 3, "idle_for": None}]}
    rows = {r["model"]: r for r in tui.merge_status(gcfg, admin)["models"]}
    assert rows["org/A"]["state"] == "busy"
    assert rows["org/A"]["inflight"] == 2
    assert rows["org/A"]["idle_remaining"] is None  # 処理中はカウントダウンしない


def test_merge_no_admin_falls_back(tmp_path):
    gcfg = _gcfg(tmp_path, _TWO_MODELS)
    view = tui.merge_status(gcfg, None)  # ゲートウェイ未応答
    assert view["uptime"] is None
    assert all(r["state"] == "unloaded" for r in view["models"])


def test_fmt_hms():
    assert tui._fmt_hms(None) == "—"
    assert tui._fmt_hms(65) == "1:05"
    assert tui._fmt_hms(3725) == "1:02:05"
    assert tui._fmt_hms(0) == "0:00"
