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
    monkeypatch.setattr(tui_app, "is_ready", lambda url, **k: False)  # 実 HTTP を叩かない

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
    # TUI テストは既定で自動更新を切る（起動時の PyPI HTTP / git を走らせず hermetic に保つ）。
    # 更新機能自体のテストは明示的に auto_update を書いて上書きする。
    if "auto_update" not in body:
        body = "auto_update = false\n" + body
    p.write_text(body, encoding="utf-8")
    return p


def test_row_click_copies_model_name(tmp_path, monkeypatch):
    # 表の行をクリックすると、その行のモデル名がコピーされること（行ハイライトは使わない）。
    from textual.widgets import DataTable
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})
    # ポーリングが表を消さないよう常に同じ admin を返す（is_ready の実 HTTP も叩かせない）。
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: admin)
    monkeypatch.setattr(tui_app, "is_ready", lambda url, **k: True)
    admin = {
        "uptime": 1.0, "requests": 0,
        "models": [{"model": "mlx-community/Foo-4bit", "backend": "mlx-vlm",
                    "loaded": True, "inflight": 0, "requests": 0, "idle_for": 1.0}],
        "available": [{"id": "unsloth/Bar-GGUF", "backend": "llama-cpp"}],
    }

    async def scenario():
        app = tui_app.GatewayMonitor(gcfg)
        copied = []
        async with app.run_test(size=(120, 30)) as pilot:
            app._copy_model = lambda m: copied.append(m)
            app._apply(admin)
            await pilot.pause()
            table = app.query_one("#models", DataTable)
            assert table.row_count == 2
            assert table.cursor_type == "none"          # 行ハイライトを出さない
            await pilot.click("#models", offset=(3, 1))  # 1 行目（ヘッダの下）= Foo
            await pilot.pause()
            await pilot.click("#models", offset=(3, 2))  # 2 行目 = Bar（未ロード候補）
            await pilot.pause()
        assert copied == ["mlx-community/Foo-4bit", "unsloth/Bar-GGUF"]

    asyncio.run(scenario())


def test_copy_model_uses_pbcopy_on_macos(tmp_path, monkeypatch):
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(tui_app, "is_ready", lambda url, **k: False)  # 実 HTTP を叩かない
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


def test_auto_update_applies_when_idle(tmp_path, monkeypatch):
    # auto_update=true で新版を検知し、ゲートウェイがアイドルなら自動適用→再起動フラグが立つ。
    from local_llm_server import tui_app, update

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "auto_update = true\nport = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})  # 自動起動させない
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: None)      # admin なし=アイドル
    monkeypatch.setattr(tui_app, "is_ready", lambda url, **k: False)
    # 新版あり・適用可能を返すようチェックを固定。
    st = update.UpdateStatus(current="0.21.0", latest="0.22.0", available=True,
                             can_apply=True, reason="ok")
    monkeypatch.setattr(tui_app.update, "check", lambda timeout=3.0: st)
    applied = {}

    def _fake_apply(*a, **k):
        applied["called"] = True
        return True, "done"

    monkeypatch.setattr(tui_app.update, "apply_update", _fake_apply)

    async def scenario():
        app = tui_app.GatewayMonitor(gcfg)
        app._kill_ports = lambda: None  # 実プロセスは触らない
        async with app.run_test() as pilot:
            for _ in range(10):
                await pilot.pause()
                if app.restart_after_exit:
                    break
        assert applied.get("called") is True
        assert app.restart_after_exit is True

    asyncio.run(scenario())


def test_auto_update_holds_when_dirty(tmp_path, monkeypatch):
    # ローカル変更ありは適用せず、バナー表示のみ（can_apply=False）。
    from local_llm_server import tui_app, update

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "auto_update = true\nport = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(tui_app, "is_ready", lambda url, **k: False)
    st = update.UpdateStatus(current="0.21.0", latest="0.22.0", available=True,
                             can_apply=False, reason="dirty")
    monkeypatch.setattr(tui_app.update, "check", lambda timeout=3.0: st)
    applied = {}

    def _fake_apply(*a, **k):
        applied["called"] = True
        return True, "x"

    monkeypatch.setattr(tui_app.update, "apply_update", _fake_apply)

    async def scenario():
        app = tui_app.GatewayMonitor(gcfg)
        app._kill_ports = lambda: None
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
        assert applied.get("called") is None       # 汚れているので適用しない
        assert app.restart_after_exit is False
        assert app._update is not None and app._update.reason == "dirty"

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


def test_merge_shows_mtp_status(tmp_path, monkeypatch):
    # 行に MTP（高速化）の利用可否が乗る: ドラフター揃い="ready"、未取得="available"、非対応=None。
    import json
    from local_llm_server import server as srv

    root = tmp_path / "hub"

    def _mk(repo, files):
        org, name = repo.split("/", 1)
        d = root / f"models--{org}--{name}" / "snapshots" / "a"
        d.mkdir(parents=True, exist_ok=True)
        for f, data in files.items():
            (d / f).write_bytes(data)

    # Qwen3.6-27B-4bit のドラフターだけ用意（本体名は対応表に在る）→ "ready"
    _mk("mlx-community/Qwen3.6-27B-MTP-4bit",
        {"config.json": json.dumps({"model_type": "qwen3"}).encode(),
         "model.safetensors": b"x"})
    monkeypatch.setattr(srv, "_hf_hub_cache", lambda: str(root))

    gcfg = _gcfg(tmp_path, _TWO_MODELS)
    admin = {
        "uptime": 1.0, "requests": 0, "models": [],
        "available": [
            {"id": "mlx-community/Qwen3.6-27B-4bit", "backend": "mlx-vlm"},   # ドラフター揃い
            {"id": "mlx-community/gemma-4-E4B-it-qat-4bit", "backend": "mlx-vlm"},  # 未取得
        ],
    }
    rows = {r["model"]: r for r in tui.merge_status(gcfg, admin)["models"]}
    assert rows["mlx-community/Qwen3.6-27B-4bit"]["mtp"] == "ready"
    assert rows["mlx-community/gemma-4-E4B-it-qat-4bit"]["mtp"] == "available"
    assert rows["org/A"]["mtp"] is None  # 対応表に無い本体は MTP 非対応


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


def test_merge_prefers_live_max_resident(tmp_path):
    # max_resident は実行中に変更できる。admin のライブ値があれば toml 値より優先する。
    gcfg = _gcfg(tmp_path, "max_resident = 1\n" + _TWO_MODELS)
    assert gcfg.max_resident == 1
    admin = {"uptime": 1.0, "requests": 0, "models": [], "max_resident": 3}
    assert tui.merge_status(gcfg, admin)["max_resident"] == 3   # ライブ値を優先
    # admin が無制限（None）ならそれを尊重（toml の 1 に戻さない）。
    admin["max_resident"] = None
    assert tui.merge_status(gcfg, admin)["max_resident"] is None
    # ゲートウェイ未応答（admin=None）なら toml の起動時値へフォールバック。
    assert tui.merge_status(gcfg, None)["max_resident"] == 1


def test_fmt_hms():
    assert tui._fmt_hms(None) == "—"
    assert tui._fmt_hms(65) == "1:05"
    assert tui._fmt_hms(3725) == "1:02:05"
    assert tui._fmt_hms(0) == "0:00"


def test_quit_stops_gateway_daemon(tmp_path, monkeypatch):
    # q（終了）でゲートウェイ・デーモンも停止する（次回起動で最新コードが反映されるように）。
    # 停止は UI スレッドを固めないよう別スレッド（@work）で走り、完了後に exit する。
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app, "server_status", lambda h, p: {"ready": True})  # 自動起動させない
    monkeypatch.setattr(tui_app, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(tui_app, "is_ready", lambda url, **k: False)  # 実 HTTP を叩かない
    killed = []
    exited = []

    async def scenario():
        app = tui_app.GatewayMonitor(gcfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            monkeypatch.setattr(app, "_kill_ports", lambda: killed.append(True))
            monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
            app.action_quit()
            await app.workers.wait_for_complete()  # 停止ワーカーの完了を待つ
            await pilot.pause()
            # 停止中は "closing…" を表示して UI が固まって見えないようにする。
            assert app.busy == "closing…"
            assert "closing" in app.query_one("#status").render().plain
            # コマンド入力（"quit"）経路も同じく停止を通すこと
            app.on_input_submitted(_FakeSubmit("quit"))
            await app.workers.wait_for_complete()
            await pilot.pause()

    asyncio.run(scenario())
    # 両経路（キー q・入力 "quit"）とも停止と終了を通す。
    assert killed == [True, True]
    assert exited == [True, True]


class _FakeSubmit:
    """on_input_submitted に渡す最小のイベント代用（value と input だけ持つ）。"""
    def __init__(self, value):
        self.value = value
        self.input = _FakeInput()


class _FakeInput:
    value = ""


def test_merge_status_ready_is_passed_in_not_probed(tmp_path):
    # ready はポーリング側（ワーカースレッド）が判定して渡す。merge_status 自身は
    # HTTP を叩かない純粋関数のまま（UI スレッドを固めない）。省略時は admin の有無。
    gcfg = _gcfg(tmp_path, _TWO_MODELS)
    admin = {"uptime": 1.0, "requests": 0, "models": []}
    assert tui.merge_status(gcfg, admin)["ready"] is True          # admin あり → 稼働中
    assert tui.merge_status(gcfg, None)["ready"] is False          # admin なし → 停止扱い
    assert tui.merge_status(gcfg, None, ready=True)["ready"] is True   # 明示指定を優先
    assert tui.merge_status(gcfg, admin, ready=False)["ready"] is False


def test_title_shows_name_and_version(tmp_path, monkeypatch):
    # TUI タイトルに製品名と稼働中バージョンを出す（自動更新でどの版が動くか分かるように）。
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app.update, "installed_version", lambda: "9.9.9")
    title = tui_app.GatewayMonitor(gcfg)._title().plain
    assert "local-llm-server" in title      # 名前
    assert "v9.9.9" in title                 # バージョン


def test_title_falls_back_to_dev_when_version_unknown(tmp_path, monkeypatch):
    # ソース実行等で版が取れないときは (dev) 表示にして落ちない。
    from local_llm_server import tui_app

    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    monkeypatch.setattr(tui_app.update, "installed_version", lambda: None)
    title = tui_app.GatewayMonitor(gcfg)._title().plain
    assert "local-llm-server" in title and "(dev)" in title
