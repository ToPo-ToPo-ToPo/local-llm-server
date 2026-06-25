"""TUI ダッシュボードの純ロジック（catalog × ライブ状態のマージ・時間整形）。

curses 描画は端末が要るのでテストしない。ここでは描画に渡す前のデータ統合（merge_status）と
整形（_fmt_hms）だけを検証する。
"""
from local_llm_server import tui
from local_llm_server.daemon import load_gateway_config


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
