import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from local_llm_server import ServerConfig
from local_llm_server import daemon as gw


# --- 設定ロード -----------------------------------------------------------

def _write(tmp_path, text):
    p = tmp_path / "gateway.toml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_load_gateway_config_assigns_internal_ports(tmp_path):
    p = _write(
        tmp_path,
        'port = 8080\ninternal_base_port = 9001\nmax_resident = 2\n'
        '[[models]]\nmodel = "org/A"\nbackend = "mlx"\n'
        '[[models]]\nmodel = "org/B"\nbackend = "llama-cpp"\n',
    )
    cfg = gw.load_gateway_config(p)
    assert cfg.port == 8080 and cfg.max_resident == 2
    assert [c.model for c in cfg.models] == ["org/A", "org/B"]
    # 内部ポートは連番で割り当てられる
    assert [c.port for c in cfg.models] == [9001, 9002]


def test_load_gateway_config_llama_draft_passthrough(tmp_path):
    # llama-cpp の draft_model は repo-id をそのまま採用（speculative decoding 用）。
    # グローバル既定 "auto" は llama-cpp では無効化される（自動解決表が無い）。
    p = _write(
        tmp_path,
        'draft_model = "auto"\n'
        '[[models]]\nmodel = "org/a-gguf"\nbackend = "llama-cpp"\n'
        'draft_model = "org/d-gguf:F16-MTP"\n'
        '[[models]]\nmodel = "org/b-gguf"\nbackend = "llama-cpp"\n',
    )
    cfg = gw.load_gateway_config(p)
    drafts = {c.model: c.draft_model for c in cfg.models}
    assert drafts["org/a-gguf"] == "org/d-gguf:F16-MTP"
    assert drafts["org/b-gguf"] is None  # "auto" 継承は無効


def test_load_gateway_config_rejects_empty_models(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        gw.load_gateway_config(_write(tmp_path, "port = 8080\n"))


def test_load_gateway_config_rejects_duplicate_and_bad_backend(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        gw.load_gateway_config(_write(
            tmp_path,
            '[[models]]\nmodel = "x"\nbackend = "mlx"\n'
            '[[models]]\nmodel = "x"\nbackend = "mlx"\n',
        ))
    with pytest.raises(ValueError, match="backend must be one of"):
        gw.load_gateway_config(_write(
            tmp_path, '[[models]]\nmodel = "x"\nbackend = "nope"\n'))


def test_load_gateway_config_rejects_internal_port_collision(tmp_path):
    # internal_base_port が公開ポートと衝突
    with pytest.raises(ValueError, match="collides"):
        gw.load_gateway_config(_write(
            tmp_path,
            'port = 9001\ninternal_base_port = 9001\n'
            '[[models]]\nmodel = "x"\nbackend = "mlx"\n'))


def test_load_gateway_config_rejects_unknown_default_model(tmp_path):
    with pytest.raises(ValueError, match="default_model"):
        gw.load_gateway_config(_write(
            tmp_path,
            'default_model = "ghost"\n[[models]]\nmodel = "x"\nbackend = "mlx"\n'))


# --- MTP（draft_model）の継承・上書き・無効化・検証 -------------------------

def test_gateway_draft_model_inherits_default_and_resolves(tmp_path):
    # トップレベル draft_model="auto" を継承し、mlx-vlm なら本体名から MTP ドラフターを解決。
    cfg = gw.load_gateway_config(_write(
        tmp_path,
        'draft_model = "auto"\n'
        '[[models]]\nmodel = "mlx-community/Qwen3.6-27B-4bit"\nbackend = "mlx-vlm"\n'))
    assert cfg.models[0].draft_model == "mlx-community/Qwen3.6-27B-MTP-4bit"


def test_gateway_draft_model_per_model_override(tmp_path):
    # 個別指定がゲートウェイ既定より優先される。
    cfg = gw.load_gateway_config(_write(
        tmp_path,
        'draft_model = "auto"\n'
        '[[models]]\nmodel = "mlx-community/Qwen3.6-27B-4bit"\nbackend = "mlx-vlm"\n'
        'draft_model = "org/custom-mtp"\n'))
    assert cfg.models[0].draft_model == "org/custom-mtp"


def test_gateway_draft_model_off_disables(tmp_path):
    # "off" でゲートウェイ既定の継承を打ち消す。
    cfg = gw.load_gateway_config(_write(
        tmp_path,
        'draft_model = "auto"\n'
        '[[models]]\nmodel = "mlx-community/Qwen3.6-27B-4bit"\nbackend = "mlx-vlm"\n'
        'draft_model = "off"\n'))
    assert cfg.models[0].draft_model is None


def test_gateway_draft_model_ignored_on_non_mlx_vlm(tmp_path):
    # MTP は mlx-vlm のみ。他バックエンドでは継承既定でも None（無視）。
    cfg = gw.load_gateway_config(_write(
        tmp_path,
        'draft_model = "auto"\n'
        '[[models]]\nmodel = "mlx-community/Qwen3.6-27B-4bit"\nbackend = "mlx"\n'))
    assert cfg.models[0].draft_model is None


def test_gateway_draft_model_auto_unknown_model_fails_fast(tmp_path):
    # mlx-vlm で auto だが MTP 対応表に無いモデル → 起動時に即エラー。
    with pytest.raises(ValueError):
        gw.load_gateway_config(_write(
            tmp_path,
            'draft_model = "auto"\n'
            '[[models]]\nmodel = "org/unknown-model"\nbackend = "mlx-vlm"\n'))


# --- ModelManager（フェイク LocalServer で遅延起動/LRU を検証）----------------

class _FakeServer:
    """LocalServer 差し替え。start/stop を記録し、即 ready になる。"""

    def __init__(self, config, log_path=None):
        self.config = config
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def wait_until_ready(self, *a, **k):
        pass

    def stop(self):
        self.stops += 1


def _patch_fake(monkeypatch):
    created = []

    def factory(config, log_path=None):
        s = _FakeServer(config, log_path)
        created.append(s)
        return s

    monkeypatch.setattr(gw, "LocalServer", factory)
    return created


def _configs():
    return [
        ServerConfig(backend="mlx", model="m1", host="127.0.0.1", port=9001),
        ServerConfig(backend="mlx", model="m2", host="127.0.0.1", port=9002),
    ]


def test_manager_lazy_loads_once(monkeypatch):
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    addr, h = mgr.acquire("m1")
    assert addr == ("127.0.0.1", 9001)
    mgr.release(h)
    # 2 回目は高速パス。新規起動しない。
    mgr.acquire("m1")
    assert len(created) == 1 and created[0].starts == 1


def test_manager_unknown_model_raises(monkeypatch):
    _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    with pytest.raises(KeyError):
        mgr.acquire("ghost")


def test_manager_lru_evicts_when_over_budget(monkeypatch):
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=1)
    _, h1 = mgr.acquire("m1")
    mgr.release(h1)
    _, h2 = mgr.acquire("m2")   # 予算 1 → m1 を退避
    mgr.release(h2)
    by_model = {s.config.model: s for s in created}
    assert by_model["m1"].stops == 1   # 退避で停止
    assert by_model["m2"].stops == 0   # 稼働中
    st = {s["model"]: s["loaded"] for s in mgr.status()}
    assert st == {"m1": False, "m2": True}


def test_load_gateway_config_idle_timeout(tmp_path):
    cfg = gw.load_gateway_config(_write(
        tmp_path,
        'idle_timeout = 600\n[[models]]\nmodel = "x"\nbackend = "mlx"\n'))
    assert cfg.idle_timeout == 600.0
    # 省略時は既定 1200、0 は無効（None）
    assert gw.load_gateway_config(_write(
        tmp_path, '[[models]]\nmodel = "x"\nbackend = "mlx"\n')).idle_timeout == 1200.0
    assert gw.load_gateway_config(_write(
        tmp_path, 'idle_timeout = 0\n[[models]]\nmodel = "x"\nbackend = "mlx"\n')).idle_timeout is None
    import pytest
    with pytest.raises(ValueError, match="idle_timeout"):
        gw.load_gateway_config(_write(
            tmp_path, 'idle_timeout = -1\n[[models]]\nmodel = "x"\nbackend = "mlx"\n'))


def test_manager_evict_idle_stops_unused(monkeypatch):
    # idle TTL: 最終利用から timeout 超のモデルだけ停止（処理中・最近使用は残す）。
    import time
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    _, h1 = mgr.acquire("m1")
    mgr.release(h1)
    # m1 の last_used を十分過去にする（idle 判定させる）。
    mgr._models["m1"].last_used = time.monotonic() - 100
    freed = mgr.evict_idle(timeout=10)
    assert freed == 1
    by_model = {s.config.model: s for s in created}
    assert by_model["m1"].stops == 1
    assert mgr.status()[0]["loaded"] is False


def test_manager_evict_idle_keeps_recent_and_inflight(monkeypatch):
    import time
    _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    # m1: 最近使った（idle でない）→ 残す
    _, h1 = mgr.acquire("m1")
    mgr.release(h1)
    # m2: 古いが処理中（inflight>0）→ 残す
    _, h2 = mgr.acquire("m2")
    mgr._models["m2"].last_used = time.monotonic() - 100
    assert mgr.evict_idle(timeout=10) == 0
    st = {s["model"]: s["loaded"] for s in mgr.status()}
    assert st == {"m1": True, "m2": True}
    mgr.release(h2)


def test_manager_capacity_timeout_when_all_busy(monkeypatch):
    # ハード上限: 全枠が処理中なら待ち、load_timeout 超過で CapacityError（→ 503）。超過は許さない。
    import pytest
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=1, load_timeout=0.2)
    _, h1 = mgr.acquire("m1")        # release しない → inflight=1（枠が埋まったまま）
    with pytest.raises(gw.CapacityError):
        mgr.acquire("m2")           # 全枠 busy → 0.2s 待って timeout
    by_model = {s.config.model: s for s in created}
    assert by_model["m1"].stops == 0           # m1 は退避されない
    assert "m2" not in by_model                # m2 はロードされない（OOM 回避）
    st = {s["model"]: s["loaded"] for s in mgr.status()}
    assert st == {"m1": True, "m2": False}
    mgr.release(h1)


def test_manager_waits_for_slot_then_loads(monkeypatch):
    # 枠が空く（release）と、待っていたロードが進む（m1 を退避して m2 をロード）。
    import threading
    import time
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=1, load_timeout=5)
    _, h1 = mgr.acquire("m1")        # 枠を占有（処理中）
    result = {}

    def worker():
        try:
            mgr.acquire("m2")       # 枠が空くまでブロック
            result["ok"] = True
        except Exception as exc:    # noqa: BLE001
            result["err"] = exc

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.3)
    assert "ok" not in result        # まだ待機中（超過ロードしない）
    mgr.release(h1)                  # 枠が空く → m1 退避 → m2 ロード
    t.join(timeout=5)
    assert result.get("ok") is True, result
    by_model = {s.config.model: s for s in created}
    assert by_model["m1"].stops == 1           # m1 は退避された
    st = {s["model"]: s["loaded"] for s in mgr.status()}
    assert st == {"m1": False, "m2": True}


def test_load_gateway_config_load_timeout(tmp_path):
    base = '[[models]]\nmodel = "x"\nbackend = "mlx"\n'
    assert gw.load_gateway_config(_write(tmp_path, base)).load_timeout == 300.0  # 既定
    assert gw.load_gateway_config(
        _write(tmp_path, "load_timeout = 30\n" + base)).load_timeout == 30.0
    import pytest
    with pytest.raises(ValueError, match="load_timeout"):
        gw.load_gateway_config(_write(tmp_path, "load_timeout = 0\n" + base))


def test_manager_start_failure_propagates_and_stays_unloaded(monkeypatch):
    class _BadServer(_FakeServer):
        def wait_until_ready(self, *a, **k):
            raise TimeoutError("never ready")

    monkeypatch.setattr(gw, "LocalServer", _BadServer)
    mgr = gw.ModelManager(_configs())
    with pytest.raises(TimeoutError):
        mgr.acquire("m1")
    assert all(not s["loaded"] for s in mgr.status())


def test_manager_shutdown_stops_all(monkeypatch):
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    mgr.release(mgr.acquire("m1")[1])
    mgr.release(mgr.acquire("m2")[1])
    mgr.shutdown()
    assert all(s.stops == 1 for s in created)
    assert all(not s["loaded"] for s in mgr.status())


# --- HTTP 振り分け（実フェイク上流 + no-op LocalServer）----------------------

def _make_upstream(name):
    class _H(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def _send(self, obj):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self._send({"backend": name})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(port, path, obj):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("POST", path, json.dumps(obj), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    return resp.status, json.loads(body)


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    return resp.status, json.loads(body)


def _start_gateway(monkeypatch):
    # 上流は実フェイクサーバー。LocalServer は no-op（既に上流が動いている）に差し替える。
    up1 = _make_upstream("m1-upstream")
    up2 = _make_upstream("m2-upstream")
    monkeypatch.setattr(gw, "LocalServer", lambda config, log_path=None: _FakeServer(config, log_path))
    configs = [
        ServerConfig(backend="mlx", model="m1", host="127.0.0.1", port=up1.server_address[1]),
        ServerConfig(backend="mlx", model="m2", host="127.0.0.1", port=up2.server_address[1]),
    ]
    mgr = gw.ModelManager(configs)
    server = gw.GatewayServer(("127.0.0.1", 0), mgr, catalog=["m1", "m2"])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, mgr, (up1, up2)


def test_gateway_routes_by_model(monkeypatch):
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        for model, expected in (("m1", "m1-upstream"), ("m2", "m2-upstream")):
            status, obj = _post(port, "/v1/chat/completions",
                                 {"model": model, "messages": [{"role": "user", "content": "hi"}]})
            assert status == 200 and obj["backend"] == expected
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_unknown_model_returns_404(monkeypatch):
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        status, obj = _post(port, "/v1/chat/completions",
                            {"model": "ghost", "messages": []})
        assert status == 404 and "error" in obj
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_models_catalog(monkeypatch):
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        status, obj = _get(port, "/v1/models")
        assert status == 200
        ids = [m["id"] for m in obj["data"]]
        assert ids == ["m1", "m2"]
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_missing_model_returns_400(monkeypatch):
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        status, obj = _post(port, "/v1/chat/completions", {"messages": []})
        assert status == 400 and "error" in obj
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()
