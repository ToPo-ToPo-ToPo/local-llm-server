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
    # dynamic を切ると [[models]] 必須（旧挙動）
    with pytest.raises(ValueError, match="non-empty"):
        gw.load_gateway_config(_write(tmp_path, "port = 8080\ndynamic = false\n"))


def test_load_gateway_config_empty_models_ok_when_dynamic(tmp_path):
    # 既定（dynamic = true）では [[models]] 無しでも OK（全て動的ロード）
    cfg = gw.load_gateway_config(_write(tmp_path, "port = 8080\n"))
    assert cfg.dynamic is True and cfg.models == []


def test_dynamic_register_infers_backend_and_allocates_port(tmp_path):
    # 動的登録: ID からバックエンド推論＋内部ポート割当（事前登録の続き番号）。
    cfg = gw.load_gateway_config(_write(
        tmp_path,
        'port = 8799\ninternal_base_port = 9001\ndisable_thinking = true\n'
        '[[models]]\nmodel = "mlx-community/A"\nbackend = "mlx"\n'))
    mgr = gw.ModelManager(
        cfg.models, dynamic=cfg.dynamic,
        default_disable_thinking=cfg.disable_thinking,
        internal_base_port=cfg.internal_base_port, public_port=cfg.port,
    )
    gguf = mgr._register_dynamic_locked("unsloth/Foo-GGUF:Q4")
    assert gguf.config.backend == "llama-cpp"   # gguf → llama-cpp
    assert gguf.config.port == 9002             # 事前登録(9001)の次
    assert gguf.config.disable_thinking is True  # 動的既定を継承
    assert gguf.dynamic is True
    mlx = mgr._register_dynamic_locked("mlx-community/Bar")
    assert mlx.config.backend == "mlx-vlm" and mlx.config.port == 9003


def test_dynamic_register_auto_enables_mtp_for_supported_mlx_vlm():
    # 事前登録なしの動的ロードでも、対応表に在る mlx-vlm モデルは MTP が自動で効く
    # （draft_model="auto" を graceful に解決）。
    mgr = gw.ModelManager([], dynamic=True)
    m = mgr._register_dynamic_locked("mlx-community/Qwen3.6-27B-4bit")
    assert m.config.draft_model == "mlx-community/Qwen3.6-27B-MTP-4bit"


def test_dynamic_register_no_mtp_for_unsupported_or_other_backends():
    # 対応表に無い mlx-vlm モデルや、MTP 非対応バックエンド（gguf/mlx）は
    # 動的ロードを失敗させずに MTP なしで起動する。
    mgr = gw.ModelManager([], dynamic=True)
    assert mgr._register_dynamic_locked("mlx-community/Unknown-99B").config.draft_model is None
    assert mgr._register_dynamic_locked("unsloth/Foo-GGUF:Q4").config.draft_model is None


def test_dynamic_register_default_draft_off_disables_mtp():
    # トップレベル draft_model="off" を継承すると、対応モデルでも動的 MTP を切れる。
    mgr = gw.ModelManager([], dynamic=True, default_draft="off")
    m = mgr._register_dynamic_locked("mlx-community/Qwen3.6-27B-4bit")
    assert m.config.draft_model is None


def test_acquire_unknown_raises_when_dynamic_disabled(tmp_path):
    cfg = gw.load_gateway_config(_write(
        tmp_path, 'dynamic = false\n[[models]]\nmodel = "org/A"\nbackend = "mlx"\n'))
    mgr = gw.ModelManager(cfg.models, dynamic=cfg.dynamic)
    with pytest.raises(KeyError):
        mgr.acquire("org/unknown")


# --- global parallel（動的ロードの並列スロット既定）-------------------------

def test_dynamic_global_parallel_applies_to_llama_cpp_only():
    # トップレベル parallel は llama-cpp 動的モデルにだけ付き、mlx/mlx-vlm では無視される。
    mgr = gw.ModelManager([], dynamic=True, default_parallel=4)
    assert mgr._register_dynamic_locked("unsloth/Foo-GGUF:Q4").config.parallel == 4
    assert mgr._register_dynamic_locked("mlx-community/Bar").config.parallel is None


def test_load_gateway_config_parses_and_validates_parallel(tmp_path):
    cfg = gw.load_gateway_config(_write(tmp_path, "parallel = 4\n"))
    assert cfg.parallel == 4
    with pytest.raises(ValueError, match="parallel must be 1 or greater"):
        gw.load_gateway_config(_write(tmp_path, "parallel = 0\n"))


# --- メモリガード（max_memory_fraction）------------------------------------

class _StubServer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def _resident(mgr, model, gb, *, inflight=0, last_used=0.0):
    """ロード済みモデルを 1 つ手で常駐させる（占有量 gb GB をキャッシュ済みにする）。"""
    srv = _StubServer()
    m = gw._Model(
        config=ServerConfig(backend="mlx-vlm", model=model, port=9000),
        server=srv, ready=True, inflight=inflight, last_used=last_used,
        footprint=int(gb * 1e9),
    )
    mgr._models[model] = m
    return m


def test_load_gateway_config_validates_memory_fraction(tmp_path):
    cfg = gw.load_gateway_config(_write(tmp_path, "max_memory_fraction = 0.66\n"))
    assert cfg.max_memory_fraction == 0.66
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        gw.load_gateway_config(_write(tmp_path, "max_memory_fraction = 1.5\n"))


def test_memory_guard_evicts_idle_to_fit():
    # 予算 66GB（総RAM 100GB × 0.66）。常駐 50GB（アイドル）＋新規 30GB → 超過するので
    # アイドルを退避して収める。
    mgr = gw.ModelManager([], dynamic=True, max_memory_fraction=0.66)
    mgr._mem_total = int(100e9)  # 決定的にするため総RAMを固定
    idle = _resident(mgr, "old/idle", gb=50)
    keep = mgr._register_dynamic_locked("new/Big-GGUF:Q4")
    keep.footprint = int(30e9)
    mgr._evict_if_needed(keep="new/Big-GGUF:Q4")
    # idle は退避された（server が外れ、FakeServer.stop が呼ばれた）。非動的なので _models には残る。
    assert idle.server is None and not idle.ready
    assert mgr._models["old/idle"] is idle


def test_memory_guard_refuses_single_oversized_model():
    # 退避できる常駐が無く、新規モデル単体で予算超過 → 即 CapacityError（503）。
    mgr = gw.ModelManager([], dynamic=True, max_memory_fraction=0.5, load_timeout=1)
    mgr._mem_total = int(40e9)  # 予算 20GB
    keep = mgr._register_dynamic_locked("huge/Model-GGUF:Q8")
    keep.footprint = int(35e9)  # 単体で予算超過
    with pytest.raises(gw.CapacityError, match="memory budget"):
        mgr._evict_if_needed(keep="huge/Model-GGUF:Q8")


def test_memory_guard_allows_when_under_budget():
    mgr = gw.ModelManager([], dynamic=True, max_memory_fraction=0.66)
    mgr._mem_total = int(100e9)
    keep = mgr._register_dynamic_locked("ok/Model-GGUF:Q4")
    keep.footprint = int(20e9)
    mgr._evict_if_needed(keep="ok/Model-GGUF:Q4")  # 予算内 → 何も起きず返る


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
    # dynamic を切ると default_model は事前登録に在ることが必須
    with pytest.raises(ValueError, match="default_model"):
        gw.load_gateway_config(_write(
            tmp_path,
            'dynamic = false\ndefault_model = "ghost"\n'
            '[[models]]\nmodel = "x"\nbackend = "mlx"\n'))


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
    # 発見（キャッシュ走査）は決定的にするため空に固定する。
    monkeypatch.setattr(gw, "discover_cached_models", lambda *a, **k: [])
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


def test_gateway_models_includes_cached_discovered(monkeypatch):
    # /v1/models は 事前登録カタログ＋キャッシュにある DL 済みモデルを重複なく合成する。
    monkeypatch.setattr(
        gw, "discover_cached_models",
        lambda *a, **k: [{"id": "m1", "backend": "mlx"},          # カタログと重複→1回だけ
                         {"id": "mlx-community/New-4bit", "backend": "mlx-vlm"}],
    )
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        status, obj = _get(port, "/v1/models")
        ids = [m["id"] for m in obj["data"]]
        assert ids == ["m1", "m2", "mlx-community/New-4bit"]
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


# --- エージェント在席（セッション）ベースの即時アンロード --------------------

def _wait_unloaded(mgr, model, timeout=2.0):
    """非同期解放スレッドの完了を待つ（指定モデルが unloaded になるまで）。"""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = {s["model"]: s["loaded"] for s in mgr.status()}
        if not st.get(model, False):
            return True
        time.sleep(0.01)
    return False


def test_load_gateway_config_session_ttl(tmp_path):
    # 明示・既定(90)・無効(0→None)・負数で ValueError。
    base = '[[models]]\nmodel = "x"\nbackend = "mlx"\n'
    assert gw.load_gateway_config(_write(tmp_path, 'session_ttl = 30\n' + base)).session_ttl == 30.0
    assert gw.load_gateway_config(_write(tmp_path, base)).session_ttl == 90.0
    assert gw.load_gateway_config(_write(tmp_path, 'session_ttl = 0\n' + base)).session_ttl is None
    with pytest.raises(ValueError, match="session_ttl"):
        gw.load_gateway_config(_write(tmp_path, 'session_ttl = -1\n' + base))


def test_session_last_agent_release_frees_immediately(monkeypatch):
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    mgr.register_session("A", "m1")
    _, h = mgr.acquire("m1")   # ロードさせる
    mgr.release(h)
    assert mgr.status()[0]["loaded"] is True
    mgr.unregister_session("A")  # 最後の在席 → 即アンロード（別スレッド）
    assert _wait_unloaded(mgr, "m1")
    assert {s.config.model: s.stops for s in created}["m1"] == 1


def test_session_kept_while_another_agent_present(monkeypatch):
    _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    mgr.register_session("A", "m1")
    mgr.register_session("B", "m1")
    _, h = mgr.acquire("m1"); mgr.release(h)
    assert {s["model"]: s["sessions"] for s in mgr.status()}["m1"] == 2
    mgr.unregister_session("A")        # B がまだ在席 → 維持
    import time; time.sleep(0.2)
    st = {s["model"]: s for s in mgr.status()}["m1"]
    assert st["loaded"] is True and st["sessions"] == 1
    mgr.unregister_session("B")        # 最後の在席 → 解放
    assert _wait_unloaded(mgr, "m1")


def test_session_not_freed_while_inflight(monkeypatch):
    _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    mgr.register_session("A", "m1")
    _, h = mgr.acquire("m1")           # inflight=1 のまま
    mgr.unregister_session("A")        # 在席0 でも処理中なので解放しない
    import time; time.sleep(0.2)
    assert mgr.status()[0]["loaded"] is True
    mgr.release(h)


def test_reap_sessions_frees_on_heartbeat_timeout(monkeypatch):
    import time
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    mgr.register_session("A", "m1")
    _, h = mgr.acquire("m1"); mgr.release(h)
    # 心拍を過去にして途絶扱いにする。
    mgr._sessions["A"].last_seen = time.monotonic() - 100
    assert mgr.reap_sessions(ttl=10) == 1
    assert mgr.status()[0]["loaded"] is False
    assert {s.config.model: s.stops for s in created}["m1"] == 1


def test_heartbeat_unknown_agent_returns_false(monkeypatch):
    _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    assert mgr.heartbeat("ghost") is False
    mgr.register_session("A", "m1")
    assert mgr.heartbeat("A") is True


def test_session_switch_model_detaches_old(monkeypatch):
    # 同じ agent が別モデルへ乗り換えたら、旧モデルから外れる（旧モデルが無人なら解放）。
    _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs())
    mgr.register_session("A", "m1")
    _, h = mgr.acquire("m1"); mgr.release(h)
    mgr.register_session("A", "m2")    # m1 から m2 へ乗り換え
    assert _wait_unloaded(mgr, "m1")
    counts = {s["model"]: s["sessions"] for s in mgr.status()}
    assert counts["m1"] == 0 and counts["m2"] == 1


def test_gateway_session_endpoints(monkeypatch):
    # HTTP 経由で register/heartbeat/release を叩き、最後の解除で即アンロードされる。
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        assert _post(port, "/admin/sessions/register", {"agent_id": "A", "model": "m1"})[0] == 200
        _, h = mgr.acquire("m1"); mgr.release(h)   # ロード
        assert _post(port, "/admin/sessions/heartbeat", {"agent_id": "A"})[0] == 200
        assert _post(port, "/admin/sessions/heartbeat", {"agent_id": "ZZ"})[0] == 404
        # admin/status に sessions と session_ttl が出る
        _, st = _get(port, "/admin/status")
        assert {m["model"]: m["sessions"] for m in st["models"]}["m1"] == 1
        assert _post(port, "/admin/sessions/release", {"agent_id": "A"})[0] == 200
        assert _wait_unloaded(mgr, "m1")
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()
