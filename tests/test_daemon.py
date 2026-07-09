import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from local_llm_server import ServerConfig
from local_llm_server import daemon as gw


@pytest.fixture(autouse=True)
def _no_reclaim(monkeypatch):
    """ワーカー起動直前の孤児回収（実 lsof/kill）を無効化する。

    テストのフェイク上流はテストプロセス内でポートを LISTEN しているので、実回収を走らせると
    無関係なプロセス探索（lsof）が毎 acquire で走る。回収ロジック自体は専用テスト
    （test_reclaim_stale_workers_kills_only_ours、server モジュールを直接叩く）で検証する。
    """
    monkeypatch.setattr(gw, "reclaim_stale_workers", lambda *a, **k: [])


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
    # （draft_model="auto" を graceful に解決）。自作 ToPo-ToPo 版も収録済み。
    mgr = gw.ModelManager([], dynamic=True)
    m = mgr._register_dynamic_locked("mlx-community/Qwen3.6-27B-4bit")
    assert m.config.draft_model == "mlx-community/Qwen3.6-27B-MTP-4bit"
    topo = mgr._register_dynamic_locked("ToPo-ToPo/Qwen3.6-27B-mlx-4bit")
    assert topo.config.draft_model == "mlx-community/Qwen3.6-27B-MTP-4bit"
    # ToPo-ToPo 版 gemma 4 は model card 推奨の Google 公式ドラフターに解決する。
    g = mgr._register_dynamic_locked("ToPo-ToPo/gemma-4-31b-it-mlx-4bit")
    assert g.config.draft_model == "google/gemma-4-31B-it-assistant"


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

    def stop(self, grace: float | None = None):
        self.stopped = True


def _resident(mgr, model, gb, *, inflight=0, last_used=0.0):
    """ロード済みモデルを 1 つ手で常駐させる（占有量 gb GB をキャッシュ済みにする）。

    1 モデル=複数インスタンス構成なので、単一インスタンスを持つ _Model を組み立てる。
    """
    srv = _StubServer()
    cfg = ServerConfig(backend="mlx-vlm", model=model, port=9000)
    inst = gw._Instance(
        config=cfg, server=srv, ready=True, inflight=inflight, last_used=last_used,
    )
    m = gw._Model(config=cfg, instances=[inst], footprint=int(gb * 1e9))
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
    idle_inst = idle.instances[0]
    keep = mgr._register_dynamic_locked("new/Big-GGUF:Q4")
    keep.footprint = int(30e9)
    mgr._evict_if_needed(keep="new/Big-GGUF:Q4")
    # idle は退避された（インスタンスが外れ、StubServer.stop が呼ばれた）。非動的なので _models には残る。
    assert idle_inst.server.stopped is True
    assert idle.instances == []
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
        self.last_grace = None  # 直近 stop() に渡された grace（全体終了は 0）

    def start(self):
        self.starts += 1

    def wait_until_ready(self, *a, **k):
        pass

    def stop(self, grace: float = 10.0):
        self.stops += 1
        self.last_grace = grace


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
    mgr._models["m1"].instances[0].last_used = time.monotonic() - 100
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
    mgr._models["m2"].instances[0].last_used = time.monotonic() - 100
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


def test_set_max_resident_trims_idle_keeps_busy(monkeypatch):
    # 実行中に max_resident を下げると、超過分をアイドルから LRU 退避する（busy は止めない）。
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=None)  # 起動時は無制限
    _, h1 = mgr.acquire("m1")
    mgr.release(h1)             # m1: アイドル
    _, h2 = mgr.acquire("m2")   # m2: 処理中（release しない＝busy）
    mgr.set_max_resident(1)     # 上限 1 に縮小 → 裏で trim
    assert _wait_unloaded(mgr, "m1")          # アイドルの m1 は退避される
    st = {s["model"]: s["loaded"] for s in mgr.status()}
    assert st == {"m1": False, "m2": True}    # busy な m2 は残る（更新で止まらない）
    by_model = {s.config.model: s for s in created}
    assert by_model["m1"].stops == 1
    assert by_model["m2"].stops == 0
    assert mgr._max_resident == 1
    mgr.release(h2)


def test_set_max_resident_does_not_stop_busy_when_all_busy(monkeypatch):
    # 全て処理中なら、上限を下げても 1 つも止めない（稼働中の生成を守る）。
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=None)
    _, h1 = mgr.acquire("m1")   # busy
    _, h2 = mgr.acquire("m2")   # busy
    mgr.set_max_resident(1)
    mgr._trim_to_limit()        # 直接呼んで決定的に検証（アイドルが無いので何も止まらない）
    st = {s["model"]: s["loaded"] for s in mgr.status()}
    assert st == {"m1": True, "m2": True}
    assert all(s.stops == 0 for s in created)
    mgr.release(h1)
    mgr.release(h2)


def test_set_max_resident_raise_allows_new_load_without_evict(monkeypatch):
    # 上限を上げれば、退避せずに新しいモデルを追加常駐できる。
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=1)
    _, h1 = mgr.acquire("m1")
    mgr.release(h1)
    mgr.set_max_resident(2)     # 枠を 2 に拡大
    _, h2 = mgr.acquire("m2")   # m1 を退避せずロードできる
    mgr.release(h2)
    st = {s["model"]: s["loaded"] for s in mgr.status()}
    assert st == {"m1": True, "m2": True}
    assert all(s.stops == 0 for s in created)  # 退避は起きていない


def _wait_instances(mgr, model, n, timeout=2.0):
    """指定モデルの ready インスタンス数が n 以上になるまで待つ（複製の非同期起動用）。"""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        mm = mgr._models.get(model)
        if mm and sum(1 for i in mm.instances if i.ready and i.server is not None) >= n:
            return True
        time.sleep(0.01)
    return False


def test_no_replica_for_sequential_requests(monkeypatch):
    # 逐次リクエスト（常に release で inflight 0 に戻る）は満杯にならず、複製しない。
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=3)
    for _ in range(3):
        _, h = mgr.acquire("m1")
        mgr.release(h)
    import time
    time.sleep(0.1)  # 複製スレッドが走る猶予（走らないはず）
    assert len(mgr._models["m1"].instances) == 1
    assert sum(1 for _ in created) == 1


def test_no_replica_for_release_race(monkeypatch):
    # 逐次クライアントのフェーズ境界レース: ストリーミングのクライアントは [DONE] を
    # 受けた直後に次のリクエストを送るため、前リクエストのゲートウェイ側 release より
    # 数 ms 早く次が届き「満杯」に見えて複製がトリガーされ得る。猶予後の再確認で
    # 競合が持続していなければ (前リクエストが解放済みなら) 複製しないこと。
    import time
    monkeypatch.setattr(gw, "_REPLICA_GRACE_S", 0.2)  # テスト高速化
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=2)
    _, h1 = mgr.acquire("m1")     # 生成中 (inflight=1, 満杯)
    _, h2 = mgr.acquire("m1")     # release の直前に次ターンが到着 → 複製トリガー
    mgr.release(h1)               # 数 ms 後に前リクエストが解放される (レース解消)
    assert not _wait_instances(mgr, "m1", 2, timeout=0.8)  # 猶予後も複製されない
    assert len(mgr._models["m1"].instances) == 1
    assert sum(1 for _ in created) == 1
    mgr.release(h2)


def test_replica_spawns_on_concurrent_load(monkeypatch):
    # 同一モデルに同時リクエストが集中して満杯になると、複製インスタンスが起動して並列化する。
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=2)  # mlx: capacity=1
    _, h1 = mgr.acquire("m1")   # inst1 へ（inflight=1、満杯）
    _, h2 = mgr.acquire("m1")   # 満杯のまま2本目 → 複製起動をトリガ（今は inst1 へ）
    assert _wait_instances(mgr, "m1", 2)          # 裏で inst2 が起動する
    _, h3 = mgr.acquire("m1")   # 3本目は空いた inst2 へ（負荷分散）
    insts = mgr._models["m1"].instances
    assert len(insts) == 2
    assert sorted(i.inflight for i in insts) == [1, 2]  # inst1=2, inst2=1
    assert sum(1 for _ in created) == 2            # プロセスが 2 つ起動している
    mgr.release(h1); mgr.release(h2); mgr.release(h3)


def test_replica_capped_by_max_resident(monkeypatch):
    # 上限に達していてアイドルも無ければ、満杯でも複製しない（busy は止めない）。
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=1)
    _, h1 = mgr.acquire("m1")
    _, h2 = mgr.acquire("m1")   # 満杯だが上限 1・退避できるアイドル無し → 複製不可
    import time
    time.sleep(0.2)  # 複製スレッドが走る猶予
    assert len(mgr._models["m1"].instances) == 1
    assert sum(1 for _ in created) == 1
    mgr.release(h1); mgr.release(h2)


def test_replica_respects_llama_parallel_slots(monkeypatch):
    # llama-cpp は 1 プロセス内の parallel スロットを使い切ってから複製する（メモリ効率優先）。
    created = _patch_fake(monkeypatch)
    cfgs = [ServerConfig(backend="llama-cpp", model="L", host="127.0.0.1",
                         port=9001, parallel=3)]
    mgr = gw.ModelManager(cfgs, max_resident=2)
    hs = [mgr.acquire("L")[1] for _ in range(3)]  # 3 本は inst1 の parallel スロット内
    import time
    time.sleep(0.15)
    assert len(mgr._models["L"].instances) == 1   # まだ複製しない
    h4 = mgr.acquire("L")[1]                       # 4 本目で満杯 → 複製
    assert _wait_instances(mgr, "L", 2)
    assert sum(1 for _ in created) == 2
    for h in hs:
        mgr.release(h)
    mgr.release(h4)


def test_load_gateway_config_api_key(tmp_path):
    # api_key を読み取る。省略や空文字は None（認証なし）。
    base = '[[models]]\nmodel = "x"\nbackend = "mlx"\n'
    assert gw.load_gateway_config(_write(tmp_path, base)).api_key is None
    assert gw.load_gateway_config(
        _write(tmp_path, 'api_key = "s3cret"\n' + base)).api_key == "s3cret"
    assert gw.load_gateway_config(
        _write(tmp_path, 'api_key = "  "\n' + base)).api_key is None  # 空白のみ→無効


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
    # 全体終了は graceful を待たず即 SIGKILL（grace=0）で畳む（quit を速くするため）。
    assert all(s.last_grace == 0.0 for s in created)
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


def _post(port, path, obj, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    conn.request("POST", path, json.dumps(obj), hdrs)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    return resp.status, json.loads(body)


def _get(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    return resp.status, json.loads(body)


def _start_gateway(monkeypatch, api_key=None):
    # 上流は実フェイクサーバー。LocalServer は no-op（既に上流が動いている）に差し替える。
    up1 = _make_upstream("m1-upstream")
    up2 = _make_upstream("m2-upstream")
    monkeypatch.setattr(gw, "LocalServer", lambda config, log_path=None: _FakeServer(config, log_path))
    configs = [
        ServerConfig(backend="mlx", model="m1", host="127.0.0.1", port=up1.server_address[1]),
        ServerConfig(backend="mlx", model="m2", host="127.0.0.1", port=up2.server_address[1]),
    ]
    mgr = gw.ModelManager(configs)
    server = gw.GatewayServer(("127.0.0.1", 0), mgr, catalog=["m1", "m2"], api_key=api_key)
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
    # /v1/models は標準どおり「事前登録カタログ＋ロード中」のみ（発見一覧は TUI 専用）。
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


def test_local_connect_host():
    # ワイルドカード bind は自己接続をループバックに寄せる。特定 IP はそのまま。
    from local_llm_server.server import local_connect_host
    assert local_connect_host("0.0.0.0") == "127.0.0.1"
    assert local_connect_host("::") == "127.0.0.1"
    assert local_connect_host("") == "127.0.0.1"
    assert local_connect_host("192.168.1.5") == "192.168.1.5"
    assert local_connect_host("127.0.0.1") == "127.0.0.1"


def test_gateway_api_key_required_for_chat_and_models(monkeypatch):
    # api_key 設定時、chat と /v1/models は Authorization: Bearer <key> が要る（無/不一致は 401）。
    server, mgr, ups = _start_gateway(monkeypatch, api_key="secret")
    try:
        port = server.server_address[1]
        chat = {"model": "m1", "messages": [{"role": "user", "content": "hi"}]}
        # キー無し → 401
        status, obj = _post(port, "/v1/chat/completions", chat)
        assert status == 401 and "error" in obj
        # 不一致 → 401
        status, _ = _post(port, "/v1/chat/completions", chat,
                          headers={"Authorization": "Bearer wrong"})
        assert status == 401
        # 正しいキー → 通る
        status, obj = _post(port, "/v1/chat/completions", chat,
                            headers={"Authorization": "Bearer secret"})
        assert status == 200 and obj["backend"] == "m1-upstream"
        # /v1/models もキーが要る
        assert _get(port, "/v1/models")[0] == 401
        status, obj = _get(port, "/v1/models", headers={"Authorization": "Bearer secret"})
        assert status == 200
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_admin_status_reports_live_model_state():
    """/admin/status が常駐状態（loaded/inflight）＋運用方針を返す。"""
    from local_llm_server.server import gateway_admin_status

    cfgs = [
        ServerConfig(backend="mlx-vlm", model="org/A", host="127.0.0.1", port=9001),
        ServerConfig(backend="mlx", model="org/B", host="127.0.0.1", port=9002),
    ]
    mgr = gw.ModelManager(cfgs, max_resident=1, load_timeout=300)
    # org/A をロード済み・処理中 2 件に見立てる（実サーバーは起動しない）。
    a = mgr._models["org/A"]
    a.instances.append(
        gw._Instance(config=a.config, server=object(), ready=True, inflight=2)
    )
    srv = gw.GatewayServer(
        ("127.0.0.1", 0), mgr, catalog=["org/A", "org/B"],
        default_model=None, max_resident=1, idle_timeout=1200, load_timeout=300,
    )
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        data = gateway_admin_status("127.0.0.1", port)
    finally:
        srv.shutdown()
        srv.server_close()

    assert data is not None
    assert data["object"] == "gateway.status"
    assert data["max_resident"] == 1
    assert data["idle_timeout"] == 1200
    assert data["uptime"] >= 0                 # 起動経過（秒）
    assert data["requests"] == 0               # acquire を通していないので 0
    by_model = {m["model"]: m for m in data["models"]}
    assert by_model["org/A"]["loaded"] is True
    assert by_model["org/A"]["inflight"] == 2
    assert by_model["org/A"]["requests"] == 0
    assert "idle_for" in by_model["org/A"]     # 処理中なので None
    assert by_model["org/B"]["loaded"] is False


def test_gateway_admin_status_none_when_down():
    """応答が無ければ None（TUI は server_status にフォールバックできる）。"""
    from local_llm_server.server import gateway_admin_status

    # 使われていないであろうポート。urlopen が失敗して None。
    assert gateway_admin_status("127.0.0.1", 6, timeout=0.5) is None


def test_gateway_admin_is_loopback_and_keyless(monkeypatch):
    # api_key 設定時でも、/admin/status・/admin/config はローカル（=テストは 127.0.0.1）から
    # キー無しで使える（ループバック限定・キーではなく接続元で保護）。
    server, mgr, ups = _start_gateway(monkeypatch, api_key="secret")
    try:
        port = server.server_address[1]
        status, obj = _get(port, "/admin/status")            # キー無しでも 200
        assert status == 200 and obj["object"] == "gateway.status"
        status, obj = _post(port, "/admin/config", {"max_resident": 2})
        assert status == 200 and obj["max_resident"] == 2
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_no_auth_when_key_unset(monkeypatch):
    # api_key 未設定なら従来どおり認証なし（後方互換）。
    server, mgr, ups = _start_gateway(monkeypatch)  # api_key=None
    try:
        port = server.server_address[1]
        status, obj = _post(port, "/v1/chat/completions",
                            {"model": "m1", "messages": []})
        assert status == 200 and obj["backend"] == "m1-upstream"
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


def test_gateway_admin_config_updates_max_resident(monkeypatch):
    # POST /admin/config で稼働中に max_resident を変更でき、/admin/status に即反映される。
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        _, st = _get(port, "/admin/status")
        assert st["max_resident"] is None            # 起動時は無制限
        status, obj = _post(port, "/admin/config", {"max_resident": 2})
        assert status == 200 and obj["max_resident"] == 2
        assert mgr._max_resident == 2
        _, st = _get(port, "/admin/status")
        assert st["max_resident"] == 2               # 表示にも反映
        status, obj = _post(port, "/admin/config", {"max_resident": None})
        assert status == 200 and obj["max_resident"] is None   # null → 無制限に戻す
        assert mgr._max_resident is None
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_admin_config_validates_input(monkeypatch):
    # 0 / off は無制限扱い。負数・非数値・キー無しは 400。
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        status, obj = _post(port, "/admin/config", {"max_resident": 0})
        assert status == 200 and obj["max_resident"] is None
        status, obj = _post(port, "/admin/config", {"max_resident": "off"})
        assert status == 200 and obj["max_resident"] is None
        status, obj = _post(port, "/admin/config", {"max_resident": -3})
        assert status == 400 and "error" in obj
        status, obj = _post(port, "/admin/config", {"max_resident": "abc"})
        assert status == 400 and "error" in obj
        status, obj = _post(port, "/admin/config", {})
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


# --- 今回の修正の回帰テスト ---------------------------------------------------

def test_load_gateway_config_request_and_start_timeout(tmp_path):
    base = '[[models]]\nmodel = "x"\nbackend = "mlx"\n'
    cfg = gw.load_gateway_config(_write(tmp_path, base))
    assert cfg.request_timeout == 600.0     # 既定: 600 秒（沈黙した上流が枠を握り続けるのを防ぐ保険）
    assert cfg.start_timeout == 120.0       # 既定: 120 秒
    cfg = gw.load_gateway_config(
        _write(tmp_path, "request_timeout = 900\nstart_timeout = 300\n" + base))
    assert cfg.request_timeout == 900.0
    assert cfg.start_timeout == 300.0
    # 0 は「無効（無制限）」
    assert gw.load_gateway_config(
        _write(tmp_path, "request_timeout = 0\n" + base)).request_timeout is None
    with pytest.raises(ValueError, match="start_timeout"):
        gw.load_gateway_config(_write(tmp_path, "start_timeout = 0\n" + base))


# --- ワーカー健全性チェック / 孤児回収 -----------------------------------------

class _HealthStub:
    """LocalServer 差し替え。is_alive / pid / stop を持ち、生死を制御できる。"""

    def __init__(self, pid=1234, alive=True):
        self._pid = pid
        self._alive = alive
        self.stops = 0

    @property
    def pid(self):
        return self._pid

    def is_alive(self):
        return self._alive

    def stop(self, grace: float = 10.0):
        self.stops += 1


def _install_instance(mgr, model, *, alive=True, inflight=0, dynamic=False, port=9000):
    srv = _HealthStub(alive=alive)
    cfg = ServerConfig(backend="mlx-vlm", model=model, port=port)
    inst = gw._Instance(config=cfg, server=srv, ready=True, inflight=inflight)
    mgr._models[model] = gw._Model(config=cfg, instances=[inst], dynamic=dynamic)
    return srv


def test_reap_dead_instances_removes_only_dead():
    mgr = gw.ModelManager([], dynamic=True)
    dead = _install_instance(mgr, "crashed", alive=False, port=9001)
    live = _install_instance(mgr, "healthy", alive=True, port=9002)
    n = mgr.reap_dead_instances()
    assert n == 1
    assert mgr._models["crashed"].instances == []   # 死んだインスタンスは外れる
    assert dead.stops == 1                            # stop でログ fd 等を掃除
    assert len(mgr._models["healthy"].instances) == 1  # 生きているものは残る
    assert live.stops == 0


def test_reap_dead_instances_drops_empty_dynamic_model():
    # クラッシュで空になった動的モデルは登録ごと消える（表示から外れる）。
    mgr = gw.ModelManager([], dynamic=True)
    _install_instance(mgr, "dyn/crashed", alive=False, dynamic=True, port=9001)
    assert mgr.reap_dead_instances() == 1
    assert "dyn/crashed" not in mgr._models


def test_reap_dead_instances_reaps_even_with_inflight():
    # プロセスが死んでいれば inflight>0 でも外す（もう進まない。枠を戻す）。
    mgr = gw.ModelManager([], dynamic=True)
    _install_instance(mgr, "crashed", alive=False, inflight=3, port=9001)
    assert mgr.reap_dead_instances() == 1
    assert mgr._models["crashed"].instances == []


def test_status_includes_worker_pids():
    mgr = gw.ModelManager([], dynamic=True)
    _install_instance(mgr, "m", alive=True, port=9001)
    st = {row["model"]: row for row in mgr.status()}
    assert st["m"]["pids"] == [1234]


def test_reclaim_stale_workers_kills_only_ours(monkeypatch):
    from local_llm_server import server as srv_mod
    killed = []
    monkeypatch.setattr(srv_mod, "find_pids_on_port", lambda port: [111, 222, 333])
    # 111/333 は our-worker、222 は無関係。
    monkeypatch.setattr(srv_mod, "pid_looks_like_ours", lambda pid: pid in (111, 333))

    def _stop(pid, timeout=10.0):
        killed.append(pid)
        return True

    monkeypatch.setattr(srv_mod, "stop_pid", _stop)
    reclaimed = srv_mod.reclaim_stale_workers(9001)
    assert reclaimed == [111, 333]   # 無関係な 222 には手を出さない
    assert killed == [111, 333]


# --- 画像入りリクエストの vision_model への振り分け ----------------------------
# 一部の vision モデル（Qwen3.6-27B/qwen3_5 等）は現行 mlx_vlm で画像入力が壊れている。
# vision_model を設定すると、画像を含むリクエストだけをそのモデル（gemma-4 系など、画像が確実に
# 動くもの）へ振り分ける。テキストは元モデルのまま。

def test_request_has_images_detects_vision_content():
    assert gw._request_has_images({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    }) is True
    assert gw._request_has_images({
        "messages": [{"role": "user", "content": [{"type": "input_image", "image": "x"}]}]
    }) is True
    assert gw._request_has_images({"images": ["data:image/png;base64,AAAA"]}) is True


def test_request_has_images_false_for_text_only():
    assert gw._request_has_images(
        {"messages": [{"role": "user", "content": "hi"}]}) is False
    assert gw._request_has_images(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}) is False
    assert gw._request_has_images({}) is False


def test_load_gateway_config_parses_vision_model(tmp_path):
    cfg = gw.load_gateway_config(_write(tmp_path, 'vision_model = "org/gemma-vision"\n'))
    assert cfg.vision_model == "org/gemma-vision"
    assert gw.load_gateway_config(_write(tmp_path, "port = 8080\n")).vision_model is None
    # 空文字は None 扱い。
    assert gw.load_gateway_config(_write(tmp_path, 'vision_model = ""\n')).vision_model is None


def _start_routing_gateway(monkeypatch):
    """text-model（MTP 有り）と vision-model を持ち、vision_model を設定したゲートウェイ。"""
    up_text = _make_upstream("text-up")
    up_vision = _make_upstream("vision-up")
    monkeypatch.setattr(gw, "LocalServer",
                        lambda config, log_path=None: _FakeServer(config, log_path))
    configs = [
        ServerConfig(backend="mlx-vlm", model="text-model", host="127.0.0.1",
                     port=up_text.server_address[1], draft_model="org/draft"),
        ServerConfig(backend="mlx-vlm", model="vision-model", host="127.0.0.1",
                     port=up_vision.server_address[1]),
    ]
    mgr = gw.ModelManager(configs, dynamic=False)
    server = gw.GatewayServer(("127.0.0.1", 0), mgr, catalog=["text-model", "vision-model"],
                              vision_model="vision-model")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, mgr, (up_text, up_vision)


def test_image_request_routed_to_vision_model(monkeypatch):
    server, mgr, ups = _start_routing_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        # テキストは元モデルへ（振り分けない）。
        s, o = _post(port, "/v1/chat/completions",
                     {"model": "text-model", "messages": [{"role": "user", "content": "hi"}]})
        assert s == 200 and o["backend"] == "text-up"
        # 画像入りは vision-model へ振り分け。
        img = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
        s, o = _post(port, "/v1/chat/completions", {"model": "text-model", "messages": img})
        assert s == 200 and o["backend"] == "vision-up"
        # 実際に vision-model がロードされ、text-model は画像では起動していない。
        assert mgr._models["vision-model"].instances
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_image_request_to_vision_model_itself_not_rerouted(monkeypatch):
    # 既に vision_model 宛のリクエストは二度振り分けしない（自分自身へループしない）。
    server, mgr, ups = _start_routing_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        img = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
        s, o = _post(port, "/v1/chat/completions", {"model": "vision-model", "messages": img})
        assert s == 200 and o["backend"] == "vision-up"
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_load_gateway_config_rejects_unsupported_wildcards(tmp_path):
    # ゲートウェイは IPv4 で bind する。"::" / "*" は bind 時の分かりにくい OSError ではなく
    # 設定読み込みで明確に断る。
    base = '[[models]]\nmodel = "x"\nbackend = "mlx"\n'
    for host in ("::", "*"):
        with pytest.raises(ValueError, match="not supported"):
            gw.load_gateway_config(_write(tmp_path, f'host = "{host}"\n' + base))
    # "0.0.0.0" は従来どおり許可
    assert gw.load_gateway_config(
        _write(tmp_path, 'host = "0.0.0.0"\n' + base)).host == "0.0.0.0"


def test_no_replica_when_no_limits_configured(monkeypatch):
    # max_resident もメモリ上限も無い構成では複製しない（重みのコピーが際限なく増えて
    # OOM する事故を防ぐ。並列化を使うにはどちらかで範囲を決める）。
    created = _patch_fake(monkeypatch)
    mgr = gw.ModelManager(_configs(), max_resident=None)  # 上限なし
    _, h1 = mgr.acquire("m1")
    _, h2 = mgr.acquire("m1")   # 満杯（mlx: capacity=1）だが上限が無い → 複製しない
    import time
    time.sleep(0.2)  # 複製スレッドが走る猶予（走らないはず）
    assert len(mgr._models["m1"].instances) == 1
    assert sum(1 for _ in created) == 1
    mgr.release(h1); mgr.release(h2)


def test_shutdown_stops_server_that_is_still_starting(monkeypatch):
    # ロード中（wait_until_ready の途中）に shutdown しても、起動しかけのモデルサーバーを
    # 取り逃さず止める（孤児プロセス化してメモリ・ポートを掴み続けない）。
    import time

    started_loading = threading.Event()
    release_load = threading.Event()
    created = []

    class _SlowServer(_FakeServer):
        def __init__(self, config, log_path=None):
            super().__init__(config, log_path)
            created.append(self)

        def wait_until_ready(self, *a, **k):
            started_loading.set()
            release_load.wait(timeout=5)

    monkeypatch.setattr(gw, "LocalServer", _SlowServer)
    mgr = gw.ModelManager(_configs())
    errors = []

    def _load():
        try:
            mgr.acquire("m1")
        except RuntimeError as exc:  # shutting down は正当な失敗
            errors.append(exc)

    t = threading.Thread(target=_load)
    t.start()
    assert started_loading.wait(timeout=5)  # ロード（wait_until_ready）中
    mgr.shutdown()                          # ← この時点で instances は空だが _starting にいる
    release_load.set()
    t.join(timeout=5)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and created[0].stops == 0:
        time.sleep(0.01)
    assert created[0].stops >= 1            # 起動途中のサーバーも止まっている
    assert errors                            # ロード側は「shutting down」で中断される
    # shutdown 後の新規ロードは拒否される
    with pytest.raises(RuntimeError):
        mgr.acquire("m2")


def test_gateway_non_ascii_bearer_token_returns_401(monkeypatch):
    # 非 ASCII のトークンでも 500（TypeError）にならず、きれいに 401 を返す。
    server, mgr, ups = _start_gateway(monkeypatch, api_key="secret")
    try:
        port = server.server_address[1]
        status, obj = _get(port, "/v1/models",
                           headers={"Authorization": "Bearer café"})
        assert status == 401 and "error" in obj
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_rejects_non_dict_json_body(monkeypatch):
    # [1] や "x" のような dict 以外の JSON は 500（AttributeError）ではなく 400。
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        status, obj = _post(port, "/v1/chat/completions", [1])
        assert status == 400 and "error" in obj
        # model が文字列でないのも 400
        status, obj = _post(port, "/v1/chat/completions", {"model": ["m1"]})
        assert status == 400 and "error" in obj
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_routes_with_query_string(monkeypatch):
    # クエリ付きでもルーティングが壊れない（GET /v1/models?limit=10 → 200）。
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        status, obj = _get(port, "/v1/models?limit=10")
        assert status == 200
        assert [m["id"] for m in obj["data"]] == ["m1", "m2"]
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_invalid_content_length_returns_400(monkeypatch):
    # Content-Length が数値でない/負でも 500 にならず 400 を返す。
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        for bad in ("abc", "-5"):
            conn = http.client.HTTPConnection("127.0.0.1", port)
            conn.putrequest("POST", "/v1/chat/completions")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", bad)
            conn.endheaders()
            resp = conn.getresponse()
            assert resp.status == 400
            conn.close()
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()


def test_gateway_oversized_content_length_returns_413(monkeypatch):
    # 巨大 Content-Length の申告は本文を読み込まず 413 で断る（メモリ DoS 防止）。
    server, mgr, ups = _start_gateway(monkeypatch)
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(10 * 1024 * 1024 * 1024))  # 10GB を申告
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
        conn.close()
    finally:
        server.shutdown(); server.server_close(); mgr.shutdown()
        for u in ups:
            u.shutdown(); u.server_close()
