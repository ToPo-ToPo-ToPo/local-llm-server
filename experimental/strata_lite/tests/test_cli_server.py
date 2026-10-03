from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from strata_lite.__main__ import main, plan
from strata_lite.engine import Engine
from strata_lite.server import normalize_messages, serve

CPU = ["--device", "cpu", "--dtype", "float32", "--kernel", "dequant", "--no-profile"]


def test_plan(converted):
    p = plan(converted, vram_gb=1.0, reserve_gb=0.0, context_gb=0.0)
    assert p["experts_total"] == 4 * 8
    assert p["gpu_slots"] == p["experts_total"]  # 小型モデルは全部載る
    assert p["active_experts_per_token"] == 4 * 2
    p0 = plan(converted, vram_gb=0.0, reserve_gb=0.0, context_gb=0.0)
    assert p0["gpu_slots"] == 0


def test_bench_compare(converted, tmp_path, capsys):
    out = tmp_path / "bench.json"
    main(["bench", converted, "--compare", "--tokens", "6", "--prompts", "2", "--warmup", "1",
          "--expert-vram-gb", "0.0001", "--json", str(out), *CPU])
    report = json.loads(out.read_text())
    assert [(r["placement"], r["miss"]) for r in report] == [("layer", "cpu"), ("hot", "cpu"), ("hot", "transfer")]
    assert all(r["decode_tps"] > 0 for r in report)
    assert "GPU ヒット率" in capsys.readouterr().out


def test_normalize_messages():
    msgs = normalize_messages([{"role": "user", "content": [{"type": "text", "text": "a"},
                                                            {"type": "image_url", "image_url": {}},
                                                            {"type": "text", "text": "b"}]}])
    assert msgs == [{"role": "user", "content": "ab"}]


@pytest.fixture(scope="module")
def server(converted):
    eng = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0.0001,
                 use_profile=False, log=lambda *_: None)
    httpd = serve(eng, "127.0.0.1", 0, "tiny")
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60)


def test_server_models_and_stats(server):
    with urllib.request.urlopen(server + "/v1/models") as r:
        assert json.load(r)["data"][0]["id"] == "tiny"
    with urllib.request.urlopen(server + "/v1/strata/stats") as r:
        st = json.load(r)
    assert st["placement"] == "hot" and "decode" in st


def test_server_chat(server):
    with _post(server + "/v1/chat/completions",
               {"model": "tiny", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}) as r:
        body = json.load(r)
    assert body["object"] == "chat.completion"
    assert body["usage"]["completion_tokens"] >= 1
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_server_stream(server):
    with _post(server + "/v1/chat/completions",
               {"model": "tiny", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4,
                "stream": True}) as r:
        lines = [ln.decode() for ln in r.read().splitlines() if ln.startswith(b"data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_server_bad_request(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server + "/v1/chat/completions", {"model": "tiny"})
    assert e.value.code == 400
