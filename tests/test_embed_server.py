"""同梱の Embeddings サーバ（embed_server）。モデルは読まない（偽のバックエンド）。"""
from __future__ import annotations

import base64
import http.client
import json
import math
import threading

import pytest

from local_llm_server import embed_server


class _FakeBackend:
    """文の長さだけで決まる 4 次元のベクトルを返す（モデル無し）。"""

    def __init__(self):
        self.dim = 4
        self.calls: list[list[str]] = []

    def load(self):
        pass

    def embed(self, texts):
        self.calls.append(list(texts))
        vecs = []
        for t in texts:
            raw = [float(len(t)), 1.0, 0.5, 0.25]
            n = math.sqrt(sum(x * x for x in raw))
            vecs.append([x / n for x in raw])
        return vecs, sum(len(t) for t in texts)


@pytest.fixture
def server():
    backend = _FakeBackend()
    srv = embed_server._Server(("127.0.0.1", 0), "org/fake-embedding", backend=backend)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, backend
    srv.shutdown()
    srv.server_close()


def _post(port: int, path: str, payload, raw: bytes | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = raw if raw is not None else json.dumps(payload).encode()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode())
    conn.close()
    return resp.status, data


def test_models_endpoint_lists_the_loaded_model(server):
    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    conn.request("GET", "/v1/models")
    resp = conn.getresponse()
    data = json.loads(resp.read().decode())
    conn.close()
    assert resp.status == 200 and data["data"][0]["id"] == "org/fake-embedding"


def test_embeddings_follow_the_openai_shape(server):
    srv, backend = server
    status, data = _post(srv.server_address[1], "/v1/embeddings",
                         {"model": "org/fake-embedding", "input": ["abc", "defgh"]})
    assert status == 200
    assert data["object"] == "list" and data["model"] == "org/fake-embedding"
    assert [d["index"] for d in data["data"]] == [0, 1] and data["data"][0]["object"] == "embedding"
    assert len(data["data"][0]["embedding"]) == 4
    assert data["usage"] == {"prompt_tokens": 8, "total_tokens": 8}
    assert backend.calls == [["abc", "defgh"]]
    # 文字列 1 本でもよい
    status, data = _post(srv.server_address[1], "/v1/embeddings", {"input": "xyz"})
    assert status == 200 and len(data["data"]) == 1


def test_dimensions_truncate_and_renormalize(server):
    srv, _ = server
    status, data = _post(srv.server_address[1], "/v1/embeddings", {"input": ["abc"], "dimensions": 2})
    assert status == 200
    vec = data["data"][0]["embedding"]
    assert len(vec) == 2 and math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)
    # モデルの次元より大きい指定は断る
    status, data = _post(srv.server_address[1], "/v1/embeddings", {"input": ["abc"], "dimensions": 9})
    assert status == 400 and "dimensions" in data["error"]


def test_base64_encoding_format(server):
    srv, _ = server
    status, data = _post(srv.server_address[1], "/v1/embeddings",
                         {"input": ["abc"], "encoding_format": "base64"})
    assert status == 200
    import array

    raw = base64.b64decode(data["data"][0]["embedding"])
    vec = array.array("f")
    vec.frombytes(raw)
    assert len(vec) == 4 and math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-5)


def test_bad_requests_are_rejected_before_the_backend(server):
    srv, backend = server
    port = srv.server_address[1]
    assert _post(port, "/v1/embeddings", {"input": []})[0] == 400
    assert _post(port, "/v1/embeddings", {"input": [1, 2, 3]})[0] == 400  # token id は受けない
    assert _post(port, "/v1/embeddings", {"input": ["ok", "  "]})[0] == 400
    assert _post(port, "/v1/embeddings", {"input": "x", "dimensions": -1})[0] == 400
    assert _post(port, "/v1/embeddings", {"input": "x", "encoding_format": "int8"})[0] == 400
    assert _post(port, "/v1/embeddings", None, raw=b"[1,2]")[0] == 400
    assert _post(port, "/v1/embeddings", None, raw=b"{not json")[0] == 400
    assert _post(port, "/v1/chat/completions", {"input": "x"})[0] == 404
    assert backend.calls == []


def test_too_many_inputs_rejected():
    with pytest.raises(ValueError):
        embed_server.parse_request({"input": ["x"] * (embed_server._MAX_INPUTS + 1)})


def test_body_limit_rejected_before_reading_payload(server):
    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=3)
    conn.putrequest("POST", "/v1/embeddings")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(embed_server._MAX_BODY_BYTES + 1))
    conn.endheaders()
    response = conn.getresponse()
    assert response.status == 413
    response.read()
    conn.close()


def test_warmup_holds_same_lock_as_requests():
    entered = threading.Event()
    release = threading.Event()

    class _Slow(_FakeBackend):
        def load(self):
            entered.set()
            release.wait(2)

    srv = embed_server._Server(("127.0.0.1", 0), "org/fake", backend=_Slow())
    thread = threading.Thread(target=embed_server._warm, args=(srv,))
    thread.start()
    try:
        assert entered.wait(1)
        assert srv.lock.acquire(blocking=False) is False
    finally:
        release.set()
        thread.join(2)
        srv.server_close()
