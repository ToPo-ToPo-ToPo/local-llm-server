"""同梱の Rerank サーバ（rerank_server）。モデルは読まない（偽のバックエンド）。"""
from __future__ import annotations

import http.client
import json
import threading

import pytest

from local_llm_server import rerank_server


class _FakeBackend:
    """問いの語が文に含まれる割合を関連度にする（モデル無し）。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def load(self):
        pass

    def score(self, query, documents, instruction=None):
        self.calls.append((query, list(documents), instruction))
        words = query.lower().split()
        scores = [sum(w in d.lower() for w in words) / max(len(words), 1) for d in documents]
        return scores, sum(len(d) for d in documents)


@pytest.fixture
def server():
    backend = _FakeBackend()
    srv = rerank_server._Server(("127.0.0.1", 0), "org/fake-reranker", backend=backend)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, backend
    srv.shutdown()
    srv.server_close()


def _post(port, payload, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = raw if raw is not None else json.dumps(payload).encode()
    conn.request("POST", "/v1/rerank", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode())
    conn.close()
    return resp.status, data


def test_results_are_sorted_by_relevance_with_original_indexes(server):
    srv, backend = server
    status, data = _post(srv.server_address[1], {
        "model": "org/fake-reranker", "query": "lattice stiffness",
        "documents": ["about pumps", "lattice stiffness study", {"text": "lattice only"}],
    })
    assert status == 200 and data["model"] == "org/fake-reranker"
    assert [r["index"] for r in data["results"]] == [1, 2, 0]
    assert data["results"][0]["relevance_score"] == 1.0 and data["results"][2]["relevance_score"] == 0.0
    assert "document" not in data["results"][0] and data["usage"]["total_tokens"] > 0
    assert backend.calls[0][1] == ["about pumps", "lattice stiffness study", "lattice only"]


def test_top_n_and_return_documents_and_instruction(server):
    srv, backend = server
    status, data = _post(srv.server_address[1], {
        "query": "lattice", "documents": ["a", "lattice b", "lattice c"], "top_n": 2,
        "return_documents": True, "instruction": "Find the passage about lattices",
    })
    assert status == 200 and len(data["results"]) == 2
    assert data["results"][0]["document"] == {"text": "lattice b"}
    assert backend.calls[-1][2] == "Find the passage about lattices"


def test_bad_requests_are_rejected_before_the_backend(server):
    srv, backend = server
    port = srv.server_address[1]
    assert _post(port, {"documents": ["a"]})[0] == 400
    assert _post(port, {"query": "q", "documents": []})[0] == 400
    assert _post(port, {"query": "q", "documents": [1, 2]})[0] == 400
    assert _post(port, {"query": "q", "documents": ["a", " "]})[0] == 400
    assert _post(port, {"query": "q", "documents": ["a"], "top_n": -1})[0] == 400
    assert _post(port, {"query": "q", "documents": ["a"], "instruction": 5})[0] == 400
    assert _post(port, None, raw=b"[1]")[0] == 400
    assert backend.calls == []
    with pytest.raises(ValueError):
        rerank_server.parse_request({"query": "q", "documents": ["x"] * (rerank_server._MAX_DOCUMENTS + 1)})


def test_models_endpoint_and_unknown_paths(server):
    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    conn.request("GET", "/v1/models")
    resp = conn.getresponse()
    assert resp.status == 200 and json.loads(resp.read())["data"][0]["id"] == "org/fake-reranker"
    conn.request("POST", "/v1/embeddings", body=b"{}", headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_body_limit_rejected_before_reading_payload(server):
    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=3)
    conn.putrequest("POST", "/v1/rerank")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(rerank_server._MAX_BODY_BYTES + 1))
    conn.endheaders()
    response = conn.getresponse()
    assert response.status == 413
    response.read()
    conn.close()
