"""ゲートウェイの Embeddings ルーティング（JSON の model で振り分け、本文をそのまま中継）。"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import local_llm_server.daemon as gw


class _EmbedUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length))
        _EmbedUpstream.seen.append((self.path, payload))
        body = json.dumps({
            "object": "list", "model": payload.get("model"),
            "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i in range(len(payload["input"]))],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a):
        pass


class _FakeManager:
    def __init__(self, addr):
        self._addr = addr
        self.acquired: list[str] = []
        self.model_ids: list[str] = []

    def acquire(self, model: str):
        self.acquired.append(model)
        return self._addr, object()

    def release(self, _handle) -> None:
        pass

    def backend_for(self, model: str) -> str:
        return "embed"


def test_embeddings_route_by_json_model_and_pass_the_body_through():
    _EmbedUpstream.seen = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _EmbedUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    mgr = _FakeManager(("127.0.0.1", upstream.server_address[1]))
    server = gw.GatewayServer(("127.0.0.1", 0), mgr, catalog=[], default_model=None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = {"model": "google/embeddinggemma-2", "input": ["title: none | text: a", "title: none | text: b"],
               "dimensions": 256}
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("POST", "/v1/embeddings", body=json.dumps(req).encode(),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        assert resp.status == 200
        assert mgr.acquired == ["google/embeddinggemma-2"]
        assert len(data["data"]) == 2 and data["model"] == "google/embeddinggemma-2"
        path, payload = _EmbedUpstream.seen[0]
        assert path.endswith("/v1/embeddings") and payload == req  # 生成向けの注入（repetition_penalty 等）は付かない
    finally:
        server.shutdown()
        server.server_close()
        upstream.shutdown()
        upstream.server_close()
