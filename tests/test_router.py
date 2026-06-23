import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm_server.router import RouterServer, needs_vision


def test_needs_vision_text_only():
    assert not needs_vision([{"role": "user", "content": "hello"}])
    assert not needs_vision([{"role": "user", "content": [{"type": "text", "text": "x"}]}])
    assert not needs_vision("not a list")
    assert not needs_vision(None)


def test_needs_vision_with_image():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "これは?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]
    assert needs_vision(messages)


def test_needs_vision_any_nontext_part():
    # 将来のメディア（音声/動画など text 以外のパート）も vision 扱い
    messages = [{"role": "user", "content": [{"type": "input_audio", "data": "x"}]}]
    assert needs_vision(messages)


def _make_upstream(name: str) -> ThreadingHTTPServer:
    """自分の名前を返すだけのフェイク OpenAI 互換上流サーバー。"""

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

        def do_GET(self):
            self._send({"backend": name, "path": self.path, "method": "GET"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self._send({"backend": name, "path": self.path, "method": "POST"})

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


def _start_router():
    text = _make_upstream("text")
    vision = _make_upstream("vision")
    router = RouterServer(
        ("127.0.0.1", 0),
        ("127.0.0.1", text.server_address[1]),
        ("127.0.0.1", vision.server_address[1]),
    )
    threading.Thread(target=router.serve_forever, daemon=True).start()
    return router, text, vision


def test_router_text_request_goes_to_text():
    router, text, vision = _start_router()
    try:
        port = router.server_address[1]
        status, obj = _post(
            port, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "こんにちは"}]},
        )
        assert status == 200
        assert obj["backend"] == "text"
        assert obj["path"] == "/v1/chat/completions"
    finally:
        for s in (router, text, vision):
            s.shutdown()
            s.server_close()


def test_router_image_request_goes_to_vision():
    router, text, vision = _start_router()
    try:
        port = router.server_address[1]
        status, obj = _post(
            port, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": [
                {"type": "text", "text": "見て"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]}]},
        )
        assert status == 200
        assert obj["backend"] == "vision"
    finally:
        for s in (router, text, vision):
            s.shutdown()
            s.server_close()


def test_router_get_models_goes_to_text():
    router, text, vision = _start_router()
    try:
        port = router.server_address[1]
        status, obj = _get(port, "/v1/models")
        assert status == 200
        assert obj["backend"] == "text"
        assert obj["method"] == "GET"
    finally:
        for s in (router, text, vision):
            s.shutdown()
            s.server_close()


def test_router_non_chat_post_goes_to_text():
    router, text, vision = _start_router()
    try:
        port = router.server_address[1]
        status, obj = _post(port, "/v1/embeddings", {"input": "x"})
        assert status == 200
        assert obj["backend"] == "text"
    finally:
        for s in (router, text, vision):
            s.shutdown()
            s.server_close()


def test_router_upstream_unreachable_returns_502():
    # 上流を立てずにルーターだけ起動 → 502 を返す
    router = RouterServer(
        ("127.0.0.1", 0), ("127.0.0.1", 1), ("127.0.0.1", 2)
    )
    threading.Thread(target=router.serve_forever, daemon=True).start()
    try:
        port = router.server_address[1]
        status, obj = _post(
            port, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 502
        assert "error" in obj
    finally:
        router.shutdown()
        router.server_close()
