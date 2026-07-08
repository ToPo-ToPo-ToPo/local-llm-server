"""ゲートウェイの STT ルーティング（multipart/form-data）を HTTP レベルで検証する。

chat（JSON）と違い body は multipart。ゲートウェイは form の `model` を見て振り分け、
multipart 本文を（境界を保ったまま）担当インスタンスへ中継する必要がある。
"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import local_llm_server.daemon as gw
from local_llm_server import multipart


class _EchoUpstream(BaseHTTPRequestHandler):
    """受け取った multipart から model を読み、そのまま JSON で返すダミー STT サーバ。"""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")
        got_model = multipart.field(body, ctype, "model")
        has_file = any(p.filename for p in multipart.parse(body, ctype))
        payload = json.dumps({
            "path": self.path,
            "got_model": got_model,
            "has_file": has_file,
            "content_type": ctype,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_a):
        pass


class _FakeManager:
    """acquire された model を記録し、echo 上流の addr を返すだけのフェイク。"""

    def __init__(self, addr: tuple[str, int]) -> None:
        self._addr = addr
        self.acquired: list[str] = []
        self.model_ids: list[str] = []

    def acquire(self, model: str):
        self.acquired.append(model)
        return self._addr, object()

    def release(self, _handle) -> None:
        pass


def _multipart_body(boundary: str, fields: dict, file_field: str | None = "file") -> bytes:
    out = b""
    for name, value in fields.items():
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n").encode()
    if file_field:
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{file_field}"; filename="a.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n").encode()
        out += b"RIFF\x00\x01\x02audio\r\ndata" + f"\r\n--{boundary}--\r\n".encode()
    else:
        out += f"--{boundary}--\r\n".encode()
    return out


def _run_gateway(default_model=None):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _EchoUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    mgr = _FakeManager(("127.0.0.1", upstream.server_address[1]))
    server = gw.GatewayServer(
        ("127.0.0.1", 0), mgr, catalog=[], default_model=default_model,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return upstream, server, mgr


def _post(port: int, path: str, body: bytes, boundary: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=body,
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode())
    conn.close()
    return resp.status, data


def test_transcriptions_routes_by_multipart_model():
    upstream, server, mgr = _run_gateway()
    try:
        b = "BND1"
        body = _multipart_body(b, {"model": "mlx-community/whisper-tiny", "language": "en"})
        status, data = _post(server.server_address[1], "/v1/audio/transcriptions", body, b)
        assert status == 200
        # ゲートウェイは form の model で acquire した。
        assert mgr.acquired == ["mlx-community/whisper-tiny"]
        # 上流には multipart が壊れず（境界・file 付きで）中継された。
        assert data["got_model"] == "mlx-community/whisper-tiny"
        assert data["has_file"] is True
        assert data["path"] == "/v1/audio/transcriptions"
        assert "multipart/form-data" in data["content_type"]
    finally:
        server.shutdown(); server.server_close()
        upstream.shutdown(); upstream.server_close()


def test_translations_endpoint_also_routes():
    upstream, server, mgr = _run_gateway()
    try:
        b = "BND2"
        body = _multipart_body(b, {"model": "org/whisper-x"})
        status, _ = _post(server.server_address[1], "/v1/audio/translations", body, b)
        assert status == 200
        assert mgr.acquired == ["org/whisper-x"]
    finally:
        server.shutdown(); server.server_close()
        upstream.shutdown(); upstream.server_close()


def test_falls_back_to_default_model_when_no_model_field():
    upstream, server, mgr = _run_gateway(default_model="org/default-whisper")
    try:
        b = "BND3"
        body = _multipart_body(b, {"language": "en"})  # model フィールド無し
        status, data = _post(server.server_address[1], "/v1/audio/transcriptions", body, b)
        assert status == 200
        assert mgr.acquired == ["org/default-whisper"]
    finally:
        server.shutdown(); server.server_close()
        upstream.shutdown(); upstream.server_close()


def test_no_model_and_no_default_is_400():
    upstream, server, mgr = _run_gateway(default_model=None)
    try:
        b = "BND4"
        body = _multipart_body(b, {"language": "en"})
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("POST", "/v1/audio/transcriptions", body=body,
                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        resp = conn.getresponse()
        resp.read(); conn.close()
        assert resp.status == 400
        assert mgr.acquired == []
    finally:
        server.shutdown(); server.server_close()
        upstream.shutdown(); upstream.server_close()
