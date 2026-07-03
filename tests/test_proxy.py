"""proxy.forward のテスト（SSE ストリーミングの逐次中継を保証する）。"""
from __future__ import annotations

import http.client
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm_server.proxy import forward

# 上流が 1 チャンク送るごとに置く間隔（秒）。逐次中継なら最初のチャンクは
# 全チャンク送信完了（CHUNKS * INTERVAL 秒後）よりずっと前に届く。
INTERVAL = 0.4
CHUNKS = 3


class _DribblingUpstream(BaseHTTPRequestHandler):
    """SSE 風に、少量のチャンクを間隔を空けて流す上流サーバー。"""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(CHUNKS):
            data = f"data: token{i}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            time.sleep(INTERVAL)
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *_args) -> None:
        pass


class _ForwardingHandler(BaseHTTPRequestHandler):
    """受けたリクエストをそのまま forward() で上流へ中継するだけのゲートウェイ。"""

    protocol_version = "HTTP/1.0"
    upstream_addr: tuple[str, int] = ("127.0.0.1", 0)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        forward(self, self.upstream_addr, body, timeout_s=10.0)

    def log_message(self, *_args) -> None:
        pass


def test_forward_streams_chunks_incrementally():
    """上流のチャンクを溜め込まず、届いた順にクライアントへ流すこと。

    以前は resp.read(8192) が 8192 バイト溜まるまでブロックしたため、SSE の
    トークンが応答完了までまとめてバッファリングされていた（回帰防止）。
    """
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _DribblingUpstream)
    _ForwardingHandler.upstream_addr = ("127.0.0.1", upstream.server_address[1])
    gw = ThreadingHTTPServer(("127.0.0.1", 0), _ForwardingHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", gw.server_address[1], timeout=10)
        t0 = time.monotonic()
        conn.request("POST", "/v1/chat/completions", body=b"{}",
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        first_chunk_at = None
        received = b""
        while True:
            chunk = resp.read1(8192)
            if not chunk:
                break
            if first_chunk_at is None:
                first_chunk_at = time.monotonic() - t0
            received += chunk
        conn.close()
    finally:
        upstream.shutdown()
        upstream.server_close()
        gw.shutdown()
        gw.server_close()

    assert received.count(b"data: token") == CHUNKS  # 全チャンクを中継し切る
    # 逐次中継なら最初のチャンクは全送信完了（CHUNKS*INTERVAL 秒）より前に届く。
    total = CHUNKS * INTERVAL
    assert first_chunk_at is not None and first_chunk_at < total * 0.6, (
        f"first chunk arrived at {first_chunk_at:.2f}s — streaming is buffered "
        f"(upstream finishes at ~{total:.1f}s)"
    )
