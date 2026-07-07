"""テキスト LLM と vision VLM を内部で振り分けるルーティングプロキシ。

1 つの OpenAI 互換エンドポイント（例 http://127.0.0.1:8080/v1）を公開し、受信した
`/v1/chat/completions` の `messages` を検査して:

  - 画像など非テキストのコンテンツパートを含む → vision バックエンド(VLM)
  - テキストのみ                              → テキスト LLM

へ転送する（ストリーミングはそのまま中継）。クライアントの base_url は
1 つのままで、内部の 2 プロセスを意識しなくてよい。
"""
from __future__ import annotations

import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def needs_vision(messages: Any) -> bool:
    """messages に非テキストのコンテンツパート（画像など）が含まれるか。

    OpenAI 互換の content は文字列か、{"type": "text"|"image_url"|...} の配列。
    type が text 以外のパートが 1 つでもあれば vision 扱いにする。
    """
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") not in (None, "text"):
                return True
    return False


class _RouterHandler(BaseHTTPRequestHandler):
    server_version = "local-llm-router"

    # HTTP/1.0: 応答ボディは接続クローズ区切り。ストリーミング/非ストリーミングを
    # 一律に「上流を読み切るまで中継して閉じる」形で扱え、Content-Length 計算が不要。
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args) -> None:  # アクセスログは出さない
        pass

    # --- ルーティング ---------------------------------------------------
    def do_GET(self) -> None:
        # is_ready 用の /v1/models など。読み取り系はテキスト上流へ。
        self._proxy(self.server.text_addr, body=b"")  # type: ignore[attr-defined]

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "invalid Content-Length header")
            return
        if length < 0:
            self._error(400, "invalid Content-Length header")
            return
        body = self.rfile.read(length) if length else b""
        target = self._select_target(body)
        self._proxy(target, body)

    def _select_target(self, body: bytes) -> tuple[str, int]:
        srv = self.server  # type: ignore[assignment]
        # chat/completions 以外（埋め込み等）はテキスト上流へ素通し。
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return srv.text_addr
        try:
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, ValueError):
            return srv.text_addr
        if needs_vision(payload.get("messages")):
            return srv.vision_addr
        return srv.text_addr

    # --- 転送 -----------------------------------------------------------
    def _proxy(self, addr: tuple[str, int], body: bytes) -> None:
        host, port = addr
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        if self.command == "POST":
            headers["Content-Length"] = str(len(body))
        try:
            conn = http.client.HTTPConnection(host, port, timeout=self.server.timeout_s)  # type: ignore[attr-defined]
            conn.request(self.command, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:
            self._error(502, f"upstream {host}:{port} unreachable: {exc}")
            return
        try:
            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type")
            if ctype:
                self.send_header("Content-Type", ctype)
            self.end_headers()
            while True:
                # read1: ソケットに届いた分だけ即返す（read(8192) は 8192 バイト溜まる
                # までブロックし、SSE のトークン逐次配信が全バッファリングされてしまう）。
                chunk = resp.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # クライアント切断
        finally:
            conn.close()

    def _error(self, status: int, message: str) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class RouterServer(ThreadingHTTPServer):
    """テキスト/vision 上流へ振り分けるプロキシ HTTP サーバー。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        addr: tuple[str, int],
        text_addr: tuple[str, int],
        vision_addr: tuple[str, int],
        timeout_s: float | None = None,
    ) -> None:
        super().__init__(addr, _RouterHandler)
        self.text_addr = text_addr
        self.vision_addr = vision_addr
        # None なら無制限（長時間生成に備える）。
        self.timeout_s = timeout_s
