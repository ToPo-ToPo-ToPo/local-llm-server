"""OpenAI 互換プロキシの共有ヘルパー（gateway が利用）。

`BaseHTTPRequestHandler` を受け取り、現在のリクエストを上流へ中継したり、
JSON でエラー/結果を返したりする。HTTP/1.0（接続クローズ区切り）前提で、
ストリーミング/非ストリーミングを一律に「上流を読み切るまで中継して閉じる」形で扱う。
"""
from __future__ import annotations

import http.client
import json
from typing import Any


def forward(handler: Any, addr: tuple[str, int], body: bytes, timeout_s: float | None) -> None:
    """handler の現在のリクエスト（command/path/headers）を上流 addr へ中継する。

    上流応答（ストリーミング含む）をそのままクライアントへ書き戻す。上流へ到達
    できなければ 502 を返す。クライアント切断（Broken pipe）は黙って無視する。
    """
    host, port = addr
    headers = {"Content-Type": handler.headers.get("Content-Type", "application/json")}
    auth = handler.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    if handler.command == "POST":
        headers["Content-Length"] = str(len(body))
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request(handler.command, handler.path, body=body or None, headers=headers)
        resp = conn.getresponse()
    except OSError as exc:
        send_error(handler, 502, f"upstream {host}:{port} unreachable: {exc}")
        return
    try:
        handler.send_response(resp.status)
        ctype = resp.getheader("Content-Type")
        if ctype:
            handler.send_header("Content-Type", ctype)
        handler.end_headers()
        while True:
            # read1: ソケットに届いた分だけ即返す（read(8192) は 8192 バイト溜まるまで
            # ブロックするため、SSE のトークン逐次配信が全バッファリングされてしまう）。
            chunk = resp.read1(8192)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass  # クライアント切断
    finally:
        conn.close()


def send_error(handler: Any, status: int, message: str) -> None:
    """{"error": message} を JSON で返す。"""
    send_json(handler, status, {"error": message})


def send_json(handler: Any, status: int, obj: Any) -> None:
    """obj を JSON 本文として status で返す（Content-Length 付き）。"""
    payload = json.dumps(obj).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass
