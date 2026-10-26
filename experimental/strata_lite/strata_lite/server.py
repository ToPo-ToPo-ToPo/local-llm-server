"""最小限の OpenAI 互換サーバー（/v1/chat/completions・/v1/models・/health）。

1 プロセス 1 モデル。生成は 1 本ずつ（ロックで直列化）。統計は GET /v1/strata/stats。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def normalize_messages(messages) -> list[dict]:
    return [{"role": m.get("role", "user"), "content": _text_of(m.get("content"))} for m in messages]


def make_handler(engine, model_id: str):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: D401 - 標準のアクセスログは出さない
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "owned_by": "strata-lite"}]})
            if self.path == "/v1/strata/stats":
                c = engine.cache
                return self._json(200, {"placement": c.placement, "miss": c.miss, "slots": c.slots,
                                        "resident_fraction": c.resident_fraction(),
                                        "per_layer": c.placement_summary(), **c.stats.as_dict()})
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                messages = normalize_messages(req["messages"])
            except (ValueError, KeyError, TypeError) as e:
                return self._json(400, {"error": {"message": f"bad request: {e}"}})
            max_tokens = int(req.get("max_completion_tokens") or req.get("max_tokens") or 512)
            opts = dict(max_new_tokens=max_tokens, temperature=float(req.get("temperature") or 0.0),
                        top_p=float(req.get("top_p") or 1.0))
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            with lock:
                ids = engine.encode_chat(messages)
                if req.get("stream"):
                    return self._stream(ids, opts, cid, created, max_tokens)
                res = engine.generate_ids(ids, **opts)
            finish = "length" if res.new_tokens >= max_tokens else "stop"
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": res.text}}],
                "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.new_tokens,
                          "total_tokens": res.prompt_tokens + res.new_tokens},
            })

        def _stream(self, ids, opts, cid, created, max_tokens):
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(engine.tokenizer, skip_prompt=True, skip_special_tokens=True)
            box = {}

            def run():
                try:
                    box["res"] = engine.generate_ids(ids, streamer=streamer, **opts)
                except Exception as e:  # noqa: BLE001 - 失敗は最後のチャンクで伝える
                    box["err"] = e
                    streamer.end()

            th = threading.Thread(target=run, daemon=True)
            th.start()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(delta, finish=None):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                send({"role": "assistant", "content": ""})
                for piece in streamer:
                    if piece:
                        send({"content": piece})
                th.join()
                res = box.get("res")
                finish = "length" if res is not None and res.new_tokens >= max_tokens else "stop"
                send({}, "error" if "err" in box else finish)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                th.join()
            self.close_connection = True

    return Handler


def serve(engine, host: str = "127.0.0.1", port: int = 8090, model_id: str | None = None):
    model_id = model_id or os.path.basename(os.path.normpath(engine.model_dir))
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, model_id))
    engine.log(f"[strata-lite] serving {model_id} on http://{host}:{port}/v1")
    return httpd
