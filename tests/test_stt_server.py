from __future__ import annotations

import http.client
import threading

from local_llm_server import stt_server


def test_warmup_holds_same_lock_as_requests(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class _Backend:
        def load_model(self, _model):
            entered.set()
            release.wait(2)

    monkeypatch.setattr(stt_server, "_backend_module", lambda: _Backend())
    server = stt_server._Server(("127.0.0.1", 0), "test/model")
    thread = threading.Thread(target=stt_server._warm, args=(server,))
    thread.start()
    try:
        assert entered.wait(1)
        assert server.lock.acquire(blocking=False) is False
    finally:
        release.set()
        thread.join(2)
        server.server_close()


def test_audio_body_limit_rejected_before_reading_payload():
    server = stt_server._Server(("127.0.0.1", 0), "test/model")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
        conn.putrequest("POST", "/v1/audio/transcriptions")
        conn.putheader("Content-Type", "multipart/form-data; boundary=x")
        conn.putheader("Content-Length", str(stt_server._MAX_AUDIO_BODY_BYTES + 1))
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 413
        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_invalid_stt_options_are_rejected_before_backend(monkeypatch):
    monkeypatch.setattr(
        stt_server, "_backend_module",
        lambda: (_ for _ in ()).throw(AssertionError("backend must not run")),
    )
    boundary = "stt-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        "yaml\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "sound\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    server = stt_server._Server(("127.0.0.1", 0), "test/model")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
        conn.request(
            "POST", "/v1/audio/transcriptions", body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        assert response.status == 400
        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
