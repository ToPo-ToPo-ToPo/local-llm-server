"""Anthropic Messages API（/v1/messages）の変換と、ゲートウェイ経由の中継の試験。

上流は試験の中の偽サーバー（OpenAI 形式で答える）。実際のモデルは使わない。
"""

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from local_llm_server import ServerConfig
from local_llm_server import anthropic_api as aa
from local_llm_server import daemon as gw

# --- 要求の変換 ------------------------------------------------------------------


def test_request_converts_system_text_and_sampling():
    chat = aa.to_chat_request({
        "model": "m1",
        "max_tokens": 100,
        "temperature": 0.5,
        "top_k": 20,
        "stop_sequences": ["END"],
        "system": [{"type": "text", "text": "rule 1", "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "rule 2"}],
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": "x"},
    })
    assert chat == {
        "model": "m1",
        "max_tokens": 100,
        "temperature": 0.5,
        "top_k": 20,
        "stop": ["END"],
        "messages": [
            {"role": "system", "content": "rule 1\n\nrule 2"},
            {"role": "user", "content": "hi"},
        ],
    }


def test_request_converts_tool_use_round_trip():
    chat = aa.to_chat_request({
        "model": "m1",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "time?"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {"city": "東京"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                    {"type": "text", "text": "12:00"}]},
                {"type": "text", "text": "thanks"},
            ]},
        ],
        "tools": [
            {"name": "get_time", "description": "now", "input_schema": {
                "type": "object", "properties": {"city": {"type": "string"}}}},
            {"type": "web_search_20250305", "name": "web_search"},  # サーバー側ツールは落とす
        ],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    })
    assert chat["messages"] == [
        {"role": "user", "content": "time?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [{
            "id": "toolu_1", "type": "function",
            "function": {"name": "get_time", "arguments": '{"city": "東京"}'}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "12:00"},
        {"role": "user", "content": "thanks"},
    ]
    assert chat["tools"] == [{"type": "function", "function": {
        "name": "get_time", "description": "now",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    assert chat["tool_choice"] == "required"
    assert chat["parallel_tool_calls"] is False


def test_request_marks_tool_errors_and_moves_images_out_of_tool_results():
    chat = aa.to_chat_request({
        "model": "m1",
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "boom"},
            {"type": "tool_result", "tool_use_id": "t2", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}}]},
        ]}],
    })
    assert chat["messages"] == [
        {"role": "tool", "tool_call_id": "t1", "content": "Error: boom"},
        {"role": "tool", "tool_call_id": "t2", "content": ""},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]},
    ]


def test_request_maps_thinking_and_stream():
    on = aa.to_chat_request({"model": "m", "messages": [], "thinking": {"type": "adaptive"},
                             "stream": True})
    off = aa.to_chat_request({"model": "m", "messages": [], "thinking": {"type": "disabled"}})
    assert on["enable_thinking"] is True
    assert on["stream"] is True and on["stream_options"] == {"include_usage": True}
    assert off["enable_thinking"] is False
    assert "enable_thinking" not in aa.to_chat_request({"model": "m", "messages": []})


@pytest.mark.parametrize("payload", [
    {"messages": "hi"},
    {"messages": [{"role": "system", "content": "x"}]},
    {"messages": [{"role": "user", "content": 3}]},
])
def test_request_rejects_malformed(payload):
    with pytest.raises(aa.RequestError):
        aa.to_chat_request(payload)


# --- 応答の変換 ------------------------------------------------------------------


def test_response_converts_text_reasoning_and_tool_calls():
    msg = aa.to_message({
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": "Let me check.",
            "reasoning_content": "need tool",
            "tool_calls": [{"id": "c1", "function": {"name": "get_time", "arguments": '{"city":"Tokyo"}'}},
                           {"id": "c2", "function": {"name": "broken", "arguments": "{oops"}}],
        }}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }, "m1")
    assert msg["type"] == "message" and msg["role"] == "assistant" and msg["model"] == "m1"
    assert msg["content"] == [
        {"type": "thinking", "thinking": "need tool", "signature": ""},
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "c1", "name": "get_time", "input": {"city": "Tokyo"}},
        {"type": "tool_use", "id": "c2", "name": "broken", "input": {"_raw_arguments": "{oops"}},
    ]
    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"] == {"input_tokens": 12, "output_tokens": 5}


@pytest.mark.parametrize("finish,expected", [
    ("stop", "end_turn"), ("length", "max_tokens"), (None, "end_turn")])
def test_response_stop_reasons(finish, expected):
    msg = aa.to_message({"choices": [{"finish_reason": finish, "message": {"content": "x"}}]}, "m")
    assert msg["stop_reason"] == expected


def _sse(*chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def _events(raw: bytes):
    out = []
    for frame in raw.decode().split("\n\n"):
        if not frame.strip():
            continue
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        data = json.loads(lines["data"])
        assert data["type"] == lines["event"]
        out.append(data)
    return out


def _delta(**delta):
    return {"choices": [{"index": 0, "finish_reason": None, "delta": delta}]}


def test_stream_translates_thinking_text_and_tool_call():
    upstream = _sse(
        _delta(role="assistant", content=None, reasoning_content="think"),
        _delta(content="I'll "),
        _delta(content="check."),
        _delta(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                            "function": {"name": "get_time", "arguments": '{"city": "Tok'}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": 'yo"}'}}]),
        {"choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}]},
        {"choices": [], "usage": {"prompt_tokens": 274, "completion_tokens": 39}},
    )
    tr = aa.StreamTranslator("m1")
    raw = tr.start()
    # SSE の行の途中で切れたチャンクでも同じ結果になる
    for i in range(0, len(upstream), 7):
        raw += tr.feed(upstream[i:i + 7])
    raw += tr.finish()
    events = _events(raw)
    kinds = [e["type"] for e in events]
    assert kinds == [
        "message_start", "ping",
        "content_block_start", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
        "message_delta", "message_stop",
    ]
    assert events[2]["content_block"]["type"] == "thinking"
    assert events[3]["delta"] == {"type": "thinking_delta", "thinking": "think"}
    assert [e["delta"]["text"] for e in events[6:8]] == ["I'll ", "check."]
    assert events[9]["content_block"] == {"type": "tool_use", "id": "c1", "name": "get_time", "input": {}}
    args = "".join(e["delta"]["partial_json"] for e in events[10:12])
    assert json.loads(args) == {"city": "Tokyo"}
    assert [e["index"] for e in events if "index" in e] == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    assert events[-2]["usage"] == {"input_tokens": 274, "output_tokens": 39}


def test_stream_tool_call_without_arguments_gets_empty_object():
    tr = aa.StreamTranslator("m")
    raw = tr.start() + tr.feed(_sse(
        _delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "now", "arguments": ""}}]),
        {"choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}]},
    )) + tr.finish()
    deltas = [e["delta"] for e in _events(raw) if e["type"] == "content_block_delta"]
    assert deltas == [{"type": "input_json_delta", "partial_json": "{}"}]


def test_stream_cut_off_reports_error_instead_of_clean_stop():
    tr = aa.StreamTranslator("m")
    raw = tr.start() + tr.feed(b'data: {"choices":[{"delta":{"content":"par"}}]}\n\n') + tr.finish()
    kinds = [e["type"] for e in _events(raw)]
    assert kinds[-1] == "error" and "message_stop" not in kinds


# --- ゲートウェイ経由（偽の上流） -------------------------------------------------


@pytest.fixture(autouse=True)
def _no_reclaim(monkeypatch):
    monkeypatch.setattr(gw, "reclaim_stale_workers", lambda *a, **k: [])


class _FakeServer:
    def __init__(self, config, log_path=None):
        self.config = config

    def start(self):
        pass

    def wait_until_ready(self, *a, **k):
        pass

    def stop(self, grace: float = 10.0):
        pass


def _upstream(reply):
    """OpenAI 形式で答える偽の上流。受けた (path, body) を記録する。reply(body) → (status, ctype, bytes)。"""
    seen = []

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            seen.append((self.path, body))
            status, ctype, data = reply(body)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.fixture
def gateway(monkeypatch):
    started = []

    def start(reply, api_key=None):
        up, seen = _upstream(reply)
        monkeypatch.setattr(gw, "LocalServer", lambda config, log_path=None: _FakeServer(config))
        mgr = gw.ModelManager([ServerConfig(backend="mlx", model="m1", host="127.0.0.1",
                                            port=up.server_address[1])])
        server = gw.GatewayServer(("127.0.0.1", 0), mgr, catalog=["m1"], api_key=api_key)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append((server, mgr, up))
        return server.server_address[1], seen

    yield start
    for server, mgr, up in started:
        server.shutdown(); server.server_close(); mgr.shutdown()
        up.shutdown(); up.server_close()


def _post(port, path, obj, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    hdrs = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    hdrs.update(headers or {})
    conn.request("POST", path, json.dumps(obj), hdrs)
    resp = conn.getresponse()
    data = resp.read()
    ctype = resp.getheader("Content-Type")
    conn.close()
    return resp.status, ctype, data


def _json_reply(obj, status=200):
    return lambda body: (status, "application/json", json.dumps(obj).encode())


def test_gateway_messages_non_stream(gateway):
    port, seen = gateway(_json_reply({
        "choices": [{"finish_reason": "stop", "message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }))
    status, ctype, data = _post(port, "/v1/messages", {
        "model": "m1", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and ctype == "application/json"
    msg = json.loads(data)
    assert msg["content"] == [{"type": "text", "text": "hello"}]
    assert msg["stop_reason"] == "end_turn" and msg["model"] == "m1"
    # 上流には OpenAI 形式で chat/completions に届く
    path, body = seen[0]
    assert path == "/v1/chat/completions"
    assert body["messages"] == [{"role": "user", "content": "hi"}] and body["max_tokens"] == 10


def test_gateway_messages_stream(gateway):
    port, seen = gateway(lambda body: (200, "text/event-stream", _sse(
        _delta(content="hel"), _delta(content="lo"),
        {"choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]})))
    status, ctype, data = _post(port, "/v1/messages", {
        "model": "m1", "max_tokens": 10, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and ctype == "text/event-stream"
    events = _events(data)
    assert events[0]["type"] == "message_start" and events[-1]["type"] == "message_stop"
    text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta")
    assert text == "hello"
    assert seen[0][1]["stream"] is True


def test_gateway_messages_stream_with_json_upstream(gateway):
    # stream を頼んでも上流が JSON で返したら、まとめて SSE にして返す
    port, _ = gateway(_json_reply({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}))
    status, ctype, data = _post(port, "/v1/messages", {
        "model": "m1", "max_tokens": 10, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and ctype == "text/event-stream"
    events = _events(data)
    assert [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"] == ["ok"]
    assert events[-1]["type"] == "message_stop"


def test_gateway_messages_upstream_error_is_anthropic_shaped(gateway):
    port, _ = gateway(_json_reply({"error": {"message": "context too long"}}, status=400))
    status, _, data = _post(port, "/v1/messages", {
        "model": "m1", "max_tokens": 10, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert status == 400
    assert json.loads(data) == {"type": "error", "error": {
        "type": "invalid_request_error", "message": "context too long"}}


def test_gateway_messages_gateway_errors_are_anthropic_shaped(gateway):
    port, _ = gateway(_json_reply({}))
    status, _, data = _post(port, "/v1/messages", {"model": "ghost", "messages": []})
    assert status == 404
    assert json.loads(data)["error"]["type"] == "not_found_error"
    status, _, data = _post(port, "/v1/messages", {"model": "m1", "messages": "hi"})
    assert status == 400 and json.loads(data)["type"] == "error"
    status, _, data = _post(port, "/v1/messages/count_tokens", {"model": "m1", "messages": []})
    assert status == 404 and json.loads(data)["type"] == "error"


def test_gateway_messages_accepts_x_api_key(gateway):
    port, _ = gateway(_json_reply({"choices": [{"message": {"content": "ok"}}]}), api_key="secret")
    req = {"model": "m1", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
    assert _post(port, "/v1/messages", req)[0] == 401
    assert _post(port, "/v1/messages", req, {"x-api-key": "wrong"})[0] == 401
    assert _post(port, "/v1/messages", req, {"x-api-key": "secret"})[0] == 200
    assert _post(port, "/v1/messages", req, {"Authorization": "Bearer secret"})[0] == 200


def test_gateway_chat_completions_unchanged(gateway):
    # 既存の OpenAI 形式の入口は変換を通らず、上流の応答をそのまま返す
    port, seen = gateway(_json_reply({"raw": True}))
    status, _, data = _post(port, "/v1/chat/completions", {
        "model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and json.loads(data) == {"raw": True}
    assert seen[0][0] == "/v1/chat/completions"
