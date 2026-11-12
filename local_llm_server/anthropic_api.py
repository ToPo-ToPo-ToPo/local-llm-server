"""Anthropic Messages API（`POST /v1/messages`）を OpenAI 互換の chat/completions に橋渡しする。

Claude Code のように Anthropic 形式しか話さないクライアントを、ゲートウェイの既存の振り分け
（`model` → ローカルのモデルサーバー）にそのまま載せるための変換層。外部ライブラリは使わない
（変換プロキシを別に立てない。第三者の中継を挟まないため）。

- 要求: Anthropic の system / messages / tools → OpenAI の messages / tools（`to_chat_request`）
- 応答: OpenAI の chat.completion → Anthropic の message（`to_message`）
- ストリーミング: OpenAI の SSE チャンク → Anthropic の SSE イベント（`StreamTranslator`）
- 中継: `proxy.forward` に渡す応答の変換器（`MessagesResponse`）

Anthropic にしか無い指定（`cache_control`・`metadata`・`anthropic-beta` ヘッダなど）は受け取って
捨てる。思考の指定（`thinking`）だけは、mlx-vlm が読む `enable_thinking` に写す。
"""

from __future__ import annotations

import json
import uuid
from typing import Any


class RequestError(ValueError):
    """Anthropic 形式として受け取れない要求（HTTP 400 にする）。"""


# OpenAI の finish_reason → Anthropic の stop_reason
_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}

# HTTP ステータス → Anthropic のエラー種別
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def error_body(status: int, message: str) -> dict:
    """Anthropic 形式のエラー本文。"""
    kind = _ERROR_TYPES.get(status, "api_error" if status >= 500 else "invalid_request_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


# --- 要求の変換 ------------------------------------------------------------------


def to_chat_request(payload: dict) -> dict:
    """Anthropic の Messages 要求を OpenAI の chat/completions 要求に変える。"""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise RequestError("'messages' must be an array")
    messages: list[dict] = []
    system = _system_text(payload.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for msg in raw_messages:
        messages.extend(_convert_message(msg))

    out: dict[str, Any] = {"model": payload.get("model"), "messages": messages}
    if "max_tokens" in payload:
        out["max_tokens"] = payload["max_tokens"]
    for key in ("temperature", "top_p", "top_k"):
        if key in payload:
            out[key] = payload[key]
    stops = payload.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = [s for s in stops if isinstance(s, str)]

    tools = [_convert_tool(t) for t in payload.get("tools") or [] if _is_custom_tool(t)]
    if tools:
        out["tools"] = tools
        choice = payload.get("tool_choice")
        if isinstance(choice, dict):
            converted = _convert_tool_choice(choice)
            if converted is not None:
                out["tool_choice"] = converted
            if choice.get("disable_parallel_tool_use"):
                out["parallel_tool_calls"] = False

    thinking = payload.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type"):
        out["enable_thinking"] = thinking["type"] != "disabled"

    if payload.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _convert_message(msg: Any) -> list[dict]:
    if not isinstance(msg, dict):
        raise RequestError("each message must be an object")
    role = msg.get("role")
    content = msg.get("content")
    if role not in ("user", "assistant"):
        raise RequestError(f"unsupported message role: {role!r}")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise RequestError("message 'content' must be a string or an array of blocks")
    blocks = [b for b in content if isinstance(b, dict)]
    if role == "assistant":
        return [_convert_assistant(blocks)]
    return _convert_user(blocks)


def _convert_assistant(blocks: list[dict]) -> dict:
    texts: list[str] = []
    tool_calls: list[dict] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            texts.append(block.get("text", ""))
        elif kind == "tool_use":
            tool_calls.append({
                "id": block.get("id") or _new_id("call"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })
        # thinking / redacted_thinking は履歴に戻さない（ローカルのテンプレートは署名を持たない）
    # content は null にしない（null を "None" と描くチャットテンプレートがある）
    out: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(t for t in texts if t)}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _convert_user(blocks: list[dict]) -> list[dict]:
    # tool_result は role="tool" のメッセージにし、直前の assistant の tool_calls の直後に置く
    # （OpenAI 形式の決まり）。同じ user の残り（文・画像）はその後ろに 1 通でまとめる。
    tool_messages: list[dict] = []
    parts: list[dict] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "tool_result":
            text, images = _tool_result_content(block.get("content"))
            if block.get("is_error") and not text.startswith("Error"):
                text = f"Error: {text}" if text else "Error"
            tool_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": text,
            })
            parts.extend(images)
        elif kind == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            part = _image_part(block)
            if part:
                parts.append(part)
        elif kind == "document":
            source = block.get("source") or {}
            if source.get("type") == "text" and source.get("data"):
                parts.append({"type": "text", "text": source["data"]})
    out = list(tool_messages)
    if parts:
        if all(p["type"] == "text" for p in parts):
            out.append({"role": "user", "content": "\n\n".join(p["text"] for p in parts)})
        else:
            out.append({"role": "user", "content": parts})
    return out


def _tool_result_content(content: Any) -> tuple[str, list[dict]]:
    """tool_result の中身を（文, 画像パーツ）に分ける。tool メッセージは画像を持てない。"""
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[dict] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "image":
            part = _image_part(block)
            if part:
                images.append(part)
    return "\n".join(texts), images


def _image_part(block: dict) -> dict | None:
    source = block.get("source") or {}
    if source.get("type") == "base64" and source.get("data"):
        media = source.get("media_type") or "image/png"
        url = f"data:{media};base64,{source['data']}"
    elif source.get("type") == "url" and source.get("url"):
        url = source["url"]
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _is_custom_tool(tool: Any) -> bool:
    # サーバー側ツール（web_search_20250305 など）は input_schema を持たない。ローカルでは動かせない。
    return isinstance(tool, dict) and bool(tool.get("name")) and "input_schema" in tool


def _convert_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _convert_tool_choice(choice: dict) -> Any:
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


# --- 応答の変換（非ストリーミング） -------------------------------------------------


def to_message(chat: dict, model: Any) -> dict:
    """OpenAI の chat.completion を Anthropic の message に変える。"""
    choices = chat.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    msg = choice.get("message") or {}
    content: list[dict] = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        content.append({
            "type": "tool_use",
            "id": call.get("id") or _new_id("toolu"),
            "name": fn.get("name", ""),
            "input": _parse_arguments(fn.get("arguments")),
        })
    has_tools = any(b["type"] == "tool_use" for b in content)
    usage = chat.get("usage") or {}
    return {
        "id": _new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason(choice.get("finish_reason"), has_tools),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        # 壊れた引数は捨てずに渡す（クライアントの検証エラーとしてモデルに返り、直せる）
        return {"_raw_arguments": str(raw)}
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": str(raw)}


def _stop_reason(finish_reason: Any, has_tools: bool) -> str:
    reason = _STOP_REASONS.get(finish_reason or "", "end_turn")
    if has_tools and reason == "end_turn":
        return "tool_use"
    return reason


# --- 応答の変換（ストリーミング） ---------------------------------------------------


def _event(kind: str, data: dict) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class StreamTranslator:
    """OpenAI の SSE（`data: {...}` 行）を Anthropic の SSE イベント列に変える。

    `start()` → `feed(chunk)` を繰り返す → `finish()` の順に呼ぶ。チャンクの境目は SSE の
    行の途中でもよい（行が揃うまで溜める）。
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._buf = b""
        self._index = -1  # 今開いているブロックの番号（-1 = なし）
        self._kind: str | None = None  # 今開いているブロックの種類
        self._tool_blocks: dict[int, int] = {}  # OpenAI の tool_calls[].index → ブロック番号
        self._tool_has_args: dict[int, bool] = {}  # ブロック番号 → 引数を 1 片でも流したか
        self._finish_reason: str | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._ended = False  # [DONE] か finish_reason を受けた
        self._closed = False

    def start(self) -> bytes:
        message = {
            "id": _new_id("msg"),
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return _event("message_start", {"type": "message_start", "message": message}) + _event(
            "ping", {"type": "ping"}
        )

    def feed(self, chunk: bytes) -> bytes:
        self._buf += chunk
        out: list[bytes] = []
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            out.append(self._line(line))
        return b"".join(out)

    def finish(self, aborted: bool = False) -> bytes:
        if self._closed:
            return b""
        out = [self._line(self._buf)] if self._buf else []
        self._buf = b""
        self._closed = True
        if aborted or not self._ended:
            out.append(_event("error", error_body(502, "upstream stream ended unexpectedly")))
            return b"".join(out)
        out.append(self._close_block())
        has_tools = bool(self._tool_blocks)
        out.append(_event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": _stop_reason(self._finish_reason, has_tools),
                "stop_sequence": None,
            },
            "usage": {"input_tokens": self._input_tokens, "output_tokens": self._output_tokens},
        }))
        out.append(_event("message_stop", {"type": "message_stop"}))
        return b"".join(out)

    # -- 内部 --

    def _line(self, raw: bytes) -> bytes:
        line = raw.strip()
        if not line.startswith(b"data:"):
            return b""
        data = line[5:].strip()
        if data == b"[DONE]":
            self._ended = True
            return b""
        try:
            obj = json.loads(data)
        except ValueError:
            return b""
        if not isinstance(obj, dict):
            return b""
        if obj.get("error"):
            err = obj["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            self._closed = True
            return _event("error", error_body(502, message or "upstream error"))
        usage = obj.get("usage")
        if isinstance(usage, dict):
            self._input_tokens = int(usage.get("prompt_tokens") or self._input_tokens)
            self._output_tokens = int(usage.get("completion_tokens") or self._output_tokens)
        out: list[bytes] = []
        for choice in (obj.get("choices") or [])[:1]:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                out.append(self._open("thinking"))
                out.append(self._delta({"type": "thinking_delta", "thinking": reasoning}))
            if delta.get("content"):
                out.append(self._open("text"))
                out.append(self._delta({"type": "text_delta", "text": delta["content"]}))
            for call in delta.get("tool_calls") or []:
                out.append(self._tool_delta(call))
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]
                self._ended = True
        return b"".join(out)

    def _open(self, kind: str) -> bytes:
        if self._kind == kind:
            return b""
        out = self._close_block()
        self._index += 1
        self._kind = kind
        block = {"type": "text", "text": ""} if kind == "text" else {
            "type": "thinking", "thinking": "", "signature": ""}
        return out + _event("content_block_start", {
            "type": "content_block_start", "index": self._index, "content_block": block})

    def _tool_delta(self, call: dict) -> bytes:
        position = call.get("index", 0)
        fn = call.get("function") or {}
        out = b""
        if position not in self._tool_blocks:
            out += self._close_block()
            self._index += 1
            self._kind = "tool_use"
            self._tool_blocks[position] = self._index
            self._tool_has_args[self._index] = False
            out += _event("content_block_start", {
                "type": "content_block_start",
                "index": self._index,
                "content_block": {
                    "type": "tool_use",
                    "id": call.get("id") or _new_id("toolu"),
                    "name": fn.get("name", ""),
                    "input": {},
                },
            })
        args = fn.get("arguments")
        if args:
            index = self._tool_blocks[position]
            self._tool_has_args[index] = True
            out += _event("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": args},
            })
        return out

    def _delta(self, delta: dict) -> bytes:
        return _event("content_block_delta", {
            "type": "content_block_delta", "index": self._index, "delta": delta})

    def _close_block(self) -> bytes:
        if self._kind is None:
            return b""
        out = b""
        if self._kind == "tool_use" and not self._tool_has_args.get(self._index):
            # 引数の無い呼び出しも JSON として読めるようにする
            out += self._delta({"type": "input_json_delta", "partial_json": "{}"})
        self._kind = None
        return out + _event("content_block_stop", {"type": "content_block_stop", "index": self._index})


def message_to_events(message: dict) -> bytes:
    """組み上がった message を Anthropic の SSE イベント列にする（上流が非ストリーミングで返したとき）。"""
    head = dict(message, content=[], stop_reason=None)
    out = [_event("message_start", {"type": "message_start", "message": head})]
    for index, block in enumerate(message.get("content") or []):
        kind = block["type"]
        if kind == "text":
            start: dict = {"type": "text", "text": ""}
            delta: dict = {"type": "text_delta", "text": block["text"]}
        elif kind == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {"type": "input_json_delta",
                     "partial_json": json.dumps(block["input"], ensure_ascii=False)}
        out.append(_event("content_block_start", {
            "type": "content_block_start", "index": index, "content_block": start}))
        out.append(_event("content_block_delta", {
            "type": "content_block_delta", "index": index, "delta": delta}))
        out.append(_event("content_block_stop", {"type": "content_block_stop", "index": index}))
    out.append(_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
        "usage": message["usage"],
    }))
    out.append(_event("message_stop", {"type": "message_stop"}))
    return b"".join(out)


# --- 中継のための応答変換器 ---------------------------------------------------------


class MessagesResponse:
    """`proxy.forward` に渡す変換器。上流（chat/completions）の応答を Anthropic 形式で返す。

    - 成功 + stream 指定: SSE を逐次変換する（上流が JSON で返したら最後にまとめてイベント化）
    - 成功 + 非 stream: 本文を読み切って message に変える
    - 失敗: 上流のエラーを Anthropic 形式のエラーに包む（ステータスはそのまま）
    """

    def __init__(self, model: Any, stream: bool) -> None:
        self.model = model
        self.stream = stream
        self._status = 200
        self._mode = "buffer"
        self._buf = b""
        self._translator: StreamTranslator | None = None

    def begin(self, status: int, content_type: str | None) -> tuple[int, str, bytes]:
        self._status = status
        if status != 200:
            return status, "application/json", b""
        if not self.stream:
            return 200, "application/json", b""
        if (content_type or "").lower().startswith("text/event-stream"):
            self._mode = "sse"
            self._translator = StreamTranslator(self.model)
            return 200, "text/event-stream", self._translator.start()
        return 200, "text/event-stream", b""

    def feed(self, chunk: bytes) -> bytes:
        if self._translator is not None:
            return self._translator.feed(chunk)
        self._buf += chunk
        return b""

    def finish(self, aborted: bool = False) -> bytes:
        if self._translator is not None:
            return self._translator.finish(aborted)
        if self._status != 200:
            return json.dumps(error_body(self._status, _upstream_error(self._buf))).encode("utf-8")
        try:
            chat = json.loads(self._buf or b"{}")
        except ValueError:
            chat = None
        if aborted or not isinstance(chat, dict):
            body = error_body(502, "upstream returned an unreadable response")
            if self.stream:
                return _event("error", body)
            return json.dumps(body).encode("utf-8")
        message = to_message(chat, self.model)
        if self.stream:
            return message_to_events(message)
        return json.dumps(message, ensure_ascii=False).encode("utf-8")


def _upstream_error(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip()
    try:
        obj = json.loads(text)
    except ValueError:
        return text[:2000] or "upstream error"
    if isinstance(obj, dict):
        err = obj.get("error", obj.get("detail"))
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    return text[:2000] or "upstream error"
