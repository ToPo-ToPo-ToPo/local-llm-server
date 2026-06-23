"""LLMClient / build_user_content / to_image_url のテスト（urllib をモック）。

openai 等の追加依存は不要（client は標準ライブラリのみで動く）。
"""
from __future__ import annotations

import base64
import io
import json

import pytest

from local_llm_server import client as client_mod
from local_llm_server.client import LLMClient, build_user_content, to_image_url


# --- マルチモーダル content 構築 -------------------------------------------
def test_build_user_content_text_only():
    assert build_user_content("hello") == "hello"


def test_build_user_content_with_image_passthrough_url():
    content = build_user_content("見て", images=["https://example.com/a.png"])
    assert content[0] == {"type": "text", "text": "見て"}
    assert content[1]["image_url"]["url"] == "https://example.com/a.png"


def test_to_image_url_local_file_becomes_data_uri(tmp_path):
    p = tmp_path / "pix.png"
    p.write_bytes(b"\x89PNG\r\n")
    url = to_image_url(str(p))
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG\r\n"


def test_to_image_url_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        to_image_url("/no/such/file.png")


# --- respond（urllib.request.urlopen をフェイクに差し替え） ----------------
class _FakeResp(io.BytesIO):
    """urlopen の戻り（context manager かつ iterable）。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def captured(monkeypatch):
    """送信ペイロードを記録し、固定レスポンスを返す urlopen を仕込む。"""
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["headers"] = dict(req.header_items())
        sent["payload"] = json.loads(req.data.decode("utf-8"))
        if sent["payload"].get("stream"):
            body = (
                'data: {"choices":[{"delta":{"content":"こん"}}]}\n'
                'data: {"choices":[{"delta":{"content":"にちは"}}]}\n'
                "data: [DONE]\n"
            ).encode("utf-8")
        else:
            body = json.dumps(
                {"choices": [{"message": {"content": "done"}}]}
            ).encode("utf-8")
        return _FakeResp(body)

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_respond_non_stream_returns_text(captured):
    llm = LLMClient(model="m")
    assert llm.respond("hi") == "done"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["messages"][-1] == {"role": "user", "content": "hi"}


def test_respond_includes_system_prompt(captured):
    LLMClient(model="m").respond("hi", system_prompt="be brief")
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "be brief"}


def test_respond_stream_yields_pieces(captured):
    out = list(LLMClient(model="m").respond("hi", stream=True))
    assert out == ["こん", "にちは"]
    assert captured["payload"]["stream"] is True


def test_respond_passes_images(captured):
    LLMClient(model="m").respond("見て", images=["https://example.com/a.png"])
    content = captured["payload"]["messages"][-1]["content"]
    assert isinstance(content, list) and content[1]["type"] == "image_url"


def test_max_tokens_forwarded(captured):
    LLMClient(model="m", max_tokens=128).respond("hi")
    assert captured["payload"]["max_tokens"] == 128


def test_authorization_header_set(captured):
    LLMClient(model="m", api_key="secret").respond("hi")
    assert captured["headers"].get("Authorization") == "Bearer secret"
