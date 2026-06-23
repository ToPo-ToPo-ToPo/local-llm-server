"""LLMClient / build_user_content / to_image_url のテスト（openai をモック）。

openai はコア依存なので常に import できる前提。実サーバーには繋がず、土台の
OpenAI クライアントをフェイクに差し替える。
"""
from __future__ import annotations

import base64

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


# --- respond（openai クライアントをフェイクに差し替え） --------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeStreamChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_FakeStreamChunk("こん"), _FakeStreamChunk("にちは")])
        return _FakeResp("done")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, *a, **k):
        self.init_kwargs = k
        self.chat = _FakeChat()


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.setattr(client_mod, "OpenAI", _FakeOpenAI)


def test_respond_non_stream_returns_text(fake_openai):
    llm = LLMClient(model="m")
    assert llm.respond("hi") == "done"
    sent = llm.openai.chat.completions.calls[0]
    assert sent["messages"][-1] == {"role": "user", "content": "hi"}


def test_respond_includes_system_prompt(fake_openai):
    llm = LLMClient(model="m")
    llm.respond("hi", system_prompt="be brief")
    msgs = llm.openai.chat.completions.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}


def test_respond_stream_yields_pieces(fake_openai):
    llm = LLMClient(model="m")
    assert list(llm.respond("hi", stream=True)) == ["こん", "にちは"]


def test_respond_passes_images(fake_openai):
    llm = LLMClient(model="m")
    llm.respond("見て", images=["https://example.com/a.png"])
    content = llm.openai.chat.completions.calls[0]["messages"][-1]["content"]
    assert isinstance(content, list) and content[1]["type"] == "image_url"


def test_max_tokens_forwarded(fake_openai):
    LLMClient(model="m", max_tokens=128).respond("hi")
    assert LLMClient(model="m", max_tokens=128).max_tokens == 128


def test_openai_client_accessible(fake_openai):
    # 土台の openai クライアントに直接アクセスできる（高度操作用）。
    llm = LLMClient(model="m", base_url="http://127.0.0.1:8080/v1")
    assert llm.openai.init_kwargs["base_url"] == "http://127.0.0.1:8080/v1"
