"""OpenAI 互換サーバーへ繋ぐ高レベルクライアント（任意 extra）。

各エージェントが個別に書いていた「system / user / 画像をまとめて chat.completions に
投げ、テキスト（or ストリーム）を受け取る」ラッパーを共通化する。`openai` パッケージ
が必要なため、コア依存には含めない:

    uv add "local-llm-server[client]"

使い方（サーバーは別途起動済み）:

    from local_llm_server import LLMClient

    llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit")
    print(llm.respond("俳句を一つ詠んでください。"))

サーバーの相乗り/自動起動までまとめて行うなら connect() を使う:

    from local_llm_server import connect

    llm = connect(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto")
    for piece in llm.respond("ローカルLLMの利点は？", stream=True):
        print(piece, end="", flush=True)
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Iterator

from .constants import DEFAULT_MODEL
from .gateway import DEFAULT_BASE_URL, ServerHandle

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:  # pragma: no cover - 案内のみ
    raise ModuleNotFoundError(
        "local_llm_server.client requires the 'openai' package. "
        'Install it with: uv add "local-llm-server[client]"'
    ) from exc


def _is_url(ref: str) -> bool:
    return ref.startswith(("http://", "https://", "data:"))


def to_image_url(ref: str) -> str:
    """画像参照（ローカルパス or URL）を OpenAI 互換の image_url 文字列に変換する。

    URL / データURI はそのまま、ローカルファイルは base64 のデータURIにする。
    """
    if _is_url(ref):
        return ref
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {ref}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_user_content(
    text: str, images: list[str] | None = None
) -> str | list[dict[str, Any]]:
    """テキスト（＋画像）を OpenAI 互換の user メッセージ content に組み立てる。

    画像が無ければ素の文字列、あれば text パート＋image_url パートの配列を返す。
    """
    if not images:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for ref in images:
        parts.append({"type": "image_url", "image_url": {"url": to_image_url(ref)}})
    return parts


class LLMClient:
    """OpenAI 互換エンドポイントへ繋ぐ最小クライアント。

    respond() は非ストリームでは生成テキスト（str）を、stream=True ではテキスト断片の
    Iterator[str] を返す。マルチモーダルは images にローカルパス/URL を渡す。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "not-needed",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._handle: ServerHandle | None = None  # connect() が起動を紐づけるとき用

    def respond(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        images: list[str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """1 ターン生成する。stream=True なら断片の Iterator[str] を返す。"""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_user_content(user_text, images)})

        params: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=self.temperature, **kwargs
        )
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if stream:
            return self._stream(params)
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    def _stream(self, params: dict[str, Any]) -> Iterator[str]:
        for chunk in self._client.chat.completions.create(stream=True, **params):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def stop(self) -> None:
        """connect() で自動起動したサーバーがあれば停止する（無ければ無害）。"""
        if self._handle is not None:
            self._handle.stop()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


def connect(
    model: str = DEFAULT_MODEL,
    *,
    base_url: str = DEFAULT_BASE_URL,
    auto_start: bool = True,
    backend: str | None = None,
    draft_model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **ensure_kwargs: Any,
) -> LLMClient:
    """サーバーを用意（相乗り or 自動起動）してから、繋がった LLMClient を返す。

    ensure_server() + LLMClient を 1 呼び出しにまとめた入口。自動起動した場合、
    返り値の .stop()（または with 文）でサーバーを停止できる。
    """
    from .gateway import ensure_server

    handle = ensure_server(
        base_url=base_url,
        model=model,
        backend=backend,
        auto_start=auto_start,
        draft_model=draft_model,
        **ensure_kwargs,
    )
    client = LLMClient(
        model,
        base_url=handle.base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    client._handle = handle
    return client
