"""OpenAI 互換サーバーへ繋ぐ高レベルクライアント（標準ライブラリのみ）。

各エージェントが個別に書いていた「system / user / 画像をまとめて chat.completions に
投げ、テキスト（or ストリーム）を受け取る」ラッパー。`/v1/chat/completions` に JSON を
POST し、SSE を読むだけなので **追加依存なし**（urllib で実装）。openai クライアントは不要。

    from local_llm_server import LLMClient

    llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit")
    print(llm.respond("俳句を一つ詠んでください。"))

サーバーの相乗り/自動起動までまとめるなら connect():

    from local_llm_server import connect

    llm = connect(model="mlx-community/Qwen3.6-27B-4bit", draft_model="auto")
    for piece in llm.respond("ローカルLLMの利点は？", stream=True):
        print(piece, end="", flush=True)
"""
from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from .constants import DEFAULT_MODEL
from .gateway import DEFAULT_BASE_URL, ServerHandle


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
    """OpenAI 互換エンドポイントへ繋ぐ最小クライアント（標準ライブラリのみ）。

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
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._handle: ServerHandle | None = None  # connect() が起動を紐づけるとき用

    # --- リクエスト組み立て ---------------------------------------------
    def _payload(self, user_text, system_prompt, images, stream, kwargs) -> dict:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_user_content(user_text, images)})
        payload: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=self.temperature, **kwargs
        )
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stream:
            payload["stream"] = True
        return payload

    def _request(self, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, headers=headers
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:  # 上流のエラーボディを見えるように
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"server returned {exc.code}: {body}") from exc

    # --- 生成 -----------------------------------------------------------
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
        payload = self._payload(user_text, system_prompt, images, stream, kwargs)
        if stream:
            return self._stream(payload)
        with self._request(payload) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"].get("content") or ""

    def _stream(self, payload: dict) -> Iterator[str]:
        # SSE: 各行が "data: {json}"、終端は "data: [DONE]"。delta.content を逐次返す。
        with self._request(payload) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield piece

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
    timeout: float | None = None,
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
        timeout=timeout,
    )
    client._handle = handle
    return client
