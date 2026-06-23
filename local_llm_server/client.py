"""OpenAI 互換サーバーへ繋ぐ高レベルクライアント。

各エージェントが個別に書いていた「system / user / 画像をまとめて chat.completions に
投げ、テキスト（or ストリーム）を受け取る」ラッパー。公式 `openai` SDK を土台にするため
自動リトライ・型付きレスポンス・ツール呼び出し/構造化出力などの高度機能もそのまま使える
（`openai` はコア依存）。

    from local_llm_server import LLMClient

    llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit")
    print(llm.respond("俳句を一つ詠んでください。"))

起動中のゲートウェイに繋ぐワンライナー（未起動なら親切なエラー。サーバーは起動しない）
なら connect():

    from local_llm_server import connect

    llm = connect(model="mlx-community/Qwen3.6-27B-4bit",
                  base_url="http://127.0.0.1:8799/v1")
    for piece in llm.respond("ローカルLLMの利点は？", stream=True):
        print(piece, end="", flush=True)

より高度な操作（embeddings / tool calling / 構造化出力 / async など）は、`LLMClient.openai`
で土台の openai クライアントに直接アクセスできる。
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

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


def thinking_extra_body(enable: bool) -> dict[str, Any]:
    """思考(thinking)モードの ON/OFF をサーバーへ渡す extra_body を作る。

    バックエンドによって解釈するキーが異なる（これはバックエンドの protocol 知識で、
    どのキーを使うかを知っているのは推論サーバー側＝本ライブラリの責務）:
      - mlx-vlm      … トップレベル enable_thinking（chat_template_kwargs は無視）
      - mlx_lm/llama … chat_template_kwargs.enable_thinking
    未知キーはどのサーバーも無視するため両形式を併記して安全。常に明示送信することで、
    既定 OFF も enable_thinking=true での ON 化も全バックエンドで確実に効く。

    chat.completions.create(..., extra_body=thinking_extra_body(False)) のように渡す。
    """
    return {
        "enable_thinking": enable,
        "chat_template_kwargs": {"enable_thinking": enable},
    }


class LLMClient:
    """OpenAI 互換エンドポイントへ繋ぐクライアント（公式 openai SDK を土台にする）。

    respond() は非ストリームでは生成テキスト（str）を、stream=True ではテキスト断片の
    Iterator[str] を返す。マルチモーダルは images にローカルパス/URL を渡す。土台の
    openai クライアントは ``self.openai`` で直接使える（embeddings / tool calling など）。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "not-needed",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        # timeout は float / httpx.Timeout / None。None のときは openai の既定に任せる
        # （ローカルの巨大モデルは初回応答が遅いので、長め/無制限を渡せるようにする）。
        client_kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self.openai = OpenAI(**client_kwargs)
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
        resp = self.openai.chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    def _stream(self, params: dict[str, Any]) -> Iterator[str]:
        for chunk in self.openai.chat.completions.create(stream=True, **params):
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
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: Any = None,
) -> LLMClient:
    """**起動中のゲートウェイ**に繋いだ LLMClient を返す（サーバーは起動しない）。

    挙動は 2 つだけ:
      1. base_url が応答する … そのゲートウェイに繋いだ LLMClient を返す。
      2. 応答しない          … ServerNotRunningError を投げる（先にゲートウェイを起動して）。

    サーバーを自前で立てることはしない（サーバーを起動するのはゲートウェイ 1 箇所だけ、
    という運用のため）。繋ぐ前に死活確認したいときの薄いワンライナーで、挙動は
    `LLMClient(model, base_url=...)` と同じだが、未起動なら早めに親切なエラーを出す。
    """
    from .server import is_ready

    if not is_ready(base_url):
        from .gateway import ServerNotRunningError

        raise ServerNotRunningError(
            f"No gateway responding at {base_url}. Start it first with `local-llm-server` "
            "in the directory holding gateway.toml, then connect again."
        )
    return LLMClient(
        model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
