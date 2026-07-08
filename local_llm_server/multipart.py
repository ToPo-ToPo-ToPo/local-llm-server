"""`multipart/form-data` の最小パーサ（標準ライブラリのみ）。

STT（`/v1/audio/transcriptions`）は OpenAI 仕様どおり JSON ではなく multipart で
送られてくる。ゲートウェイのルータは本文から `model` フィールドだけを取り出して振り分け、
STT サーバは `file`（音声）＋任意フィールドを取り出す。Python 3.13 で `cgi` が削除された
ため、その用途に必要な範囲だけを自前で解析する（RFC 7578 のサブセット）。

大きな音声本文（数十 MB）を丸ごとコピーしないよう、境界の探索は `bytes.find` の
インデックス操作だけで行い、各パートの値はスライスで一度だけ切り出す。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Part:
    """multipart の 1 パート。"""

    name: str                 # フォームフィールド名（Content-Disposition の name）
    filename: str | None      # ファイルパートなら元ファイル名、テキストフィールドなら None
    content_type: str | None  # パートの Content-Type（あれば）
    value: bytes              # パート本体（生バイト）

    def text(self, encoding: str = "utf-8") -> str:
        return self.value.decode(encoding, "replace")


def _boundary(content_type: str) -> bytes | None:
    """Content-Type ヘッダから boundary を取り出す（無ければ None）。"""
    for token in content_type.split(";"):
        token = token.strip()
        if token.lower().startswith("boundary="):
            b = token[len("boundary="):].strip().strip('"')
            return b.encode("latin-1") if b else None
    return None


def _parse_headers(raw: bytes) -> dict[str, str]:
    """パート冒頭のヘッダブロック（`\\r\\n\\r\\n` より前）を小文字キーの dict にする。"""
    headers: dict[str, str] = {}
    for line in raw.split(b"\r\n"):
        if not line or b":" not in line:
            continue
        key, _, val = line.partition(b":")
        headers[key.decode("latin-1").strip().lower()] = val.decode("latin-1").strip()
    return headers


def _disposition_params(value: str) -> dict[str, str]:
    """`form-data; name="x"; filename="y"` を {name, filename} に分解する。"""
    params: dict[str, str] = {}
    for token in value.split(";"):
        token = token.strip()
        if "=" in token:
            k, _, v = token.partition("=")
            params[k.strip().lower()] = v.strip().strip('"')
    return params


def parse(body: bytes, content_type: str) -> list[Part]:
    """multipart/form-data の本文を Part のリストに分解する。

    パースできない（boundary 無し・形式不正）ときは空リストを返す（呼び出し側は
    フィールド不在として扱う）。想定外の形でも例外を投げないことを優先する。
    """
    if not body or "multipart/form-data" not in content_type.lower():
        return []
    boundary = _boundary(content_type)
    if not boundary:
        return []
    delimiter = b"--" + boundary
    parts: list[Part] = []
    # 最初の境界の位置。前段プリアンブルは無視する。
    idx = body.find(delimiter)
    if idx < 0:
        return []
    idx += len(delimiter)
    while True:
        # 境界直後は CRLF（次パート）か "--"（終端）。
        if body[idx:idx + 2] == b"--":
            break
        if body[idx:idx + 2] == b"\r\n":
            idx += 2
        # 次の境界までが 1 パート。
        nxt = body.find(b"\r\n" + delimiter, idx)
        if nxt < 0:
            break
        segment = body[idx:nxt]
        head, _, value = segment.partition(b"\r\n\r\n")
        headers = _parse_headers(head)
        disp = _disposition_params(headers.get("content-disposition", ""))
        name = disp.get("name")
        if name is not None:
            parts.append(Part(
                name=name,
                filename=disp.get("filename"),
                content_type=headers.get("content-type"),
                value=value,
            ))
        idx = nxt + 2 + len(delimiter)  # \r\n + delimiter を飛ばして次パートへ
    return parts


def field(body: bytes, content_type: str, name: str) -> str | None:
    """テキストフィールド 1 つの値だけを取り出す（ルータの model 抽出用）。"""
    for part in parse(body, content_type):
        if part.name == name and part.filename is None:
            return part.text()
    return None
