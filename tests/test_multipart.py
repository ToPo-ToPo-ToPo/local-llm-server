"""multipart.parse / multipart.field のテスト（STT の multipart/form-data 解析）。"""
from __future__ import annotations

from local_llm_server import multipart


def _build(boundary: str, parts: list[tuple[str, str | None, str | None, bytes]]) -> bytes:
    """(name, filename, content_type, value) の並びから multipart 本文を組む。"""
    out = b""
    for name, filename, ctype, value in parts:
        out += f"--{boundary}\r\n".encode()
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disp += f'; filename="{filename}"'
        out += (disp + "\r\n").encode()
        if ctype:
            out += f"Content-Type: {ctype}\r\n".encode()
        out += b"\r\n" + value + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return out


def test_parse_fields_and_file():
    b = "X-B0undary-123"
    ct = f"multipart/form-data; boundary={b}"
    body = _build(b, [
        ("model", None, None, b"org/whisper"),
        ("response_format", None, None, b"text"),
        ("file", "a.wav", "audio/wav", b"RIFF\x00\x01audio"),
    ])
    parts = multipart.parse(body, ct)
    names = {p.name: p for p in parts}
    assert names["model"].filename is None
    assert names["model"].text() == "org/whisper"
    assert names["file"].filename == "a.wav"
    assert names["file"].content_type == "audio/wav"
    assert names["file"].value == b"RIFF\x00\x01audio"


def test_field_helper_returns_only_text_field():
    b = "b"
    ct = f"multipart/form-data; boundary={b}"
    body = _build(b, [("model", None, None, b"m1"), ("file", "x", "audio/wav", b"...")])
    assert multipart.field(body, ct, "model") == "m1"
    # ファイルパートは field() の対象外（filename 付きは除外）。
    assert multipart.field(body, ct, "file") is None
    assert multipart.field(body, ct, "missing") is None


def test_binary_payload_with_embedded_crlf_and_boundary_text():
    """境界ではない CRLF や、値中に境界文字列を含んでも壊れないこと。"""
    b = "sep"
    ct = f"multipart/form-data; boundary={b}"
    payload = b"\x00line1\r\nline2--sep-not-a-boundary\r\n\xff"
    body = _build(b, [("file", "a.bin", "application/octet-stream", payload)])
    parts = multipart.parse(body, ct)
    assert len(parts) == 1
    assert parts[0].value == payload


def test_quoted_boundary_in_content_type():
    b = "with space"
    ct = f'multipart/form-data; boundary="{b}"'
    body = _build(b, [("model", None, None, b"m")])
    assert multipart.field(body, ct, "model") == "m"


def test_non_multipart_or_malformed_returns_empty():
    assert multipart.parse(b"{}", "application/json") == []
    assert multipart.parse(b"", "multipart/form-data; boundary=b") == []
    # boundary 指定なし。
    assert multipart.parse(b"whatever", "multipart/form-data") == []
    # 本文に境界が出てこない。
    assert multipart.parse(b"no-boundary-here", "multipart/form-data; boundary=zzz") == []
