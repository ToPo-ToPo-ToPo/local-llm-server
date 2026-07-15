"""画像入力の長辺リサイズ（image.py）のテスト。

実モデルは使わず、Pillow で合成した画像を data URL / トップレベル images に載せ、
downscale_image_parts が「大きい画像だけを長辺 max_edge に縮小し、小さい画像・リモート URL・
壊れた入力・無効(0)は素通しする」ことを検証する。
"""
from __future__ import annotations

import base64
import io

from PIL import Image

from local_llm_server import image


def _data_url(w: int, h: int, fmt: str = "PNG") -> str:
    img = Image.new("RGB", (w, h), (120, 30, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _dims_of_data_url(url: str) -> tuple[int, int]:
    raw = base64.b64decode(url.split(",", 1)[1])
    return Image.open(io.BytesIO(raw)).size


def _dims_of_b64(b64: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def test_downscales_large_image_url_in_content():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": _data_url(4032, 3024)}},
    ]}]}
    assert image.downscale_image_parts(payload, 1568) is True
    url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert max(_dims_of_data_url(url)) == 1568          # 長辺が上限に一致
    assert _dims_of_data_url(url) == (1568, 1176)       # アスペクト比維持


def test_small_image_is_untouched():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url(400, 300)}}]}]}
    assert image.downscale_image_parts(payload, 1568) is False


def test_string_form_image_url_part():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": _data_url(3000, 2000, "JPEG")}]}]}
    assert image.downscale_image_parts(payload, 1568) is True
    url = payload["messages"][0]["content"][0]["image_url"]
    assert url.startswith("data:image/jpeg;base64,")     # 元が JPEG なら JPEG のまま
    assert max(_dims_of_data_url(url)) == 1568


def test_top_level_images_raw_base64():
    img = Image.new("RGB", (2000, 1000))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = {"images": [base64.b64encode(buf.getvalue()).decode()]}
    assert image.downscale_image_parts(payload, 1568) is True
    assert max(_dims_of_b64(payload["images"][0])) == 1568


def test_top_level_images_dict_url():
    payload = {"images": [{"url": _data_url(2500, 2500)}]}
    assert image.downscale_image_parts(payload, 1568) is True
    assert max(_dims_of_data_url(payload["images"][0]["url"])) == 1568


def test_remote_url_untouched():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]}
    assert image.downscale_image_parts(payload, 1568) is False


def test_disabled_when_max_edge_zero():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url(4000, 3000)}}]}]}
    assert image.downscale_image_parts(payload, 0) is False


def test_broken_data_url_is_passed_through():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-valid-b64!!"}}]}]}
    # 壊れた画像は縮小せず素通し（例外を投げない・無変更）
    assert image.downscale_image_parts(payload, 1568) is False


def test_no_upscale_of_medium_image():
    # 長辺 1200 の画像を max_edge 1568 に「拡大」しない
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url(1200, 800)}}]}]}
    assert image.downscale_image_parts(payload, 1568) is False


def test_png_with_alpha_downscaled_stays_png():
    img = Image.new("RGBA", (3000, 2000), (10, 20, 30, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}]}]}
    assert image.downscale_image_parts(payload, 1568) is True
    out = payload["messages"][0]["content"][0]["image_url"]["url"]
    assert out.startswith("data:image/png;base64,")
    assert max(_dims_of_data_url(out)) == 1568
