"""画像入力の長辺リサイズ（ゲートウェイ側の前処理）。

解像度上限の無い VLM（例: Qwen3.6 / Ornith 系の qwen3_5・qwen3_5_moe。preprocessor が
`max_pixels` 未設定で `longest_edge` が 16MP）に巨大画像を渡すと、1 枚が数千〜万の vision
トークンに展開され、prefill と get_rope_index（O(n) の Python ループ）で数十秒〜数分かかる。
外からは「異常に遅い / 動作が止まる」ように見える（12MP 写真で実測 46s）。

動画入力が `video_max_edge` でフレームを縮小するのと同じ発想で、静止画も上流へ渡す前に
**長辺 `max_edge` に収まるよう縮小**する（拡大はしない）。バックエンド非依存（mlx-vlm /
llama-cpp どちらでも効く）。対象は data URL（`data:image/...;base64,...`）と、トップレベル
`images=[...]` の base64 文字列。リモート URL（http/https）は上流が自分で取得するため対象外
（ゲートウェイが代理取得すると SSRF/プライバシー面の別懸念が出る）。
"""
from __future__ import annotations

import base64
import binascii
import io
import re

# data URL のヘッダを緩く拾う（`data:image/png;base64,....`）。base64 以外の稀な形は対象外。
_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+)?;base64,(.*)$", re.IGNORECASE | re.DOTALL)

# PIL.format → data URL の mime。JPEG は screenshot 的な線画/文字を劣化させにくいよう PNG を優先し、
# 元が JPEG のときだけ JPEG のまま返す（写真のペイロード肥大を避ける）。
_FMT_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def _resize_encoded(raw: bytes, max_edge: int) -> bytes | None:
    """画像バイト列を長辺 max_edge に縮小して返す。縮小不要・失敗時は None（＝無変更）。

    PIL が無い環境（Pillow 未導入）では import 失敗を握って None を返し、ゲートウェイ自体は
    そのまま動かす（最適化が効かないだけ）。
    """
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - Pillow 未導入なら縮小せず素通し
        return None
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:  # noqa: BLE001 - 壊れた/未対応画像は素通し（上流に委ねる）
        return None
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return None  # 既に十分小さい
    scale = max_edge / float(longest)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    fmt = (img.format or "PNG").upper()
    if fmt not in _FMT_MIME:
        fmt = "PNG"
    try:
        resized = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        if fmt == "JPEG":
            # JPEG は alpha 非対応。RGBA/P はの RGB に落としてから保存する。
            if resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            resized.save(buf, format="JPEG", quality=90)
        else:
            resized.save(buf, format=fmt)
    except Exception:  # noqa: BLE001 - エンコード不能なら無変更
        return None
    return buf.getvalue()


def _mime_for(raw: bytes) -> str:
    try:
        from PIL import Image
        fmt = (Image.open(io.BytesIO(raw)).format or "PNG").upper()
        return _FMT_MIME.get(fmt, "image/png")
    except Exception:  # noqa: BLE001
        return "image/png"


def _shrink_data_url(value: str, max_edge: int) -> str | None:
    """data URL を縮小した data URL にして返す。data URL でない/縮小不要なら None。"""
    m = _DATA_URL_RE.match(value.strip())
    if not m:
        return None
    try:
        raw = base64.b64decode(m.group(2), validate=False)
    except (binascii.Error, ValueError):
        return None
    small = _resize_encoded(raw, max_edge)
    if small is None:
        return None
    mime = m.group(1) or _mime_for(small)
    return f"data:{mime};base64," + base64.b64encode(small).decode("ascii")


def _shrink_bare_or_data(value: str, max_edge: int) -> str | None:
    """トップレベル images=[...] 用。data URL か生 base64 のどちらでも縮小を試みる。"""
    out = _shrink_data_url(value, max_edge)
    if out is not None:
        return out
    if value.startswith("data:"):
        return None  # data URL だが対象外 mime／縮小不要
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None  # 生 base64 でない（URL 等）→ 対象外
    small = _resize_encoded(raw, max_edge)
    if small is None:
        return None
    return base64.b64encode(small).decode("ascii")


def _shrink_part(part: dict, max_edge: int) -> bool:
    """content パーツ（dict）内の画像 URL を in-place で縮小。変更したら True。"""
    t = part.get("type")
    if not (isinstance(t, str) and "image" in t):
        return False
    # 形の揺れを吸収: image_url / image / input_image。値は {"url": ...} か文字列。
    for key in ("image_url", "image", "input_image"):
        if key not in part:
            continue
        v = part[key]
        if isinstance(v, dict):
            url = v.get("url")
            if isinstance(url, str):
                new = _shrink_data_url(url, max_edge)
                if new is not None:
                    v["url"] = new
                    return True
        elif isinstance(v, str):
            new = _shrink_data_url(v, max_edge)
            if new is not None:
                part[key] = new
                return True
    return False


def downscale_image_parts(payload: dict, max_edge: int) -> bool:
    """payload 内の画像を長辺 max_edge に縮小する（in-place）。1 つでも縮小したら True。

    max_edge <= 0 は無効（何もしない）。data URL / トップレベル images の base64 が対象。
    リモート URL は対象外（上流が取得する）。壊れた画像・未対応形式・Pillow 未導入は素通し。
    """
    if not max_edge or max_edge <= 0:
        return False
    changed = False
    for msg in payload.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and _shrink_part(part, max_edge):
                changed = True
    images = payload.get("images")
    if isinstance(images, list):
        for i, item in enumerate(images):
            if isinstance(item, str):
                new = _shrink_bare_or_data(item, max_edge)
                if new is not None:
                    images[i] = new
                    changed = True
            elif isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str):
                    new = _shrink_data_url(url, max_edge)
                    if new is not None:
                        item["url"] = new
                        changed = True
    return changed
