"""動画入力のゲートウェイ側フレーム展開。

llama-server も mlx-vlm も動画そのものは受けられないので、ゲートウェイが `video_url` を受けて
ffmpeg で等間隔にフレーム抽出し、`image_url`（base64 PNG）の列に展開してから上流へ渡す。
バックエンド非依存（llama-cpp / mlx-vlm どちらでも同じく効く）。ffmpeg は STT と同じく
システム PATH → pip 同梱 imageio-ffmpeg の順で解決する（brew/apt 不要）。

ffmpeg 呼び出し（run）と抽出（extract）は差し替え可能にしてあり、実 ffmpeg 無しで
本体ロジック（部品検出・置換・タイムスタンプ計算）をユニット検証できる。
"""
from __future__ import annotations

import base64
import binascii
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse

from . import net_safety

# ffmpeg が stderr に出す "  Duration: 00:00:12.34, ..." をパースする。
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def ffmpeg_exe() -> str | None:
    """使う ffmpeg のパス（システム PATH 優先、無ければ pip 同梱 imageio-ffmpeg）。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 未導入/取得失敗。呼び出し側で分かりやすく扱う
        return None


class VideoError(RuntimeError):
    """動画のフレーム展開に失敗した（ffmpeg 不在・デコード失敗など）。"""


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


# 動画はデコード後に PNG/base64 の複製も作るため、HTTP 本文上限（100 MiB）より低くする。
_MAX_REMOTE_BYTES = 64 * 1024 * 1024
_MAX_INLINE_BYTES = 32 * 1024 * 1024
_MAX_FRAME_BYTES = 24 * 1024 * 1024
_FFMPEG_INPUT_PROTOCOLS = ("-protocol_whitelist", "file,pipe")


def fetch_remote(url: str, max_bytes: int, *, timeout: float = 120.0) -> bytes:
    """SSRF検査済みのHTTP(S) URLをDNS固定・上限付きで一度だけ取得する。"""
    try:
        return net_safety.fetch_remote(url, max_bytes, timeout=timeout)
    except net_safety.RemoteFetchError as exc:
        raise VideoError(str(exc)) from exc


def _source_to_path(url: str, *, allow_local: bool = False) -> tuple[str, bool]:
    """video の url を ffmpeg が読めるローカルパスにする。戻り値 (path, is_temp)。

    - data URI（data:video/...;base64,...）→ 一時ファイルに書き出す（is_temp=True）。
    - http(s) URL → **一度だけ**一時ファイルへダウンロードする（is_temp=True）。ffmpeg に
      URL を直接渡すと probe + フレームごとに再取得され、8 フレーム設定で 9 回 DL になるため。
    - ローカルパス → allow_local=True の場合だけそのまま。
    壊れた base64・取得失敗は VideoError（呼び出し側が 400 に変換できるよう集約する）。
    """
    if url.startswith("data:"):
        _, _, b64 = url.partition(",")
        if len(b64) > ((_MAX_INLINE_BYTES + 2) // 3) * 4:
            raise VideoError(f"video の data URI が大きすぎます（> {_MAX_INLINE_BYTES} bytes）")
        try:
            raw = base64.b64decode(b64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise VideoError(f"video の data URI が壊れています（base64 不正）: {exc}") from exc
        fd, path = tempfile.mkstemp(suffix=".video")
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        return path, True
    if _is_url(url):
        fd, path = tempfile.mkstemp(suffix=".video")
        try:
            raw = fetch_remote(url, _MAX_REMOTE_BYTES)
            with os.fdopen(fd, "wb") as fh:
                fd = -1
                fh.write(raw)
        except VideoError:
            if fd >= 0:
                os.close(fd)
            _try_remove(path)
            raise
        except (OSError, ValueError) as exc:
            if fd >= 0:
                os.close(fd)
            _try_remove(path)
            raise VideoError(f"video URL を取得できません: {exc}") from exc
        return path, True
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme:
        raise VideoError(f"video URL のプロトコルは許可されていません: {scheme}")
    if not allow_local:
        raise VideoError("ローカル動画パスは同一マシンのクライアントだけが使用できます")
    return os.path.abspath(os.path.expanduser(url)), False


def _try_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _probe_duration(exe: str, src: str, run) -> float:
    """ffmpeg の情報出力から尺（秒）を得る。取れなければ 0.0（先頭付近から抜く）。"""
    try:
        proc = run(
            [exe, "-nostdin", *_FFMPEG_INPUT_PROTOCOLS, "-i", src],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    text = (getattr(proc, "stderr", b"") or b"").decode("utf-8", "replace")
    m = _DURATION_RE.search(text)
    if not m:
        return 0.0
    h, mnt, sec = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(sec)


def _frame_timestamps(duration: float, frames: int) -> list[float]:
    """尺を frames 等分した各区間の中央の時刻列（尺不明時は 0 を並べる＝先頭付近）。"""
    if frames <= 0:
        return []
    if duration <= 0:
        return [0.0] * frames
    return [duration * (i + 0.5) / frames for i in range(frames)]


def extract_frames(
    src: str, frames: int, max_edge: int, *, exe: str | None = None, run=subprocess.run,
    allow_local: bool = False,
) -> list[bytes]:
    """動画から frames 枚を等間隔で抜き、各 PNG のバイト列を返す（長辺 max_edge に縮小）。"""
    if not (1 <= frames <= 32):
        raise VideoError("video frames must be between 1 and 32")
    if not (64 <= max_edge <= 2048):
        raise VideoError("video max edge must be between 64 and 2048")
    exe = exe or ffmpeg_exe()
    if not exe:
        raise VideoError(
            "ffmpeg が見つかりません（システム PATH にも imageio-ffmpeg にも無い）。"
            "動画入力には ffmpeg が要ります。"
        )
    path, is_temp = _source_to_path(src, allow_local=allow_local)
    # 長辺を max_edge に収める（縦横どちら向きでも）。拡大はしない。filtergraph 中の
    # min() のカンマは \, でエスケープが必要（カンマはフィルタ区切りのため）。
    vf = (f"scale=min(iw\\,{max_edge}):min(ih\\,{max_edge})"
          f":force_original_aspect_ratio=decrease")
    out: list[bytes] = []
    try:
        duration = _probe_duration(exe, path, run)
        for t in _frame_timestamps(duration, frames):
            try:
                proc = run(
                    [exe, "-nostdin", "-ss", f"{t:.3f}",
                     *_FFMPEG_INPUT_PROTOCOLS, "-i", path,
                     "-frames:v", "1", "-vf", vf, "-f", "image2",
                     "-c:v", "png", "-"],
                    capture_output=True, timeout=60,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise VideoError(f"ffmpeg の実行に失敗: {exc}") from exc
            data = getattr(proc, "stdout", b"") or b""
            if len(data) > _MAX_FRAME_BYTES:
                raise VideoError(f"decoded video frame is too large (> {_MAX_FRAME_BYTES} bytes)")
            if getattr(proc, "returncode", 1) == 0 and data:
                out.append(data)
    finally:
        if is_temp:
            _try_remove(path)
    if not out:
        raise VideoError("動画からフレームを抽出できませんでした（ffmpeg 失敗）")
    return out


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _video_url_of(part: dict) -> str | None:
    """content パーツが動画なら url を返す（そうでなければ None）。

    受ける形（image_url にならう）: {"type": "video_url", "video_url": {"url": ...}}
    もしくは {"type": "input_video", "video": "..."} 等の緩い揺れも拾う。
    """
    ptype = part.get("type") or ""
    if "video" not in ptype:
        return None
    v = part.get("video_url") or part.get("video")
    if isinstance(v, dict):
        return v.get("url")
    if isinstance(v, str):
        return v
    return None


def request_has_video(payload: dict) -> bool:
    for msg in payload.get("messages", []) or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and _video_url_of(part):
                    return True
    return False


def expand_video_parts(
    payload: dict, frames: int = 8, max_edge: int = 768, *, extract=None,
    allow_local: bool = False,
) -> bool:
    """payload 内の video パーツをフレーム画像（image_url）の列に置き換える。

    その場で payload を書き換え、1 つでも展開したら True。動画が無ければ False（無変更）。
    抽出失敗（VideoError）は上位へ伝える（呼び出し側が 400 等に変換）。
    """
    # 既定は呼び出し時に解決する（module 属性の差し替え＝テストの monkeypatch を効かせるため）。
    custom_extract = extract is not None
    extract = extract or extract_frames
    changed = False
    for msg in payload.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_parts: list = []
        for part in content:
            url = _video_url_of(part) if isinstance(part, dict) else None
            if not url:
                new_parts.append(part)
                continue
            extracted = (
                extract(url, frames, max_edge)
                if custom_extract
                else extract(url, frames, max_edge, allow_local=allow_local)
            )
            for png in extracted:
                new_parts.append(
                    {"type": "image_url",
                     "image_url": {"url": _png_data_url(png)}})
            changed = True
        if changed:
            msg["content"] = new_parts
    return changed
