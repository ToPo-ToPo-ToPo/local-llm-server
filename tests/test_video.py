"""動画入力のフレーム展開（video.py）のテスト。

実 ffmpeg・実動画は使わず、フレーム抽出（extract / run）を差し替えて本体ロジック
（部品検出・タイムスタンプ計算・video_url→image_url 置換）を検証する。
"""
from __future__ import annotations

import base64

from local_llm_server import video


class _Proc:
    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_frame_timestamps_even_spacing():
    ts = video._frame_timestamps(10.0, 4)
    assert ts == [1.25, 3.75, 6.25, 8.75]           # 各 1/4 区間の中央
    assert video._frame_timestamps(0.0, 3) == [0.0, 0.0, 0.0]  # 尺不明は先頭付近
    assert video._frame_timestamps(10.0, 0) == []


def test_request_has_video_detects_various_shapes():
    assert video.request_has_video({"messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]}]})
    assert video.request_has_video({"messages": [{"role": "user", "content": [
        {"type": "input_video", "video": "http://x/v.mp4"}]}]})
    assert not video.request_has_video({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]})
    assert not video.request_has_video({"messages": [{"role": "user", "content": "hi"}]})


def test_extract_frames_calls_ffmpeg_per_timestamp(monkeypatch):
    calls = []

    def fake_run(cmd, capture_output=False, timeout=None):
        calls.append(cmd)
        if "-i" in cmd and cmd[-1] != "-":            # probe（尺取得）
            return _Proc(stderr=b"  Duration: 00:00:08.00, start: 0.0")
        return _Proc(stdout=b"PNGDATA", returncode=0)  # フレーム抽出

    frames = video.extract_frames("http://x/v.mp4", 4, 512, exe="/ff", run=fake_run)
    assert frames == [b"PNGDATA"] * 4
    # probe 1 回 + フレーム 4 回。scale フィルタに max_edge が入る。
    extract_cmds = [c for c in calls if c[-1] == "-"]
    assert len(extract_cmds) == 4
    assert any("min(iw,512)" in " ".join(c) for c in extract_cmds)


def test_extract_frames_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: None)
    try:
        video.extract_frames("x", 4, 512)
        assert False, "should raise"
    except video.VideoError:
        pass


def test_expand_video_parts_replaces_with_images():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "video_url", "video_url": {"url": "http://x/v.mp4"}},
    ]}]}
    changed = video.expand_video_parts(
        payload, frames=3, max_edge=256,
        extract=lambda url, n, edge: [b"a", b"b", b"c"])
    assert changed is True
    parts = payload["messages"][0]["content"]
    # text は残り、video は 3 枚の image_url に展開される。
    assert parts[0]["type"] == "text"
    imgs = [p for p in parts if p["type"] == "image_url"]
    assert len(imgs) == 3
    assert imgs[0]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"a").decode())


def test_expand_video_parts_noop_without_video():
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    assert video.expand_video_parts(payload, extract=lambda *a: [b"x"]) is False
