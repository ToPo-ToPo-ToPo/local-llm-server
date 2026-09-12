"""動画入力のフレーム展開（video.py）のテスト。

実 ffmpeg・実動画は使わず、フレーム抽出（extract / run）を差し替えて本体ロジック
（部品検出・タイムスタンプ計算・video_url→image_url 置換）を検証する。
"""
from __future__ import annotations

import base64
import pytest

from local_llm_server import net_safety, video


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

    frames = video.extract_frames(
        "/local/v.mp4", 4, 512, exe="/ff", run=fake_run, allow_local=True,
    )
    assert frames == [b"PNGDATA"] * 4
    # probe 1 回 + フレーム 4 回。scale フィルタに max_edge（長辺キャップ・\, エスケープ）が入る。
    extract_cmds = [c for c in calls if c[-1] == "-"]
    assert len(extract_cmds) == 4
    assert any("min(iw\\,512)" in " ".join(c) for c in extract_cmds)
    assert all("file,pipe" in c for c in calls)  # 入力からネットワークプロトコルを起動させない


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


# --- 総点検で見つかった不具合の回帰テスト ------------------------------------------

def test_scale_filter_caps_long_edge_both_orientations():
    # 縦長動画で「長辺 max_edge」が効くこと（旧実装は幅しか制限せず 240x1000 が素通りした）。
    # フィルタ文字列に width/height 両方の min とアスペクト維持縮小が入ることを確認する。
    calls = []

    def fake_run(cmd, capture_output=False, timeout=None):
        calls.append(cmd)
        if cmd[-1] != "-":
            return _Proc(stderr=b"  Duration: 00:00:02.00,")
        return _Proc(stdout=b"PNG", returncode=0)

    video.extract_frames("x.mp4", 1, 512, exe="/ff", run=fake_run, allow_local=True)
    vf = [c[c.index("-vf") + 1] for c in calls if "-vf" in c][0]
    assert "min(iw\\,512)" in vf and "min(ih\\,512)" in vf
    assert "force_original_aspect_ratio=decrease" in vf


def test_broken_data_uri_raises_video_error():
    # 壊れた base64 は VideoError（旧実装は binascii.Error が漏れてハンドラを貫通した）。
    try:
        video.extract_frames("data:video/mp4;base64,!!!bad!!!", 2, 256, exe="/ff",
                             run=lambda *a, **k: _Proc())
        assert False, "should raise"
    except video.VideoError:
        pass


@pytest.mark.parametrize("frames, edge", [(0, 512), (33, 512), (1, 63), (1, 2049)])
def test_extract_frames_rejects_resource_exhausting_settings(frames, edge):
    with pytest.raises(video.VideoError):
        video.extract_frames("x.mp4", frames, edge, exe="/ff", run=lambda *_a, **_k: _Proc())


def test_remote_url_downloaded_once_not_per_frame(monkeypatch, tmp_path):
    # http URL は一度だけ一時ファイルへ落とす（旧実装は probe+フレームごとに再取得 = N+1 回DL）。
    fetches = []

    def fake_fetch(url, _max_bytes, **_kw):
        fetches.append(url)
        return b"VIDEODATA"

    monkeypatch.setattr(video, "fetch_remote", fake_fetch)
    runs = []

    def fake_run(cmd, capture_output=False, timeout=None):
        runs.append(cmd)
        if cmd[-1] != "-":
            return _Proc(stderr=b"  Duration: 00:00:04.00,")
        return _Proc(stdout=b"PNG", returncode=0)

    frames = video.extract_frames("http://x/v.mp4", 4, 256, exe="/ff", run=fake_run)
    assert len(frames) == 4
    # URL の取得は 1 回だけ。ffmpeg にはローカル一時ファイルが渡る（URL は渡らない）。
    assert fetches == ["http://x/v.mp4"]
    assert all("http://x/v.mp4" not in " ".join(c) for c in runs)


def test_private_and_unknown_video_sources_are_rejected(monkeypatch):
    monkeypatch.setattr(
        net_safety.socket, "getaddrinfo",
        lambda *_a, **_k: [(2, 1, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(video.VideoError):
        video._source_to_path("http://localhost/private.mp4")
    with pytest.raises(video.VideoError):
        video._source_to_path("rtsp://camera/video")
    with pytest.raises(video.VideoError):
        video._source_to_path("/tmp/private.mp4", allow_local=False)


def test_remote_fetch_pins_validated_dns_address(monkeypatch):
    used = {}

    class _Response:
        status = 200

        def getheader(self, _name):
            return None

        def read(self, _size=-1):
            if used.get("read"):
                return b""
            used["read"] = True
            return b"media"

    class _Connection:
        sock = None

        def request(self, method, path, headers):
            used.update(method=method, path=path, headers=headers)

        def getresponse(self):
            return _Response()

        def close(self):
            used["closed"] = True

    monkeypatch.setattr(
        net_safety, "resolve_public_addresses", lambda _parsed: ["93.184.216.34"],
    )

    def connection(parsed, address, timeout):
        used.update(host=parsed.hostname, address=address, timeout=timeout)
        return _Connection()

    monkeypatch.setattr(net_safety, "_connection", connection)
    assert net_safety.fetch_remote("https://example.com/a?q=1", 100) == b"media"
    assert used["host"] == "example.com"
    assert used["address"] == "93.184.216.34"
    assert used["headers"]["Host"] == "example.com"
    assert used["closed"] is True


def test_remote_fetch_revalidates_redirect_target(monkeypatch):
    class _Redirect:
        status = 302

        def getheader(self, name):
            return "http://127.0.0.1/private" if name == "Location" else None

    class _Connection:
        sock = None

        def request(self, *_a, **_k):
            pass

        def getresponse(self):
            return _Redirect()

        def close(self):
            pass

    def resolve(parsed):
        if parsed.hostname == "example.com":
            return ["93.184.216.34"]
        raise net_safety.RemoteFetchError("private redirect blocked")

    monkeypatch.setattr(net_safety, "resolve_public_addresses", resolve)
    monkeypatch.setattr(net_safety, "_connection", lambda *_a: _Connection())
    with pytest.raises(net_safety.RemoteFetchError, match="private redirect"):
        net_safety.fetch_remote("https://example.com/video", 100)


def test_subprocess_failure_wrapped_as_video_error():
    # subprocess の失敗（Timeout 等）は VideoError に包む（生で漏らさない）。
    import subprocess as sp

    def boom(cmd, capture_output=False, timeout=None):
        if cmd[-1] == "-":
            raise sp.TimeoutExpired(cmd, timeout)
        return _Proc(stderr=b"  Duration: 00:00:02.00,")

    try:
        video.extract_frames("x.mp4", 1, 256, exe="/ff", run=boom)
        assert False, "should raise"
    except video.VideoError:
        pass
