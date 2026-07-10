"""llama.cpp 自動導入（provisioner）のテスト。

実ダウンロード・実 GPU は使わず、検出（OS/arch/accel）・アセット名解決・導入フロー
（download/verify を差し替え）をユニットで検証する。アセット名は実 Releases（b9946）の
命名と一致することを固定値で確認する（上流の命名変更を検知できる）。
"""
from __future__ import annotations

import io
import os
import tarfile
import zipfile

import pytest

from local_llm_server import provisioner as pv


# --- 検出 --------------------------------------------------------------------

def test_detect_os(monkeypatch):
    monkeypatch.setattr(pv.sys, "platform", "darwin")
    assert pv.detect_os() == "macos"
    monkeypatch.setattr(pv.sys, "platform", "linux")
    assert pv.detect_os() == "linux"
    monkeypatch.setattr(pv.sys, "platform", "win32")
    assert pv.detect_os() == "windows"


def test_detect_arch(monkeypatch):
    monkeypatch.setattr(pv.platform, "machine", lambda: "arm64")
    assert pv.detect_arch() == "arm64"
    monkeypatch.setattr(pv.platform, "machine", lambda: "aarch64")
    assert pv.detect_arch() == "arm64"
    monkeypatch.setattr(pv.platform, "machine", lambda: "x86_64")
    assert pv.detect_arch() == "x64"
    monkeypatch.setattr(pv.platform, "machine", lambda: "AMD64")
    assert pv.detect_arch() == "x64"


def test_detect_accelerator_macos_is_metal():
    assert pv.detect_accelerator("macos") == "metal"


def test_detect_accelerator_gpu_prefers_vulkan(monkeypatch):
    # NVIDIA でも Vulkan（universal な GPU 経路）を選ぶ。CUDA は自動では選ばない。
    monkeypatch.setattr(pv, "_has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(pv, "_has_vulkan", lambda: False)
    assert pv.detect_accelerator("linux") == "vulkan"


def test_detect_accelerator_no_gpu_is_cpu(monkeypatch):
    monkeypatch.setattr(pv, "_has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(pv, "_has_vulkan", lambda: False)
    assert pv.detect_accelerator("windows") == "cpu"


# --- アセット名（実 Releases b9946 の命名と一致すること）---------------------

@pytest.mark.parametrize("os_name,arch,accel,expected", [
    ("macos", "arm64", "metal", "llama-b9946-bin-macos-arm64.tar.gz"),
    ("macos", "x64", "metal", "llama-b9946-bin-macos-x64.tar.gz"),
    ("linux", "x64", "cpu", "llama-b9946-bin-ubuntu-x64.tar.gz"),
    ("linux", "x64", "vulkan", "llama-b9946-bin-ubuntu-vulkan-x64.tar.gz"),
    ("linux", "arm64", "vulkan", "llama-b9946-bin-ubuntu-vulkan-arm64.tar.gz"),
    ("windows", "x64", "cpu", "llama-b9946-bin-win-cpu-x64.zip"),
    ("windows", "x64", "vulkan", "llama-b9946-bin-win-vulkan-x64.zip"),
    ("windows", "x64", "cuda", "llama-b9946-bin-win-cuda-12.4-x64.zip"),
    ("windows", "arm64", "cpu", "llama-b9946-bin-win-cpu-arm64.zip"),
])
def test_asset_name_matches_real_releases(os_name, arch, accel, expected):
    assert pv.asset_name("b9946", os_name, arch, accel) == expected


def test_asset_url():
    url = pv.asset_url("b9946", "llama-b9946-bin-macos-arm64.tar.gz")
    assert url == (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        "b9946/llama-b9946-bin-macos-arm64.tar.gz"
    )


# --- 管理ディレクトリ --------------------------------------------------------

def test_managed_root_windows_uses_localappdata(monkeypatch):
    monkeypatch.setattr(pv.os, "name", "nt")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\me\AppData\Local")
    root = pv.managed_root()
    assert root.endswith(os.path.join("local-llm-server", "llama.cpp"))
    assert "AppData" in root


def test_managed_root_unix_uses_xdg(monkeypatch):
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setenv("XDG_CACHE_HOME", "/home/me/.cache")
    assert pv.managed_root() == "/home/me/.cache/local-llm-server/llama.cpp"


def test_install_dir_is_unique_per_combo():
    a = pv.install_dir("b9946", "linux", "x64", "vulkan")
    b = pv.install_dir("b9946", "linux", "x64", "cpu")
    assert a != b and a.endswith("b9946-linux-vulkan-x64")


# --- 導入フロー（download/verify を差し替え）--------------------------------

def _fake_tarball(path, exe_name="llama-server"):
    """llama-server を1つ含む tar.gz を作る（ダウンロード結果の代役）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = b"#!/bin/sh\necho version\n"
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name=f"build/bin/{exe_name}")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))


def test_ensure_downloads_extracts_and_returns_path(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv, "detect_os", lambda: "linux")
    monkeypatch.setattr(pv, "detect_arch", lambda: "x64")

    downloaded = {}

    def fake_download(url, dest, timeout=300.0):
        downloaded["url"] = url
        _fake_tarball(dest)

    path = pv.ensure_llama_server(
        provision="auto", accel="cpu", build="b9946",
        download=fake_download, verify=lambda p, **k: True,
    )
    assert path.endswith(os.path.join("bin", "llama-server"))
    assert os.path.exists(path)
    assert "llama-b9946-bin-ubuntu-x64.tar.gz" in downloaded["url"]


def test_ensure_reuses_existing_install(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv, "detect_os", lambda: "linux")
    monkeypatch.setattr(pv, "detect_arch", lambda: "x64")
    calls = {"n": 0}

    def fake_download(url, dest, timeout=300.0):
        calls["n"] += 1
        _fake_tarball(dest)

    kw = dict(provision="auto", accel="cpu", build="b9946",
              download=fake_download, verify=lambda p, **k: True)
    p1 = pv.ensure_llama_server(**kw)
    p2 = pv.ensure_llama_server(**kw)  # 2 回目はダウンロードしない
    assert p1 == p2
    assert calls["n"] == 1


def test_ensure_system_uses_path(monkeypatch):
    monkeypatch.setattr(pv.shutil, "which", lambda name: "/usr/local/bin/llama-server")
    assert pv.ensure_llama_server(provision="system") == "/usr/local/bin/llama-server"


def test_ensure_system_missing_raises(monkeypatch):
    monkeypatch.setattr(pv.shutil, "which", lambda name: None)
    with pytest.raises(pv.ProvisionError):
        pv.ensure_llama_server(provision="system")


def test_ensure_verify_failure_raises(tmp_path, monkeypatch):
    # 導入できても --version に失敗（accel 不一致など）なら ProvisionError。
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv, "detect_os", lambda: "linux")
    monkeypatch.setattr(pv, "detect_arch", lambda: "x64")
    with pytest.raises(pv.ProvisionError):
        pv.ensure_llama_server(
            provision="auto", accel="vulkan", build="b9946",
            download=lambda url, dest, timeout=300.0: _fake_tarball(dest),
            verify=lambda p, **k: False,
        )


def test_ensure_download_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv, "detect_os", lambda: "linux")
    monkeypatch.setattr(pv, "detect_arch", lambda: "x64")

    def boom(url, dest, timeout=300.0):
        raise OSError("network down")

    with pytest.raises(pv.ProvisionError):
        pv.ensure_llama_server(provision="auto", accel="cpu", build="b9946",
                               download=boom, verify=lambda p, **k: True)


def test_extract_zip(tmp_path):
    # Windows 経路（.zip）の展開も動くこと。
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bin/llama-server.exe", "x")
    pv._extract(str(archive), str(tmp_path / "out"))
    assert (tmp_path / "out" / "bin" / "llama-server.exe").exists()
