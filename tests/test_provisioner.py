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
    # 区切りは実行 OS の os.path に従うので os.path.join で期待値を組む（Windows でも一致）。
    assert pv.managed_root() == os.path.join(
        "/home/me/.cache", "local-llm-server", "llama.cpp")


def test_install_dir_is_unique_per_combo():
    a = pv.install_dir("b9946", "linux", "x64", "vulkan")
    b = pv.install_dir("b9946", "linux", "x64", "cpu")
    assert a != b and a.endswith("b9946-linux-vulkan-x64")


# --- 導入フロー（download/verify を差し替え）--------------------------------

def _fake_tarball(path, exe_name=None):
    """llama-server を1つ含む tar.gz を作る（ダウンロード結果の代役）。

    実行ファイル名は実行 OS の期待値（pv._EXE = Windows は llama-server.exe）に合わせる。
    そうしないと _find_llama_server が Windows で見つけられない。
    """
    exe_name = exe_name or pv._EXE
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
    assert path.endswith(os.path.join("bin", pv._EXE))
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


# --- ソースビルド（provision="build"）。実ビルドはせず subprocess を差し替える ------

def test_accel_cmake_flags():
    assert pv.accel_cmake_flags("cuda") == ["-DGGML_CUDA=ON"]
    assert pv.accel_cmake_flags("vulkan") == ["-DGGML_VULKAN=ON"]
    assert pv.accel_cmake_flags("metal") == []   # macOS 既定で有効
    assert pv.accel_cmake_flags("cpu") == []


def test_build_from_source_runs_cmake_and_returns_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv.shutil, "which", lambda n: f"/usr/bin/{n}")  # cmake/git あり
    ran = []

    def fake_run(cmd, check=False):
        ran.append(cmd)
        # cmake --build のときにビルド成果物（bin/llama-server）を作る。
        if cmd[:2] == ["cmake", "--build"]:
            binroot = os.path.join(cmd[2], "bin")
            os.makedirs(binroot, exist_ok=True)
            with open(os.path.join(binroot, pv._EXE), "w") as fh:
                fh.write("x")

    dest = pv.install_dir("b9946", "linux", "x64", "vulkan")
    out = pv.build_from_source("b9946", "linux", "x64", "vulkan", dest,
                               run=fake_run, verify=lambda p, **k: True)
    assert out.endswith(os.path.join("bin", pv._EXE)) and os.path.exists(out)
    # clone → configure（vulkan フラグ）→ build の順で走った。
    assert ran[0][0] == "git" and "--branch" in ran[0] and "b9946" in ran[0]
    assert "-DGGML_VULKAN=ON" in ran[1]
    assert ran[2][:2] == ["cmake", "--build"]


def test_build_from_source_without_toolchain_raises(monkeypatch):
    monkeypatch.setattr(pv.shutil, "which", lambda n: None)  # cmake/git 無し
    with pytest.raises(pv.BuildUnavailable):
        pv.build_from_source("b9946", "linux", "x64", "cpu", "/tmp/x",
                             run=lambda *a, **k: None)


def test_ensure_build_falls_back_to_prebuilt(tmp_path, monkeypatch):
    # provision="build" でビルド不可 → プリビルト自動導入にフォールバックする。
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv, "detect_os", lambda: "linux")
    monkeypatch.setattr(pv, "detect_arch", lambda: "x64")

    def no_build(*a, **k):
        raise pv.BuildUnavailable("no cmake")

    monkeypatch.setattr(pv, "build_from_source", no_build)
    downloaded = {}
    path = pv.ensure_llama_server(
        provision="build", accel="cpu", build="b9946",
        download=lambda url, dest, timeout=300.0: (
            downloaded.setdefault("url", url), _fake_tarball(dest)),
        verify=lambda p, **k: True,
    )
    assert os.path.exists(path)                      # プリビルトで導入できた
    assert "ubuntu-x64.tar.gz" in downloaded["url"]  # プリビルト経路を通った


# --- 総点検で見つかった不具合の回帰テスト ------------------------------------------

def test_ensure_reuses_installed_build_without_network(tmp_path, monkeypatch):
    # pin 未指定の 2 回目以降はネットワーク（latest_build）に触れず導入済みを再利用する。
    # これが無いと (1) オフラインで起動不能 (2) 上流の新リリースごとに再DL、が起きる。
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv, "detect_os", lambda: "linux")
    monkeypatch.setattr(pv, "detect_arch", lambda: "x64")

    # 1 回目: latest=b9946 を導入。
    monkeypatch.setattr(pv, "latest_build", lambda timeout=5.0: "b9946")
    kw = dict(provision="auto", accel="cpu",
              download=lambda url, dest, timeout=300.0: _fake_tarball(dest),
              verify=lambda p, **k: True)
    p1 = pv.ensure_llama_server(build=None, **kw)

    # 2 回目: ネットワーク断＋上流には新ビルドがある想定 → それでも導入済みを使う。
    def offline(timeout=5.0):
        raise OSError("network unreachable")

    monkeypatch.setattr(pv, "latest_build", offline)
    p2 = pv.ensure_llama_server(build=None, **kw)
    assert p2 == p1
    assert pv.last_info()["build"] == "b9946"


def test_ensure_offline_without_install_raises_provision_error(tmp_path, monkeypatch):
    # オフラインで未導入なら、素の URLError ではなく案内付き ProvisionError（起動側で握れる）。
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    monkeypatch.setattr(pv, "detect_os", lambda: "linux")
    monkeypatch.setattr(pv, "detect_arch", lambda: "x64")

    def offline(timeout=5.0):
        raise OSError("network unreachable")

    monkeypatch.setattr(pv, "latest_build", offline)
    with pytest.raises(pv.ProvisionError):
        pv.ensure_llama_server(provision="auto", accel="cpu", build=None,
                               download=lambda *a, **k: None,
                               verify=lambda p, **k: True)


def test_installed_builds_sorted_newest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pv.os, "name", "posix")
    root = pv.managed_root()
    for name in ("b9900-linux-cpu-x64", "b10002-linux-cpu-x64", "b9946-linux-cpu-x64",
                 "b9999-linux-vulkan-x64",   # accel 違いは含めない
                 "src-b9946", "junk"):        # 形式外は無視
        os.makedirs(os.path.join(root, name), exist_ok=True)
    assert pv.installed_builds("linux", "x64", "cpu") == ["b10002", "b9946", "b9900"]


def test_last_info_records_system_without_accel(monkeypatch):
    # system はユーザー管理バイナリ＝素性不明。accel を None にして auto フラグの対象外にする。
    monkeypatch.setattr(pv.shutil, "which", lambda name: "/usr/bin/llama-server")
    pv.ensure_llama_server(provision="system")
    info = pv.last_info()
    assert info["provision"] == "system" and info["accel"] is None
