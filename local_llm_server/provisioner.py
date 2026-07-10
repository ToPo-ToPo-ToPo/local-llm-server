"""llama.cpp（llama-server）バイナリの自動導入（プロビジョナ）。

Linux / Windows / macOS のどれでも `uv run gw` するだけで llama-server が使えるように、
OS・CPU アーキ・アクセラレータを自動判定して **ggml-org/llama.cpp の GitHub Releases から
プリビルトバイナリをダウンロード**し、管理ディレクトリへ展開して絶対パスで起動する。
PATH は汚さない。PATH に既に llama-server があるなら `provision = "system"` で従来どおり使う。

設計方針（実アセット命名 b9946 時点の調査に基づく）:
  - 拡張子は macOS/Linux が .tar.gz、Windows が .zip。
  - **auto の GPU 経路は Vulkan**。NVIDIA/AMD/Intel 共通の単一アセットで追加ランタイム不要。
    Linux には CUDA プリビルトが無く、Windows CUDA は別途 cudart DLL が要るため、
    「導入が簡単・確実に動く」を優先して Vulkan を既定の GPU 経路にする。
  - macOS は Metal がバイナリに内蔵（accel トークン無し）。
  - CUDA は Windows 限定の明示 opt-in（accel = "cuda"）。

ソースビルド（provision = "build"）は別フェーズ。ここでは auto / system を実装する。
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

_REPO = "ggml-org/llama.cpp"
_RELEASES_API = f"https://api.github.com/repos/{_REPO}/releases"
# Releases のダウンロード URL。<build> はタグ（例 "b9946"）、<name> はアセット名。
_DL_URL = f"https://github.com/{_REPO}/releases/download/{{build}}/{{name}}"

# llama-server 実行ファイル名（OS 依存）。
_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


# --- プラットフォーム検出 ----------------------------------------------------

def detect_os() -> str:
    """"macos" | "linux" | "windows"。未知は ValueError。"""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    if os.name == "nt" or sys.platform.startswith("win"):
        return "windows"
    raise ValueError(f"unsupported OS: {sys.platform!r}")


def detect_arch() -> str:
    """"x64" | "arm64"。llama.cpp のアセット命名に合わせて正規化する。"""
    m = platform.machine().lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("x86_64", "amd64", "x64"):
        return "x64"
    raise ValueError(f"unsupported CPU arch: {platform.machine()!r}")


def _has_nvidia_gpu() -> bool:
    """NVIDIA GPU の存在を nvidia-smi で確認する（無ければ False）。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run(
            [exe, "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _has_vulkan() -> bool:
    """Vulkan 対応 GPU/ローダの存在を vulkaninfo で確認する（無ければ False）。"""
    exe = shutil.which("vulkaninfo")
    if not exe:
        return False
    try:
        return subprocess.run(
            [exe, "--summary"], capture_output=True, timeout=5
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def detect_accelerator(os_name: str | None = None) -> str:
    """auto 用のアクセラレータ自動判定。"metal" | "vulkan" | "cpu"。

    macOS は常に metal（バイナリ内蔵）。Linux/Windows は GPU を検出できれば
    vulkan（universal な GPU 経路）、できなければ cpu。誤検出しても accel を明示すれば
    上書きできる。CUDA は自動では選ばない（Windows 限定・cudart 依存のため明示 opt-in）。
    """
    os_name = os_name or detect_os()
    if os_name == "macos":
        return "metal"
    if _has_nvidia_gpu() or _has_vulkan():
        return "vulkan"
    return "cpu"


# --- アセット名の解決 --------------------------------------------------------

# CUDA を明示指定したときの既定 CUDA バージョン（Windows のみ）。実アセット: win-cuda-12.4-x64。
_DEFAULT_CUDA = "12.4"


def asset_name(build: str, os_name: str, arch: str, accel: str) -> str:
    """Releases のアセットファイル名を組み立てる。

    実アセット例（b9946）:
      macos-arm64            → llama-<b>-bin-macos-arm64.tar.gz
      linux cpu   x64        → llama-<b>-bin-ubuntu-x64.tar.gz
      linux vulkan x64       → llama-<b>-bin-ubuntu-vulkan-x64.tar.gz
      windows cpu  x64       → llama-<b>-bin-win-cpu-x64.zip
      windows vulkan x64     → llama-<b>-bin-win-vulkan-x64.zip
      windows cuda x64       → llama-<b>-bin-win-cuda-12.4-x64.zip
    """
    if os_name == "macos":
        # macOS は accel トークン無し（Metal 内蔵）。
        return f"llama-{build}-bin-macos-{arch}.tar.gz"
    if os_name == "linux":
        # Releases 上は "ubuntu" 命名。CPU は accel トークン無し。
        token = "" if accel in ("cpu", "metal") else f"{accel}-"
        return f"llama-{build}-bin-ubuntu-{token}{arch}.tar.gz"
    if os_name == "windows":
        if accel == "cuda":
            return f"llama-{build}-bin-win-cuda-{_DEFAULT_CUDA}-{arch}.zip"
        # Windows は CPU も明示トークン（win-cpu-...）。GPU は vulkan。
        token = "cpu" if accel in ("cpu", "metal") else accel
        return f"llama-{build}-bin-win-{token}-{arch}.zip"
    raise ValueError(f"unsupported os: {os_name!r}")


def asset_url(build: str, name: str) -> str:
    return _DL_URL.format(build=build, name=name)


def latest_build(timeout: float = 5.0) -> str:
    """最新リリースのタグ（例 "b9946"）を GitHub API から取得する。"""
    req = urllib.request.Request(
        f"{_RELEASES_API}/latest", headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["tag_name"]


# --- 管理ディレクトリと導入 --------------------------------------------------

def managed_root() -> str:
    """バイナリを置く管理ディレクトリ（PATH は汚さない）。

    Windows は %LOCALAPPDATA%、その他は XDG（~/.cache）に置く。
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    return os.path.join(base, "local-llm-server", "llama.cpp")


def install_dir(build: str, os_name: str, arch: str, accel: str) -> str:
    """この (build, os, arch, accel) 組み合わせの展開先ディレクトリ。"""
    return os.path.join(managed_root(), f"{build}-{os_name}-{accel}-{arch}")


def installed_builds(os_name: str, arch: str, accel: str) -> list[str]:
    """管理ディレクトリに導入済みのビルドタグ一覧（新しい順）。

    `pin` 未指定の 2 回目以降の起動はここから再利用する——毎回 GitHub API に最新を
    照会すると、オフラインで起動できず、llama.cpp の頻繁なリリース（日に数回）のたびに
    数十 MB を再ダウンロードしてしまうため。更新したいときは `pin` を変えるか、
    管理ディレクトリを消して次回起動で最新を取らせる。
    """
    suffix = f"-{os_name}-{accel}-{arch}"
    try:
        names = os.listdir(managed_root())
    except OSError:
        return []
    builds = [n[: -len(suffix)] for n in names if n.endswith(suffix)]
    builds = [b for b in builds if re.fullmatch(r"b\d+", b)]
    return sorted(builds, key=lambda b: int(b[1:]), reverse=True)


# 直近に ensure_llama_server が解決した素性（build/accel/provision）。TUI・/admin/status の
# 表示用に、呼び出し側（daemon）が last_info() で取得する。
_LAST_INFO: dict | None = None


def last_info() -> dict | None:
    return _LAST_INFO


def _find_llama_server(root: str) -> str | None:
    """展開ディレクトリ配下から llama-server 実行ファイルを探す（無ければ None）。"""
    for dirpath, _dirs, files in os.walk(root):
        if _EXE in files:
            return os.path.join(dirpath, _EXE)
    return None


def _download(url: str, dest: str, timeout: float = 300.0) -> None:
    """url を dest へダウンロードする（親ディレクトリは作成）。"""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "local-llm-server"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def _extract(archive: str, dest: str) -> None:
    """.tar.gz / .zip を dest へ展開する。"""
    os.makedirs(dest, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            # filter="data" はパストラバーサル等を弾く（信頼できない DL 物の安全な展開）。
            # Python 3.12+ で追加。古い版では TypeError になるのでフォールバックする。
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                tf.extractall(dest)


def _verify(binary: str, timeout: float = 15.0) -> bool:
    """llama-server --version が動くか（正しく展開・実行できるか）を確認する。"""
    try:
        proc = subprocess.run(
            [binary, "--version"], capture_output=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return False
    # llama-server --version は "version: ..." を stderr に出して 0 で終わる（版により stdout）。
    out = (proc.stdout + proc.stderr).decode("utf-8", "replace").lower()
    return proc.returncode == 0 and "version" in out


class ProvisionError(RuntimeError):
    """llama-server の自動導入に失敗した（ダウンロード・展開・検証のいずれか）。"""


class BuildUnavailable(RuntimeError):
    """ソースビルドができない/失敗した（ツールチェーン不在・ビルド失敗など）。

    provision='build' でこれが起きたときは、プリビルト自動導入（auto）へフォールバックする。
    """


# accel → cmake のアクセラレータ有効化フラグ。metal は macOS 既定で有効なので追加不要。
def accel_cmake_flags(accel: str) -> list[str]:
    return {
        "cuda": ["-DGGML_CUDA=ON"],
        "vulkan": ["-DGGML_VULKAN=ON"],
        "hip": ["-DGGML_HIP=ON"],
    }.get(accel, [])


def _build_toolchain_ok() -> bool:
    """ソースビルドに必要な最低限（cmake・git）が揃っているか。"""
    return bool(shutil.which("cmake") and shutil.which("git"))


def build_from_source(
    build: str, os_name: str, arch: str, accel: str, dest: str,
    *, run=subprocess.run, verify=_verify,
) -> str:
    """llama.cpp を指定ビルドタグからソースビルドし、成果物を dest へ入れてパスを返す。

    cmake/git が要る。ビルドは共有ライブラリごと必要なので bin ディレクトリ一式を dest/bin へ
    コピーする。ツールチェーン不在・cmake/ビルド失敗・検証失敗は BuildUnavailable（呼び出し側で
    プリビルトへフォールバックする）。
    """
    if not _build_toolchain_ok():
        raise BuildUnavailable("cmake / git が見つからない（プリビルトへフォールバック）")
    src = os.path.join(managed_root(), f"src-{build}")
    bdir = os.path.join(src, "build")
    try:
        if not os.path.isdir(os.path.join(src, ".git")):
            run(["git", "clone", "--depth", "1", "--branch", build,
                 f"https://github.com/{_REPO}", src], check=True)
        run(["cmake", "-S", src, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release",
             *accel_cmake_flags(accel)], check=True)
        run(["cmake", "--build", bdir, "--config", "Release",
             "--target", "llama-server", "-j"], check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildUnavailable(f"ソースビルドに失敗: {exc}") from exc
    binary = _find_llama_server(bdir)
    if not binary:
        raise BuildUnavailable("ビルドしたが llama-server が見つからない")
    # 共有ライブラリごと dest/bin へ配置（binary 単体だと実行時に .so/.dll が欠ける）。
    out_bin = os.path.join(dest, "bin")
    shutil.copytree(os.path.dirname(binary), out_bin, dirs_exist_ok=True)
    out = os.path.join(out_bin, _EXE)
    if not verify(out):
        raise BuildUnavailable("ビルドした llama-server の --version 検証に失敗")
    return out


def ensure_llama_server(
    *,
    provision: str = "auto",
    accel: str = "auto",
    build: str | None = None,
    download=_download,
    verify=_verify,
) -> str:
    """起動に使う llama-server の絶対パスを返す（必要なら自動導入する）。

    - provision="system": PATH の llama-server を使う（無ければ ProvisionError）。
    - provision="auto"  : 管理ディレクトリに導入済みならそれを、無ければ Releases から
                          プリビルトを取得して展開・検証する。
    - provision="build" : ソースから cmake ビルドする。ツールチェーン不在・失敗時は
                          プリビルト自動導入（auto）へフォールバックする（ゲートウェイを
                          立たなくしない）。

    download/verify は差し替え可能（テスト用）。
    """
    global _LAST_INFO
    if provision == "system":
        found = shutil.which("llama-server")
        if not found:
            raise ProvisionError(
                "provision='system' だが PATH に llama-server が見つからない。"
                "provision='auto'（自動導入）にするか、llama-server を PATH に置く。"
            )
        # system はユーザー管理のバイナリ（accel 等の素性は不明）。auto フラグ付与の対象外。
        _LAST_INFO = {"provision": "system", "build": None, "accel": None}
        return found

    os_name = detect_os()
    arch = detect_arch()
    accel = detect_accelerator(os_name) if accel == "auto" else accel

    if build is None:
        # pin 未指定: まず導入済みを再利用する（オフラインでも起動でき、上流の頻繁な
        # リリースのたびに再ダウンロードしない）。無いときだけ最新を照会する。
        for installed in installed_builds(os_name, arch, accel):
            binary = _find_llama_server(install_dir(installed, os_name, arch, accel))
            if binary and verify(binary):
                _LAST_INFO = {"provision": provision, "build": installed,
                              "accel": accel}
                return binary
        try:
            build = latest_build()
        except (OSError, ValueError, KeyError) as exc:  # URLError も OSError の subclass
            raise ProvisionError(
                f"llama.cpp の最新ビルド番号を取得できない（オフライン?）: {exc}. "
                f"[llama_cpp] pin でビルド番号を固定するか、provision='system' で "
                f"手動導入の llama-server を使う。"
            ) from exc

    target = install_dir(build, os_name, arch, accel)
    existing = _find_llama_server(target)
    if existing and verify(existing):
        _LAST_INFO = {"provision": provision, "build": build, "accel": accel}
        return existing

    if provision == "build":
        try:
            built = build_from_source(build, os_name, arch, accel, target,
                                      verify=verify)
            _LAST_INFO = {"provision": "build", "build": build, "accel": accel}
            return built
        except BuildUnavailable as exc:
            print(f"llama.cpp source build unavailable, falling back to prebuilt: "
                  f"{exc}", file=sys.stderr)
            # プリビルト導入へフォールバック（下へ続く）。

    name = asset_name(build, os_name, arch, accel)
    url = asset_url(build, name)
    archive = os.path.join(managed_root(), name)
    try:
        download(url, archive)
        _extract(archive, target)
    except Exception as exc:  # noqa: BLE001 - まとめて ProvisionError に包む
        raise ProvisionError(
            f"llama.cpp の自動導入に失敗（{name}）: {exc}. "
            f"手動導入は docs/llama-cpp.md を参照、または gateway.toml で "
            f"[llama_cpp] provision='system' / accel を指定する。URL: {url}"
        ) from exc
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass

    binary = _find_llama_server(target)
    if not binary:
        raise ProvisionError(f"展開したが llama-server が見つからない（{target}）")
    if not verify(binary):
        raise ProvisionError(
            f"llama-server を導入したが --version に失敗（{binary}）。"
            f"accel={accel} が環境に合っていない可能性（accel を cpu 等に変えて再試行）"
        )
    _LAST_INFO = {"provision": provision, "build": build, "accel": accel}
    return binary
