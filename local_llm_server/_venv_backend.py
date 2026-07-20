"""隔離 venv バックエンド（vLLM / SGLang）の共通プロビジョナ。

vLLM も SGLang も「torch+CUDA を含む重量級 Python パッケージを本体に混ぜず、管理ディレクトリ
配下の専用 venv へ隔離導入し、そこの python から OpenAI 互換サーバを起動する」という**まったく
同じ導入ロジック**を持つ。その共通部分（GPU/OS ガード・system 判定・venv 作成/再利用/
pip install/import 検証）をここに一本化し、各バックエンドは package 名・import 名・venv パス・
例外型・GPU 判定だけを与える薄いラッパにする（→ vllm_provisioner / sglang_provisioner）。

install（実 pip）と verify（実 import）は差し替え可能で、実パッケージ/GPU 無しで
本体ロジックをユニット検証できる。
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv

from . import provisioner  # detect_os / _has_nvidia_gpu を再利用


def gpu_available() -> bool:
    """実用になる NVIDIA GPU があるか（vLLM/SGLang は GPU 前提）。"""
    return provisioner._has_nvidia_gpu()


def venv_python(venv_dir: str) -> str:
    """venv 内の python 実行ファイルのパス（OS 依存）。"""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def make_importable(import_name: str):
    """`import <import_name>` が通るかで導入済み判定する検証関数を作る。"""
    def _check(py: str, run) -> bool:
        try:
            return run([py, "-c", f"import {import_name}"], capture_output=True,
                       timeout=120).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return _check


def ensure_backend(
    *,
    package: str,
    import_name: str,
    venv_dir: str,
    human_name: str,
    unavailable: type[RuntimeError],
    gpu_check,
    require_gpu: bool = True,
    run=subprocess.run,
    create_venv=None,
    importable=None,
) -> str:
    """隔離 venv バックエンドを起動できる python の絶対パスを返す（必要なら導入する）。

    ルートは 1 つだけ（導入方法をユーザーに選ばせない）。解決順は決定的:
    1. 現在の python 環境に package があればそれを使う（extras で導入済みなら数 GB の
       再ダウンロードをしない）。
    2. 無ければ隔離 venv（venv_dir）を再利用、それも無ければ venv 作成＋pip install。
    GPU 非検出・macOS は `unavailable`（require_gpu=False で GPU ガードを外せる＝テスト用）。
    `unavailable` は各バックエンドの例外型（VllmUnavailable / SglangUnavailable）。
    """
    importable = importable or make_importable(import_name)
    if require_gpu and not gpu_check():
        raise unavailable(
            f"{human_name} には NVIDIA GPU が要ります（CPU 実行は実用外）。GPU のある Linux か "
            f"WSL2 で使うか、backend='llama-cpp' を選んでください。"
        )
    if provisioner.detect_os() == "macos":
        raise unavailable(
            f"{human_name} は macOS では非対応です（Apple Silicon は backend='mlx-vlm' を使用）。"
        )

    if importable(sys.executable, run):
        return sys.executable

    vdir = os.path.normpath(venv_dir)
    py = venv_python(vdir)
    if os.path.exists(py) and importable(py, run):
        return py

    try:
        if create_venv is not None:
            create_venv(vdir)
        else:
            venv.create(vdir, with_pip=True)
        run([py, "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True, timeout=300)
        proc = run([py, "-m", "pip", "install", package],
                   capture_output=True, timeout=3600)
        if getattr(proc, "returncode", 1) != 0:
            err = (getattr(proc, "stderr", b"") or b"").decode("utf-8", "replace")
            raise unavailable(f"pip install {package} に失敗: {err[-400:]}")
    except unavailable:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise unavailable(f"{human_name} 用 venv の作成/導入に失敗: {exc}") from exc

    if not importable(py, run):
        raise unavailable(
            f"{human_name} を導入したが import できない（{py}）。CUDA/torch の不整合の可能性。"
        )
    return py
