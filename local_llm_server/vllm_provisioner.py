"""vLLM バックエンドの自動導入（隔離 venv プロビジョナ）。

vLLM は torch+CUDA を含む重量級 Python パッケージ（数 GB）で、local-llm-server 本体の
依存に混ぜると macOS の mlx 環境を壊す・パッケージが巨大化する。そこで **管理ディレクトリに
隔離した専用 venv** を作り、そこへ `pip install vllm` して、その venv の python から
`python -m vllm.entrypoints.openai.api_server` を起動する（llama.cpp プロビジョナと同じ
「本体を汚さず管理ディレクトリに隔離」の思想を、バイナリでなく venv で実現）。

vLLM は Linux/NVIDIA が対象（Windows はネイティブ非対応 → WSL2 内の Linux で動かす）。
GPU 非検出時は明示エラーにする（CPU 実行は実用外）。

install（実 pip）と verify（実 import）は差し替え可能にしてあり、実 vLLM 無しで
本体ロジック（venv パス解決・導入フロー・GPU ガード）をユニット検証できる。
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv

from . import provisioner  # managed_root / detect_os / _has_nvidia_gpu を再利用

# vLLM の起動モジュール（OpenAI 互換 API サーバ）。
VLLM_SERVER_MODULE = "vllm.entrypoints.openai.api_server"


class VllmUnavailable(RuntimeError):
    """vLLM を使えない/導入できない（GPU 非検出・pip 失敗・非対応 OS など）。

    起動側はこれを捕まえてゲートウェイを止めずに続行し、vllm モデルの要求時に
    分かりやすいエラーにする。"""


def vllm_venv_dir() -> str:
    """vLLM 専用 venv の置き場（管理ディレクトリ配下。PATH は汚さない）。"""
    return os.path.join(provisioner.managed_root(), "..", "vllm-venv")


def _norm(path: str) -> str:
    return os.path.normpath(path)


def venv_python(venv_dir: str) -> str:
    """venv 内の python 実行ファイルのパス（OS 依存）。"""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _vllm_importable(py: str, run) -> bool:
    """その python で vllm が import できるか（導入済み判定・自己検証）。"""
    try:
        return run([py, "-c", "import vllm"], capture_output=True,
                   timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def gpu_available() -> bool:
    """vLLM が実用になる NVIDIA GPU があるか。"""
    return provisioner._has_nvidia_gpu()


def ensure_vllm(
    *,
    provision: str = "auto",
    require_gpu: bool = True,
    run=subprocess.run,
    create_venv=None,
    importable=_vllm_importable,
) -> str:
    """vLLM を起動できる python の絶対パスを返す（必要なら隔離 venv へ導入する）。

    - provision="system": 現在の python 環境に vllm があればそれを使う（無ければ VllmUnavailable）。
    - provision="auto"  : 隔離 venv に導入済みならそれを、無ければ venv 作成＋pip install する。
    GPU 非検出は VllmUnavailable（require_gpu=False で回避可＝テスト用）。
    create_venv/run/importable は差し替え可能（テスト用）。
    """
    if require_gpu and not gpu_available():
        raise VllmUnavailable(
            "vLLM には NVIDIA GPU が要ります（CPU 実行は実用外）。GPU のある Linux か "
            "WSL2 で使うか、backend='llama-cpp' を選んでください。"
        )
    if provisioner.detect_os() == "macos":
        raise VllmUnavailable(
            "vLLM は macOS では非対応です（Apple Silicon は backend='mlx-vlm' を使用）。"
        )

    if provision == "system":
        if importable(sys.executable, run):
            return sys.executable
        raise VllmUnavailable(
            "provision='system' だが現在の環境に vllm が無い。provision='auto'（自動導入）に "
            "するか、この環境へ vllm を入れる。"
        )

    venv_dir = _norm(vllm_venv_dir())
    py = venv_python(venv_dir)
    if os.path.exists(py) and importable(py, run):
        return py

    # venv を作成して vLLM を導入する。
    try:
        if create_venv is not None:
            create_venv(venv_dir)
        else:
            venv.create(venv_dir, with_pip=True)
        run([py, "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True, timeout=300)
        proc = run([py, "-m", "pip", "install", "vllm"],
                   capture_output=True, timeout=3600)
        if getattr(proc, "returncode", 1) != 0:
            err = (getattr(proc, "stderr", b"") or b"").decode("utf-8", "replace")
            raise VllmUnavailable(f"pip install vllm に失敗: {err[-400:]}")
    except VllmUnavailable:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise VllmUnavailable(f"vLLM 用 venv の作成/導入に失敗: {exc}") from exc

    if not importable(py, run):
        raise VllmUnavailable(
            f"vLLM を導入したが import できない（{py}）。CUDA/torch の不整合の可能性。"
        )
    return py
