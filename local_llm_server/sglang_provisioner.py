"""SGLang バックエンドの自動導入（隔離 venv プロビジョナ）。

vllm_provisioner と同じ思想・構造（重量級 Python パッケージを本体に混ぜず、管理ディレクトリ
配下の専用 venv へ隔離導入し、そこの python から OpenAI 互換サーバを起動する）。SGLang は
RadixAttention（プレフィックスキャッシュ）で、共有プレフィックスの多い用途（同じシステム
プロンプト/ツール定義を毎回送るエージェント運用）に強い。vLLM と同じく Linux/NVIDIA が対象
（Windows は WSL2 内の Linux）。

install（実 pip）と verify（実 import）は差し替え可能で、実 SGLang/GPU 無しで本体ロジックを
ユニット検証できる。vLLM 経路とは独立（片方の変更が他方に波及しない）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv

from . import provisioner  # managed_root / detect_os / _has_nvidia_gpu を再利用

# SGLang の起動モジュール（OpenAI 互換 API サーバ）。
SGLANG_SERVER_MODULE = "sglang.launch_server"


class SglangUnavailable(RuntimeError):
    """SGLang を使えない/導入できない（GPU 非検出・pip 失敗・非対応 OS など）。

    起動側はこれを捕まえてゲートウェイを止めずに続行し、sglang モデルの要求時に
    分かりやすいエラーにする。"""


def sglang_venv_dir() -> str:
    """SGLang 専用 venv の置き場（管理ディレクトリ配下。PATH は汚さない）。"""
    return os.path.join(provisioner.managed_root(), "..", "sglang-venv")


def venv_python(venv_dir: str) -> str:
    """venv 内の python 実行ファイルのパス（OS 依存）。"""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _sglang_importable(py: str, run) -> bool:
    """その python で sglang が import できるか（導入済み判定・自己検証）。"""
    try:
        return run([py, "-c", "import sglang"], capture_output=True,
                   timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def gpu_available() -> bool:
    """SGLang が実用になる NVIDIA GPU があるか。"""
    return provisioner._has_nvidia_gpu()


def ensure_sglang(
    *,
    provision: str = "auto",
    require_gpu: bool = True,
    run=subprocess.run,
    create_venv=None,
    importable=_sglang_importable,
) -> str:
    """SGLang を起動できる python の絶対パスを返す（必要なら隔離 venv へ導入する）。

    - provision="system": 現在の python 環境に sglang があればそれを使う（無ければ SglangUnavailable）。
    - provision="auto"  : 隔離 venv に導入済みならそれを、無ければ venv 作成＋pip install する。
    GPU 非検出は SglangUnavailable（require_gpu=False で回避可＝テスト用）。
    create_venv/run/importable は差し替え可能（テスト用）。
    """
    if require_gpu and not gpu_available():
        raise SglangUnavailable(
            "SGLang には NVIDIA GPU が要ります（CPU 実行は実用外）。GPU のある Linux か "
            "WSL2 で使うか、backend='llama-cpp' を選んでください。"
        )
    if provisioner.detect_os() == "macos":
        raise SglangUnavailable(
            "SGLang は macOS では非対応です（Apple Silicon は backend='mlx-vlm' を使用）。"
        )

    if provision == "system":
        if importable(sys.executable, run):
            return sys.executable
        raise SglangUnavailable(
            "provision='system' だが現在の環境に sglang が無い。provision='auto'（自動導入）に "
            "するか、この環境へ sglang を入れる。"
        )

    venv_dir = os.path.normpath(sglang_venv_dir())
    py = venv_python(venv_dir)
    if os.path.exists(py) and importable(py, run):
        return py

    try:
        if create_venv is not None:
            create_venv(venv_dir)
        else:
            venv.create(venv_dir, with_pip=True)
        run([py, "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True, timeout=300)
        proc = run([py, "-m", "pip", "install", "sglang"],
                   capture_output=True, timeout=3600)
        if getattr(proc, "returncode", 1) != 0:
            err = (getattr(proc, "stderr", b"") or b"").decode("utf-8", "replace")
            raise SglangUnavailable(f"pip install sglang に失敗: {err[-400:]}")
    except SglangUnavailable:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise SglangUnavailable(f"SGLang 用 venv の作成/導入に失敗: {exc}") from exc

    if not importable(py, run):
        raise SglangUnavailable(
            f"SGLang を導入したが import できない（{py}）。CUDA/torch の不整合の可能性。"
        )
    return py
