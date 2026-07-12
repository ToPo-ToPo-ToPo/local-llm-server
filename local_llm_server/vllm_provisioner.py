"""vLLM バックエンドの自動導入（隔離 venv プロビジョナ）。

vLLM は torch+CUDA を含む重量級 Python パッケージ（数 GB）で、local-llm-server 本体の依存に
混ぜると macOS の mlx 環境を壊す・パッケージが巨大化する。そこで **管理ディレクトリに隔離した
専用 venv** へ導入し、その venv の python から `python -m vllm.entrypoints.openai.api_server`
を起動する。venv 作成/導入の共通ロジックは _venv_backend に一本化してある（SGLang と共有）。

vLLM は Linux/NVIDIA が対象（Windows はネイティブ非対応 → WSL2 内の Linux で動かす）。
GPU 非検出・macOS は明示エラー（VllmUnavailable）。install/verify は差し替え可能（テスト用）。
"""
from __future__ import annotations

import os
import subprocess
import sys  # noqa: F401 - テスト（sys.executable 参照）と後方互換のため公開しておく

from . import _venv_backend, provisioner  # noqa: F401 - provisioner はテストが monkeypatch する

# vLLM の起動モジュール（OpenAI 互換 API サーバ）。build_command が使う。
VLLM_SERVER_MODULE = "vllm.entrypoints.openai.api_server"

# 共通ヘルパの再公開（テスト・呼び出し側が vllm_provisioner 名前空間から使えるように）。
venv_python = _venv_backend.venv_python
gpu_available = _venv_backend.gpu_available


class VllmUnavailable(RuntimeError):
    """vLLM を使えない/導入できない（GPU 非検出・pip 失敗・非対応 OS など）。

    起動側はこれを捕まえてゲートウェイを止めずに続行し、vllm モデルの要求時に
    分かりやすいエラーにする。"""


def vllm_venv_dir() -> str:
    """vLLM 専用 venv の置き場（管理ディレクトリ配下。PATH は汚さない）。"""
    return os.path.join(provisioner.managed_root(), "..", "vllm-venv")


def _vllm_importable(py: str, run) -> bool:
    """その python で vllm が import できるか（導入済み判定・自己検証）。"""
    return _venv_backend.make_importable("vllm")(py, run)


def ensure_vllm(
    *,
    provision: str = "auto",
    require_gpu: bool = True,
    run=subprocess.run,
    create_venv=None,
    importable=_vllm_importable,
) -> str:
    """vLLM を起動できる python の絶対パスを返す（必要なら隔離 venv へ導入する）。→ _venv_backend。"""
    return _venv_backend.ensure_backend(
        package="vllm", import_name="vllm", venv_dir=vllm_venv_dir(),
        human_name="vLLM", unavailable=VllmUnavailable,
        gpu_check=gpu_available, provision=provision, require_gpu=require_gpu,
        run=run, create_venv=create_venv, importable=importable,
    )
