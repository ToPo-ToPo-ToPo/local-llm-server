"""SGLang バックエンドの自動導入（隔離 venv プロビジォナ）。

vLLM と同じ思想・構造で、重量級 Python パッケージを本体に混ぜず、管理ディレクトリ配下の専用
venv へ隔離導入し、その python から `python -m sglang.launch_server` を起動する。venv 作成/
導入の共通ロジックは _venv_backend に一本化してある（vLLM と共有）。SGLang は RadixAttention
（プレフィックスキャッシュ）で、共有プレフィックスの多い用途（同じシステムプロンプト/ツール
定義を毎回送るエージェント運用）に強い。Linux/NVIDIA が対象（Windows は WSL2 内の Linux）。

GPU 非検出・macOS は明示エラー（SglangUnavailable）。install/verify は差し替え可能（テスト用）。
"""
from __future__ import annotations

import os
import subprocess
import sys  # noqa: F401 - テスト（sys.executable 参照）と後方互換のため公開しておく

from . import _venv_backend, provisioner  # noqa: F401 - provisioner はテストが monkeypatch する

# SGLang の起動モジュール（OpenAI 互換 API サーバ）。build_command が使う。
SGLANG_SERVER_MODULE = "sglang.launch_server"

# 共通ヘルパの再公開（テスト・呼び出し側が sglang_provisioner 名前空間から使えるように）。
venv_python = _venv_backend.venv_python
gpu_available = _venv_backend.gpu_available


class SglangUnavailable(RuntimeError):
    """SGLang を使えない/導入できない（GPU 非検出・pip 失敗・非対応 OS など）。

    起動側はこれを捕まえてゲートウェイを止めずに続行し、sglang モデルの要求時に
    分かりやすいエラーにする。"""


def sglang_venv_dir() -> str:
    """SGLang 専用 venv の置き場（管理ディレクトリ配下。PATH は汚さない）。"""
    return os.path.join(provisioner.managed_root(), "..", "sglang-venv")


def _sglang_importable(py: str, run) -> bool:
    """その python で sglang が import できるか（導入済み判定・自己検証）。"""
    return _venv_backend.make_importable("sglang")(py, run)


def ensure_sglang(
    *,
    provision: str = "auto",
    require_gpu: bool = True,
    run=subprocess.run,
    create_venv=None,
    importable=_sglang_importable,
) -> str:
    """SGLang を起動できる python の絶対パスを返す（必要なら隔離 venv へ導入する）。→ _venv_backend。"""
    return _venv_backend.ensure_backend(
        package="sglang", import_name="sglang", venv_dir=sglang_venv_dir(),
        human_name="SGLang", unavailable=SglangUnavailable,
        gpu_check=gpu_available, provision=provision, require_gpu=require_gpu,
        run=run, create_venv=create_venv, importable=importable,
    )
