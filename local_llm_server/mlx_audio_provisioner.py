"""mlx-audio（音声合成 TTS）バックエンドの自動導入（隔離 venv プロビジョナ）。

mlx-audio のサーバーは本体と依存がぶつかる（例: 本体は macOS で setuptools>=83 を要求し、
mlx-audio の server extra は setuptools<81 を要求する。mlx / transformers の版も引きずられる）。
本体の環境に混ぜると mlx-vlm（チャット推論）まで壊しうるので、vLLM / SGLang と同じく
**管理ディレクトリに隔離した専用 venv** へ導入し、その python から `python -m mlx_audio.server`
を起動する（共通ロジックは _venv_backend）。Apple Silicon の macOS 専用。

導入の時期は 2 つ。gateway.toml に backend="mlx-audio" の登録があれば起動時（daemon）、
動的ロード（未登録）なら最初の読み上げ要求のとき（server._build_mlx_audio）。どちらも初回だけ
数分かかり、以後は venv を再利用する。
"""
from __future__ import annotations

import os
import subprocess

from . import _venv_backend, provisioner  # noqa: F401 - provisioner はテストが monkeypatch する

# 起動モジュール（OpenAI 互換の POST /v1/audio/speech）。build_command が使う。
MLX_AUDIO_SERVER_MODULE = "mlx_audio.server"
# 導入する版。上げるときは実機で読み上げを確かめてから（再現性のため固定する）。
MLX_AUDIO_PACKAGE = "mlx-audio[tts,server]==0.5.5"

venv_python = _venv_backend.venv_python


class MlxAudioUnavailable(RuntimeError):
    """mlx-audio を使えない/導入できない（Apple Silicon 以外・pip 失敗など）。"""


def mlx_audio_venv_dir() -> str:
    """mlx-audio 専用 venv の置き場（管理ディレクトリ配下。PATH は汚さない）。"""
    return os.path.join(provisioner.managed_root(), "..", "mlx-audio-venv")


def _mlx_audio_importable(py: str, run) -> bool:
    """その python でサーバーモジュールまで import できるか（server extra の導入も確かめる）。"""
    return _venv_backend.make_importable(MLX_AUDIO_SERVER_MODULE)(py, run)


def ensure_mlx_audio(
    *,
    run=subprocess.run,
    create_venv=None,
    importable=_mlx_audio_importable,
) -> str:
    """mlx-audio のサーバーを起動できる python の絶対パスを返す（必要なら隔離 venv へ導入する）。"""
    return _venv_backend.ensure_backend(
        package=MLX_AUDIO_PACKAGE, import_name=MLX_AUDIO_SERVER_MODULE,
        venv_dir=mlx_audio_venv_dir(), human_name="mlx-audio (TTS)",
        unavailable=MlxAudioUnavailable, gpu_check=lambda: True,
        require_gpu=False, apple_silicon_only=True,
        run=run, create_venv=create_venv, importable=importable,
    )
