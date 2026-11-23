"""`gw claude`: ゲートウェイの /v1/messages を使って Claude Code を起動する。

Claude Code の CLI は環境変数 `ANTHROPIC_BASE_URL` で接続先を切り替えられる。ここではその
環境変数一式をゲートウェイ向けに組み立てて `claude` を起動するだけで、Claude Code 本体も
その設定ファイルも書き換えない。

- 設定・履歴は `~/.config/local-llm-server/claude-code`（`CLAUDE_CONFIG_DIR`）に分ける。
  普段の `claude`（`~/.claude`）やデスクトップアプリとは混ざらない。
- 使うモデルは `--model` で指定し、そのディレクトリに覚えておく（次からは省略できる）。
  Claude Code が裏の処理（要約・サブエージェント等）に使うモデルも同じものにそろえ、
  別のモデルを読み込ませない。
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence

_MODEL_FILE = "gw-model"
# ゲートウェイに api_key が無いときに送る値。Claude Code は認証情報が空だと
# Anthropic へのログインを求めるので、何かしらの値が要る（ゲートウェイは見ない）。
_NO_KEY = "local"


def config_dir(gateway_config_path: str) -> str:
    """Claude Code 用の設定ディレクトリ（gateway.toml と同じ場所の `claude-code`）。"""
    return os.path.join(os.path.dirname(gateway_config_path), "claude-code")


def split_model_arg(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """`--model X` / `--model=X` を取り出し、残りの引数はそのまま claude に渡す。"""
    model: str | None = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            rest.extend(argv[i:])
            break
        if arg == "--model":
            if i + 1 >= len(argv):
                raise ValueError("--model にはモデル ID が要ります")
            model = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1
    if model is not None and not model.strip():
        raise ValueError("--model にはモデル ID が要ります")
    return (model.strip() if model else None), rest


def read_saved_model(cdir: str) -> str | None:
    try:
        with open(os.path.join(cdir, _MODEL_FILE), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def save_model(cdir: str, model: str) -> None:
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, _MODEL_FILE), "w", encoding="utf-8") as fh:
        fh.write(model + "\n")


def build_env(
    base: Mapping[str, str], *, base_url: str, api_key: str | None, model: str, cdir: str
) -> dict[str, str]:
    """claude に渡す環境変数。元の環境を写し、ゲートウェイ向けの値で上書きする。"""
    env = dict(base)
    # Anthropic の API キーが残っていると x-api-key として本物のキーが送られてしまう
    env.pop("ANTHROPIC_API_KEY", None)
    env.update(
        {
            "CLAUDE_CONFIG_DIR": cdir,
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_AUTH_TOKEN": api_key or _NO_KEY,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            # 更新確認・テレメトリ・エラー報告など、Anthropic への付随通信を止める
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    return env


def launch(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """claude を起動する。POSIX ではこのプロセスを置き換える（Ctrl+C を claude だけが受ける）。"""
    exe = shutil.which("claude", path=env.get("PATH"))
    if exe is None:
        print(
            "claude（Claude Code の CLI）が見つかりません。先にインストールしてください。",
            file=sys.stderr,
        )
        return 127
    if os.name == "posix":
        os.execve(exe, [exe, *argv], dict(env))
    # Windows: Ctrl+C は claude 側が処理する。ここで受けて先に終わると claude だけが残る
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return subprocess.call([exe, *argv], env=dict(env))
    finally:
        signal.signal(signal.SIGINT, previous)
