"""モデルサーバーをデーモンへ「繋留」する監視ラッパー（POSIX 専用・内部実行口）。

デーモン（ゲートウェイ）はモデルサーバーを直接ではなくこのラッパー経由で起動し、
デーモンだけが書き込み端を握るパイプの**読み取り端**を `--fd` で渡す
（→ server.enable_child_tethering / LocalServer.start）。デーモンが `kill -9` を含む
どんな形で死んでも OS がパイプを閉じるので、ラッパーは EOF を検知して自分の
プロセスグループ（＝実サーバーとその孫）へ SIGTERM →（猶予後）SIGKILL を送る。
これで「デーモンだけ死んでモデルサーバーが GPU/メモリを掴んだまま残る」事態を
構造的に防ぐ（docs/ollama-clone-plan.md Phase 0b）。

ラッパー自身は SIGTERM/SIGINT/SIGHUP を無視する——デーモンの stop() は killpg で
グループ全体へ送るため実サーバーには直接届き、ラッパーは実サーバーの終了を看取って
同じ終了コードで exit する。LocalServer から見れば従来どおり「1 つの子プロセス
（＝グループリーダー）」であり、poll/stop の扱いは何も変わらない。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

# 親の死を検知してからの graceful 猶予（SIGTERM → この秒数 → SIGKILL）。
# LocalServer.stop(grace=10.0) の既定と同じ感覚値。
_GRACE_S = 10.0


def _shutdown_group(proc: subprocess.Popen) -> None:
    """親（デーモン）の死後、自分のグループ（実サーバーと孫を含む）を畳む。

    自分は SIGTERM を無視しているので killpg しても生き残り、猶予後の SIGKILL まで
    見届けられる（SIGKILL は自分にも効くが、その時点で仕事は終わっている）。
    """
    try:
        os.killpg(os.getpgid(0), signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + _GRACE_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            os._exit(1)
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except OSError:
        pass
    os._exit(1)


def _watch_parent(fd: int, proc: subprocess.Popen) -> None:
    """繋留パイプの EOF（＝デーモンの死）を待つ。書き込みは来ない——閉じられるだけ。"""
    while True:
        try:
            data = os.read(fd, 4096)
        except InterruptedError:
            continue
        except OSError:
            break
        if not data:
            break
    _shutdown_group(proc)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: python -m local_llm_server.tether --fd N -- <cmd> [args...]"
    try:
        sep = argv.index("--")
        opts, cmd = argv[:sep], argv[sep + 1:]
        fd = int(opts[opts.index("--fd") + 1])
    except (ValueError, IndexError):
        print(usage, file=sys.stderr)
        return 2
    if not cmd:
        print(usage, file=sys.stderr)
        return 2
    # 停止シグナルは無視する（実サーバーへは killpg で直接届く。自分が先に死ぬと
    # 「実サーバーの終了を看取って同じコードで exit する」役目を果たせない）。
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (OSError, ValueError):
            pass
    try:
        proc = subprocess.Popen(cmd)  # 同一グループ（リーダーは自分）。stdout/err は継承
    except FileNotFoundError as exc:
        print(f"tether: backend executable not found: {exc.filename}", file=sys.stderr)
        return 127
    threading.Thread(target=_watch_parent, args=(fd, proc), daemon=True).start()
    rc = proc.wait()
    # 負値（シグナル死）はシェル慣習の 128+n へ写して返す。
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    raise SystemExit(main())
