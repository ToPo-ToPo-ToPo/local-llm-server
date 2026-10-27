"""Windows の日本語環境で、UTF-8 モード(PYTHONUTF8=1)の親からでもポートの持ち主を見つけられる。

日本語 Windows の netstat は cp932 で出力する。UTF-8 モードで読むと読み取りが失敗して出力が空(None)になり、
`gw restart` が AttributeError で落ちていた(実機 2026-10-27)。
"""
import os
import socket
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != "nt", reason="netstat による探索は Windows の経路")
def test_finds_the_listener_even_in_utf8_mode():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        code = ("from local_llm_server.process_control import find_pids_on_port; "
                f"print(find_pids_on_port({port}))")
        out = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert out.returncode == 0, out.stderr
        assert str(os.getpid()) in out.stdout
