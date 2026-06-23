from types import SimpleNamespace

import local_llm_server as srv
from local_llm_server import (
    BACKENDS,
    MTP_DRAFTERS,
    ServerConfig,
    ServerPool,
    build_command,
    build_pool_configs,
    default_backend,
    parallel_supported,
    parse_host_port,
    resolve_drafter,
)


def test_build_command_mlx():
    cmd = build_command(ServerConfig("mlx", "model", port=8080))
    assert cmd[0] == "mlx_lm.server" and "--model" in cmd and "8080" in cmd


def test_build_command_llama_parallel_and_thinking():
    c = ServerConfig("llama-cpp", "/m.gguf", parallel=4, disable_thinking=True)
    cmd = build_command(c)
    assert cmd[0] == "llama-server"
    assert "--parallel" in cmd and "4" in cmd
    assert "--chat-template-kwargs" in cmd


def test_build_command_mlx_disable_thinking():
    cmd = build_command(ServerConfig("mlx", "m", disable_thinking=True))
    assert "--chat-template-args" in cmd


def test_build_command_mlx_vlm():
    cmd = build_command(ServerConfig("mlx-vlm", "vision-model", port=8080))
    # python -m mlx_vlm.server でモデル/ホスト/ポートを渡して起動する
    assert cmd[1:3] == ["-m", "mlx_vlm.server"]
    assert "--model" in cmd and "vision-model" in cmd and "8080" in cmd
    # vision は逐次処理。thinking テンプレート引数は付けない
    assert "--chat-template-args" not in cmd


def test_build_command_mlx_vlm_mtp_drafter():
    # Gemma 4 MTP: mlx-vlm に --draft-model を渡し、--draft-kind は mtp 固定。
    c = ServerConfig(
        "mlx-vlm", "mlx-community/gemma-4-E4B-it-qat-4bit",
        draft_model="mlx-community/gemma-4-E4B-it-qat-assistant-bf16",
    )
    cmd = build_command(c)
    assert "--draft-model" in cmd
    assert "mlx-community/gemma-4-E4B-it-qat-assistant-bf16" in cmd
    assert cmd[cmd.index("--draft-kind") + 1] == "mtp"  # 公式対応の Gemma 4 MTP に限定


def test_resolve_drafter_auto_qwen36():
    # 既定モデル Qwen3.6-27B-4bit は auto で MTP ドラフターに解決する（実機検証済み）。
    assert resolve_drafter("mlx-community/Qwen3.6-27B-4bit", "auto") == (
        "mlx-community/Qwen3.6-27B-MTP-4bit"
    )


def test_build_command_mlx_no_draft_support():
    # テキスト専用 mlx には MTP を渡さない（mlx-vlm のみ対応）。
    cmd = build_command(ServerConfig("mlx", "m", draft_model="d"))
    assert "--draft-model" not in cmd and "--draft-kind" not in cmd


def test_build_command_no_draft_by_default():
    # ドラフター未指定なら draft 系フラグは出ない。
    assert "--draft-model" not in build_command(ServerConfig("mlx-vlm", "m"))


def test_resolve_drafter_passthrough_and_none():
    # 明示指定はそのまま、未指定は None。
    assert resolve_drafter("any/model", "x/drafter") == "x/drafter"
    assert resolve_drafter("any/model", None) is None
    assert resolve_drafter("any/model", "") is None


def test_resolve_drafter_auto_known():
    # "auto" は本体名から対応表で引く。
    target = "mlx-community/gemma-4-E4B-it-qat-4bit"
    assert resolve_drafter(target, "auto") == MTP_DRAFTERS[target]


def test_resolve_drafter_auto_nonqat_8bit():
    # 非QAT 8bit（26B-A4B）も auto で引ける。
    target = "mlx-community/gemma-4-26b-a4b-it-8bit"
    assert resolve_drafter(target, "auto") == \
        "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"


def test_resolve_drafter_auto_unknown_raises():
    # 未収載モデルの "auto" は明示指定を促すエラー。
    import pytest
    with pytest.raises(ValueError):
        resolve_drafter("some/unknown-model", "auto")


def test_build_command_mlx_vlm_draft_auto():
    # build_command でも "auto" が解決され、対応ドラフター＋mtp が付く。
    target = "mlx-community/gemma-4-12B-it-qat-4bit"
    cmd = build_command(ServerConfig("mlx-vlm", target, draft_model="auto"))
    assert cmd[cmd.index("--draft-model") + 1] == MTP_DRAFTERS[target]
    assert cmd[cmd.index("--draft-kind") + 1] == "mtp"


def test_mlx_vlm_in_backends_and_not_parallel():
    assert "mlx-vlm" in BACKENDS
    assert not parallel_supported("mlx-vlm")


def test_default_backend_and_supported():
    assert default_backend() in BACKENDS
    assert parallel_supported("llama-cpp") and not parallel_supported("mlx")


def test_default_backend_is_vision_on_apple_silicon(monkeypatch):
    # Apple Silicon では既定を vision 対応の mlx-vlm にする（既定モデルが多モーダル）。
    monkeypatch.setattr("local_llm_server.sys.platform", "darwin")
    monkeypatch.setattr("local_llm_server.platform.machine", lambda: "arm64")
    assert default_backend() == "mlx-vlm"
    # それ以外は llama.cpp
    monkeypatch.setattr("local_llm_server.sys.platform", "linux")
    assert default_backend() == "llama-cpp"


def test_parse_host_port():
    assert parse_host_port("http://127.0.0.1:8081/v1") == ("127.0.0.1", 8081)


def test_build_pool_configs():
    base = ServerConfig("mlx", "m", port=8080)
    cfgs = build_pool_configs(base, 3)
    assert [c.port for c in cfgs] == [8080, 8081, 8082]
    assert base.port == 8080  # 元は不変


def test_server_pool_order(monkeypatch):
    events = []

    class FakeLocal:
        def __init__(self, cfg):
            self.cfg = cfg

        @property
        def base_url(self):
            return self.cfg.base_url

        def start(self):
            events.append(("start", self.cfg.port))

        def wait_until_ready(self, timeout=120.0):
            events.append(("ready", self.cfg.port))

        def stop(self):
            events.append(("stop", self.cfg.port))

    # ServerPool 等は実装モジュール local_llm_server.server 内の名前を参照するため、
    # パッケージ再エクスポート名ではなく実装モジュール側を差し替える。
    monkeypatch.setattr("local_llm_server.server.LocalServer", FakeLocal)
    pool = ServerPool(build_pool_configs(ServerConfig("mlx", "m", port=8080), 2))
    pool.start()
    pool.wait_until_ready()
    pool.stop()
    assert events == [
        ("start", 8080), ("start", 8081),
        ("ready", 8080), ("ready", 8081),
        ("stop", 8080), ("stop", 8081),
    ]


def test_local_server_redirects_output_to_log(monkeypatch, tmp_path):
    # 自動起動サーバーの出力は端末に流さず、ログファイルへ逃がす（対話画面を汚さない）。
    import io
    import contextlib
    from local_llm_server import server as srv_mod
    from local_llm_server import LocalServer, ServerConfig

    monkeypatch.setattr(
        srv_mod, "build_command",
        lambda cfg: ["python", "-c", "import sys; print('NOISE_OUT'); sys.stderr.write('NOISE_ERR\\n')"],
    )
    log = tmp_path / "srv.log"
    server = LocalServer(ServerConfig("mlx", "dummy", "127.0.0.1", 9), log_path=str(log))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        server.start()
        server.wait()
        server.stop()
    text = log.read_text(encoding="utf-8")
    assert "NOISE_OUT" in text and "NOISE_ERR" in text  # ログへ
    assert "NOISE" not in out.getvalue() and "NOISE" not in err.getvalue()  # 端末へは出さない


def test_install_shutdown_handlers_converts_sigterm(monkeypatch):
    # SIGTERM を KeyboardInterrupt に変換して、各エントリポイントの finally（stop）を通す。
    import os
    import signal
    import sys
    from local_llm_server import install_shutdown_handlers

    if not hasattr(signal, "SIGTERM"):
        return  # POSIX 以外はスキップ
    original = signal.getsignal(signal.SIGTERM)
    try:
        install_shutdown_handlers()
        raised = False
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            # シグナル配送を確実にするため少しだけ待つ（POSIX では即時）
            for _ in range(1000):
                pass
        except KeyboardInterrupt:
            raised = True
        assert raised, "SIGTERM should be re-raised as KeyboardInterrupt"
    finally:
        signal.signal(signal.SIGTERM, original)


def test_stop_kills_process_group(tmp_path):
    # stop() はプロセスグループ全体を止め、バックエンドが起こした孫プロセスも残さない。
    import os
    import time
    from local_llm_server import server as srv_mod
    from local_llm_server import LocalServer, ServerConfig

    if os.name != "posix":
        return
    pidfile = tmp_path / "grandchild.pid"
    # 子(sh)が孫(sleep)を起こし、その PID を書き出す。
    cmd = ["sh", "-c", f"sleep 300 & echo $! > {pidfile}; wait"]
    orig = srv_mod.build_command
    srv_mod.build_command = lambda cfg: cmd
    try:
        server = LocalServer(ServerConfig("mlx", "dummy", "127.0.0.1", 9),
                             log_path=str(tmp_path / "s.log"))
        server.start()
        for _ in range(50):
            if pidfile.exists():
                break
            time.sleep(0.05)
        grand = int(pidfile.read_text().strip())
        server.stop()
        time.sleep(0.3)

        def alive(pid):
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False

        assert not alive(grand), "grandchild process leaked after stop()"
    finally:
        srv_mod.build_command = orig


def test_find_pids_on_port_filters_to_listener():
    # find_pids_on_port は対象ポートを LISTEN するプロセスだけを返す（全 TCP を拾わない）。
    import os
    import subprocess
    import sys
    import time
    from local_llm_server import find_pids_on_port

    if os.name != "posix":
        return
    code = ('import http.server,socketserver;'
            'socketserver.TCPServer(("127.0.0.1",8097),'
            'http.server.BaseHTTPRequestHandler).serve_forever()')
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        for _ in range(50):
            if find_pids_on_port(8097):
                break
            time.sleep(0.05)
        assert find_pids_on_port(8097) == [proc.pid]
        assert find_pids_on_port(8096) == []  # 別ポートは拾わない
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_models_match():
    from local_llm_server import models_match
    assert models_match("a/Foo", "a/Foo")
    assert models_match("/abs/Qwen3.6-27B-4bit", "mlx-community/Qwen3.6-27B-4bit")  # path vs repo
    assert not models_match("org/gemma-4-31B", "org/Qwen3.6-27B")
    assert models_match(None, "x") and models_match("x", None)  # 不明なら一致扱い（警告しない）


def test_running_model_parses_v1_models(monkeypatch):
    import io
    import local_llm_server as srv

    class _Resp:
        def __init__(self, body):
            self._b = body.encode("utf-8")
        def read(self):
            return self._b
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        srv.urllib.request, "urlopen",
        lambda url, timeout=5.0: _Resp('{"data": [{"id": "mlx-community/Qwen3.6-27B-4bit"}]}'),
    )
    assert srv.running_model("http://x/v1") == "mlx-community/Qwen3.6-27B-4bit"

    def _boom(url, timeout=5.0):
        raise OSError("down")
    monkeypatch.setattr(srv.urllib.request, "urlopen", _boom)
    assert srv.running_model("http://x/v1") is None  # 取得失敗は None


class _Resp:
    def __init__(self, body):
        self._b = body.encode("utf-8")
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _patch_models(monkeypatch, body):
    import local_llm_server as srv
    monkeypatch.setattr(
        srv.urllib.request, "urlopen", lambda url, timeout=5.0: _Resp(body)
    )


def test_list_models_returns_all_ids(monkeypatch):
    import local_llm_server as srv
    _patch_models(monkeypatch, '{"data": [{"id": "a/Foo"}, {"id": "b/Bar"}, {"id": "c/Baz"}]}')
    assert srv.list_models("http://x/v1") == ["a/Foo", "b/Bar", "c/Baz"]

    def _boom(url, timeout=5.0):
        raise OSError("down")
    monkeypatch.setattr(srv.urllib.request, "urlopen", _boom)
    assert srv.list_models("http://x/v1") == []  # 取得失敗は []


def test_model_available_against_router_catalog(monkeypatch):
    import local_llm_server as srv
    # ルーター型サーバー: 先頭は別モデルだが、設定モデルはカタログに含まれる → True（誤警告しない）
    _patch_models(
        monkeypatch,
        '{"data": [{"id": "mlx-community/Qwen3.5-27B-4bit"},'
        ' {"id": "mlx-community/Qwen3.6-27B-4bit"}, {"id": "org/other"}]}',
    )
    assert srv.model_available("http://x/v1", "mlx-community/Qwen3.6-27B-4bit") is True
    # カタログに無いモデル → False
    assert srv.model_available("http://x/v1", "mlx-community/does-not-exist") is False
    # model 未指定は判定不能 → None（警告しない）
    assert srv.model_available("http://x/v1", None) is None

    # 一覧を取得できない場合も None
    def _boom(url, timeout=5.0):
        raise OSError("down")
    monkeypatch.setattr(srv.urllib.request, "urlopen", _boom)
    assert srv.model_available("http://x/v1", "x/y") is None
