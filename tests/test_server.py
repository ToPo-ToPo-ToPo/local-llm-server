from types import SimpleNamespace

import pytest

import local_llm_server.server as srv
from local_llm_server import (
    BACKENDS,
    MTP_DRAFTERS,
    ServerConfig,
    build_command,
    default_backend,
    parallel_supported,
    parse_host_port,
    resolve_drafter,
)


def test_build_command_mlx(stub_cache):
    cmd = build_command(ServerConfig("mlx", "model", port=8080))
    assert cmd[0] == "mlx_lm.server" and "--model" in cmd and "8080" in cmd


def _make_hf_cache(tmp_path, repo, files):
    """HF キャッシュ風のレイアウトを作る: models--org--name/snapshots/rev/<files>。"""
    org, name = repo.split("/", 1)
    snap = tmp_path / "hub" / f"models--{org}--{name}" / "snapshots" / "rev1"
    for f in files:
        p = snap / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return snap


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """HF_HUB_CACHE を隔離し、`make(repo, files)` で repo を用意できるようにする。"""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return lambda repo, files: _make_hf_cache(tmp_path, repo, files)


@pytest.fixture
def stub_cache(monkeypatch):
    """mlx 系の事前 DL チェック（ensure_cached）を無効化し、コマンド構築だけを検証する。

    ensure_cached 自体の挙動は test_ensure_cached_* で（実物のまま）検証する。
    """
    monkeypatch.setattr(srv, "ensure_cached", lambda repo, **_kw: repo)


def test_build_command_llama_parallel_and_thinking(hf_cache):
    hf_cache("org/m-gguf", ["m-Q4_K_M.gguf"])
    c = ServerConfig("llama-cpp", "org/m-gguf", parallel=4, disable_thinking=True)
    cmd = build_command(c)
    assert cmd[0] == "llama-server"
    assert "--parallel" in cmd and "4" in cmd
    assert "--chat-template-kwargs" in cmd


def test_resolve_gguf_rejects_non_repo_id(hf_cache):
    # model は HF repo-id 専用。実パスや repo-id 形式でないものは弾く
    for bad in ["/abs/model.gguf", "./rel.gguf", "just-a-name", "a/b/c"]:
        with pytest.raises(ValueError):
            srv.resolve_gguf(bad)


def test_resolve_gguf_errors_when_not_cached(hf_cache):
    # repo-id だがキャッシュに無ければエラー
    with pytest.raises(ValueError):
        srv.resolve_gguf("org/not-cached")
    # repo はあるがセレクタに一致する GGUF が無ければエラー
    hf_cache("org/m-gguf", ["m-Q4_K_M.gguf"])
    with pytest.raises(ValueError):
        srv.resolve_gguf("org/m-gguf:NOPE")


def test_resolve_gguf_repo_id_single_base(hf_cache):
    snap = hf_cache("google/gemma-4-x-gguf", ["gemma-4-x_q4_0.gguf", "gemma-4-x-mmproj.gguf"])
    # mmproj/MTP を除いた本体を選ぶ
    assert srv.resolve_gguf("google/gemma-4-x-gguf") == str(snap / "gemma-4-x_q4_0.gguf")


def test_resolve_gguf_selector_and_ambiguity(hf_cache):
    snap = hf_cache("unsloth/g-gguf", ["base-Q4_K_XL.gguf", "base-Q8_0.gguf", "MTP/g-F16-MTP.gguf"])
    # セレクタでファイルを 1 つに絞れる（MTP ヘッドも選べる）
    assert srv.resolve_gguf("unsloth/g-gguf:F16-MTP") == str(snap / "MTP" / "g-F16-MTP.gguf")
    # セレクタ無しで本体が複数あると曖昧エラー
    with pytest.raises(ValueError):
        srv.resolve_gguf("unsloth/g-gguf")


def test_build_command_llama_resolves_repo_id(hf_cache):
    snap = hf_cache("google/g-gguf", ["g_q4_0.gguf", "g-mmproj.gguf"])
    cmd = build_command(ServerConfig("llama-cpp", "google/g-gguf"))
    assert "-m" in cmd and str(snap / "g_q4_0.gguf") in cmd
    # 同スナップショットの mmproj も自動検出される
    assert "--mmproj" in cmd and str(snap / "g-mmproj.gguf") in cmd


def test_build_command_llama_mtp_draft(hf_cache):
    # MTP ヘッド repo-id を draft に指定 → -md ＋ --spec-type draft-mtp
    hf_cache("org/m-gguf", ["m.gguf"])
    hf_cache("org/d-gguf", ["d-F16-MTP.gguf"])
    c = ServerConfig("llama-cpp", "org/m-gguf", draft_model="org/d-gguf:F16-MTP")
    cmd = build_command(c)
    assert "-md" in cmd
    assert "--spec-type" in cmd and "draft-mtp" in cmd


def test_build_command_llama_embedded_mtp(hf_cache):
    # draft_model="self"/"mtp" は埋め込み MTP。-md 無しで --spec-type draft-mtp のみ。
    # vision(--mmproj) と parallel>1 は llama.cpp 側未対応なので付けない。
    hf_cache("org/m-gguf", ["m.gguf"])
    c = ServerConfig("llama-cpp", "org/m-gguf", parallel=4, draft_model="self")
    cmd = build_command(c)
    assert "--spec-type" in cmd and "draft-mtp" in cmd
    assert "-md" not in cmd
    assert "--parallel" not in cmd
    assert build_command(ServerConfig("llama-cpp", "org/m-gguf", draft_model="mtp")).count("--spec-type") == 1


def test_build_command_llama_plain_draft(hf_cache):
    # mtp を含まないドラフトは -md のみ（spec-type は llama.cpp 既定の draft-simple）
    hf_cache("org/m-gguf", ["m.gguf"])
    hf_cache("org/draft-gguf", ["small-draft.gguf"])
    c = ServerConfig("llama-cpp", "org/m-gguf", draft_model="org/draft-gguf")
    cmd = build_command(c)
    assert "-md" in cmd
    assert "draft-mtp" not in cmd


def test_find_sibling_mmproj(tmp_path):
    model = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    model.write_bytes(b"")
    # 隣に mmproj が無ければ None
    assert srv.find_sibling_mmproj(str(model)) is None
    mm = tmp_path / "mmproj-Qwen3.6-27B-BF16.gguf"
    mm.write_bytes(b"")
    assert srv.find_sibling_mmproj(str(model)) == str(mm)


def test_build_command_llama_auto_mmproj(hf_cache):
    snap = hf_cache("org/m-gguf", ["model.gguf", "mmproj-model.gguf"])
    # 隣の mmproj を自動検出して --mmproj に渡す（手動設定不要）
    cmd = build_command(ServerConfig("llama-cpp", "org/m-gguf"))
    assert "--mmproj" in cmd and str(snap / "mmproj-model.gguf") in cmd


def test_build_command_llama_no_mmproj_optout(hf_cache):
    hf_cache("org/m-gguf", ["model.gguf", "mmproj-model.gguf"])
    # 明示的に --no-mmproj を渡したら自動付与しない
    c = ServerConfig("llama-cpp", "org/m-gguf", extra_args=["--no-mmproj"])
    cmd = build_command(c)
    assert "--mmproj" not in cmd and "--no-mmproj" in cmd


def test_build_command_mlx_disable_thinking(stub_cache):
    cmd = build_command(ServerConfig("mlx", "m", disable_thinking=True))
    assert "--chat-template-args" in cmd


def test_build_command_mlx_vlm(stub_cache):
    cmd = build_command(ServerConfig("mlx-vlm", "vision-model", port=8080))
    # python -m mlx_vlm.server でモデル/ホスト/ポートを渡して起動する
    assert cmd[1:3] == ["-m", "mlx_vlm.server"]
    assert "--model" in cmd and "vision-model" in cmd and "8080" in cmd
    # vision は逐次処理。thinking テンプレート引数は付けない
    assert "--chat-template-args" not in cmd


def test_build_command_mlx_vlm_mtp_drafter(stub_cache):
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


def test_build_command_mlx_no_draft_support(stub_cache):
    # テキスト専用 mlx には MTP を渡さない（mlx-vlm のみ対応）。
    cmd = build_command(ServerConfig("mlx", "m", draft_model="d"))
    assert "--draft-model" not in cmd and "--draft-kind" not in cmd


def test_build_command_no_draft_by_default(stub_cache):
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


def test_build_command_mlx_vlm_draft_auto(stub_cache):
    # build_command でも "auto" が解決され、対応ドラフター＋mtp が付く。
    target = "mlx-community/gemma-4-12B-it-qat-4bit"
    cmd = build_command(ServerConfig("mlx-vlm", target, draft_model="auto"))
    assert cmd[cmd.index("--draft-model") + 1] == MTP_DRAFTERS[target]
    assert cmd[cmd.index("--draft-kind") + 1] == "mtp"


# --- 事前 DL 必須（自動ダウンロード無効）の検証 ----------------------------

def test_ensure_cached_ok_when_present(hf_cache):
    # 重み（safetensors）が揃っていれば OK（スナップショットを返す）。
    snap = hf_cache("org/mlx-model", ["config.json", "model.safetensors"])
    assert srv.ensure_cached("org/mlx-model") == str(snap)


def test_ensure_cached_errors_when_not_cached(hf_cache):
    # キャッシュに無ければ案内付きでエラー（自動 DL しない）。
    with pytest.raises(ValueError, match="hf download"):
        srv.ensure_cached("org/not-cached")


def test_ensure_cached_errors_on_incomplete(hf_cache, tmp_path):
    # DL 途中（*.incomplete 残存）は「未取得」と同じ扱い（今回の不具合の主症状）。
    hf_cache("org/mlx-model", ["config.json", "model.safetensors"])
    blobs = tmp_path / "hub" / "models--org--mlx-model" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "abc123.incomplete").write_bytes(b"")
    with pytest.raises(ValueError, match="hf download"):
        srv.ensure_cached("org/mlx-model")


def test_ensure_cached_rejects_bare_name(hf_cache):
    # repo-id（org/repo）形式でないものは弾く。
    with pytest.raises(ValueError):
        srv.ensure_cached("bare-name")


def test_build_command_mlx_requires_predownload(hf_cache):
    # 配線確認: 未取得モデルでは build_command がそのまま起動せずエラーにする。
    with pytest.raises(ValueError, match="hf download"):
        build_command(ServerConfig("mlx-vlm", "org/uncached-model"))


def test_mlx_vlm_in_backends_and_not_parallel():
    assert "mlx-vlm" in BACKENDS
    assert not parallel_supported("mlx-vlm")


def test_default_backend_and_supported():
    assert default_backend() in BACKENDS
    assert parallel_supported("llama-cpp") and not parallel_supported("mlx")


def test_default_backend_is_vision_on_apple_silicon(monkeypatch):
    # Apple Silicon では既定を vision 対応の mlx-vlm にする（既定モデルが多モーダル）。
    monkeypatch.setattr("local_llm_server.server.sys.platform", "darwin")
    monkeypatch.setattr("local_llm_server.server.platform.machine", lambda: "arm64")
    assert default_backend() == "mlx-vlm"
    # それ以外は llama.cpp
    monkeypatch.setattr("local_llm_server.server.sys.platform", "linux")
    assert default_backend() == "llama-cpp"


def test_parse_host_port():
    assert parse_host_port("http://127.0.0.1:8081/v1") == ("127.0.0.1", 8081)


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


class _Resp:
    """urlopen の戻り値スタブ（/v1/models の JSON を返す context manager）。"""
    def __init__(self, body):
        self._b = body.encode("utf-8")
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _patch_models(monkeypatch, body):
    import local_llm_server.server as srv
    monkeypatch.setattr(
        srv.urllib.request, "urlopen", lambda url, timeout=5.0: _Resp(body)
    )


def test_running_model_parses_v1_models(monkeypatch):
    import local_llm_server.server as srv

    _patch_models(monkeypatch, '{"data": [{"id": "mlx-community/Qwen3.6-27B-4bit"}]}')
    assert srv.running_model("http://x/v1") == "mlx-community/Qwen3.6-27B-4bit"

    def _boom(url, timeout=5.0):
        raise OSError("down")
    monkeypatch.setattr(srv.urllib.request, "urlopen", _boom)
    assert srv.running_model("http://x/v1") is None  # 取得失敗は None


def test_list_models_returns_all_ids(monkeypatch):
    import local_llm_server.server as srv
    _patch_models(monkeypatch, '{"data": [{"id": "a/Foo"}, {"id": "b/Bar"}, {"id": "c/Baz"}]}')
    assert srv.list_models("http://x/v1") == ["a/Foo", "b/Bar", "c/Baz"]

    def _boom(url, timeout=5.0):
        raise OSError("down")
    monkeypatch.setattr(srv.urllib.request, "urlopen", _boom)
    assert srv.list_models("http://x/v1") == []  # 取得失敗は []


def test_model_available_against_router_catalog(monkeypatch):
    import local_llm_server.server as srv
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


def test_estimate_model_bytes_llama_cpp(monkeypatch, tmp_path):
    # llama-cpp は本体 GGUF（＋隣の mmproj）のファイルサイズ合計を概算に使う。
    gguf = tmp_path / "model-Q4.gguf"
    gguf.write_bytes(b"x" * 1000)
    mmproj = tmp_path / "mmproj-F16.gguf"
    mmproj.write_bytes(b"y" * 200)
    monkeypatch.setattr(srv, "resolve_gguf", lambda m: str(gguf))
    cfg = ServerConfig(backend="llama-cpp", model="org/repo:Q4")
    assert srv.estimate_model_bytes(cfg) == 1200  # 本体 1000 ＋ mmproj 200


def test_estimate_model_bytes_unknown_returns_none(monkeypatch):
    # 解決できない（未キャッシュ）なら None（メモリガードはスキップ）。
    def _boom(m):
        raise ValueError("not cached")
    monkeypatch.setattr(srv, "resolve_gguf", _boom)
    assert srv.estimate_model_bytes(ServerConfig(backend="llama-cpp", model="org/x")) is None
    # mlx で未DL（スナップショット無し）なら None
    assert srv.estimate_model_bytes(
        ServerConfig(backend="mlx-vlm", model="mlx-community/Definitely-Not-Downloaded-xyz")
    ) is None


def _touch(path, data=b"x"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _config(path, model_type, architectures):
    import json
    _touch(path, json.dumps(
        {"model_type": model_type, "architectures": architectures}).encode())


def test_discover_cached_models(monkeypatch, tmp_path):
    # 疑似 HF キャッシュを作り、ダウンロード済みのチャットモデルだけが発見されることを確認する。
    root = tmp_path / "hub"
    # mlx 系（生成アーキ）→ 採用
    _config(str(root / "models--mlx-community--Qwen3.6-27B-4bit/snapshots/a/config.json"),
            "qwen3", ["Qwen3ForCausalLM"])
    _touch(str(root / "models--mlx-community--Qwen3.6-27B-4bit/snapshots/a/model.safetensors"))
    # llama-cpp（本体 1 つ＋ mmproj は除外）
    _touch(str(root / "models--unsloth--Foo-GGUF/snapshots/a/Foo-Q4.gguf"))
    _touch(str(root / "models--unsloth--Foo-GGUF/snapshots/a/mmproj-F16.gguf"))
    # llama-cpp（本体が複数 → 量子化ごとにセレクタ付きで列挙）
    _touch(str(root / "models--multi--Bar-GGUF/snapshots/a/Bar-Q4.gguf"))
    _touch(str(root / "models--multi--Bar-GGUF/snapshots/a/Bar-Q8.gguf"))
    # 埋め込みモデル（非チャット）→ 除外
    _config(str(root / "models--intfloat--e5-large/snapshots/a/config.json"),
            "xlm-roberta", ["XLMRobertaModel"])
    _touch(str(root / "models--intfloat--e5-large/snapshots/a/model.safetensors"))
    # 本体の無い repo（mmproj だけ）→ 除外
    _touch(str(root / "models--x--OnlyProj-GGUF/snapshots/a/mmproj.gguf"))
    # モデルでない repo（重みなし）→ 除外
    _touch(str(root / "models--y--JustReadme/snapshots/a/README.md"))
    # Qwen3.6-27B-4bit の MTP ドラフター（揃っている）→ 一覧から除外しつつ本体は mtp="ready"
    _config(str(root / "models--mlx-community--Qwen3.6-27B-MTP-4bit/snapshots/a/config.json"),
            "qwen3", ["Qwen3ForCausalLM"])
    _touch(str(root / "models--mlx-community--Qwen3.6-27B-MTP-4bit/snapshots/a/model.safetensors"))
    # MTP 対応だがドラフター未取得の本体 → mtp="available"
    _config(str(root / "models--mlx-community--gemma-4-E4B-it-qat-4bit/snapshots/a/config.json"),
            "gemma3", ["Gemma3ForCausalLM"])
    _touch(str(root / "models--mlx-community--gemma-4-E4B-it-qat-4bit/snapshots/a/model.safetensors"))

    monkeypatch.setattr(srv, "_hf_hub_cache", lambda: str(root))
    srv._DISCOVER_CACHE["t"] = -1e9  # キャッシュを無効化して再走査させる
    items = srv.discover_cached_models(ttl=0)
    found = {d["id"]: d["backend"] for d in items}
    mtp = {d["id"]: d["mtp"] for d in items}

    assert found["mlx-community/Qwen3.6-27B-4bit"] == "mlx-vlm"
    assert found["unsloth/Foo-GGUF"] == "llama-cpp"
    assert found["multi/Bar-GGUF:Bar-Q4"] == "llama-cpp"
    assert found["multi/Bar-GGUF:Bar-Q8"] == "llama-cpp"
    assert "intfloat/e5-large" not in found     # 埋め込みは除外
    assert "x/OnlyProj-GGUF" not in found
    assert "y/JustReadme" not in found
    # ドラフターは「使えるモデル」一覧に出さない
    assert "mlx-community/Qwen3.6-27B-MTP-4bit" not in found
    # MTP の利用可否: ドラフターが揃う本体は "ready"、未取得は "available"、非対応は None
    assert mtp["mlx-community/Qwen3.6-27B-4bit"] == "ready"
    assert mtp["mlx-community/gemma-4-E4B-it-qat-4bit"] == "available"
    assert mtp["unsloth/Foo-GGUF"] is None


# --- is_ready: 認証ゲート越しのヘルスチェック --------------------------------

def test_is_ready_treats_auth_gated_as_up():
    """api_key 付きゲートウェイ（/v1/models が 401）も「稼働中」と判定する。

    ここで False になると、TUI の自己ヘルスチェック（起動判定・常駐ポーリング）が
    正常起動したゲートウェイを「応答なし」と誤判定してしまう（回帰防止）。
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _AuthGated(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"error": "missing or invalid API key"}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthGated)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert srv.is_ready(f"http://127.0.0.1:{server.server_address[1]}/v1") is True
    finally:
        server.shutdown()
        server.server_close()


def test_is_ready_false_when_down():
    # 誰も LISTEN していないポートは従来どおり False。
    assert srv.is_ready("http://127.0.0.1:9/v1", timeout=0.3) is False
