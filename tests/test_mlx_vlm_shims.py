"""_mlx_vlm_shims が mlx-vlm の残る穴（content マーカー）を埋めることを確認する。

mlx-vlm は Apple Silicon でしか入らないので、未導入環境ではスキップする。
上流が修正を入れたらパッチは no-op になる（既存チェック）ので、「パッチ後に期待する
状態になっている」ことだけを検証する。
"""

import pytest

from local_llm_server import _mlx_vlm_shims


def test_apply_is_safe_without_mlx_vlm(monkeypatch):
    """mlx-vlm が無い環境でも apply() は例外を出さない（起動を止めない）。"""
    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", None)
    _mlx_vlm_shims.apply()  # 例外が出ないこと


def test_inkling_content_markers_stripped():
    responses_state = pytest.importorskip("mlx_vlm.server.responses_state")
    _mlx_vlm_shims.apply()
    markers = responses_state._CONTENT_MARKERS
    for marker in ("<|message_model|>", "<|content_text|>", "<|end_message|>"):
        assert marker in markers, marker
    # 上流の既定を消していないこと
    assert "<|START_TEXT|>" in markers


def test_apply_is_idempotent():
    responses_state = pytest.importorskip("mlx_vlm.server.responses_state")
    _mlx_vlm_shims.apply()
    once = responses_state._CONTENT_MARKERS
    _mlx_vlm_shims.apply()
    assert responses_state._CONTENT_MARKERS == once


def test_upstream_fixed_gaps_are_not_repatched():
    """0.6.9 が直した 2 点（sub-config 再エクスポート / MODEL_CONFIG 登録）は上流に在ること。

    ここが落ちたら mlx-vlm が 0.6.9 未満に落ちている（pyproject のピンを確認する）。
    その状態では Inkling は公開ローダー経路で読めず、変換物の重みも欠ける。
    """
    inkling = pytest.importorskip("mlx_vlm.models.inkling")
    prompt_utils = pytest.importorskip("mlx_vlm.prompt_utils")
    utils = pytest.importorskip("mlx_vlm.utils")
    for name in ("TextConfig", "VisionConfig", "AudioConfig"):
        assert hasattr(inkling, name), f"{name} が未エクスポート（mlx-vlm < 0.6.9？）"
    assert "inkling" in prompt_utils.MODEL_CONFIG
    assert utils.MODEL_REMAPPING.get("inkling_mm_model") == "inkling"


def test_apc_extra_hash_ignores_embeddings_and_masks():
    """APC の照合キーが inputs_embeds / attention_mask に左右されないこと。

    上流はサーバーの連続バッチング経路で全リクエストに inputs_embeds を積み、
    semantic_extra_hash がそれを内容ごとハッシュする。テキストの埋め込みは
    トークン列から決定的に決まるため、これは照合キーへプロンプト全体を焼き込む
    のと同じで、exact キャッシュの前方一致が完全一致に退化していた
    （実測: exact_stores だけ増えて exact_hits が 0）。
    """
    apc = pytest.importorskip("mlx_vlm.apc")
    mx = pytest.importorskip("mlx.core")
    _mlx_vlm_shims.apply()

    base = apc.semantic_extra_hash(tenant=None, image_hash=7, media=None)
    with_embeds = apc.semantic_extra_hash(
        tenant=None, image_hash=7,
        media={"embeddings": mx.ones((1, 8, 4)), "masks": mx.ones((1, 8))},
    )
    assert with_embeds == base  # 埋め込み・マスクはキーに影響しない

    # 音声・動画は従来どおりキーに効く（別コンテンツを同一視しない）
    with_audio = apc.semantic_extra_hash(
        tenant=None, image_hash=7, media={"audio": mx.ones((1, 4))},
    )
    assert with_audio != base


def test_apc_patch_is_idempotent():
    apc = pytest.importorskip("mlx_vlm.apc")
    _mlx_vlm_shims.apply()
    once = apc.semantic_extra_hash
    _mlx_vlm_shims.apply()
    assert apc.semantic_extra_hash is once   # 二重ラップしない


def _bare_manager(apc):
    """_observe_cache_size だけを動かすための最小の APCManager（Metal を触らない）。"""
    import threading
    if "_observe_cache_size" not in vars(apc.APCManager):
        pytest.skip("上流がこの関数を持たない版（修正済み or 実装変更）。パッチは no-op")
    m = object.__new__(apc.APCManager)
    m.lock = threading.RLock()
    m._bytes_per_token = 0.0
    m._prefill_tokens = 0
    m._prefill_reserve_bytes = 0
    return m


_REAL_RATE = 254660      # 実測: 64 層 4bit の 27B は 1 トークン約 249KB
_SHORT_RATE = 1670000    # 65 トークンの要求では確保の固定費が乗って 1.6MB 相当に見える


def _turn(m, tokens, rate):
    """1 リクエストぶんの流れ（プリフィルの申告 → 実際の大きさの観測）を真似る。"""
    m._prefill_tokens = tokens
    m._prefill_reserve_bytes = int(2 * tokens * m._bytes_per_token)
    m._observe_cache_size(tokens * rate, tokens)


def test_apc_rate_is_not_poisoned_by_a_short_prompt():
    """短いプロンプト 1 回で 1 トークンあたりの見積りが跳ね上がらないこと。

    上流は max(これまでの値, size / token_count) で覚えるので、65 トークンのような
    短い要求（会話の題を付ける等）が 1 回来ると実勢の数倍で固定される。その値から
    計算する予約がメモリ上限を超え、以後すべての保存が見送られて、モデルサーバーを
    再起動するまで前方一致が効かなくなる（実測 9.9 秒 → 162 秒、予約 2.7GB → 81.5GB）。
    """
    apc = pytest.importorskip("mlx_vlm.apc")
    _mlx_vlm_shims.apply()
    m = _bare_manager(apc)

    _turn(m, 18000, _REAL_RATE)          # 実用のプロンプト
    healthy_rate = m._bytes_per_token
    healthy_reserve = m._prefill_reserve_bytes
    assert healthy_rate > 0

    _turn(m, 65, _SHORT_RATE)            # 会話の題を付ける短い要求
    assert m._bytes_per_token == healthy_rate      # 見積りを汚さない

    _turn(m, 18000, _REAL_RATE)          # 次の手番
    assert m._bytes_per_token == healthy_rate
    assert m._prefill_reserve_bytes == healthy_reserve   # 予約も元のまま


def test_apc_rate_still_rises_for_long_prompts():
    """長いプロンプトの観測は従来どおり見積りを上げる（過小評価へ倒さない）。"""
    apc = pytest.importorskip("mlx_vlm.apc")
    _mlx_vlm_shims.apply()
    m = _bare_manager(apc)
    _turn(m, 18000, _REAL_RATE)
    before = m._bytes_per_token
    _turn(m, 20000, _REAL_RATE * 2)
    assert m._bytes_per_token > before


def test_apc_rate_patch_is_idempotent():
    apc = pytest.importorskip("mlx_vlm.apc")
    _mlx_vlm_shims.apply()
    once = apc.APCManager._observe_cache_size
    _mlx_vlm_shims.apply()
    assert apc.APCManager._observe_cache_size is once   # 二重ラップしない


# ---- stream_tool_calls: ツール呼び出しの生成中トークンを流す ----------------------

import sys as _sys
import types as _types


def _fake_openai_module(*, with_class: bool, with_func: bool):
    """mlx_vlm.server.openai の偽物。0.7 系(クラス)か 0.6 系(関数)のどちらかを持たせる。"""
    mod = _types.ModuleType("mlx_vlm.server.openai")
    if with_class:
        class ToolCallStreamState:
            def __init__(self, tc_start, tc_end):
                self.tc_start, self.tc_end = tc_start, tc_end
            def feed(self, text, last=False):
                return None  # 上流: ツール呼び出し中は捨てる
        mod.ToolCallStreamState = ToolCallStreamState
    if with_func:
        def suppress_tool_call_content(full_output, in_tool_call, tc_start, delta_content):
            return True, None  # 上流: 捨てる
        mod.suppress_tool_call_content = suppress_tool_call_content
    pkg_server = _types.ModuleType("mlx_vlm.server")
    pkg_server.openai = mod
    pkg = _types.ModuleType("mlx_vlm")
    pkg.server = pkg_server
    return pkg, pkg_server, mod


def _install_fake(monkeypatch, **kw):
    pkg, pkg_server, mod = _fake_openai_module(**kw)
    monkeypatch.setitem(_sys.modules, "mlx_vlm", pkg)
    monkeypatch.setitem(_sys.modules, "mlx_vlm.server", pkg_server)
    monkeypatch.setitem(_sys.modules, "mlx_vlm.server.openai", mod)
    return mod


def test_stream_tool_calls_patches_07_class(monkeypatch):
    mod = _install_fake(monkeypatch, with_class=True, with_func=False)
    monkeypatch.setenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, "1")      # モデルの既定 on
    result = _mlx_vlm_shims._patch_stream_tool_calls()
    assert result.startswith("applied (mlx-vlm 0.7")
    st = mod.ToolCallStreamState("<tool_call>", "</tool_call>")
    assert st.feed('<tool_call>{"name": "write_file"') == '<tool_call>{"name": "write_file"'
    assert st.feed(None) is None
    assert _mlx_vlm_shims._patch_stream_tool_calls().startswith("already applied")


def test_stream_tool_calls_patches_06_function(monkeypatch):
    mod = _install_fake(monkeypatch, with_class=False, with_func=True)
    monkeypatch.setenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, "1")      # モデルの既定 on
    result = _mlx_vlm_shims._patch_stream_tool_calls()
    assert result.startswith("applied (mlx-vlm 0.6")
    assert mod.suppress_tool_call_content("x", False, "<tool_call>", "abc") == (False, "abc")


def test_stream_tool_calls_reports_when_hook_missing(monkeypatch):
    _install_fake(monkeypatch, with_class=False, with_func=False)
    assert _mlx_vlm_shims._patch_stream_tool_calls().startswith("not applied")


def test_apply_patches_but_keeps_the_upstream_behavior_by_default(monkeypatch, capsys):
    """パッチは常に当てる。指定が無ければ(ヘッダーもモデルの既定も無し)上流どおり捨てる。"""
    mod = _install_fake(monkeypatch, with_class=True, with_func=False)
    monkeypatch.delenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, raising=False)
    _mlx_vlm_shims.apply()
    assert getattr(mod.ToolCallStreamState, "_llmserver_passthrough", False)
    assert mod.ToolCallStreamState("<tool_call>", "</tool_call>").feed("<tool_call>{") is None
    err = capsys.readouterr().err
    assert "stream_tool_calls: applied" in err and "default off" in err


def test_model_default_on_streams_without_a_header(monkeypatch):
    mod = _install_fake(monkeypatch, with_class=True, with_func=False)
    monkeypatch.setenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, "1")
    _mlx_vlm_shims.apply()
    assert mod.ToolCallStreamState("<tool_call>", "</tool_call>").feed("<tool_call>{") == "<tool_call>{"


def test_per_request_flag_overrides_the_model_default(monkeypatch):
    """リクエストごとの指定(X-Stream-Tool-Calls)がモデルの既定より勝つ。流すのは頼んだリクエストだけ。"""
    mod = _install_fake(monkeypatch, with_class=True, with_func=False)
    monkeypatch.delenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, raising=False)
    _mlx_vlm_shims._patch_stream_tool_calls()
    var = _mlx_vlm_shims._REQUEST_STREAM_TOOL_CALLS
    tok = var.set(True)
    try:
        on = mod.ToolCallStreamState("<tool_call>", "</tool_call>")
    finally:
        var.reset(tok)
    off = mod.ToolCallStreamState("<tool_call>", "</tool_call>")
    assert on.feed("<tool_call>{") == "<tool_call>{"     # 頼んだリクエスト
    assert off.feed("<tool_call>{") is None               # 頼んでいないリクエスト(上流どおり)

    monkeypatch.setenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, "1")
    tok = var.set(False)                                   # 既定 on でも、断ったリクエストは流さない
    try:
        assert mod.ToolCallStreamState("<tool_call>", "</tool_call>").feed("<tool_call>{") is None
    finally:
        var.reset(tok)


def test_per_request_flag_for_the_06_function(monkeypatch):
    mod = _install_fake(monkeypatch, with_class=False, with_func=True)
    monkeypatch.delenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, raising=False)
    _mlx_vlm_shims._patch_stream_tool_calls()
    assert mod.suppress_tool_call_content("x", False, "<tool_call>", "abc") == (True, None)
    tok = _mlx_vlm_shims._REQUEST_STREAM_TOOL_CALLS.set(True)
    try:
        assert mod.suppress_tool_call_content("x", False, "<tool_call>", "abc") == (False, "abc")
    finally:
        _mlx_vlm_shims._REQUEST_STREAM_TOOL_CALLS.reset(tok)


def test_middleware_reads_the_header_for_the_request_only():
    """ASGI ミドルウェアがヘッダーを読み、そのリクエストの処理中だけ文脈に置く。"""
    import asyncio

    seen = []

    async def app(scope, receive, send):
        seen.append(_mlx_vlm_shims._REQUEST_STREAM_TOOL_CALLS.get())

    mw = _mlx_vlm_shims._StreamToolCallsFlag(app)

    async def run(headers):
        await mw({"type": "http", "headers": headers}, None, None)

    asyncio.run(run([(b"x-stream-tool-calls", b"1")]))
    asyncio.run(run([(b"x-stream-tool-calls", b"0")]))
    asyncio.run(run([]))
    assert seen == [True, False, None]
    assert _mlx_vlm_shims._REQUEST_STREAM_TOOL_CALLS.get() is None


def test_install_request_flag_adds_the_middleware_once(monkeypatch):
    _install_fake(monkeypatch, with_class=True, with_func=False)
    added = []

    class _App:
        def add_middleware(self, cls):
            added.append(cls)

    _sys.modules["mlx_vlm.server"].app = _App()
    assert _mlx_vlm_shims._install_request_flag().startswith("per-request enabled")
    assert _mlx_vlm_shims._install_request_flag() == "per-request already installed"
    assert added == [_mlx_vlm_shims._StreamToolCallsFlag]
