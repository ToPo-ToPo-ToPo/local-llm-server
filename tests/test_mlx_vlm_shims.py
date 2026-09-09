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
    result = _mlx_vlm_shims._patch_stream_tool_calls()
    assert result.startswith("applied (mlx-vlm 0.7")
    st = mod.ToolCallStreamState("<tool_call>", "</tool_call>")
    assert st.feed('<tool_call>{"name": "write_file"') == '<tool_call>{"name": "write_file"'
    assert st.feed(None) is None
    assert _mlx_vlm_shims._patch_stream_tool_calls().startswith("already applied")


def test_stream_tool_calls_patches_06_function(monkeypatch):
    mod = _install_fake(monkeypatch, with_class=False, with_func=True)
    result = _mlx_vlm_shims._patch_stream_tool_calls()
    assert result.startswith("applied (mlx-vlm 0.6")
    assert mod.suppress_tool_call_content("x", False, "<tool_call>", "abc") == (False, "abc")


def test_stream_tool_calls_reports_when_hook_missing(monkeypatch):
    _install_fake(monkeypatch, with_class=False, with_func=False)
    assert _mlx_vlm_shims._patch_stream_tool_calls().startswith("not applied")


def test_apply_does_not_patch_without_env(monkeypatch):
    mod = _install_fake(monkeypatch, with_class=True, with_func=False)
    monkeypatch.delenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, raising=False)
    _mlx_vlm_shims.apply()
    assert not getattr(mod.ToolCallStreamState, "_llmserver_passthrough", False)


def test_apply_patches_with_env(monkeypatch, capsys):
    mod = _install_fake(monkeypatch, with_class=True, with_func=False)
    monkeypatch.setenv(_mlx_vlm_shims.STREAM_TOOL_CALLS_ENV, "1")
    _mlx_vlm_shims.apply()
    assert getattr(mod.ToolCallStreamState, "_llmserver_passthrough", False)
    assert "stream_tool_calls: applied" in capsys.readouterr().err
