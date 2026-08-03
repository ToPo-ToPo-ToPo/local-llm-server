"""_mlx_vlm_shims が mlx-vlm の Inkling 対応の穴を埋めることを確認する。

mlx-vlm は Apple Silicon でしか入らないので、未導入環境ではスキップする。
上流が修正を入れたらパッチは no-op になる（setdefault / 既存チェック）ので、
「パッチ後に期待する状態になっている」ことだけを検証する。
"""

import pytest

from local_llm_server import _mlx_vlm_shims


def test_apply_is_safe_without_mlx_vlm(monkeypatch):
    """mlx-vlm が無い環境でも apply() は例外を出さない（起動を止めない）。"""
    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", None)
    _mlx_vlm_shims.apply()  # 例外が出ないこと


def test_inkling_subconfigs_exported():
    inkling = pytest.importorskip("mlx_vlm.models.inkling")
    _mlx_vlm_shims.apply()
    # 汎用ローダー（utils.update_module_configs）が getattr する名前
    for name in ("TextConfig", "VisionConfig", "AudioConfig"):
        assert hasattr(inkling, name), name


def test_inkling_registered_in_message_format():
    prompt_utils = pytest.importorskip("mlx_vlm.prompt_utils")
    _mlx_vlm_shims.apply()
    # 未登録だと apply_chat_template が text-only 扱いで画像パートを捨てる
    assert "inkling" in prompt_utils.MODEL_CONFIG
    assert prompt_utils.MODEL_CONFIG["inkling"] is (
        prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST
    )


def test_inkling_content_markers_stripped():
    responses_state = pytest.importorskip("mlx_vlm.server.responses_state")
    _mlx_vlm_shims.apply()
    markers = responses_state._CONTENT_MARKERS
    for marker in ("<|message_model|>", "<|content_text|>", "<|end_message|>"):
        assert marker in markers, marker
    # 上流の既定を消していないこと
    assert "<|START_TEXT|>" in markers


def test_apply_is_idempotent():
    prompt_utils = pytest.importorskip("mlx_vlm.prompt_utils")
    responses_state = pytest.importorskip("mlx_vlm.server.responses_state")
    _mlx_vlm_shims.apply()
    markers_once = responses_state._CONTENT_MARKERS
    fmt_once = prompt_utils.MODEL_CONFIG["inkling"]
    _mlx_vlm_shims.apply()
    assert responses_state._CONTENT_MARKERS == markers_once
    assert prompt_utils.MODEL_CONFIG["inkling"] is fmt_once
