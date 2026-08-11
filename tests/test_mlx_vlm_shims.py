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
