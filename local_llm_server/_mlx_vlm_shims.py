"""mlx-vlm の上流未修正部分を、モデルサーバー起動時にだけ補うシム。

ゲートウェイは mlx-vlm のモデルサーバーを別プロセスで起動する（server.build_command）。
その起動を `python -m mlx_vlm.server` から `python -m local_llm_server._mlx_vlm_shims` へ
差し替え、パッチを当ててから同じ引数で mlx_vlm.server を __main__ として実行する。

site-packages を直接書き換えない理由: 自動更新（auto_update）が `uv sync` を走らせるため、
venv への直接パッチは黙って失われる。ここに置けばソース側と一緒に追従する。

現在のパッチ:

1. models/inkling が sub-config クラスを再エクスポートしていない（mlx-vlm 0.6.7 / 0.6.8）
   `utils.load_model` は `modules = ["text", "vision", "perceiver", "projector", "audio"]` を
   ハードコードし、`update_module_configs` が `model_class.TextConfig` 等を getattr する。
   ところが `mlx_vlm/models/inkling/__init__.py` は Model / ModelConfig / LanguageModel しか
   公開していないため、Inkling 系は公開ローダー経路に載せた瞬間 AttributeError で落ちる
   （＝素の mlx-vlm では Inkling をロードできない）。config.py には定義自体はあるので、
   モジュール属性として補うだけで通る。上流が再エクスポートを追加したら本パッチは no-op。

2. content マーカーの除去表が固定（mlx-vlm 0.6.7 / 0.6.8）
   `server/responses_state.py` の `_CONTENT_MARKERS` は ("<|START_TEXT|>", "<|END_TEXT|>")
   のハードコードで、env にもリクエストにも逃げ道が無い。Inkling は本文を
   `<|message_model|><|content_text|>…<|end_message|>` という構造トークンで囲んで出力する
   ため、思考の分離（MLX_VLM_THINKING_* で対応済み）をしても content にこれらが残る。
   除去表へ追記して本文だけを返す。他モデルはこれらの文字列を出さないので無害。

3. prompt_utils.MODEL_CONFIG に inkling が無い（mlx-vlm 0.6.7 / 0.6.8）
   `apply_chat_template` は `model_type not in MODEL_CONFIG` のモデルを「text-only」と
   みなし、`extract_text_from_content` でメッセージから画像・音声パートを**捨てる**。
   その結果 Inkling では画像を送っても chat template まで届かず（実測: 画像あり/なし/
   別画像で prompt_tokens が 41 のまま完全一致、応答も同一の幻覚）、エラーも出ない。
   Inkling の chat template は part.type ∈ (image, input_image, image_url) を受け付けて
   <|content_image|><|unused_200054|> を出すので、画像を先頭に並べる LIST_WITH_IMAGE_FIRST
   を登録すれば正しく渡る（同フォーマッタは num_audios も扱うので音声入力も同経路）。
"""

from __future__ import annotations

import runpy
import sys

_INKLING_SUBCONFIGS = ("TextConfig", "VisionConfig", "AudioConfig")

# Inkling が本文の周りに出す構造トークン。content から取り除く対象。
# <|end_message|> は思考の終端マーカーでもあるが、思考の切り出しはこの除去より前に
# 行われる（_split_thinking → _strip_content_markers の順）ので競合しない。
_INKLING_CONTENT_MARKERS = (
    "<|message_model|>",
    "<|content_text|>",
    "<|end_message|>",
)


def _patch_inkling_subconfig_exports() -> None:
    """mlx_vlm.models.inkling に TextConfig / VisionConfig / AudioConfig を生やす。"""
    try:
        from mlx_vlm.models import inkling
        from mlx_vlm.models.inkling import config as inkling_config
    except Exception:
        return  # inkling 非対応版の mlx-vlm。何もしない
    for name in _INKLING_SUBCONFIGS:
        if not hasattr(inkling, name) and hasattr(inkling_config, name):
            setattr(inkling, name, getattr(inkling_config, name))


def _patch_content_markers() -> None:
    """Inkling の構造トークンを content 除去表へ追加する。"""
    try:
        from mlx_vlm.server import responses_state
    except Exception:
        return
    markers = getattr(responses_state, "_CONTENT_MARKERS", None)
    if markers is None:
        return  # 上流が実装を変えた。触らない
    missing = tuple(m for m in _INKLING_CONTENT_MARKERS if m not in markers)
    if missing:
        responses_state._CONTENT_MARKERS = tuple(markers) + missing


def _patch_inkling_message_format() -> None:
    """prompt_utils.MODEL_CONFIG に inkling を登録し、画像・音声パートを捨てさせない。"""
    try:
        from mlx_vlm import prompt_utils
    except Exception:
        return
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if model_config is None or message_format is None:
        return  # 上流が実装を変えた。触らない
    fmt = getattr(message_format, "LIST_WITH_IMAGE_FIRST", None)
    if fmt is not None:
        model_config.setdefault("inkling", fmt)


def apply() -> None:
    """既知のパッチを全て適用する。失敗しても起動は止めない。"""
    _patch_inkling_subconfig_exports()
    _patch_content_markers()
    _patch_inkling_message_format()


def main() -> None:
    apply()
    # sys.argv はそのまま（argv[0] は argparse が見ない）。mlx_vlm.server を __main__ として実行。
    sys.argv[0] = "mlx_vlm.server"
    runpy.run_module("mlx_vlm.server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
