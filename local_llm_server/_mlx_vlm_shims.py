"""mlx-vlm の上流未修正部分を、モデルサーバー起動時にだけ補うシム。

ゲートウェイは mlx-vlm のモデルサーバーを別プロセスで起動する（server.build_command）。
その起動を `python -m mlx_vlm.server` から `python -m local_llm_server._mlx_vlm_shims` へ
差し替え、パッチを当ててから同じ引数で mlx_vlm.server を __main__ として実行する。

site-packages を直接書き換えない理由: 自動更新（auto_update）が `uv sync` を走らせるため、
venv への直接パッチは黙って失われる。ここに置けばソース側と一緒に追従する。

現在のパッチ:

1. Inkling の content マーカーが除去表に無い（mlx-vlm 0.6.9 時点）
   `server/responses_state.py` の `_CONTENT_MARKERS` は ("<|START_TEXT|>", "<|END_TEXT|>")
   のハードコードで、env にもリクエストにも逃げ道が無い。Inkling は本文を
   `<|message_model|><|content_text|>…<|end_message|>` という構造トークンで囲んで出力する
   ため、思考の分離（MLX_VLM_THINKING_* で対応済み）をしても content にこれらが残る。
   除去表へ追記して本文だけを返す。他モデルはこれらの文字列を出さないので無害。

（履歴）0.6.7 / 0.6.8 向けに当てていた次の 2 つは **0.6.9 で上流が修正したため削除した**:
  - models/inkling が sub-config クラス（TextConfig 等）を再エクスポートせず、
    汎用ローダーの getattr が必ず AttributeError になる問題
  - prompt_utils.MODEL_CONFIG に inkling が無く、apply_chat_template が text-only 扱いで
    画像・音声パートを黙って捨てる問題
0.6.9 は公式 config をそのまま読む（model_type=inkling_mm_model の別名登録あり）ので、
変換時の config 翻訳も不要になった。→ pyproject の mlx-vlm ピン（>=0.6.9）
"""

from __future__ import annotations

import os
import runpy
import sys

# Inkling が本文の周りに出す構造トークン。content から取り除く対象。
# <|end_message|> は思考の終端でもあるが、思考の切り出しはこの除去より前に
# 行われる（_split_thinking → _strip_content_markers の順）ので競合しない。
_INKLING_CONTENT_MARKERS = (
    "<|message_model|>",
    "<|content_text|>",
    "<|end_message|>",
)


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


def _patch_apc_extra_hash() -> None:
    """APC の照合キーから inputs_embeds / attention_mask を除外する。

    上流の問題（mlx-vlm 0.6.9 時点）: サーバーの連続バッチング経路は**全リクエスト**で
    get_input_embeddings を呼び、prompt_kwargs に inputs_embeds を積む（BatchGenerator の
    前提。server/generation.py の _gpu_embed）。一方 APC の照合キー semantic_extra_hash は
    media["embeddings"] として inputs_embeds を、media["masks"] として attention_mask を
    **テンソル内容ごと**ハッシュする。テキストの埋め込みはトークン列から決定的に決まる
    ので、これはプロンプト全体を照合キーへ焼き込むのと同じ — 1 トークンでも違えば
    キーが変わり、exact キャッシュの**前方一致が完全一致に退化**する
    （実測: exact_stores だけ増えて exact_hits は 0。同一プロンプトの再送だけ当たる）。

    埋め込みとマスクを除いても照合の安全性は落ちない:
      - exact キャッシュのエントリは**トークン列そのもの**を持ち、照合は
        token_tuple[:n] == entry.token_ids で行われる。テキストの同一性はここで担保される
      - 画像は image_hash（pixel_values の内容ハッシュ）が別枠で残る
      - 音声・動画（input_features / pixel_values_videos）もそのまま残す
    """
    try:
        from mlx_vlm import apc as _apc
    except Exception:
        return
    original = getattr(_apc, "semantic_extra_hash", None)
    if original is None or getattr(original, "_llmserver_patched", False):
        return  # 上流が実装を変えた / 適用済み。触らない

    def semantic_extra_hash(*, media=None, **kwargs):
        if media:
            media = {k: v for k, v in media.items()
                     if k not in ("embeddings", "masks")}
        return original(media=media, **kwargs)

    semantic_extra_hash._llmserver_patched = True  # type: ignore[attr-defined]
    _apc.semantic_extra_hash = semantic_extra_hash
    # ar.py は `from .. import apc as _apc` のモジュール参照経由で呼ぶため、
    # モジュール属性の差し替えだけで全呼び出し箇所に効く


STREAM_TOOL_CALLS_ENV = "LOCAL_LLM_STREAM_TOOL_CALLS"


def _patch_stream_tool_calls(log=None) -> str:
    """ツール呼び出しの生成中トークンを捨てずに流す(環境変数 LOCAL_LLM_STREAM_TOOL_CALLS=1 のとき)。

    上流の挙動(mlx-vlm 0.6/0.7): ストリーミング中に生成テキストへ <tool_call> が現れると、
    以後の delta を content から**捨て**、生成終了後に全文を解析した tool_calls を最後の
    1 チャンクにまとめて出す。そのため「ファイル本文を書いている最中の文字」はどのクライアント
    にも届かない(会社リポジトリのエディタのライブ表示が成立しない)。

    このパッチは「捨てる」部分だけを素通しに変える。最後の解析済み tool_calls チャンクは
    full_output(生成全文)から作られるので従来どおり出る=ツール呼び出しの正しさは不変。
    副作用として <tool_call>…</tool_call> の生テキストが delta.content に流れるため、
    受け側(local-llm-client 0.8+)が本文から剥がして途中経過として扱う。知らない
    クライアントには生 JSON が本文に見えるので、既定 off(設定 stream_tool_calls)。

    上流の構造が変わるとパッチが当たらない。その場合は従来の一括挙動に戻るだけで
    壊れはしないが、黙って劣化しないよう結果を文字列で返し、起動ログに出す。
      - mlx-vlm 0.6.x: openai.suppress_tool_call_content(関数)を素通し版に差し替える
      - mlx-vlm 0.7.x: openai.ToolCallStreamState(クラス)を feed が素通しの派生に差し替える
    """
    try:
        from mlx_vlm.server import openai as _oa
    except Exception as exc:  # noqa: BLE001 - mlx-vlm 無し等。起動は止めない
        return f"not applied (mlx_vlm.server.openai unavailable: {exc})"

    cls = getattr(_oa, "ToolCallStreamState", None)
    if cls is not None:
        if getattr(cls, "_llmserver_passthrough", False):
            return "already applied (0.7 class)"

        class _PassthroughToolCallStreamState(cls):  # type: ignore[misc,valid-type]
            _llmserver_passthrough = True

            def feed(self, text, last: bool = False):
                # 状態(in_tool_call)は追跡しない: 呼び出し側は戻り値の本文しか使わない。
                # None はそのまま(空 delta の抑止は呼び出し側が行う)
                return text

        _oa.ToolCallStreamState = _PassthroughToolCallStreamState
        return "applied (mlx-vlm 0.7: ToolCallStreamState passthrough)"

    fn = getattr(_oa, "suppress_tool_call_content", None)
    if fn is not None:
        if getattr(fn, "_llmserver_passthrough", False):
            return "already applied (0.6 function)"

        def suppress_tool_call_content(full_output, in_tool_call, tc_start, delta_content):
            return in_tool_call, delta_content

        suppress_tool_call_content._llmserver_passthrough = True  # type: ignore[attr-defined]
        _oa.suppress_tool_call_content = suppress_tool_call_content
        return "applied (mlx-vlm 0.6: suppress_tool_call_content passthrough)"

    return "not applied (no known suppression hook in mlx_vlm.server.openai)"


def apply() -> None:
    """既知のパッチを全て適用する。失敗しても起動は止めない。"""
    _patch_content_markers()
    _patch_apc_extra_hash()
    if os.environ.get(STREAM_TOOL_CALLS_ENV) == "1":
        result = _patch_stream_tool_calls()
        print(f"[local-llm-server shim] stream_tool_calls: {result}", file=sys.stderr, flush=True)


def main() -> None:
    apply()
    # sys.argv はそのまま（argv[0] は argparse が見ない）。mlx_vlm.server を __main__ として実行。
    sys.argv[0] = "mlx_vlm.server"
    runpy.run_module("mlx_vlm.server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
