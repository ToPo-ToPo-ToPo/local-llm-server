"""mlx-vlm の上流未修正部分を、モデルサーバー起動時にだけ補うシム。

ゲートウェイは mlx-vlm のモデルサーバーを別プロセスで起動する（server.build_command）。
その起動を `python -m mlx_vlm.server` から `python -m local_llm_server._mlx_vlm_shims` へ
差し替え、パッチを当ててから同じ引数で mlx_vlm.server を __main__ として実行する。

site-packages を直接書き換えない理由: 手動更新が `uv sync` を走らせるため、
venv への直接パッチは黙って失われる。ここに置けばソース側と一緒に追従する。

現在のパッチ:

1. Inkling の content マーカーが除去表に無い（mlx-vlm 0.6.9 時点）
   `server/responses_state.py` の `_CONTENT_MARKERS` は ("<|START_TEXT|>", "<|END_TEXT|>")
   のハードコードで、env にもリクエストにも逃げ道が無い。Inkling は本文を
   `<|message_model|><|content_text|>…<|end_message|>` という構造トークンで囲んで出力する
   ため、思考の分離（MLX_VLM_THINKING_* で対応済み）をしても content にこれらが残る。
   除去表へ追記して本文だけを返す。他モデルはこれらの文字列を出さないので無害。

2. APC の 1 トークンあたりバイト数が短いプロンプトで汚染される（mlx-vlm 0.7.1 時点）
   `_observe_cache_size` は下がることのない最大値で覚えるため、65 トークンのような
   短い要求が 1 回来ると実勢の数倍で固定され、そこから計算する予約がメモリ上限を
   超えて以後すべての保存が見送られる。再起動するまで直らない
   （→ `_patch_apc_rate_poisoning`）。

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
from contextvars import ContextVar
from typing import Optional

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


#: この数より短いプロンプトの観測は「1 トークンあたりのバイト数」の見積りに採らない。
#: KV は一定の粒度（数百トークン単位）でまとめて確保されるので、短いプロンプトでは
#: その固定費が少ないトークン数で割られ、実勢より何倍も大きい値になる。1024 なら
#: 確保の粒度が紛れても 2 割強の過大に収まり、実用のプロンプトはまず上回る。
_APC_RATE_MIN_TOKENS = 1024


def _patch_apc_rate_poisoning() -> None:
    """短いプロンプト 1 回で APC が止まるのを防ぐ（mlx-vlm 0.7.1 時点）。

    上流の問題: ``APCManager._observe_cache_size`` は 1 トークンあたりのバイト数を
    ``max(これまでの値, size / token_count)`` で覚える。**下がることのない最大値**なので、
    短いプロンプトが 1 回来ると実勢よりはるかに大きい値で固定される。この値は
    ``prefill_reserve = 2 × トークン数 × 1トークンのバイト数`` に使われ、そこが
    メモリ上限を超えると以後**すべての保存が見送られる**（``memory_skips`` が増え続ける）。
    最大値は下がらないので、モデルサーバーを再起動するまで直らない。

    実測（64 層・4bit の 27B、18k トークンのプロンプト）:

    ========================================  ==========  ==================
    条件                                      見積り      同じプロンプトの再送
    ========================================  ==========  ==================
    起動直後                                  2.7 GB      9.9 秒（前方一致あり）
    間に 65 トークンの要求を 1 回挟む         81.5 GB     162 秒（前方一致なし）
    ========================================  ==========  ==================

    65 トークンの要求は珍しくない（会話の題を付ける・要約するといった短い呼び出し）。
    エージェントは会話の頭でそれを送ることがあり、その 1 回で以降の全手番が
    毎回フルのプリフィルになる。

    直し方: 短いプロンプトの観測では見積りを上げない。``_prefill_reserve_bytes`` は
    上流の式のまま引き直したいので、``size=0`` でもう一度呼ぶ（最大値なので見積りは
    動かず、予約だけが信頼できる値で計算し直される）。
    """
    try:
        from mlx_vlm import apc as _apc
    except Exception:
        return
    manager = getattr(_apc, "APCManager", None)
    original = getattr(manager, "_observe_cache_size", None)
    if original is None or getattr(original, "_llmserver_patched", False):
        return  # 上流が実装を変えた / 適用済み。触らない

    def _observe_cache_size(self, size: int, token_count: int) -> None:
        before = getattr(self, "_bytes_per_token", None)
        original(self, size, token_count)
        if before is None or token_count <= 0 or token_count >= _APC_RATE_MIN_TOKENS:
            return
        if getattr(self, "_bytes_per_token", before) <= before:
            return  # 短くても見積りを上げていないなら触らない
        self._bytes_per_token = before
        original(self, 0, token_count)  # 予約だけを元の見積りで引き直す

    _observe_cache_size._llmserver_patched = True  # type: ignore[attr-defined]
    manager._observe_cache_size = _observe_cache_size


STREAM_TOOL_CALLS_ENV = "LOCAL_LLM_STREAM_TOOL_CALLS"
#: リクエストごとの指定（クライアントがこのヘッダーで on/off を選ぶ。ゲートウェイが中継する）。
STREAM_TOOL_CALLS_HEADER = "x-stream-tool-calls"

# いま処理中のリクエストの指定（True / False / None = 指定なし＝モデルの既定）。
_REQUEST_STREAM_TOOL_CALLS: ContextVar[Optional[bool]] = ContextVar(
    "llmserver_stream_tool_calls", default=None)


def _default_stream_tool_calls() -> bool:
    """モデルの既定（gateway.toml の stream_tool_calls → 環境変数 LOCAL_LLM_STREAM_TOOL_CALLS=1）。"""
    return os.environ.get(STREAM_TOOL_CALLS_ENV) == "1"


def _stream_tool_calls_now() -> bool:
    """このリクエストでツール呼び出しの生成中トークンを流すか。指定が無ければモデルの既定。"""
    flag = _REQUEST_STREAM_TOOL_CALLS.get()
    return _default_stream_tool_calls() if flag is None else flag


def _parse_flag(raw: bytes | str | None) -> Optional[bool]:
    if raw is None:
        return None
    v = (raw.decode("latin-1") if isinstance(raw, bytes) else str(raw)).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


class _StreamToolCallsFlag:
    """ASGI ミドルウェア: リクエストの ``X-Stream-Tool-Calls`` を読み、処理中の間だけ文脈に置く。

    ストリーム応答の生成は同じ文脈（から作られたタスク）で走るので、ツール呼び出しの抑止を
    決める箇所（_patch_stream_tool_calls）がリクエストごとの指定を読める。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        raw = None
        for k, v in scope.get("headers") or []:
            if k.lower() == STREAM_TOOL_CALLS_HEADER.encode():
                raw = v
                break
        token = _REQUEST_STREAM_TOOL_CALLS.set(_parse_flag(raw))
        try:
            await self.app(scope, receive, send)
        finally:
            _REQUEST_STREAM_TOOL_CALLS.reset(token)


def _install_request_flag() -> str:
    """mlx-vlm の app にリクエストごとの指定を読むミドルウェアを足す（起動前に 1 回）。"""
    try:
        import mlx_vlm.server as _srv
    except Exception as exc:  # noqa: BLE001 - mlx-vlm 無し等。起動は止めない
        return f"per-request not available (mlx_vlm.server unavailable: {exc})"
    app = getattr(_srv, "app", None)
    if app is None or not hasattr(app, "add_middleware"):
        return "per-request not available (no app)"
    if getattr(app, "_llmserver_stream_flag", False):
        return "per-request already installed"
    app.add_middleware(_StreamToolCallsFlag)
    app._llmserver_stream_flag = True
    return "per-request enabled (header X-Stream-Tool-Calls)"


def _patch_stream_tool_calls(log=None) -> str:
    """ツール呼び出しの生成中トークンを、流すと決まったリクエストでは捨てずに流す。

    上流の挙動(mlx-vlm 0.6/0.7): ストリーミング中に生成テキストへ <tool_call> が現れると、
    以後の delta を content から**捨て**、生成終了後に全文を解析した tool_calls を最後の
    1 チャンクにまとめて出す。そのため「ファイル本文を書いている最中の文字」はどのクライアント
    にも届かない(会社リポジトリのエディタのライブ表示が成立しない)。

    このパッチは「捨てる」部分を、流すと決まったリクエストでだけ素通しに変える。流すかどうかは
    リクエストの ``X-Stream-Tool-Calls`` ヘッダー(_install_request_flag)、無ければモデルの既定
    (gateway.toml の stream_tool_calls)。最後の解析済み tool_calls チャンクは full_output(生成全文)
    から作られるので従来どおり出る=ツール呼び出しの正しさは不変。副作用として
    <tool_call>…</tool_call> の生テキストが delta.content に流れるため、受け側
    (local-llm-client 0.8+)が本文から剥がして途中経過として扱う。知らないクライアントには生 JSON が
    本文に見えるので、既定は off で、使うクライアントだけがヘッダーで頼む。

    上流の構造が変わるとパッチが当たらない。その場合は従来の一括挙動に戻るだけで
    壊れはしないが、黙って劣化しないよう結果を文字列で返し、起動ログに出す。
      - mlx-vlm 0.6.x: openai.suppress_tool_call_content(関数)を条件つき素通し版に差し替える
      - mlx-vlm 0.7.x: openai.ToolCallStreamState(クラス)を条件つき素通しの派生に差し替える
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

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # 流すかどうかは作られた時点(=リクエストの開始)で決める
                self._llmserver_stream = _stream_tool_calls_now()

            def feed(self, text, last: bool = False):
                if not self._llmserver_stream:
                    return super().feed(text, last)
                # 状態(in_tool_call)は追跡しない: 呼び出し側は戻り値の本文しか使わない。
                # None はそのまま(空 delta の抑止は呼び出し側が行う)
                return text

        _oa.ToolCallStreamState = _PassthroughToolCallStreamState
        return "applied (mlx-vlm 0.7: ToolCallStreamState passthrough)"

    fn = getattr(_oa, "suppress_tool_call_content", None)
    if fn is not None:
        if getattr(fn, "_llmserver_passthrough", False):
            return "already applied (0.6 function)"
        orig = fn

        def suppress_tool_call_content(full_output, in_tool_call, tc_start, delta_content):
            if not _stream_tool_calls_now():
                return orig(full_output, in_tool_call, tc_start, delta_content)
            return in_tool_call, delta_content

        suppress_tool_call_content._llmserver_passthrough = True  # type: ignore[attr-defined]
        _oa.suppress_tool_call_content = suppress_tool_call_content
        return "applied (mlx-vlm 0.6: suppress_tool_call_content passthrough)"

    return "not applied (no known suppression hook in mlx_vlm.server.openai)"


def apply() -> None:
    """既知のパッチを全て適用する。失敗しても起動は止めない。"""
    _patch_content_markers()
    _patch_apc_extra_hash()
    _patch_apc_rate_poisoning()
    # ツール呼び出しの生成中トークン: パッチは常に当て、流すかはリクエストごとに決める
    # (ヘッダー X-Stream-Tool-Calls。無ければモデルの既定 = gateway.toml の stream_tool_calls)
    result = _patch_stream_tool_calls()
    flag = _install_request_flag()
    default = "on" if _default_stream_tool_calls() else "off"
    print(f"[local-llm-server shim] stream_tool_calls: {result}; default {default}; {flag}",
          file=sys.stderr, flush=True)


def main() -> None:
    apply()
    # sys.argv はそのまま（argv[0] は argparse が見ない）。mlx_vlm.server を __main__ として実行。
    sys.argv[0] = "mlx_vlm.server"
    runpy.run_module("mlx_vlm.server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
