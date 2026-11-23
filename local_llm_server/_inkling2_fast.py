"""Inkling 2 Mini の生成（1 トークンずつ進む段階）を速くする、mlx-vlm への差し替え。

Inkling 2 Mini は中段 12 層を 3 周するので、1 トークンに 48 回層を通る。1 回ごとに小さな演算が
数十個あり、M2 Ultra では GPU に送る回数そのものが時間の大半を占める（実測 40.4 ms/トークンのうち、
重みの読み出しに要るのは 10 ms 程度）。ここでは、その小さな演算の並びを ``mx.compile`` で
ひとまとめにし、送る回数を減らす。考え方は sybil-solutions/local-ai-registry#461（mlx-lm 向け）から
借り、mlx-vlm の実装に合わせて書き直した。

- 生成時（新しいトークンが 1 つ）だけ差し替える。依頼文の読み込み（prefill）は mlx-vlm のまま
  （まとめると読み込みが遅くなると #461 が報告している）。
- 計算の中身・型・順序は mlx-vlm と同じにする。まとめても結果が変わらないことを
  tests/test_inkling2_fast.py と実機の比較（貪欲生成のトークン列）で確かめる。

``install`` は mlx_vlm の Inkling 2 実装が見つかったときだけ当て、当てたものの名前を返す。
上流が実装を変えて前提が崩れていれば、何もしない（遅いだけで、壊さない）。
"""

from __future__ import annotations

import os
from collections.abc import Iterable

ALL = (
    "gates",
    "attention",
    "prefill_bound",
    "head",
    "fused_residual",
    "fused_router",
    "fused_gates",
    "step",
)

# ゲートウェイが Inkling 2 のモデルサーバーにだけ渡す環境変数（未設定の分だけ。利用者の指定が優先）。
# MLX_MAX_*_PER_BUFFER: 1 つの GPU コマンドバッファに積む演算の数と量の上限。既定のままだと
# 1 トークンの数千個の小さな演算が細かく区切って送られ、その区切りごとの待ちが積み上がる。
# M2 Ultra の実測（1 トークンあたり、5 回交互に測った中央値）: 既定 44.59 ms → 300/1000 で 37.01 ms
# → 下の差し替えと合わせて 31.25 ms。値を上げても速くはならず（1000/2000 で 31.1〜35.2 ms と不安定）、
# メモリだけが増える（300/1000 で生成時 +3 GB）。
SERVER_ENV = {
    "MLX_MAX_OPS_PER_BUFFER": "300",
    "MLX_MAX_MB_PER_BUFFER": "1000",
    "LOCAL_LLM_INKLING2_FAST": ",".join(ALL),
}


def is_inkling2(model_type: str | None) -> bool:
    """config.json の model_type が Inkling 2 系（inkling2 / inkling2_loop）か。"""
    return model_type is not None and model_type.startswith("inkling2")


def _decode_only(cls, name: str, fast, length_of) -> None:
    """生成時（長さ 1）だけ fast を使い、それ以外は元の実装に任せる。"""
    original = getattr(cls, name)
    if _wrapped_by(original, "_inkling2_decode_only"):
        return  # 二重に当てない

    def dispatch(self, *args, **kwargs):
        if length_of(*args, **kwargs) == 1:
            return fast(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    dispatch._inkling2_fast = True  # type: ignore[attr-defined]
    dispatch._inkling2_decode_only = True  # type: ignore[attr-defined]
    dispatch.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(cls, name, dispatch)


def _install_gates(language) -> bool:
    """MessageController（前の周の情報を混ぜる門。1 トークンに 144 回）を 1 つにまとめる。"""
    import mlx.core as mx

    controller = getattr(language, "MessageController", None)
    if controller is None:
        return False

    @mx.compile
    def gate(x, message, down, both_w, eps):
        # language.MessageController.__call__ と同じ演算・同じ型。forget と gain の射影だけは
        # 重みを縦に並べた 1 回の積にする（出力の各行は同じ内積なので値は変わらない）。
        pair = mx.concatenate(
            [mx.fast.rms_norm(x, None, eps), mx.fast.rms_norm(message, None, eps)],
            axis=-1,
        )
        hidden = (pair.astype(down.dtype) @ down.T).astype(mx.float32)
        both = mx.tanh(hidden @ both_w.T)
        width = x.shape[-1]
        return (1 - both[..., :width]) * x + both[..., width:] * message

    def fast(self, x, message):
        both = self.__dict__.get("_inkling2_both")
        if both is None or both[0] is not self.forget_weight or both[1] is not self.gain_weight:
            joined = mx.concatenate([self.forget_weight, self.gain_weight], axis=0)
            mx.eval(joined)
            both = (self.forget_weight, self.gain_weight, joined)
            # nn.Module の属性にすると parameters() に混ざるので、__dict__ に直接置く
            object.__setattr__(self, "_inkling2_both", both)
        return gate(x, message, self.down_weight, both[2], float(self.eps))

    _decode_only(controller, "__call__", fast, lambda x, message: x.shape[-2])
    return True


def _install_attention(language) -> bool:
    """注意機構の「沈み込み（sink）」を、K/V に零の行を継ぎ足す代わりに SDPA の sinks で渡す。

    mlx-vlm は、問い合わせごとの sink を「零の K/V 1 行＋その行のロジット」で表すため、生成の
    1 トークンごとに K/V キャッシュ全体（スライド窓なら最大 1,024 行、全体注意なら履歴すべて）と
    マスクを複製していた。1 トークンに 48 回あるので、文脈が長いほど生成が遅くなる。
    mx.fast.scaled_dot_product_attention の sinks（頭ごとのロジットを分母に足す）は同じ計算で、
    結果も一致する（tests/test_inkling2_fast.py）。sinks は頭ごとの 1 次元なので、1 系列・1 トークンの
    ときだけ使い、それ以外（複数系列の同時生成・読み込み・分散）は mlx-vlm の実装に任せる。
    """
    import mlx.core as mx

    attention = getattr(language, "Attention", None)
    needed = ("l2_clip", "banded_additive_mask", "RotatingKVCache", "BatchRotatingKVCache")
    if attention is None or not all(hasattr(language, n) for n in needed):
        return False
    l2_clip = language.l2_clip
    banded_additive_mask = language.banded_additive_mask
    rotating, batch_rotating = language.RotatingKVCache, language.BatchRotatingKVCache
    original = attention.__call__
    if _wrapped_by(original, "_inkling2_attention"):
        return True

    from ._inkling2_step import _FixedKV

    def fast(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        b, length, _ = x.shape
        # step（丸ごとコンパイル）の代役キャッシュは、内側の step の実装に任せる
        if (
            b != 1
            or length != 1
            or self.sharding_group is not None
            or (cache is not None and isinstance(cache[0], _FixedKV))
        ):
            return original(self, x, key_input, value_input, cache, conv_mask)
        # ここから先は language.Attention.__call__ と同じ（sink の扱いだけが違う）。
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        k = self.k_proj(key_input)
        v = self.v_proj(value_input)
        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        q = l2_clip(
            self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim),
            cfg.qk_l2_clip_threshold,
            self.q_scale,
        )
        k = l2_clip(k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold)
        v = v.reshape(b, length, self.n_kv, self.head_dim)
        if cfg.use_value_norm_no_gamma:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(x.dtype)
        r = l2_clip(
            self.r_proj(x).reshape(b, length, self.n_heads, cfg.d_rel),
            cfg.rel_l2_clip_threshold,
            self.rel_scale,
        )
        if not self.is_sliding and cfg.log_scaling_n_floor is not None:
            offset = kv.offset if kv is not None else 0
            positions = mx.arange(length) + mx.array(offset)[..., None] + 1
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1)
            )
            q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
            r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            if isinstance(kv, rotating):
                k, v = kv._temporal_order(k), kv._temporal_order(v)
            elif isinstance(kv, batch_rotating) and kv.rotated:
                k, v = mx.roll(k, -kv._idx, 2), mx.roll(v, -kv._idx, 2)
        size = k.shape[2]
        mask = banded_additive_mask(
            r,
            self.rel_proj.astype(x.dtype),
            size - length,
            size,
            self.sliding,
            self.rel_extent,
            left_padding=getattr(kv, "left_padding", None),
            shape_ref=k,
        )
        # 1 系列・1 トークンなので、問い合わせごとの sink は頭ごとの値になる。
        sinks = self.sink_proj(x).reshape(self.n_heads)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask, sinks=sinks
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        out = (out.astype(mx.float32) * mx.sigmoid(self.gate_proj(x).astype(mx.float32))).astype(
            x.dtype
        )
        return self.o_proj(out)

    fast._inkling2_fast = True  # type: ignore[attr-defined]
    fast._inkling2_attention = True  # type: ignore[attr-defined]
    fast.__wrapped__ = original  # type: ignore[attr-defined]
    attention.__call__ = fast
    return True


def _install_prefill_bound(language) -> bool:
    """読み込み（17 トークン以上）では、層を 2 回通るごとに計算を確定させてメモリを抑える。

    生成を速くするためにコマンドバッファを大きくする（→ server.py の MLX_MAX_*_PER_BUFFER）と、
    途中の計算結果がバッファの完了まで解放されず、8K トークンの読み込み（2,048 トークンずつ）で
    最大使用量が 30.2 GB → 51.3 GB に増えた。確定させる間隔と、読み込みの速さ・最大使用量（実測）:
    毎回 580 tok/s・31.1 GB、2 回ごと 609 tok/s・33.8 GB、4 回ごと 627 tok/s・38.8 GB
    （差し替えなし 644 tok/s・30.2 GB）。2 回ごとを既定にする。計算そのものは同じで、確定させる
    時点が変わるだけ。間隔は LOCAL_LLM_INKLING2_PREFILL_EVAL_EVERY で変えられる。
    """
    import mlx.core as mx

    layer = getattr(language, "DecoderLayer", None)
    if layer is None:
        return False
    original = layer.__call__
    if getattr(original, "_inkling2_fast", False):
        return True

    every = int(os.environ.get("LOCAL_LLM_INKLING2_PREFILL_EVAL_EVERY", "2"))
    visits = [0]

    def bounded(self, x, *args, **kwargs):
        out = original(self, x, *args, **kwargs)
        # DSpark の検証（最大 8 トークン）は読み込みではないので触らない。
        if x.shape[1] > 16:
            visits[0] += 1
            if visits[0] % every == 0:
                # async_eval では確定が追いつかず、かえって 119 GB まで膨らんだ（実測）。待つ eval にする。
                mx.eval(out)
        return out

    bounded._inkling2_fast = True  # type: ignore[attr-defined]
    bounded.__wrapped__ = original  # type: ignore[attr-defined]
    layer.__call__ = bounded
    return True


def _wrapped_by(fn, marker: str) -> bool:
    """fn（差し替えが重なっていれば内側まで）に marker の差し替えが含まれるか。"""
    while fn is not None and getattr(fn, "_inkling2_fast", False):
        if getattr(fn, marker, False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def _install_head(language) -> bool:
    from . import _inkling2_step

    return _inkling2_step.install_head(language)


def _install_fused_residual(language) -> bool:
    from . import _inkling2_fused

    return _inkling2_fused.install_residual(language)


def _install_fused_router(language) -> bool:
    from . import _inkling2_fused

    return _inkling2_fused.install_router(language)


def _install_fused_gates(language) -> bool:
    from . import _inkling2_fused

    return _inkling2_fused.install_gates(language)


def _install_step(language) -> bool:
    from . import _inkling2_step

    return _inkling2_step.install_step(language)


_INSTALLERS = {
    "gates": _install_gates,
    "attention": _install_attention,
    "prefill_bound": _install_prefill_bound,
    "head": _install_head,
    "fused_residual": _install_fused_residual,
    "fused_router": _install_fused_router,
    "fused_gates": _install_fused_gates,
    "step": _install_step,
}


def install(names: Iterable[str] = ALL) -> list[str]:
    """指定した差し替えを当て、当てたものの名前を返す（mlx-vlm に Inkling 2 が無ければ空）。"""
    try:
        from mlx_vlm.models.inkling2 import language
    except Exception:
        return []
    wanted = list(ALL) if "all" in names else [n for n in names if n]
    applied = []
    for name in wanted:
        installer = _INSTALLERS.get(name)
        if installer is None:
            raise ValueError(f"unknown Inkling 2 fast path: {name!r} (known: {', '.join(ALL)})")
        if installer(language):
            applied.append(name)
    return applied


def uninstall() -> None:
    """当てた差し替えを外して mlx-vlm の実装に戻す（試験用）。"""
    try:
        from mlx_vlm.models.inkling2 import language
    except Exception:
        return
    targets = (
        (language, "_route"),
        (language.AttentionResidual, "__call__"),
        (language.MessageController, "__call__"),
        (language.Attention, "__call__"),
        (language.DecoderLayer, "__call__"),
        (language.LanguageModel, "__call__"),
        (language.LanguageModel, "_logits_from_norm"),
    )
    for cls, name in targets:
        current = vars(cls).get(name)
        while getattr(current, "_inkling2_fast", False):
            current = current.__wrapped__
        if current is not None:
            setattr(cls, name, current)

