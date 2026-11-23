"""Inkling 2 Mini の生成 1 ステップを速くする差し替えのうち、_inkling2_fast より大きいもの。

- ``head``: 出力層（語彙 201,024 × 2,304）を bf16 のまま読み、fp32 で積和する専用の行列×ベクトル。
  mlx-vlm は精度のために fp32 に広げた複製（1.85 GB）を毎ステップ読んでいる。bf16 → fp32 の変換は
  正確なので、違うのは和の順番だけ（実モデルで |差| ≤ 2e-6、選ばれるトークンは同じ）。
  M2 Ultra で 2.63 → 1.39 ms/ステップ。
- ``step``: 生成 1 ステップ（1 系列・1 トークン）を丸ごと ``mx.compile`` する。Python で計算の
  グラフを組む時間（約 7 ms/ステップ）と、細かい要素ごとの演算の起動が減る。CPU が他の仕事で
  混んでいるときほど効く（M2 Ultra の実測: 空いているとき +4%、他の計算が 5 コア使っているとき +17%）。

``step`` は mlx-vlm のキャッシュ（KVCache / RotatingKVCache + ArraysCache）をそのまま使い続け、
各ステップでその配列を入力に渡し、更新後の配列を書き戻す（続きの会話の再利用や、読み込み・
DSpark の検証が使う通常の経路と食い違わない）。形を変えないため、全体注意の KV は容量を 2 の冪
（最小 4,096）まで、スライド窓の KV は窓の大きさまで先に広げ、書く位置と有効な長さは配列で渡す。
注意のマスクは mlx-vlm の _mask_v2_kernel と同じ式で、その配列から作る。

計算は mlx-vlm とほぼ同じだが、ビット一致はしない: 注意の SDPA に渡す鍵の本数（容量まで）と、
スライド窓の鍵の並び（循環したまま）が違い、和の順番が変わる（それ以外の部分は一致を確かめた）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

_MIN_CAPACITY = 4096

_HEAD_SRC = r"""
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int row0 = (threadgroup_position_in_grid.y * SG + sg) * R;
    if (row0 >= N) return;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int k = lane * 8; k < K; k += 256) {
        float4 xa = *(const device float4*)(x + k);
        float4 xb = *(const device float4*)(x + k + 4);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            int row = min(row0 + r, N - 1);
            const device bfloat4* wp = (const device bfloat4*)(w + (size_t)row * K + k);
            acc[r] += dot(xa, float4(wp[0])) + dot(xb, float4(wp[1]));
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float t = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < N) out[row0 + r] = t;
    }
"""

# 書く枠 s・有効な長さ valid・枠の数 M（meta）から、各枠が「何トークン前」かを求めてマスクを作る。
# 帯の中の相対位置の値は mlx-vlm の _MASK_V2_SRC と同じ式・同じ順番で計算する。
_MASK_SRC = r"""
    const int H = rel_shape[2];
    const int S = kshape_shape[2];
    uint j = thread_position_in_grid.x;
    uint h = thread_position_in_grid.z;
    if ((int)j >= S || (int)h >= H) return;
    int s = meta[0], valid = meta[1], M = meta[2];
    int back = (s - (int)j) % M;
    if (back < 0) back += M;
    T val;
    if (back >= valid) {
        val = (T)(-1e30f);
    } else if (SLIDING > 0 && back >= (int)SLIDING) {
        val = (T)(-1e30f);
    } else if (back < (int)REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)h * rel_strides[2];
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0] + (size_t)back * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[(size_t)h * S + j] = val;
"""


@lru_cache(maxsize=1)
def _head_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="local_llm_inkling2_head_bf16", input_names=["x", "w"], output_names=["out"],
        source=_HEAD_SRC,
    )


@lru_cache(maxsize=1)
def _mask_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="local_llm_inkling2_slot_mask", input_names=["rel", "proj", "kshape", "meta"],
        output_names=["out"], source=_MASK_SRC, ensure_row_contiguous=False,
    )


def head_matvec(x, weight):
    """x: (K,) fp32、weight: (N, K) bf16 → (N,) fp32。K は 8 の倍数。"""
    import mlx.core as mx

    n, k = weight.shape
    simdgroups, rows = 4, 4
    return _head_kernel()(
        inputs=[x, weight],
        template=[("K", k), ("N", n), ("SG", simdgroups), ("R", rows)],
        grid=(32, ((n + rows * simdgroups - 1) // (rows * simdgroups)) * simdgroups, 1),
        threadgroup=(32, simdgroups, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]


def install_head(language) -> bool:
    import mlx.core as mx
    from mlx import nn

    model_cls: Any = getattr(language, "LanguageModel", None)
    original = getattr(model_cls, "_logits_from_norm", None)
    if original is None:
        return False
    if getattr(original, "_inkling2_head", False):
        return True

    def logits_from_norm(self, h):
        head = self.lm_head
        weight = head.weight if type(head) is nn.Linear and "bias" not in head else None
        if (
            weight is None
            or self.training
            or self.sharding_group is not None
            or h.ndim != 3
            or h.shape[0] * h.shape[1] != 1
            or weight.dtype != mx.bfloat16
            or weight.shape[1] % 8
            or mx.default_device() != mx.gpu
        ):
            return original(self, h)
        # mlx-vlm と同じく、埋め込みの正規化の型を経てから fp32 にする
        h = h.astype(self.model.embed_norm.weight.dtype).astype(mx.float32)
        logits = head_matvec(h.reshape(-1), weight).reshape(*h.shape[:-1], weight.shape[0])
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    logits_from_norm._inkling2_fast = True  # type: ignore[attr-defined]
    logits_from_norm._inkling2_head = True  # type: ignore[attr-defined]
    logits_from_norm.__wrapped__ = original  # type: ignore[attr-defined]
    model_cls._logits_from_norm = logits_from_norm
    return True


# --- step: 1 ステップを丸ごとコンパイルする ---


class _FixedKV:
    """トレース中だけ使う KV キャッシュの代役。meta = [書く枠, 有効な長さ, 枠の数]。"""

    def __init__(self, keys, values, meta, position):
        self.keys, self.values, self.meta = keys, values, meta
        self.offset = position  # 書く前の位置（全体注意の対数スケーリングに使う）

    def write(self, k, v):
        import mlx.core as mx

        start = self.meta[:1]
        self.keys = mx.slice_update(self.keys, k, start, axes=(2,))
        self.values = mx.slice_update(self.values, v, start, axes=(2,))
        return self.keys, self.values


class _FixedConv:
    """畳み込みの状態（ArraysCache）の代役。詰め物も長さの指定もない 1 系列だけを扱う。"""

    left_padding = None
    lengths = None
    is_speculating = False

    def __init__(self, arrays):
        self.cache = list(arrays)

    def __getitem__(self, i):
        return self.cache[i]

    def __setitem__(self, i, v):
        self.cache[i] = v

    def advance(self, n):
        pass


def _attention_fixed(language, self, x, key_input, value_input, kv, conv, conv_mask):
    """_inkling2_fast の生成時の注意と同じ計算。KV の書き込みとマスクだけを固定の枠で行う。"""
    import mlx.core as mx

    b, length, _ = x.shape
    cfg = self.config
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
    q = language.l2_clip(
        self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim),
        cfg.qk_l2_clip_threshold,
        self.q_scale,
    )
    k = language.l2_clip(k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold)
    v = v.reshape(b, length, self.n_kv, self.head_dim)
    if cfg.use_value_norm_no_gamma:
        v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(x.dtype)
    r = language.l2_clip(
        self.r_proj(x).reshape(b, length, self.n_heads, cfg.d_rel),
        cfg.rel_l2_clip_threshold,
        self.rel_scale,
    )
    if not self.is_sliding and cfg.log_scaling_n_floor is not None:
        positions = mx.arange(length) + kv.offset[..., None] + 1
        tau = 1 + cfg.log_scaling_alpha * mx.log(mx.maximum(positions / cfg.log_scaling_n_floor, 1))
        q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
    q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
    k, v = kv.write(k, v)
    _, _, heads, d_rel = r.shape
    size = k.shape[2]
    mask = _mask_kernel()(
        inputs=[r, self.rel_proj.astype(x.dtype), k, kv.meta],
        template=[("T", r.dtype), ("D_REL", d_rel), ("REL_EXTENT", int(self.rel_extent)),
                  ("SLIDING", int(self.sliding))],
        grid=(((size + 31) // 32) * 32, 1, heads),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, heads, 1, size)],
        output_dtypes=[r.dtype],
    )[0]
    sinks = self.sink_proj(x).reshape(self.n_heads)
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=self.head_dim**-0.5, mask=mask, sinks=sinks
    )
    out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
    out = (out.astype(mx.float32) * mx.sigmoid(self.gate_proj(x).astype(mx.float32))).astype(x.dtype)
    return self.o_proj(out)


def _capacity(n: int) -> int:
    cap = _MIN_CAPACITY
    while cap < n:
        cap *= 2
    return cap


def _grow(a, size):
    import mlx.core as mx

    if a.shape[2] >= size:
        return a
    pad = mx.zeros((*a.shape[:2], size - a.shape[2], a.shape[3]), a.dtype)
    return mx.concatenate([a, pad], axis=2)


class _Engine:
    """1 つの LanguageModel に付く、コンパイル済みの生成 1 ステップ（形ごとに 1 つ）。"""

    def __init__(self, lm, cache_types):
        self.lm = lm
        self.kv_full, self.kv_rotating, self.conv_type = cache_types
        self.fns = {}

    def eligible(self, inputs, inputs_embeds, cache, kw) -> bool:
        if inputs is None or inputs_embeds is not None or not cache or tuple(inputs.shape) != (1, 1):
            return False
        if any(kw.get(k) for k in ("return_hidden", "return_shared_kv", "skip_logits", "speculative_verify")):
            return False
        if kw.get("capture_layer_ids") is not None or self.lm.sharding_group is not None or self.lm.training:
            return False
        offset = None
        for entry in cache:
            kv, conv = entry[0], entry[1]
            if (
                type(conv) is not self.conv_type
                or conv.left_padding is not None
                or conv.lengths is not None
                or conv.is_speculating
                or any(a is None for a in conv.cache)
            ):
                return False
            if type(kv) is self.kv_rotating:
                if kv.keep != 0:
                    return False
            elif type(kv) is not self.kv_full:
                return False
            if kv.keys is None or (offset is not None and kv.offset != offset):
                return False
            offset = kv.offset
        return True

    def _prepare(self, cache):
        """形をそろえる。戻り値は (全体注意の容量, 位置, スライド窓の書く枠)。そろえられなければ None。"""
        pos = cache[0][0].offset
        need = _capacity(pos + 1)
        cap, slot = 0, None
        for entry in cache:
            kv = entry[0]
            if type(kv) is self.kv_full:
                cap = max(cap, need, kv.keys.shape[2])
                continue
            w = kv.max_size
            if kv.keys.shape[2] > w:
                return None  # 読み込み直後で窓より長い。1 回だけ通常の経路に任せると窓の形になる
            if kv.keys.shape[2] < w:
                if kv._idx != kv.offset or kv.offset >= w:
                    return None
                # まだ育ちきっていない窓を最終の大きさに先に広げる（RotatingKVCache が育つのと同じ形）
                kv.keys, kv.values = _grow(kv.keys, w), _grow(kv.values, w)
            s = kv.keep if kv._idx == w else kv._idx
            if slot is not None and s != slot:
                return None
            slot = s
        for entry in cache:
            kv = entry[0]
            if type(kv) is self.kv_full and kv.keys.shape[2] != cap:
                kv.keys, kv.values = _grow(kv.keys, cap), _grow(kv.values, cap)
        return cap, pos, slot

    def _build(self, kinds):
        import mlx.core as mx

        lm = self.lm

        def step(inputs, full_meta, window_meta, position, *state):
            it = iter(state)
            standins = []
            for full, nconv in kinds:
                keys, values = next(it), next(it)
                kv = _FixedKV(keys, values, full_meta if full else window_meta, position)
                standins.append((kv, _FixedConv([next(it) for _ in range(nconv)])))
            h = lm.model(inputs, cache=standins, skip_final_norm=True)
            outs = [lm._logits_from_norm(lm.model.norm(h))]
            for kv, conv in standins:
                outs += [kv.keys, kv.values, *conv.cache]
            return outs

        return mx.compile(step)

    def __call__(self, inputs, cache):
        import mlx.core as mx

        prepared = self._prepare(cache)
        if prepared is None:
            return None
        cap, pos, slot = prepared
        kinds = tuple((type(e[0]) is self.kv_full, len(e[1].cache)) for e in cache)
        window = next((e[0].max_size for e in cache if type(e[0]) is self.kv_rotating), 1)
        key = (kinds, cap, window)
        fn = self.fns.get(key)
        if fn is None:
            fn = self.fns[key] = self._build(kinds)
        full_meta = mx.array([pos, pos + 1, max(cap, 1)], mx.int32)
        window_meta = mx.array([slot or 0, min(pos + 1, window), window], mx.int32)
        state = []
        for entry in cache:
            state += [entry[0].keys, entry[0].values, *entry[1].cache]
        outs = fn(inputs, full_meta, window_meta, mx.array(pos, mx.int32), *state)
        logits, rest, i = outs[0], outs[1:], 0
        for entry in cache:
            kv, conv = entry[0], entry[1]
            n = len(conv.cache)
            kv.keys, kv.values = rest[i], rest[i + 1]
            conv.cache = list(rest[i + 2 : i + 2 + n])
            i += 2 + n
            if type(kv) is self.kv_rotating:
                kv._idx = (kv.keep if kv._idx == kv.max_size else kv._idx) + 1
            kv.offset += 1
        return logits


def install_step(language) -> bool:
    model_cls = getattr(language, "LanguageModel", None)
    attention = getattr(language, "Attention", None)
    needed = ("l2_clip", "KVCache", "RotatingKVCache", "ArraysCache")
    if model_cls is None or attention is None or not all(hasattr(language, n) for n in needed):
        return False
    try:
        from mlx_vlm.models.base import LanguageModelOutput
    except Exception:
        return False
    original_call = model_cls.__call__
    if getattr(original_call, "_inkling2_step", False):
        return True
    original_attention = attention.__call__
    cache_types = (language.KVCache, language.RotatingKVCache, language.ArraysCache)

    def attention_call(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        if cache is not None and isinstance(cache[0], _FixedKV):
            return _attention_fixed(language, self, x, key_input, value_input, cache[0], cache[1], conv_mask)
        return original_attention(self, x, key_input, value_input, cache, conv_mask)

    def call(self, inputs=None, inputs_embeds=None, cache=None, **kwargs):
        engine = self.__dict__.get("_inkling2_engine")
        if engine is None:
            engine = _Engine(self, cache_types)
            object.__setattr__(self, "_inkling2_engine", engine)  # parameters() に混ぜない
        if engine.eligible(inputs, inputs_embeds, cache, kwargs):
            logits = engine(inputs, cache)
            if logits is not None:
                return LanguageModelOutput(logits=logits)
        return original_call(self, inputs=inputs, inputs_embeds=inputs_embeds, cache=cache, **kwargs)

    wrappers: list[tuple[Any, Any]] = [(attention_call, original_attention), (call, original_call)]
    for fn, orig in wrappers:
        fn._inkling2_fast = True
        fn._inkling2_step = True
        fn.__wrapped__ = orig
    attention.__call__ = attention_call
    model_cls.__call__ = call
    return True
