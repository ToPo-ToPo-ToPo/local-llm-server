"""Inkling 2 Mini の生成（1 系列・1 トークン）で、待ちの連なりを短くする専用の計算。

M2 Ultra では、1 トークンの計算は「前の結果を待ってから次を始める」小さな処理が約 1,900 段連なり、
その 1 段ごとの待ちが積み上がって時間を決めている（重みを読む時間は 1 トークン 10 ms 程度）。
ここでは、連なりの中の数段を 1〜2 回の起動にまとめる（Strata の Metal 版と同じ考え方）。

- ``fused_residual``: 層の入口で前の段を混ぜる AttentionResidual（stack → rms_norm → スコア →
  softmax → 重み付き和、1 トークン 97 回）を 1 回の起動で。
- ``fused_router``: 専門家の振り分け（スコア → sigmoid → 上位 8 → 重みの正規化、48 回）を 2 回の起動で。
- ``fused_gates``: 前の周の情報を混ぜる門（rms_norm 2 つ → 64 次元へ → 戻す → tanh → 混ぜる、96 回）を
  2 回の起動で。

値の丸め（bf16 にする箇所）と式は mlx-vlm と同じにし、違うのは和の順番だけ（単体で相対差 1e-7 程度、
振り分けの選ぶ専門家は同じ）。ただし 48 回層を通るうちに増えるので、ロジットはビット一致しない。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

_THREADS = 256


def _residual_source(count: int) -> str:
    loads = "\n".join(f"        v[{i}] = float(s{i}[k]);" for i in range(count))
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint j = threadgroup_position_in_grid.z;
    constexpr int NB = {count};
    constexpr int NSG = {_THREADS // 32};
    threadgroup float part[NB][NSG];
    threadgroup float inv[NB];
    threadgroup float wts[NB];
    float acc[NB];
    float v[NB];
    // 1) 各行の rms（mlx の rms_norm と同じく 1 / sqrt(平均二乗 + 1e-6)）
    for (int i = 0; i < NB; ++i) acc[i] = 0.0f;
    for (int k = tid; k < H; k += {_THREADS}) {{
{loads}
        for (int i = 0; i < NB; ++i) acc[i] += v[i] * v[i];
    }}
    for (int i = 0; i < NB; ++i) {{ float t = simd_sum(acc[i]); if (lane == 0) part[i][sg] = t; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NB) {{
        float t = 0.0f;
        for (int g = 0; g < NSG; ++g) t += part[tid][g];
        inv[tid] = metal::precise::rsqrt(t / float(H) + 1e-6f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2) 正規化した各行と問い合わせの内積 → 3) softmax
    for (int i = 0; i < NB; ++i) acc[i] = 0.0f;
    const device auto* qj = q + (size_t)j * H;
    for (int k = tid; k < H; k += {_THREADS}) {{
{loads}
        float qk = float(qj[k]);
        for (int i = 0; i < NB; ++i) acc[i] += (v[i] * inv[i]) * qk;
    }}
    for (int i = 0; i < NB; ++i) {{ float t = simd_sum(acc[i]); if (lane == 0) part[i][sg] = t; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float sc[NB];
        float m = -INFINITY;
        for (int i = 0; i < NB; ++i) {{
            float t = 0.0f;
            for (int g = 0; g < NSG; ++g) t += part[i][g];
            sc[i] = t * metal::precise::rsqrt(float(H));
            m = max(m, sc[i]);
        }}
        float z = 0.0f;
        for (int i = 0; i < NB; ++i) {{ sc[i] = metal::precise::exp(sc[i] - m); z += sc[i]; }}
        for (int i = 0; i < NB; ++i) wts[i] = sc[i] / z;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4) 重み付き和（normalize_values なら正規化した行の和）
    for (int k = tid; k < H; k += {_THREADS}) {{
{loads}
        float o = 0.0f;
        for (int i = 0; i < NB; ++i) o += wts[i] * (NORMV ? v[i] * inv[i] : v[i]);
        out[(size_t)j * H + k] = o;
    }}
"""


@lru_cache(maxsize=None)
def _residual_kernel(count: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name=f"local_llm_inkling2_residual_{count}",
        input_names=[f"s{i}" for i in range(count)] + ["q"],
        output_names=["out"],
        source=_residual_source(count),
    )


_GATE_DOWN_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    constexpr int NSG = 8;
    threadgroup float part[2][NSG];
    threadgroup float inv[2];
    threadgroup float rs[NSG];
    float a0 = 0.0f, a1 = 0.0f;
    for (int k = tid; k < W; k += 256) { float u = float(x[k]); float v = float(m[k]); a0 += u * u; a1 += v * v; }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0) { part[0][sg] = a0; part[1][sg] = a1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) {
        float t = 0.0f;
        for (int g = 0; g < NSG; ++g) t += part[tid][g];
        inv[tid] = metal::precise::rsqrt(t / float(W) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int row = threadgroup_position_in_grid.x;
    const device bfloat* wr = down + (size_t)row * 2 * W;
    float acc = 0.0f;
    // mlx-vlm と同じく、正規化した 2 本を bf16 に丸めてから bf16 の重みと掛ける
    for (int k = tid; k < W; k += 256) {
        acc += float(bfloat(float(x[k]) * inv[0])) * float(wr[k]);
        acc += float(bfloat(float(m[k]) * inv[1])) * float(wr[W + k]);
    }
    acc = simd_sum(acc);
    if (lane == 0) rs[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float t = 0.0f; for (int g = 0; g < NSG; ++g) t += rs[g]; hidden[row] = float(bfloat(t)); }
"""

_GATE_UP_SRC = """
    uint o = thread_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float hs[R];
    for (uint r = tid; r < (uint)R; r += 256) hs[r] = hidden[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (o >= W) return;
    const device bfloat4* fw = (const device bfloat4*)(both + (size_t)o * R);
    const device bfloat4* gw = (const device bfloat4*)(both + (size_t)(W + o) * R);
    float f = 0.0f, g = 0.0f;
    #pragma unroll
    for (int r = 0; r < R / 4; ++r) {
        float4 h = float4(hs[4 * r], hs[4 * r + 1], hs[4 * r + 2], hs[4 * r + 3]);
        f += dot(h, float4(fw[r]));
        g += dot(h, float4(gw[r]));
    }
    f = metal::precise::tanh(f);
    g = metal::precise::tanh(g);
    out[o] = (1.0f - f) * float(x[o]) + g * float(m[o]);
"""


@lru_cache(maxsize=1)
def _gate_kernels():
    import mlx.core as mx

    down = mx.fast.metal_kernel(
        name="local_llm_inkling2_gate_down", input_names=["x", "m", "down", "eps"],
        output_names=["hidden"], source=_GATE_DOWN_SRC,
    )
    up = mx.fast.metal_kernel(
        name="local_llm_inkling2_gate_up", input_names=["x", "m", "hidden", "both"],
        output_names=["out"], source=_GATE_UP_SRC,
    )
    return down, up


_ROUTER_SCORE_SRC = """
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    int e = threadgroup_position_in_grid.x * 4 + sg;
    if (e >= E) return;
    const device auto* wr = w + (size_t)e * H;
    float acc = 0.0f;
    for (int k = lane * 8; k < H; k += 256) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc += float(x[k + i]) * float(wr[k + i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) sc[e] = 1.0f / (1.0f + metal::precise::exp(-acc));
"""

# 上位 K 個: 1 スレッドが 1 専門家を持ち、simd_max で K 回選ぶ（同じ値なら番号の小さいほう）
_ROUTER_TOPK_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float gmax[32];
    threadgroup int gidx[32];
    threadgroup int picks[K];
    float mine = (tid < (uint)E) ? sc[tid] + float(bias[tid]) : -INFINITY;
    for (int t = 0; t < K; ++t) {
        float m = simd_max(mine);
        int cand = (mine == m) ? (int)tid : 0x7fffffff;
        cand = simd_min(cand);
        if (lane == 0) { gmax[sg] = m; gidx[sg] = cand; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float bm = -INFINITY; int bi = 0x7fffffff;
            for (int g = 0; g < (E + 31) / 32; ++g)
                if (gmax[g] > bm || (gmax[g] == bm && gidx[g] < bi)) { bm = gmax[g]; bi = gidx[g]; }
            picks[t] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)tid == picks[t]) mine = -INFINITY;
    }
    if (tid == 0) {
        float total = 0.0f;
        for (int t = 0; t < K; ++t) total += sc[picks[t]];
        for (int t = 0; t < K; ++t) { idx[t] = uint(picks[t]); wts[t] = sc[picks[t]] / (total + 1e-20f) * scale[0]; }
    }
"""


@lru_cache(maxsize=1)
def _router_kernels():
    import mlx.core as mx

    score = mx.fast.metal_kernel(
        name="local_llm_inkling2_router_score", input_names=["x", "w"], output_names=["sc"],
        source=_ROUTER_SCORE_SRC,
    )
    topk = mx.fast.metal_kernel(
        name="local_llm_inkling2_router_topk", input_names=["sc", "bias", "scale"],
        output_names=["idx", "wts"], source=_ROUTER_TOPK_SRC,
    )
    return score, topk


def _single(x) -> bool:
    import mlx.core as mx

    return x.ndim == 3 and x.shape[0] * x.shape[1] == 1 and mx.default_device() == mx.gpu


def _mark(fn, original, marker: str):
    fn._inkling2_fast = True
    setattr(fn, marker, True)
    fn.__wrapped__ = original
    return fn


def install_residual(language) -> bool:
    import mlx.core as mx

    cls: Any = getattr(language, "AttentionResidual", None)
    if cls is None:
        return False
    original = cls.__call__
    if getattr(original, "_inkling2_fused_residual", False):
        return True

    def call(self, blocks, partial):
        states = [*blocks, partial]
        if not (_single(partial) and len(states) <= 16 and all(s.shape == partial.shape for s in states)):
            return original(self, blocks, partial)
        h = partial.shape[-1]
        queries = 1 if self.query.ndim == 1 else self.query.shape[0]
        return _residual_kernel(len(states))(
            inputs=[*states, self.query],
            template=[("H", h), ("NORMV", bool(self.normalize_values))],
            grid=(_THREADS, 1, queries),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(*partial.shape[:-1], queries, h)],
            output_dtypes=[mx.float32],
        )[0]

    cls.__call__ = _mark(call, original, "_inkling2_fused_residual")
    return True


def install_gates(language) -> bool:
    import mlx.core as mx

    cls: Any = getattr(language, "MessageController", None)
    if cls is None:
        return False
    original = cls.__call__
    if _has_marker(original, "_inkling2_fused_gates"):
        return True

    def call(self, x, message):
        down = self.down_weight
        if not (
            _single(x)
            and message.shape == x.shape
            and x.dtype == mx.float32
            and message.dtype == mx.float32
            and down.dtype == mx.bfloat16
            and self.forget_weight.dtype == mx.bfloat16
            and self.gain_weight.dtype == mx.bfloat16
            and down.shape[0] % 4 == 0
        ):
            return original(self, x, message)
        cached = self.__dict__.get("_inkling2_fused_both")
        if cached is None or cached[0] is not self.forget_weight or cached[1] is not self.gain_weight:
            joined = mx.concatenate([self.forget_weight, self.gain_weight], axis=0)
            eps = mx.array([self.eps], mx.float32)
            mx.eval(joined, eps)
            cached = (self.forget_weight, self.gain_weight, joined, eps)
            object.__setattr__(self, "_inkling2_fused_both", cached)  # parameters() に混ぜない
        width, rank = x.shape[-1], down.shape[0]
        k_down, k_up = _gate_kernels()
        hidden = k_down(
            inputs=[x.reshape(-1), message.reshape(-1), down, cached[3]],
            template=[("W", width), ("R", rank)],
            grid=(rank * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(rank,)], output_dtypes=[mx.float32],
        )[0]
        out = k_up(
            inputs=[x.reshape(-1), message.reshape(-1), hidden, cached[2]],
            template=[("W", width), ("R", rank)],
            grid=(((width + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(width,)], output_dtypes=[mx.float32],
        )[0]
        return out.reshape(x.shape)

    cls.__call__ = _mark(call, original, "_inkling2_fused_gates")
    return True


def install_router(language) -> bool:
    import mlx.core as mx

    original = getattr(language, "_route", None)
    if original is None:
        return False
    if getattr(original, "_inkling2_fused_router", False):
        return True

    def route(x, weight, bias, top_k, route_scale):
        experts, width = weight.shape
        k = int(top_k)
        if not (_single(x) and experts <= 1024 and width % 8 == 0 and 0 < k <= experts):
            return original(x, weight, bias, top_k, route_scale)
        k_score, k_topk = _router_kernels()
        scores = k_score(
            inputs=[x.reshape(-1), weight], template=[("E", experts), ("H", width)],
            grid=(((experts + 3) // 4) * 128, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(experts,)], output_dtypes=[mx.float32],
        )[0]
        threads = ((experts + 31) // 32) * 32
        idx, wts = k_topk(
            inputs=[scores, bias, mx.array([route_scale], mx.float32)],
            template=[("E", experts), ("K", k)],
            grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(k,), (k,)], output_dtypes=[mx.uint32, mx.float32],
        )
        shape = (*x.shape[:-1], k)
        return idx.reshape(shape), wts.reshape(shape)

    language._route = _mark(route, original, "_inkling2_fused_router")
    return True


def _has_marker(fn, marker: str) -> bool:
    while fn is not None and getattr(fn, "_inkling2_fast", False):
        if getattr(fn, marker, False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False
