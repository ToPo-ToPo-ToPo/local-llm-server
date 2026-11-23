"""_inkling2_fast（Inkling 2 Mini の生成を速くする mlx-vlm への差し替え）の試験。

差し替えは計算を変えないことが前提なので、小さな乱数の Inkling 2 を組み、差し替えの前後で
ロジットを比べる。20 GB の実モデルでの速さ・一致（貪欲生成のトークン列）は docs/mtp.md に記録した。
"""
from __future__ import annotations

import sys

import pytest

from local_llm_server import _inkling2_fast as fast
from local_llm_server import _mlx_vlm_shims


def test_is_inkling2():
    assert fast.is_inkling2("inkling2_loop") and fast.is_inkling2("inkling2")
    assert not fast.is_inkling2("inkling") and not fast.is_inkling2(None) and not fast.is_inkling2("")


def test_server_env_requests_every_fast_path():
    names = fast.SERVER_ENV["LOCAL_LLM_INKLING2_FAST"].split(",")
    assert set(names) == set(fast.ALL)
    assert fast.SERVER_ENV["MLX_MAX_OPS_PER_BUFFER"] == "300"
    assert fast.SERVER_ENV["MLX_MAX_MB_PER_BUFFER"] == "1000"


def test_install_without_mlx_vlm_does_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    assert fast.install(["gates"]) == []


def test_shim_installs_only_when_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(fast, "install", lambda names: calls.append(list(names)) or list(names))
    monkeypatch.delenv("LOCAL_LLM_INKLING2_FAST", raising=False)
    assert _mlx_vlm_shims._install_inkling2_fast() is None
    monkeypatch.setenv("LOCAL_LLM_INKLING2_FAST", "off")
    assert _mlx_vlm_shims._install_inkling2_fast() is None
    monkeypatch.setenv("LOCAL_LLM_INKLING2_FAST", "gates,attention")
    assert _mlx_vlm_shims._install_inkling2_fast() == "gates, attention"
    assert calls == [["gates", "attention"]]


def test_shim_survives_a_failing_install(monkeypatch):
    def boom(names):
        raise RuntimeError("upstream changed")

    monkeypatch.setattr(fast, "install", boom)
    monkeypatch.setenv("LOCAL_LLM_INKLING2_FAST", "gates")
    assert _mlx_vlm_shims._install_inkling2_fast().startswith("failed:")


# --- 小さな Inkling 2 で、差し替えの前後の一致を確かめる（Apple Silicon のみ） ---


@pytest.fixture
def tiny():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal が無い")
    pytest.importorskip("mlx_vlm.models.inkling2")
    import mlx.utils as mu
    from mlx_vlm.models.inkling2 import language
    from mlx_vlm.models.inkling2.config import TextConfig

    # 実物と同じ形（スライド窓と全体注意の混在・中段を 3 周）を小さくしたもの。
    # 読み込み 40 トークンはスライド窓（16）より長く、循環するキャッシュも通る。
    cfg = TextConfig.from_dict(dict(
        model_type="inkling2_loop_text", hidden_size=128, num_hidden_layers=6, vocab_size=256,
        unpadded_vocab_size=250, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        swa_head_dim=32, swa_num_attention_heads=4, swa_num_key_value_heads=2, d_rel=8,
        rel_extent=64, sliding_window_size=16, local_layer_ids=[0, 1, 2, 4], n_routed_experts=8,
        num_experts_per_tok=2, intermediate_size=64, n_layers_per_attn_res_block=2,
        loop_config=dict(core_start=2, core_end=4, num_passes=3, gate_rank=16),
    ))
    mx.random.seed(0)
    model = language.LanguageModel(cfg)
    model.update(mu.tree_map(
        lambda p: (mx.random.normal(p.shape) * 0.1).astype(mx.bfloat16), model.parameters()))
    model.eval()  # 実モデルと同じく推論の状態にする（学習中は差し替えも mlx-vlm の速い経路も使われない）
    fast.uninstall()
    yield model
    fast.uninstall()


def _logits(model, batch=1, steps=24, tail=False):
    import mlx.core as mx
    import numpy as np

    cache = model.make_cache()
    prompt = mx.array([list(range(3, 43))] * batch)
    rows = [model(prompt, cache=cache).logits[:, -1]]
    for t in range(steps):
        rows.append(model(mx.array([[(t * 7) % 250]] * batch), cache=cache).logits[:, -1])
    if tail:
        # 生成のあとに通常の経路（複数トークン）で続ける。キャッシュの書き戻しが正しいかを見る
        rows.extend(model(mx.array([list(range(60, 68))] * batch), cache=cache).logits.swapaxes(0, 1))
    out = mx.stack(rows).astype(mx.float32)
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("name", ["gates", "prefill_bound"])
def test_exact_fast_paths_keep_logits_bit_identical(tiny, name):
    import numpy as np

    before = _logits(tiny)
    assert fast.install([name]) == [name]
    assert np.array_equal(before, _logits(tiny))


def test_attention_sinks_match_the_zero_row_formulation(tiny):
    """sink を SDPA の sinks で渡しても、零の K/V を継ぎ足す元の計算と一致する。"""
    import numpy as np

    before = _logits(tiny)
    assert fast.install(["attention"]) == ["attention"]
    after = _logits(tiny)
    # 実モデルでは鍵の本数が 1 本減ることで和の順序が変わり、ごく僅かにずれうる（docs/mtp.md）。
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-2)


def test_attention_falls_back_for_batched_decode(tiny):
    """複数系列の同時生成は sinks（頭ごとの 1 次元）で表せないので、元の実装に任せる。"""
    import numpy as np

    before = _logits(tiny, batch=2)
    fast.install(["attention"])
    assert np.array_equal(before, _logits(tiny, batch=2))


def test_head_matches_the_fp32_copy(tiny):
    """出力層を bf16 のまま読んでも、fp32 の複製を読む元の計算と和の順番の差しか出ない。"""
    import numpy as np

    before = _logits(tiny)
    assert fast.install(["head"]) == ["head"]
    np.testing.assert_allclose(_logits(tiny), before, rtol=0, atol=1e-4)


def test_step_matches_and_keeps_the_cache_usable(tiny):
    """丸ごとコンパイルしたステップは元の計算とほぼ一致し（読み込み 40 トークンは窓 16 より長く、
    24 ステップで窓が何周も回る）、そのあとの通常の経路でも同じキャッシュをそのまま使える。"""
    import numpy as np

    before = _logits(tiny, tail=True)
    assert fast.install(["step"]) == ["step"]
    after = _logits(tiny, tail=True)
    assert tiny.__dict__["_inkling2_engine"].fns  # 通常の経路に逃げず、コンパイルした経路を通った
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-2)
    assert np.array_equal(after.argmax(-1), before.argmax(-1))


@pytest.mark.parametrize(
    ("name", "cache"),
    [("fused_residual", "_residual_kernel"), ("fused_router", "_router_kernels"), ("fused_gates", "_gate_kernels")],
)
def test_fused_kernels_match_and_are_used(tiny, name, cache):
    """専用の計算は元の計算と和の順番の差しか出ず、選ばれるトークンも同じ。実際に専用の計算を通る。"""
    import numpy as np

    from local_llm_server import _inkling2_fused

    getattr(_inkling2_fused, cache).cache_clear()
    before = _logits(tiny, tail=True)
    assert fast.install([name]) == [name]
    after = _logits(tiny, tail=True)
    assert getattr(_inkling2_fused, cache).cache_info().currsize > 0
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-2)
    assert np.array_equal(after.argmax(-1), before.argmax(-1))


def test_fused_router_picks_the_same_experts(tiny):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.models.inkling2 import language

    original = language._route
    fast.install(["fused_router"])
    mx.random.seed(1)
    for _ in range(20):
        x = mx.random.normal((1, 1, 2304)).astype(mx.bfloat16)
        w = (mx.random.normal((128, 2304)) * 0.05).astype(mx.bfloat16)
        bias = mx.random.normal((128,)) * 0.01
        ri, rw = original(x, w, bias, 8, 2.5)
        fi, fw = language._route(x, w, bias, 8, 2.5)
        ri, rw, fi, fw = (np.array(a).reshape(-1) for a in (ri, rw, fi, fw))
        assert set(ri.tolist()) == set(fi.tolist())
        np.testing.assert_allclose(fw[np.argsort(fi)], rw[np.argsort(ri)], rtol=1e-5, atol=1e-6)


def test_step_with_every_fast_path(tiny):
    import numpy as np

    before = _logits(tiny, tail=True)
    assert fast.install(fast.ALL) == list(fast.ALL)
    np.testing.assert_allclose(_logits(tiny, tail=True), before, rtol=0, atol=1e-2)


def test_step_falls_back_for_batched_decode(tiny):
    import numpy as np

    before = _logits(tiny, batch=2)
    fast.install(["step", "head"])
    assert np.array_equal(before, _logits(tiny, batch=2))


def test_prefill_bound_only_evaluates_long_inputs(tiny, monkeypatch):
    import mlx.core as mx

    fast.install(["prefill_bound"])
    calls = []
    real_eval = mx.eval
    monkeypatch.setattr(mx, "eval", lambda *a: calls.append(1) or real_eval(*a))
    cache = tiny.make_cache()
    tiny(mx.array([list(range(3, 43))]), cache=cache)   # 40 トークン: 読み込み
    long_calls = len(calls)
    calls.clear()
    tiny(mx.array([list(range(3, 11))]), cache=cache)   # 8 トークン: DSpark の検証と同じ長さ
    tiny(mx.array([[5]]), cache=cache)                   # 1 トークン: 生成
    assert long_calls > 0 and calls == []


def test_install_is_idempotent_and_uninstall_restores(tiny):
    from mlx_vlm.models.inkling2 import language

    targets = [
        (language, "_route"),
        (language.AttentionResidual, "__call__"),
        (language.MessageController, "__call__"),
        (language.Attention, "__call__"),
        (language.LanguageModel, "__call__"),
        (language.LanguageModel, "_logits_from_norm"),
    ]
    originals = [vars(cls)[name] for cls, name in targets]
    fast.install(fast.ALL)
    patched = [vars(cls)[name] for cls, name in targets]
    fast.install(fast.ALL)
    assert [vars(cls)[name] for cls, name in targets] == patched
    fast.uninstall()
    assert [vars(cls)[name] for cls, name in targets] == originals


def test_unknown_fast_path_is_rejected(tiny):
    with pytest.raises(ValueError, match="unknown"):
        fast.install(["nope"])
