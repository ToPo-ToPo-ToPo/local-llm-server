"""変換→ロード→順伝播が、同じ量子化を施した transformers の参照モデルと一致するか。

置き場所（GPU スロット／CPU）や方針を変えても計算結果は変わらないはずなので、
全部の組み合わせで参照と比べる。GPU が無い環境では「速い側」も CPU（展開方式）で動かす。
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from strata_lite import int4
from strata_lite.engine import Engine

QUIET = dict(log=lambda *_: None)


def fake_quant(w, group=32, scheme="q4_0"):
    q, sz = int4.quantize(w, group, scheme)
    return int4.dequantize(q, sz, group).to(w.dtype)


@pytest.fixture(scope="module")
def reference(tiny_checkpoint):
    """エンジンと同じ箇所を量子化→逆量子化した参照モデル（fp32）。"""
    _, model = tiny_checkpoint
    ref = copy.deepcopy(model).float().eval()
    lm = ref.model.language_model
    with torch.no_grad():
        head = nn.Linear(lm.config.hidden_size, lm.config.vocab_size, bias=False)
        head.weight.copy_(fake_quant(lm.embed_tokens.weight))
        ref.lm_head = head
        for name, mod in lm.named_modules():
            if isinstance(mod, nn.Linear) and not name.endswith("router.proj"):
                mod.weight.copy_(fake_quant(mod.weight))
        for layer in lm.layers:
            ex = layer.experts
            for e in range(ex.num_experts):
                ex.gate_up_proj[e].copy_(fake_quant(ex.gate_up_proj[e]))
                ex.down_proj[e].copy_(fake_quant(ex.down_proj[e]))
    return ref


def ids(n=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(6, 500, (1, n), generator=g)


def ref_logits(ref, x):
    with torch.no_grad():
        return ref(input_ids=x).logits.float()


def close(a, b):
    rel = ((a - b).abs().max() / b.abs().max()).item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return rel, agree


def gb_for_slots(slots, slot_bytes):
    return slots * slot_bytes / 2**30


@pytest.mark.parametrize("placement,miss,frac", [
    ("hot", "cpu", 0.0),        # 全部 CPU（GPU スロット無し）
    ("hot", "cpu", 0.5),
    ("hot", "transfer", 0.25),
    ("layer", "cpu", 0.5),      # llama.cpp の --n-cpu-moe 相当
    ("hot", "cpu", 1.0),        # 全部 GPU
])
def test_matches_reference(converted, reference, placement, miss, frac):
    probe = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0,
                   use_profile=False, **QUIET)
    total = probe.cache.num_layers * probe.cache.num_experts
    gb = gb_for_slots(round(total * frac), probe.slot_bytes) + 1e-9
    eng = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=gb,
                 placement=placement, miss=miss, use_profile=False, **QUIET)
    assert eng.cache.slots == round(total * frac)
    for seed, n in ((1, 12), (2, 1)):
        x = ids(n, seed)
        rel, agree = close(eng.logits(x), ref_logits(reference, x))
        assert rel < 3e-2, (rel, agree)
        assert agree >= 0.9, (rel, agree)


def test_layer_placement_is_whole_layers(converted):
    eng = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0,
                 use_profile=False, **QUIET)
    e = eng.cache.num_experts
    gb = gb_for_slots(e + e // 2, eng.slot_bytes) + 1e-9  # 1.5 層分 → 丸ごと載るのは 1 層だけ
    eng.expert_vram_gb = gb
    eng.set_policy("layer", "cpu")
    assert eng.cache.placement_summary() == [0, 0, 0, e]


def test_hot_placement_learns_skew(converted):
    """偏ったルーター（conftest で 2 個のエキスパートを強くしてある）を hot 配置が学ぶ。"""
    eng = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0,
                 use_profile=False, rebalance_every=4, **QUIET)
    e = eng.cache.num_experts
    eng.expert_vram_gb = gb_for_slots(eng.cache.num_layers * 2, eng.slot_bytes) + 1e-9
    eng.set_policy("hot", "cpu")
    for seed in range(6):
        eng.generate_ids(ids(8, seed), max_new_tokens=8)
    hot_rate = eng.cache.stats.decode.as_dict()["gpu_hit_rate"]
    eng.set_policy("layer", "cpu")
    for seed in range(6):
        eng.generate_ids(ids(8, seed), max_new_tokens=8)
    layer_rate = eng.cache.stats.decode.as_dict()["gpu_hit_rate"]
    assert eng.cache.slots == eng.cache.num_layers * 2 < e * eng.cache.num_layers
    assert hot_rate > layer_rate, (hot_rate, layer_rate)


def test_profile_roundtrip(converted, tmp_path):
    eng = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0,
                 use_profile=False, **QUIET)
    eng.generate_ids(ids(8), max_new_tokens=4)
    eng.save_profile()
    eng2 = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0,
                  use_profile=True, **QUIET)
    assert torch.allclose(eng2.cache.counts, eng.cache.counts)


def test_chat_generates_text(converted):
    eng = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0.0001,
                 use_profile=False, **QUIET)
    res = eng.chat([{"role": "user", "content": "hello"}], max_new_tokens=5)
    assert res.new_tokens >= 1
    assert res.prompt_tokens > 0
    assert isinstance(res.text, str)


def test_cpu_kernel_as_fast_side(converted, reference):
    """速い側に CPU の int4 カーネルを使う（全体 bf16）。

    ランダム重みの小型モデルは bf16 で動かすだけで fp32 から大きくずれるので、
    「参照モデルを bf16 で動かしたときのずれ」と同程度に収まるかで判定する。
    """
    eng = Engine(converted, device="cpu", dtype="bfloat16", kernel="cpu", expert_vram_gb=0,
                 use_profile=False, **QUIET)
    eng.expert_vram_gb = gb_for_slots(12, eng.slot_bytes) + 1e-9
    eng.set_policy("hot", "cpu")
    assert 0 < eng.cache.slots < eng.cache.num_layers * eng.cache.num_experts
    x = ids(10, 3)
    r32 = ref_logits(reference, x)
    r16 = ref_logits(copy.deepcopy(reference).to(torch.bfloat16), x)
    base, _ = close(r16, r32)
    rel, agree = close(eng.logits(x), r32)
    assert rel < 1.5 * base + 1e-2 and agree >= 0.9, (rel, base, agree)
