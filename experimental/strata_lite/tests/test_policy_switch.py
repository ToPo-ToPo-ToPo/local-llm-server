"""配置方針の切り替え（set_policy）で、古いキャッシュが確実に手放されること。

bench --compare は 1 つのエンジンで方針を順に切り替える。古い ExpertCache への参照が各層の
HybridExperts に残っていると GPU スロットが解放されず、次の方針のスロット数が 0 になる
（実機で発覚）。CPU でも「参照が残らない」ことは weakref で確かめられる。
"""
from __future__ import annotations

import gc
import weakref

import torch

from strata_lite.engine import Engine


def test_old_cache_is_released_on_set_policy(converted):
    eng = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0.001,
                 placement="hot", use_profile=False, log=lambda *_: None)
    first = weakref.ref(eng.cache)
    eng.logits(torch.randint(5, 500, (1, 6)))  # 一度使って HybridExperts に行き渡らせる
    eng.set_policy("layer", "cpu")
    gc.collect()
    assert first() is None, "the previous ExpertCache is still referenced after set_policy"
    # 新しい方針が全層に行き渡り、同じ入力で引き続き動く
    for lid in eng.meta["moe_layers"]:
        assert eng.model.model.layers[lid].experts.cache is eng.cache
    eng.logits(torch.randint(5, 500, (1, 6)))
