from __future__ import annotations

import pytest
import torch

from strata_lite import int4


@pytest.mark.parametrize("scheme", int4.SCHEMES)
@pytest.mark.parametrize("n,k", [(64, 128), (128, 64), (48, 96), (256, 704)])
def test_cpu_kernel_matches_dequant(scheme, n, k):
    w = torch.randn(n, k)
    q, sz = int4.quantize(w, 32, scheme)
    assert q.min() >= 0 and q.max() <= 15
    deq = int4.dequantize(q, sz, 32)
    assert ((deq - w).norm() / w.norm()) < 0.12
    x = torch.randn(3, k)
    ref = x @ deq.t()
    y = int4.cpu_mm(x, int4.cpu_pack(q), sz, 32).float()
    assert ((y - ref).abs().max() / ref.abs().max()) < 1e-2


def test_q4_0_matches_ggml_rule():
    # 群 [-2, ..., 1]: 絶対値最大は -2 → d = -2 / -8 = 0.25、q = round(w / d + 8)
    w = torch.linspace(-2, 1, 32).reshape(1, 32)
    q, sz = int4.quantize(w, 32, "q4_0")
    assert sz[0, 0, 0].item() == pytest.approx(0.25)
    assert sz[0, 0, 1].item() == 0
    assert q[0, 0].item() == 0  # -2 / 0.25 + 8 = 0


@pytest.mark.parametrize("n,k", [(64, 128), (128, 64), (96, 160)])
def test_unpack_roundtrip(n, k):
    q = torch.randint(0, 16, (n, k), dtype=torch.int32)
    perm = int4.cpu_pack_permutation(n, k)
    assert torch.equal(int4.unpack_cpu_packed(int4.cpu_pack(q), perm, n, k), q)
    batched = torch.stack([int4.cpu_pack(q), int4.cpu_pack(15 - q)])
    out = int4.unpack_cpu_packed(batched, perm, n, k)
    assert torch.equal(out[1], 15 - q)


def test_dequant_kernel_and_qlinear():
    kern = int4.DequantKernel("cpu", torch.float32)
    assert int4.self_test(kern, [(64, 128)], 32) is None
    w = torch.randn(64, 128)
    lin = int4.QLinear(w, kern, 32, "asym")
    x = torch.randn(2, 5, 128)
    y = lin(x)
    assert y.shape == (2, 5, 64)
    q, sz = int4.quantize(w, 32, "asym")
    assert torch.allclose(y, x @ int4.dequantize(q, sz, 32).t(), atol=1e-4)


def test_select_kernel_on_cpu():
    assert int4.select_kernel("cpu", torch.bfloat16, [(64, 128)], 32).name == "cpu"
    assert int4.select_kernel("cpu", torch.float32, [(64, 128)], 32, "dequant").name == "dequant"


def test_bad_group():
    with pytest.raises(ValueError):
        int4.quantize(torch.randn(4, 48), 32)
