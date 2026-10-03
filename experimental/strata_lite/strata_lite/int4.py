"""int4 の群量子化と、CPU / GPU それぞれの int4 行列積。

重みは群（group 個ずつ）ごとに `w ≈ (q - 8) * s + z`（q は 0〜15）で持つ。s と z は
torch の int4 カーネル（tinygemm 系）の形式に合わせ、bf16 の `[K/group, N, 2]` に詰める。

- CPU: `aten._weight_int4pack_mm_for_cpu`。展開してから掛ける方式は数百倍遅いので、
  CPU 側はこのカーネルが使えることを前提にする。
- GPU: `aten._weight_int4pack_mm`（CUDA の tinygemm。Ampere 以降）。使えない GPU や
  自己検査に落ちたときは、展開してから掛ける方式（DequantKernel）に切り替える。

CPU カーネルのパック形式は単純な詰め方ではない（64 行ずつのブロックで並べ替わる）。
式で再現する代わりに、起動時にビット面ごとの探針で並び順を実測する
（`cpu_pack_permutation`）。CPU の命令セットで形式が変わっても追従できる。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

_ops = torch.ops.aten

SCHEMES = ("q4_0", "asym")


def quantize(w: torch.Tensor, group: int, scheme: str = "q4_0"):
    """[N, K] の重みを群量子化する。戻り値は (q int32 [N, K], sz bf16 [K/group, N, 2])。

    - q4_0: ggml の Q4_0 と同じ対称量子化（z = 0、s = 絶対値最大の値 / -8）。
      Gemma 4 の QAT 版（q4_0 前提で学習）の重みと相性が良い。
    - asym: 群ごとの最小・最大から決める非対称量子化（一般のモデル向け）。
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r} (choose from {SCHEMES})")
    n, k = w.shape
    if k % group:
        raise ValueError(f"K={k} is not a multiple of group={group}")
    wg = w.float().reshape(n, k // group, group)
    if scheme == "q4_0":
        amax_idx = wg.abs().argmax(dim=-1, keepdim=True)
        m = torch.gather(wg, -1, amax_idx).squeeze(-1)
        s = m / -8.0
        inv = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
        q = torch.clamp(torch.floor(wg * inv[..., None] + 8.5), 0, 15)
        z = torch.zeros_like(s)
    else:
        mn = wg.amin(dim=-1)
        mx = wg.amax(dim=-1)
        s = (mx - mn) / 15.0
        inv = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
        q = torch.clamp(torch.round((wg - mn[..., None]) * inv[..., None]), 0, 15)
        z = mn + 8.0 * s
    q = q.to(torch.int32).reshape(n, k)
    sz = torch.stack([s, z], dim=-1).transpose(0, 1).contiguous().to(torch.bfloat16)
    return q, sz


def dequantize(q: torch.Tensor, sz: torch.Tensor, group: int, dtype=torch.float32) -> torch.Tensor:
    """quantize の逆（参照実装・テスト用）。q は int32 [N, K]。"""
    n, k = q.shape
    s = sz[..., 0].transpose(0, 1).to(dtype)  # [N, K/group]
    z = sz[..., 1].transpose(0, 1).to(dtype)
    qg = q.reshape(n, k // group, group).to(dtype)
    return ((qg - 8) * s[..., None] + z[..., None]).reshape(n, k)


# ---------------------------------------------------------------------------
# CPU カーネル


def cpu_pack(q: torch.Tensor) -> torch.Tensor:
    """int32 [N, K] → CPU カーネル形式の uint8 [N, K/2]。"""
    return _ops._convert_weight_to_int4pack_for_cpu(q.to(torch.int32).contiguous(), 1)


def cpu_mm(x: torch.Tensor, packed: torch.Tensor, sz: torch.Tensor, group: int) -> torch.Tensor:
    """CPU の int4 行列積。x は [M, K]（bf16 に揃える）、戻り値は bf16 [M, N]。"""
    return _ops._weight_int4pack_mm_for_cpu(x.to(torch.bfloat16).contiguous(), packed, group, sz)


@lru_cache(maxsize=None)
def cpu_pack_permutation(n: int, k: int) -> torch.Tensor:
    """CPU 形式のニブル列の j 番目に、元の [N, K] のどの要素（平坦化した番号）が入るか。

    ニブル列は「バイト b の下位 4bit が 2b 番、上位 4bit が 2b+1 番」とする。元の番号の
    各ビットを 0/1 の重みとしてパックし、ビット面ごとに読み戻して番号を組み立てる。
    """
    idx = torch.arange(n * k, dtype=torch.int64).reshape(n, k)
    acc = torch.zeros(n * k, dtype=torch.int64)
    for b in range((n * k - 1).bit_length()):
        plane = ((idx >> b) & 1).to(torch.int32)
        packed = cpu_pack(plane)
        nib = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(-1).to(torch.int64)
        acc |= nib << b
    if not torch.equal(torch.sort(acc).values, torch.arange(n * k)):
        raise RuntimeError(f"CPU int4 pack layout for [{n}, {k}] is not a permutation")
    return acc


def cpu_layout_fingerprint(shapes) -> str:
    """CPU パック形式の指紋。変換した機械と実行する機械で形式が同じかを確かめる。"""
    h = hashlib.sha256()
    for n, k in shapes:
        h.update(f"{n}x{k}:".encode())
        h.update(cpu_pack_permutation(n, k).numpy().tobytes())
    return h.hexdigest()[:16]


def unpack_cpu_packed(packed: torch.Tensor, perm: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """CPU 形式の uint8 [..., N, K/2] を int32 [..., N, K] に戻す（どのデバイス上でも動く）。"""
    lead = packed.shape[:-2]
    nib = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(*lead, n * k).to(torch.int32)
    out = torch.empty_like(nib)
    out[..., perm] = nib
    return out.reshape(*lead, n, k)


# ---------------------------------------------------------------------------
# 速い側のデバイス（通常は GPU）のカーネル


class DequantKernel:
    """展開してから掛ける方式。どの GPU でも動く（tinygemm が使えないときの代替）。

    保持形式は単純な詰め方の uint8 [N, K/2]（下位 4bit が偶数列）。
    """

    name = "dequant"

    def __init__(self, device, dtype):
        self.device = torch.device(device)
        self.dtype = dtype

    def packed_shape(self, n, k):
        return (n, k // 2), torch.uint8

    def pack(self, q: torch.Tensor) -> torch.Tensor:
        q = q.to(torch.uint8)
        return (q[..., ::2] | (q[..., 1::2] << 4)).contiguous()

    def mm(self, x, packed, sz, group):
        n = packed.shape[-2]
        k = packed.shape[-1] * 2
        q = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(n, k // group, group)
        s = sz[..., 0].transpose(0, 1).to(self.dtype)
        z = sz[..., 1].transpose(0, 1).to(self.dtype)
        w = ((q.to(self.dtype) - 8) * s[..., None] + z[..., None]).reshape(n, k)
        return F.linear(x.to(self.dtype), w)


class TinyGemmKernel:
    """CUDA の tinygemm（`_weight_int4pack_mm`）。bf16 と Ampere 以降の GPU が前提。"""

    name = "tinygemm"

    def __init__(self, device, dtype):
        self.device = torch.device(device)
        self.dtype = dtype
        self._shape_cache = {}

    @staticmethod
    def inner_k_tiles(k):
        for t in (8, 4, 2):
            if k % (t * 16) == 0:
                return t
        raise ValueError(f"K={k} is not supported by tinygemm")

    def packed_shape(self, n, k):
        if (n, k) not in self._shape_cache:
            probe = self.pack(torch.zeros(n, k, dtype=torch.int32, device=self.device))
            self._shape_cache[(n, k)] = (tuple(probe.shape), probe.dtype)
        return self._shape_cache[(n, k)]

    def pack(self, q):
        k = q.shape[-1]
        q = q.to(torch.int32)
        u8 = ((q[..., ::2] << 4) | q[..., 1::2]).to(torch.uint8).contiguous()
        return _ops._convert_weight_to_int4pack(u8, self.inner_k_tiles(k))

    def mm(self, x, packed, sz, group):
        return _ops._weight_int4pack_mm(x.to(torch.bfloat16).contiguous(), packed, group, sz)


class CpuKernel:
    """CPU を「速い側」として使う（テスト用。GPU が無い環境で経路全体を通す）。"""

    name = "cpu"

    def __init__(self, device, dtype):
        self.device = torch.device("cpu")
        self.dtype = dtype

    def packed_shape(self, n, k):
        return (n, k // 2), torch.uint8

    def pack(self, q):
        return cpu_pack(q)

    def mm(self, x, packed, sz, group):
        return cpu_mm(x, packed, sz, group)


KERNELS = {"dequant": DequantKernel, "tinygemm": TinyGemmKernel, "cpu": CpuKernel}


def self_test(kernel, shapes, group: int, tol: float = 3e-2) -> str | None:
    """実際の形で量子化→パック→積を回し、参照（展開して fp32 で掛ける）と比べる。

    問題が無ければ None、あれば理由の文字列を返す。
    """
    gen = torch.Generator().manual_seed(0)
    for n, k in shapes:
        try:
            w = torch.randn(n, k, generator=gen)
            q, sz = quantize(w, group)
            x = torch.randn(3, k, generator=gen)
            ref = x @ dequantize(q, sz, group).t()
            packed = kernel.pack(q.to(kernel.device))
            y = kernel.mm(x.to(kernel.device), packed, sz.to(kernel.device), group).float().cpu()
        except Exception as e:  # noqa: BLE001 - カーネル非対応は理由を返して代替へ
            return f"{kernel.name} failed on [{n}, {k}]: {e}"
        err = (y - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        if not err < tol:
            return f"{kernel.name} mismatch on [{n}, {k}]: rel err {err:.3g}"
    return None


def select_kernel(device, dtype, shapes, group: int, prefer: str = "auto", log=print):
    """速い側のデバイスで使うカーネルを選ぶ。auto は tinygemm → dequant の順に自己検査で決める。"""
    device = torch.device(device)
    if prefer != "auto":
        kern = KERNELS[prefer](device, dtype)
        why = self_test(kern, shapes, group)
        if why:
            raise RuntimeError(why)
        return kern
    if device.type == "cpu":
        return CpuKernel(device, dtype)
    candidates = ["dequant"]
    if device.type == "cuda" and dtype == torch.bfloat16:
        candidates.insert(0, "tinygemm")
    for name in candidates:
        kern = KERNELS[name](device, dtype)
        why = self_test(kern, shapes, group)
        if why is None:
            return kern
        log(f"[strata-lite] {why}; falling back")
    raise RuntimeError("no int4 kernel works on this device")


@dataclass
class QuantizedWeight:
    packed: torch.Tensor
    sz: torch.Tensor
    n: int
    k: int


class QLinear(nn.Module):
    """バイアス無しの Linear を int4 で置き換える（速い側のデバイスに常駐）。"""

    def __init__(self, weight: torch.Tensor, kernel, group: int, scheme: str):
        super().__init__()
        n, k = weight.shape
        q, sz = quantize(weight.to(kernel.device), group, scheme)
        self.packed = nn.Buffer(kernel.pack(q))
        self.sz = nn.Buffer(sz)
        self.in_features = k
        self.out_features = n
        self.group = group
        self.kernel = kernel

    def forward(self, x):
        shape = x.shape
        y = self.kernel.mm(x.reshape(-1, shape[-1]), self.packed, self.sz, self.group)
        return y.to(x.dtype).reshape(*shape[:-1], self.out_features)
