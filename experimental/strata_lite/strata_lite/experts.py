"""エキスパートの置き場所を管理する（このエンジンの中心）。

- ExpertStore: 全エキスパートを RAM に int4 で持つ（CPU で計算するときはここから直接掛ける）。
- ExpertCache: GPU 上に「スロット」を S 個確保し、そこに載せたエキスパートだけ GPU で計算する。
  どれを載せるかが配置方針（placement）:
    - "hot":   使われた回数（減衰付き）の多い順に載せる（Strata 流。層をまたいで取り合う）
    - "layer": 後ろの層から丸ごと載せる（llama.cpp の --n-cpu-moe と同じ考え方。比較用）
  GPU に無いエキスパートが選ばれたときの扱いが miss 方針:
    - "cpu":      CPU で計算する（llama.cpp と同じ）
    - "transfer": GPU へ転送して GPU で計算する（スコアが勝てばそのままスロットに残す）
  プリフィルのように 1 つのエキスパートに多数のトークンが来るときは、方針によらず
  転送して GPU で計算する（gpu_min_tokens 以上）。
- HybridExperts: transformers の Gemma4TextExperts と同じ呼び出し口で、上の 2 つを使って計算する。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from . import int4

PLACEMENTS = ("hot", "layer")
MISS_POLICIES = ("cpu", "transfer")


class ExpertStore:
    """RAM 側の全エキスパート（CPU カーネル形式の int4）。

    layers[l] は gate_up [E, 2I, H/2] / gate_up_sz [E, H/g, 2I, 2] /
    down [E, H, I/2] / down_sz [E, I/g, H, 2] を持つ。
    """

    def __init__(self, layers: dict[int, dict[str, torch.Tensor]], *, group: int, hidden: int,
                 inter: int, act_fn, pin: bool = False, log=print):
        self.group = group
        self.hidden = hidden
        self.inter = inter
        self.act_fn = act_fn
        self.layers = layers
        self.pinned = False
        if pin and torch.cuda.is_available():
            try:
                for tensors in layers.values():
                    for name in list(tensors):
                        tensors[name] = tensors[name].pin_memory()
                self.pinned = True
            except RuntimeError as e:  # Windows などで大きな固定メモリが取れないとき
                log(f"[strata-lite] pin_memory failed ({e}); using pageable memory")
        first = next(iter(layers.values()))
        self.num_experts = first["gate_up"].shape[0]

    @property
    def layer_ids(self) -> list[int]:
        return sorted(self.layers)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in self.layers.values() for t in d.values())

    def cpu_expert(self, layer: int, e: int, x: torch.Tensor) -> torch.Tensor:
        t = self.layers[layer]
        gu = int4.cpu_mm(x, t["gate_up"][e], t["gate_up_sz"][e], self.group)
        gate, up = gu.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        return int4.cpu_mm(h, t["down"][e], t["down_sz"][e], self.group)


@dataclass
class PhaseStats:
    routed: int = 0       # 選ばれた (トークン, エキスパート) の数
    gpu_hits: int = 0     # そのうち GPU に常駐していたもの
    cpu: int = 0          # CPU で計算したもの
    transferred: int = 0  # 転送して GPU で計算したもの
    tokens: int = 0

    def as_dict(self):
        d = dict(self.__dict__)
        d["gpu_hit_rate"] = self.gpu_hits / self.routed if self.routed else 0.0
        return d


@dataclass
class CacheStats:
    decode: PhaseStats = field(default_factory=PhaseStats)
    prefill: PhaseStats = field(default_factory=PhaseStats)
    uploads: int = 0
    upload_bytes: int = 0
    swaps: int = 0
    rebalances: int = 0
    rebalance_seconds: float = 0.0

    def as_dict(self):
        return {
            "decode": self.decode.as_dict(), "prefill": self.prefill.as_dict(),
            "uploads": self.uploads, "upload_mb": self.upload_bytes / 2**20,
            "swaps": self.swaps, "rebalances": self.rebalances,
            "rebalance_seconds": self.rebalance_seconds,
        }


class ExpertCache:
    def __init__(self, store: ExpertStore, kernel, *, slots: int, placement: str = "hot",
                 miss: str = "cpu", gpu_min_tokens: int = 4, rebalance_every: int = 16,
                 max_swaps: int = 64, decay: float = 0.95, hysteresis: float = 0.1,
                 scratch: int = 2, profile: torch.Tensor | None = None):
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        if miss not in MISS_POLICIES:
            raise ValueError(f"miss must be one of {MISS_POLICIES}")
        self.store = store
        self.kernel = kernel
        self.device = kernel.device
        self.group = store.group
        self.placement = placement
        self.miss = miss
        self.gpu_min_tokens = gpu_min_tokens
        self.rebalance_every = rebalance_every
        self.max_swaps = max_swaps
        self.decay = decay
        self.hysteresis = hysteresis
        self.layer_ids = store.layer_ids
        self.first_layer = self.layer_ids[0]
        self.num_layers = len(self.layer_ids)
        self.num_experts = store.num_experts
        self.row = {lid: i for i, lid in enumerate(self.layer_ids)}
        total = self.num_layers * self.num_experts
        self.slots = max(0, min(slots, total))
        self.scratch = max(1, scratch)

        h, i = store.hidden, store.inter
        self.gu_shape = (2 * i, h)
        self.down_shape = (h, i)
        self.perm_gu = int4.cpu_pack_permutation(*self.gu_shape).to(self.device)
        self.perm_down = int4.cpu_pack_permutation(*self.down_shape).to(self.device)
        n_pool = self.slots + self.scratch
        gu_ps, gu_dt = kernel.packed_shape(*self.gu_shape)
        dn_ps, dn_dt = kernel.packed_shape(*self.down_shape)
        g = self.group
        self.gu_pool = torch.empty((n_pool, *gu_ps), dtype=gu_dt, device=self.device)
        self.down_pool = torch.empty((n_pool, *dn_ps), dtype=dn_dt, device=self.device)
        self.gu_sz = torch.empty((n_pool, h // g, 2 * i, 2), dtype=torch.bfloat16, device=self.device)
        self.down_sz = torch.empty((n_pool, i // g, h, 2), dtype=torch.bfloat16, device=self.device)

        self.slot_of = [[-1] * self.num_experts for _ in range(self.num_layers)]
        self.owner: list[tuple[int, int] | None] = [None] * self.slots
        # owner と同じ内容をテンソルでも持つ（row * E + e。空きは -1）。入れ替え先の選択をまとめて計算する
        self.owner_flat = torch.full((self.slots,), -1, dtype=torch.int64)
        self.counts = torch.zeros(self.num_layers, self.num_experts, dtype=torch.float64)
        if profile is not None and tuple(profile.shape) == tuple(self.counts.shape):
            self.counts += profile.to(torch.float64)
        self._next_scratch = 0
        self.steps = 0
        self.next_rebalance = rebalance_every
        self.stats = CacheStats()
        self._initial_placement(profile is not None)

    # -- 大きさの見積もり --------------------------------------------------

    @staticmethod
    def slot_bytes(kernel, hidden: int, inter: int, group: int) -> int:
        gu_ps, gu_dt = kernel.packed_shape(2 * inter, hidden)
        dn_ps, dn_dt = kernel.packed_shape(hidden, inter)
        el = lambda shape, dt: int(torch.tensor(shape).prod().item()) * torch.empty((), dtype=dt).element_size()  # noqa: E731
        sz = (hidden // group) * 2 * inter * 2 * 2 + (inter // group) * hidden * 2 * 2
        return el(gu_ps, gu_dt) + el(dn_ps, dn_dt) + sz

    # -- 配置 --------------------------------------------------------------

    def _initial_placement(self, have_profile: bool):
        e_count = self.num_experts
        if self.placement == "layer":
            gpu_layers = self.slots // e_count
            wanted = [(r, e) for r in range(self.num_layers - gpu_layers, self.num_layers)
                      for e in range(e_count)]
        elif have_profile:
            top = torch.topk(self.counts.reshape(-1), self.slots).indices.tolist() if self.slots else []
            wanted = [divmod(i, e_count) for i in top]
        else:
            # まだ使われ方が分からないので、各層に均等に配る（使いながら入れ替わる）
            wanted = []
            per, extra = divmod(self.slots, self.num_layers)
            for r in range(self.num_layers):
                wanted += [(r, e) for e in range(per + (1 if r < extra else 0))]
        for slot, (r, e) in enumerate(wanted[: self.slots]):
            self._assign(slot, r, e)
        self.stats.uploads = 0
        self.stats.upload_bytes = 0

    def _assign(self, slot: int, r: int, e: int):
        old = self.owner[slot]
        if old is not None:
            self.slot_of[old[0]][old[1]] = -1
        self._load(slot, r, e)
        self.owner[slot] = (r, e)
        self.owner_flat[slot] = r * self.num_experts + e
        self.slot_of[r][e] = slot

    def _load(self, slot: int, r: int, e: int):
        """RAM のエキスパート (r, e) を GPU のスロットへ書き込む。"""
        t = self.store.layers[self.layer_ids[r]]
        nb = self.store.pinned
        gu = t["gate_up"][e].to(self.device, non_blocking=nb)
        dn = t["down"][e].to(self.device, non_blocking=nb)
        self.gu_pool[slot].copy_(self.kernel.pack(int4.unpack_cpu_packed(gu, self.perm_gu, *self.gu_shape)))
        self.down_pool[slot].copy_(self.kernel.pack(int4.unpack_cpu_packed(dn, self.perm_down, *self.down_shape)))
        self.gu_sz[slot].copy_(t["gate_up_sz"][e], non_blocking=nb)
        self.down_sz[slot].copy_(t["down_sz"][e], non_blocking=nb)
        self.stats.uploads += 1
        self.stats.upload_bytes += sum(x.numel() * x.element_size() for x in (
            t["gate_up"][e], t["down"][e], t["gate_up_sz"][e], t["down_sz"][e]))

    def resident_fraction(self) -> float:
        return sum(o is not None for o in self.owner) / (self.num_layers * self.num_experts)

    def placement_summary(self) -> list[int]:
        """層ごとに GPU に載っているエキスパート数。"""
        out = [0] * self.num_layers
        for o in self.owner:
            if o is not None:
                out[o[0]] += 1
        return out

    def rebalance(self):
        """減衰付きの使用回数で上位 S 個を求め、足りないものを入れ替える（hot のみ）。"""
        if self.placement != "hot" or self.slots == 0:
            return
        t0 = time.perf_counter()
        self.counts *= self.decay
        flat = self.counts.reshape(-1)
        e_count = self.num_experts
        want = set(torch.topk(flat, self.slots).indices.tolist())
        resident = {o[0] * e_count + o[1]: s for s, o in enumerate(self.owner) if o is not None}
        add = sorted((i for i in want if i not in resident), key=lambda i: -flat[i].item())
        evict = sorted((i for i in resident if i not in want), key=lambda i: flat[i].item())
        swaps = 0
        for a, v in zip(add, evict):
            if swaps >= self.max_swaps or flat[a] <= flat[v] * (1 + self.hysteresis):
                break
            self._assign(resident[v], *divmod(a, e_count))
            swaps += 1
        self.stats.swaps += swaps
        self.stats.rebalances += 1
        self.stats.rebalance_seconds += time.perf_counter() - t0

    def _admit(self, r: int, e: int, protect: set[int]) -> int:
        """transfer 方針: スコアが一番低い常駐エキスパートより使われていれば入れ替えて残す。"""
        if self.placement != "hot" or self.slots == 0:
            return -1
        flat = self.counts.reshape(-1)
        scores = torch.where(self.owner_flat >= 0, flat[self.owner_flat.clamp(min=0)],
                             torch.full_like(self.owner_flat, -1, dtype=torch.float64))
        if protect:
            # 今まさにこの層で使うエキスパートは追い出さない
            keep = torch.tensor([r * self.num_experts + x for x in protect])
            scores = torch.where(torch.isin(self.owner_flat, keep),
                                 torch.full_like(scores, float("inf")), scores)
        slot = int(torch.argmin(scores).item())
        if not flat[r * self.num_experts + e].item() >= scores[slot].item():
            return -1
        self._assign(slot, r, e)
        return slot

    def scratch_upload(self, r: int, e: int) -> int:
        slot = self.slots + self._next_scratch
        self._next_scratch = (self._next_scratch + 1) % self.scratch
        self._load(slot, r, e)
        return slot

    # -- 計算 --------------------------------------------------------------

    def gpu_expert(self, slot: int, x: torch.Tensor) -> torch.Tensor:
        gu = self.kernel.mm(x, self.gu_pool[slot], self.gu_sz[slot], self.group)
        gate, up = gu.chunk(2, dim=-1)
        h = self.store.act_fn(gate) * up
        return self.kernel.mm(h, self.down_pool[slot], self.down_sz[slot], self.group)

    def save_profile(self, path):
        torch.save({"counts": self.counts.clone(), "layers": self.layer_ids,
                    "experts": self.num_experts}, path)

    @staticmethod
    def load_profile(path, num_layers: int, num_experts: int) -> torch.Tensor | None:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError):
            return None
        counts = data.get("counts") if isinstance(data, dict) else None
        if counts is None or tuple(counts.shape) != (num_layers, num_experts):
            return None
        return counts


class HybridExperts(nn.Module):
    """Gemma4TextExperts の差し替え。forward(hidden [T, H], top_k_index [T, K], top_k_weights [T, K])。"""

    def __init__(self, cache: ExpertCache, layer_id: int):
        super().__init__()
        self.cache = cache
        self.layer_id = layer_id
        self.row = cache.row[layer_id]

    def _tick(self, t: int):
        cache = self.cache
        if self.layer_id == cache.first_layer:
            if cache.steps >= cache.next_rebalance:
                cache.rebalance()
                cache.next_rebalance = cache.steps + cache.rebalance_every
            cache.steps += t
            (cache.stats.decode if t == 1 else cache.stats.prefill).tokens += t

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if top_k_index.shape[0] == 1:
            return self._decode(hidden_states, top_k_index, top_k_weights)
        return self._batch(hidden_states, top_k_index, top_k_weights)

    def _decode(self, hidden_states, top_k_index, top_k_weights):
        """1 トークンの生成（ここが速度を決める）。小さな torch 演算の回数を減らした経路。"""
        cache = self.cache
        r = self.row
        ex = top_k_index.reshape(-1).tolist()  # ここで GPU を待つ（どのエキスパートか分からないと始まらない）
        w = top_k_weights.reshape(-1).tolist()
        self._tick(1)
        cache.counts[r, ex] += 1.0
        phase = cache.stats.decode
        phase.routed += len(ex)
        gpu, cpu = [], []
        for j, e in enumerate(ex):
            slot = cache.slot_of[r][e]
            if slot >= 0:
                phase.gpu_hits += 1
                gpu.append((j, e, slot))
            elif cache.miss == "transfer":
                phase.transferred += 1
                gpu.append((j, e, cache._admit(r, e, set(ex))))
            else:
                phase.cpu += 1
                cpu.append((j, e))
        if cpu:
            x_cpu = hidden_states.to("cpu", torch.bfloat16)
        acc = None
        if gpu:
            # GPU 側を先に積んでおく（非同期）。その間に CPU 側を計算する
            ys = []
            for j, e, slot in gpu:
                if slot < 0:
                    slot = cache.scratch_upload(r, e)
                ys.append(cache.gpu_expert(slot, hidden_states))
            wt = torch.tensor([w[j] for j, _, _ in gpu], dtype=torch.float32).to(hidden_states.device)
            acc = (torch.stack(ys).to(torch.float32) * wt[:, None, None]).sum(0)
        if cpu:
            y = None
            for j, e in cpu:
                part = cache.store.cpu_expert(self.layer_id, e, x_cpu).to(torch.float32) * w[j]
                y = part if y is None else y + part
            y = y.to(hidden_states.device)
            acc = y if acc is None else acc + y
        return acc.to(hidden_states.dtype)

    def _batch(self, hidden_states, top_k_index, top_k_weights):
        """複数トークン（プリフィル）。エキスパートごとにトークンをまとめて計算する。"""
        cache = self.cache
        r = self.row
        t, k = top_k_index.shape
        idx_cpu = top_k_index.to("cpu")
        self._tick(t)
        flat = idx_cpu.reshape(-1)
        cache.counts[r] += torch.bincount(flat, minlength=cache.num_experts).to(torch.float64)
        phase = cache.stats.prefill
        order = torch.argsort(flat, stable=True)
        uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
        protect = set(uniq.tolist())

        gpu_jobs, cpu_jobs = [], []
        pos = 0
        for e, c in zip(uniq.tolist(), counts.tolist()):
            phase.routed += c
            slot = cache.slot_of[r][e]
            if slot >= 0:
                phase.gpu_hits += c
                gpu_jobs.append((e, slot, pos, pos + c))
            elif c >= cache.gpu_min_tokens or cache.miss == "transfer":
                phase.transferred += c
                slot = cache._admit(r, e, protect) if cache.miss == "transfer" else -1
                gpu_jobs.append((e, slot, pos, pos + c))  # slot < 0 は計算直前に作業スロットへ転送
            else:
                phase.cpu += c
                cpu_jobs.append((e, pos, pos + c))
            pos += c

        out = torch.zeros_like(hidden_states)
        if cpu_jobs:
            # GPU 側の計算を積む前に CPU へ写す（後だと GPU の計算完了を待ってしまう）
            hidden_cpu = hidden_states.to("cpu", torch.bfloat16)
            w_cpu = top_k_weights.reshape(-1).to("cpu", torch.float32)
        if gpu_jobs:
            order_dev = order.to(hidden_states.device)
            w_flat = top_k_weights.reshape(-1)
            for e, slot, a, b in gpu_jobs:
                if slot < 0:
                    slot = cache.scratch_upload(r, e)
                p = order_dev[a:b]
                tok = torch.div(p, k, rounding_mode="floor")
                y = cache.gpu_expert(slot, hidden_states[tok])
                y = y.to(torch.float32) * w_flat[p, None].to(torch.float32)
                out.index_add_(0, tok, y.to(out.dtype))
        if cpu_jobs:
            out_cpu = torch.zeros(t, hidden_states.shape[-1], dtype=torch.float32)
            for e, a, b in cpu_jobs:
                p = order[a:b]
                tok = torch.div(p, k, rounding_mode="floor")
                y = cache.store.cpu_expert(self.layer_id, e, hidden_cpu[tok]).to(torch.float32)
                out_cpu.index_add_(0, tok, y * w_cpu[p, None])
            out += out_cpu.to(out.device, out.dtype)
        return out
