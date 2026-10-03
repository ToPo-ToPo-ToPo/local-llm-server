"""変換済みモデルを読み込み、GPU と CPU にまたがって生成するエンジン。

配置:
- GPU（速い側）: 注意機構・共有 MLP・ルーター・正規化・lm_head（int4）と、エキスパートの GPU スロット
- CPU（RAM）: 単語埋め込み（引くだけなので CPU で十分）と、全エキスパート（int4）

transformers の Gemma4ForCausalLM を meta デバイス上に組み、重みを必要な形で差し込む。
注意機構・KV キャッシュ・生成ループは transformers のものをそのまま使い、エキスパート部分だけ
HybridExperts に差し替える。
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from . import int4
from .convert import FORMAT_VERSION
from .experts import ExpertCache, ExpertStore, HybridExperts

PROFILE_NAME = "strata-profile.pt"
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def physical_cores() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


class CpuEmbedding(nn.Module):
    """単語埋め込みを RAM に置き、引いた行だけ GPU へ送る（Gemma の埋め込み倍率込み）。"""

    def __init__(self, weight: torch.Tensor, scale: float, device):
        super().__init__()
        # parameter にすると model.device が CPU と判定されるので buffer で持つ
        self.weight = nn.Buffer(weight, persistent=False)
        self.scale = torch.tensor(scale, dtype=weight.dtype)
        self.out_device = torch.device(device)

    def forward(self, input_ids):
        out = F.embedding(input_ids.to("cpu"), self.weight) * self.scale
        return out.to(self.out_device)


class ChunkedQLinear(nn.Module):
    """出力次元の大きい Linear（lm_head など）を行方向に分けて int4 化する（量子化中の一時メモリを抑える）。"""

    def __init__(self, weight: torch.Tensor, kernel, group, scheme, chunk_rows: int = 32768):
        super().__init__()
        self.parts = nn.ModuleList(
            int4.QLinear(weight[i:i + chunk_rows], kernel, group, scheme)
            for i in range(0, weight.shape[0], chunk_rows))
        self.out_features = weight.shape[0]

    def forward(self, x):
        return torch.cat([p(x) for p in self.parts], dim=-1)


def quantized_linear(weight, kernel, group, scheme, device, dtype):
    """int4 化できる形なら int4、できなければ bf16 のまま速い側へ置く。"""
    n, k = weight.shape
    try:
        if n * k > 64 * 2**20:
            return ChunkedQLinear(weight, kernel, group, scheme)
        return int4.QLinear(weight, kernel, group, scheme)
    except (ValueError, RuntimeError):
        lin = nn.Linear(k, n, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_(weight.to(device))
        return lin


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    new_tokens: int
    prefill_seconds: float
    decode_seconds: float

    @property
    def prefill_tps(self):
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tps(self):
        return (self.new_tokens - 1) / self.decode_seconds if self.decode_seconds and self.new_tokens > 1 else 0.0


class _TimingStreamer:
    """generate の進み具合から、プリフィルと生成の時間を測る（transformers の streamer 規約）。"""

    def __init__(self, inner=None):
        self.inner = inner
        self.calls = 0
        self.t_start = self.t_first = self.t_last = None
        self.tokens = 0

    def put(self, value):
        now = time.perf_counter()
        if self.calls == 0:
            self.t_start = now
        else:
            self.tokens += value.numel()
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
        self.calls += 1
        if self.inner is not None:
            self.inner.put(value)

    def end(self):
        if self.inner is not None:
            self.inner.end()


class Engine:
    def __init__(self, model_dir: str, *, device: str | None = None, dtype: str = "bfloat16",
                 placement: str = "hot", miss: str = "cpu", expert_vram_gb: float | None = None,
                 reserve_gb: float = 1.5, threads: int | None = None, pin: bool = True,
                 kernel: str = "auto", gpu_min_tokens: int = 4, rebalance_every: int = 16,
                 max_swaps: int = 64, decay: float = 0.95, use_profile: bool = True, log=print):
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        t0 = time.perf_counter()
        self.log = log
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "strata.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("format_version") != FORMAT_VERSION:
            raise RuntimeError("converted with an incompatible strata-lite version; re-run convert")
        self.group = self.meta["group"]
        self.scheme = self.meta["scheme"]
        hidden, inter = self.meta["hidden_size"], self.meta["moe_intermediate_size"]
        fp = int4.cpu_layout_fingerprint([(2 * inter, hidden), (hidden, inter)])
        if fp != self.meta["cpu_layout"]:
            raise RuntimeError("the int4 CPU layout of this PC differs from the one used for convert; "
                               "re-run convert on this PC")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = _DTYPES[dtype]
        torch.set_num_threads(threads or physical_cores())
        self.kernel = int4.select_kernel(self.device, self.dtype, [(2 * inter, hidden), (hidden, inter)],
                                         self.group, kernel, log=log)
        log(f"[strata-lite] device={self.device} kernel={self.kernel.name} threads={torch.get_num_threads()}")

        cfg = AutoConfig.from_pretrained(model_dir)
        cfg = cfg.get_text_config()
        cfg._attn_implementation = "sdpa"
        self.config = cfg
        with torch.device("meta"):
            model = Gemma4ForCausalLM(cfg)
        model.to(self.dtype)
        self._load_dense(model)
        self.model = model.eval()

        # エキスパート（RAM）
        layers = {}
        for lid in self.meta["moe_layers"]:
            layers[lid] = load_file(os.path.join(model_dir, f"experts-{lid:03d}.safetensors"))
        act_fn = model.model.layers[self.meta["moe_layers"][0]].mlp.act_fn
        self.store = ExpertStore(layers, group=self.group, hidden=hidden, inter=inter, act_fn=act_fn,
                                 pin=pin and self.device.type == "cuda", log=log)
        self.slot_bytes = ExpertCache.slot_bytes(self.kernel, hidden, inter, self.group)
        self.expert_vram_gb = expert_vram_gb
        self.reserve_gb = reserve_gb
        self.cache_opts = dict(gpu_min_tokens=gpu_min_tokens, rebalance_every=rebalance_every,
                               max_swaps=max_swaps, decay=decay)
        self.use_profile = use_profile
        self.cache = None
        self.set_policy(placement, miss)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_dir)
        except OSError:
            self.generation_config = GenerationConfig.from_model_config(cfg)
        log(f"[strata-lite] ready in {time.perf_counter() - t0:.0f}s: experts on GPU "
            f"{self.cache.slots}/{self.cache.num_layers * self.cache.num_experts} "
            f"({self.cache.resident_fraction():.0%}), RAM experts {self.store.nbytes() / 2**30:.1f} GiB")

    # -- ロード ------------------------------------------------------------

    def _load_dense(self, model):
        dev, dt = self.device, self.dtype
        path = os.path.join(self.model_dir, "dense.safetensors")
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())  # noqa: SIM118
            used = set()

            def get(name):
                if name not in keys:
                    raise RuntimeError(f"{name} is missing from dense.safetensors")
                used.add(name)
                return f.get_tensor(name)

            # 単語埋め込み（RAM）と lm_head（共有重みを int4 で GPU へ）
            emb = get("model.embed_tokens.weight").to(dt)
            scale = model.model.embed_tokens.scalar_embed_scale
            model.model.embed_tokens = CpuEmbedding(emb, scale, dev)
            head_w = get("lm_head.weight") if "lm_head.weight" in keys else emb
            # 重みは CPU のまま渡す（量子化は分割して速い側で行うので、GPU に丸ごと載せない）
            model.lm_head = quantized_linear(head_w, self.kernel, self.group, self.scheme, dev, dt)

            for name, mod in list(model.named_modules()):
                if name == "lm_head" or name.startswith("lm_head.") or not isinstance(mod, nn.Linear):
                    continue
                w = get(name + ".weight")
                if name.endswith("router.proj"):
                    # どのエキスパートを選ぶかを決める層。小さいので量子化しない
                    new = nn.Linear(mod.in_features, mod.out_features, bias=False, device=dev, dtype=dt)
                    with torch.no_grad():
                        new.weight.copy_(w)
                else:
                    new = quantized_linear(w, self.kernel, self.group, self.scheme, dev, dt)
                model.set_submodule(name, new)

            # 残り（正規化・スカラーなど）
            rest = {}
            for name, t in model.state_dict().items():
                if name in keys and name not in used and t.device.type == "meta":
                    rest[name] = get(name).to(dev, t.dtype if t.is_floating_point() else None)
            model.load_state_dict(rest, strict=False, assign=True)

        # 状態辞書に入らない値（RoPE の周波数）は作り直す
        rotary = model.model.rotary_emb
        model.model.rotary_emb = type(rotary)(self.config, device=dev)
        for lid, layer in enumerate(model.model.layers):
            if getattr(layer, "enable_moe_block", False):
                layer.experts = nn.Identity()  # 後で HybridExperts に差し替える
        leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                    if t.device.type == "meta"]
        if leftover:
            raise RuntimeError(f"weights missing from dense.safetensors: {leftover[:8]}")

    # -- 方針の切り替え ----------------------------------------------------

    def _slot_count(self) -> int:
        if self.expert_vram_gb is not None:
            budget = self.expert_vram_gb * 2**30
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(self.device)
            budget = free - self.reserve_gb * 2**30
        else:
            budget = 0
        return max(0, int(budget // self.slot_bytes))

    def set_policy(self, placement: str, miss: str):
        """配置方針を切り替える（GPU スロットを作り直す）。"""
        old = self.cache
        profile = None
        if old is not None:
            profile = old.counts.clone()
            self.cache = None
            del old
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        elif self.use_profile:
            profile = ExpertCache.load_profile(os.path.join(self.model_dir, PROFILE_NAME),
                                               len(self.meta["moe_layers"]), self.meta["num_experts"])
            if profile is not None:
                self.log("[strata-lite] using saved expert usage profile")
        self.cache = ExpertCache(self.store, self.kernel, slots=self._slot_count(), placement=placement,
                                 miss=miss, profile=profile, **self.cache_opts)
        for lid in self.meta["moe_layers"]:
            self.model.model.layers[lid].experts = HybridExperts(self.cache, lid)

    def save_profile(self):
        if self.cache is not None:
            self.cache.save_profile(os.path.join(self.model_dir, PROFILE_NAME))

    # -- 生成 --------------------------------------------------------------

    def encode_chat(self, messages, enable_thinking: bool = False):
        enc = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                 return_dict=True, enable_thinking=enable_thinking)
        return enc["input_ids"]

    @torch.inference_mode()
    def generate_ids(self, input_ids, *, max_new_tokens=256, temperature=0.0, top_p=1.0, streamer=None):
        timing = _TimingStreamer(streamer)
        input_ids = input_ids.to(self.device)
        gc = copy.deepcopy(self.generation_config)
        gc.max_new_tokens = max_new_tokens
        if temperature and temperature > 0:
            gc.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=None)
        else:
            gc.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = self.model.generate(input_ids, generation_config=gc, streamer=timing,
                                  attention_mask=torch.ones_like(input_ids))
        new = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        prefill = (timing.t_first - timing.t_start) if timing.t_first else 0.0
        decode = (timing.t_last - timing.t_first) if timing.t_first else 0.0
        return GenResult(text, input_ids.shape[1], int(new.numel()), prefill, decode)

    def chat(self, messages, **kw) -> GenResult:
        return self.generate_ids(self.encode_chat(messages), **kw)

    @torch.inference_mode()
    def logits(self, input_ids):
        """1 回の順伝播のロジット（テスト用）。"""
        return self.model(input_ids.to(self.device), use_cache=False).logits.float().cpu()
