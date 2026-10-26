"""GGUF（llama.cpp 形式）の Gemma 4 MoE を strata-lite 形式に取り込む。

`convert` の入力が `.gguf` のときに使われる。出力は HF safetensors からの変換と同じ
（strata.json / dense.safetensors / experts-LLL.safetensors / config.json・tokenizer*）。

狙いは **QAT 版の Q4_0 GGUF をそのまま使う**こと。Gemma 4 の QAT 版は Q4_0 の格子
（32 個ずつ・対称・`(q - 8) * d`）で学習されており、strata-lite の `q4_0` 方式と格子が同じなので、
エキスパートは**再量子化せずに**ニブルとスケールを詰め替えるだけで取り込める（bf16 から
丸め直す HF 経路より品質が良い）。スケールは GGUF が fp16、strata-lite のカーネルが bf16 なので、
そこだけ bf16 への丸めが入る（相対 2^-9 程度。ニブルは変えない）。

- エキスパート以外（attention・共有 MLP・norm・ルーター・埋め込み）は展開して bf16 で
  dense.safetensors に書く。Linear の重みはロード時に改めて int4 化される（HF 経路と同じ）。
- Q4_0 以外の量子化（Q4_K など）のエキスパートは、展開してから strata-lite の方式で量子化し直す
  （落ちる分は Q4_0 の格子にはめ直す誤差。ログに出す）。
- config.json / tokenizer は GGUF に無いので、元の HF リポジトリ（`--base`。既定は
  google/gemma-4-26B-A4B-it）から小さなファイルだけ取る。GGUF のメタデータと寸法を突き合わせて、
  違うモデルの設定を取り違えていないかを確かめる。
- `rope_freqs` は llama.cpp が変換時に生成する補助テンソルで HF には無いので読み飛ばす。
  Gemma 4 の norm 重みは llama.cpp の変換で値を変えない（norm_shift = 0）ので、そのまま使える。
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import time

import numpy as np
import torch
from safetensors.torch import save_file

from . import int4

ARCH = "gemma4"
Q4_0_BLOCK = 32
Q4_0_BLOCK_BYTES = 18  # fp16 スケール 2 バイト + 32 ニブル 16 バイト

_COPY_PATTERNS = ("config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
                  "chat_template*", "added_tokens.json")
_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# GGUF 名 → Gemma4ForCausalLM の名前（層のものは "model.layers.{L}." を前に付ける）。
# llama.cpp の gguf-py tensor_mapping（gemma4）と conversion/gemma.py（router.scale → ffn_gate_inp.scale、
# per_expert_scale → ffn_down_exps.scale、layer_scalar → layer_output_scale）から起こした。
_TOP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_TOP_SKIP = {"rope_freqs.weight"}
_BLK = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "post_ffw_norm_1.weight": "post_feedforward_layernorm_1.weight",
    "post_ffw_norm_2.weight": "post_feedforward_layernorm_2.weight",
    "pre_ffw_norm_2.weight": "pre_feedforward_layernorm_2.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "router.proj.weight",
    "ffn_gate_inp.scale": "router.scale",
    "ffn_down_exps.scale": "router.per_expert_scale",
    "layer_output_scale.weight": "layer_scalar",
}
_EXPERTS = {"ffn_gate_up_exps.weight": "gate_up", "ffn_down_exps.weight": "down"}

# テスト（GGUF を書く側）が使う逆引き。
HF_TO_GGUF_TOP = {v: k for k, v in _TOP.items()}
HF_TO_GGUF_BLK = {v: k for k, v in _BLK.items()}
HF_TO_GGUF_EXPERTS = {"experts.gate_up_proj": "ffn_gate_up_exps.weight", "experts.down_proj": "ffn_down_exps.weight"}


def _field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    v = f.parts[f.data[0]]
    if v.dtype.kind == "u" and v.dtype.itemsize == 1 and len(f.data) == 1 and f.types and f.types[0].name == "STRING":
        return bytes(v).decode("utf-8")
    return v.tolist()[0] if v.size == 1 else v.tolist()


def q4_0_unpack(raw: np.ndarray, k: int):
    """Q4_0 の生ブロック（uint8 [..., K/32*18]）→ (q int32 [..., K] (0〜15), d float32 [..., K/32])。

    ggml の Q4_0: 1 ブロック = fp16 の d ＋ 16 バイト。バイト j の下位 4bit が要素 j、上位 4bit が
    要素 j+16。値は (q - 8) * d。
    """
    lead = raw.shape[:-1]
    blocks = raw.reshape(*lead, k // Q4_0_BLOCK, Q4_0_BLOCK_BYTES)
    d = blocks[..., :2].copy().view(np.float16).reshape(*lead, k // Q4_0_BLOCK).astype(np.float32)
    qs = blocks[..., 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(*lead, k)
    return q.astype(np.int32), d


def q4_0_to_strata(q: np.ndarray, d: np.ndarray):
    """Q4_0 の (q, d) を strata-lite の (q int32 [N, K], sz bf16 [K/32, N, 2]) に詰め替える（格子は同じ）。"""
    qt = torch.from_numpy(np.ascontiguousarray(q))
    s = torch.from_numpy(np.ascontiguousarray(d)).transpose(0, 1)  # [K/32, N]
    sz = torch.stack([s, torch.zeros_like(s)], dim=-1).contiguous().to(torch.bfloat16)
    return qt, sz


def _to_float(tensor) -> np.ndarray:
    """GGUF テンソルを float32 の numpy（HF の並び＝reader が返す並び）に展開する。"""
    from gguf import GGMLQuantizationType, quants

    data = np.asarray(tensor.data)
    if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
        return data.astype(np.float32)
    return quants.dequantize(data, tensor.tensor_type)


def resolve_gguf(src: str, gguf_file: str | None, log=print) -> str:
    """ローカルの .gguf パスはそのまま。HF repo-id なら `--gguf-file` のファイルだけ取得する。"""
    if os.path.isfile(src):
        return src
    if not gguf_file:
        raise FileNotFoundError(f"{src} is not a file; for a HF repo-id pass --gguf-file <name.gguf>")
    from huggingface_hub import hf_hub_download

    log(f"[convert] downloading {src}/{gguf_file}")
    return hf_hub_download(src, gguf_file)


def resolve_base(base: str, log=print) -> str:
    """config.json / tokenizer を取る元（ローカルディレクトリか HF repo-id）。重みは取らない。"""
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download

    log(f"[convert] downloading config/tokenizer from {base}")
    return snapshot_download(base, allow_patterns=["*.json", "*.jinja", "tokenizer*"])


def convert_gguf(gguf_path: str, out: str, *, base: str, group: int = 32, scheme: str = "q4_0", log=print) -> str:
    from gguf import GGMLQuantizationType, GGUFReader

    from .convert import FORMAT_VERSION, text_config_dict

    t0 = time.perf_counter()
    reader = GGUFReader(gguf_path)
    arch = _field(reader, "general.architecture")
    if arch != ARCH:
        raise ValueError(f"{gguf_path}: architecture is {arch!r}, expected {ARCH!r}")
    base_dir = resolve_base(base, log)
    tc = text_config_dict(base_dir)
    if not tc.get("enable_moe_block"):
        raise ValueError("base config has no MoE block (enable_moe_block is false)")
    hidden, inter, num_experts = tc["hidden_size"], tc["moe_intermediate_size"], tc["num_experts"]
    n_layers = tc["num_hidden_layers"]
    checks = {
        "gemma4.embedding_length": hidden,
        "gemma4.expert_feed_forward_length": inter,
        "gemma4.expert_count": num_experts,
        "gemma4.block_count": n_layers,
    }
    for key, want in checks.items():
        got = _field(reader, key)
        if got is not None and int(got) != want:
            raise ValueError(f"{key}={got} in the GGUF but {want} in {base} config.json; wrong --base?")
    if scheme != "q4_0":
        raise ValueError("GGUF import keeps the Q4_0 grid; only scheme='q4_0' is supported")
    for name, k in (("hidden_size", hidden), ("moe_intermediate_size", inter)):
        if k % group:
            raise ValueError(f"{name}={k} is not a multiple of group={group}")
    os.makedirs(out, exist_ok=True)

    dense: dict[str, torch.Tensor] = {}
    experts: dict[int, dict[str, torch.Tensor]] = {}
    requantized = 0
    source_types = set()
    for tensor in reader.tensors:
        name = tensor.name
        source_types.add(tensor.tensor_type.name)
        m = _BLK_RE.match(name)
        if m:
            layer, rest = int(m.group(1)), m.group(2)
            if rest in _EXPERTS:
                kind = _EXPERTS[rest]
                data = np.asarray(tensor.data)
                if data.shape[0] != num_experts:
                    raise ValueError(f"{name}: expected {num_experts} experts, got {data.shape}")
                n = data.shape[1]
                k = inter if kind == "down" else hidden
                packed, szs = [], []
                if tensor.tensor_type == GGMLQuantizationType.Q4_0 and group == Q4_0_BLOCK:
                    for e in range(num_experts):  # 1 個ずつ: 詰め替えだけなので再量子化しない
                        q, d = q4_0_unpack(data[e], k)
                        qt, sz = q4_0_to_strata(q, d)
                        packed.append(int4.cpu_pack(qt))
                        szs.append(sz)
                else:
                    requantized += 1
                    w = torch.from_numpy(_to_float(tensor))
                    for e in range(num_experts):
                        q, sz = int4.quantize(w[e], group, scheme)
                        packed.append(int4.cpu_pack(q))
                        szs.append(sz)
                experts.setdefault(layer, {})[kind] = torch.stack(packed)
                experts[layer][kind + "_sz"] = torch.stack(szs)
                log(f"[convert] layer {layer:3d} {kind:7s} [{num_experts}, {n}, {k}] {tensor.tensor_type.name} "
                    f"({time.perf_counter() - t0:.0f}s)")
                if len(experts[layer]) == 4:
                    save_file(experts.pop(layer), os.path.join(out, f"experts-{layer:03d}.safetensors"))
                continue
            if rest not in _BLK:
                raise ValueError(f"{name}: unknown tensor for {ARCH} (update gguf_import._BLK)")
            hf = f"model.layers.{layer}.{_BLK[rest]}"
        elif name in _TOP_SKIP:
            continue
        elif name in _TOP:
            hf = _TOP[name]
        else:
            raise ValueError(f"{name}: unknown tensor for {ARCH} (update gguf_import._TOP)")
        dense[hf] = torch.from_numpy(_to_float(tensor)).to(torch.bfloat16).contiguous()
    if experts:
        raise ValueError(f"incomplete expert tensors for layers {sorted(experts)}")
    emb = dense.get("model.embed_tokens.weight")
    if emb is None:
        raise ValueError("token_embd not found in the GGUF")
    if emb.shape[0] != tc["vocab_size"]:
        raise ValueError(f"token_embd has {emb.shape[0]} rows but config vocab_size is {tc['vocab_size']}")
    save_file(dense, os.path.join(out, "dense.safetensors"))
    for pattern in _COPY_PATTERNS:
        for p in glob.glob(os.path.join(base_dir, pattern)):
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(out, os.path.basename(p)))
    layers = sorted(int(os.path.basename(p)[8:11]) for p in glob.glob(os.path.join(out, "experts-*.safetensors")))
    meta = {
        "format_version": FORMAT_VERSION,
        "group": group,
        "scheme": scheme,
        "hidden_size": hidden,
        "moe_intermediate_size": inter,
        "num_experts": num_experts,
        "moe_layers": layers,
        "cpu_layout": int4.cpu_layout_fingerprint([(2 * inter, hidden), (hidden, inter)]),
        "source": os.path.abspath(gguf_path),
        "source_format": "gguf",
        "source_quant_types": sorted(source_types),
        "experts_requantized": requantized,
        "base": base,
        "torch": torch.__version__,
    }
    with open(os.path.join(out, "strata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if requantized:
        log(f"[convert] note: {requantized} expert tensors were not Q4_0 and were re-quantized to the q4_0 grid")
    log(f"[convert] done in {time.perf_counter() - t0:.0f}s -> {out}")
    return out
