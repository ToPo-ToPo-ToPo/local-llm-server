"""HF の safetensors（Gemma 4 MoE）を strata-lite 用の形式に変換する。

出力ディレクトリ:
- strata.json                 … 形式の情報（群サイズ・量子化方式・CPU パック形式の指紋など）
- dense.safetensors           … エキスパート以外のテキスト部分（bf16。名前は Gemma4ForCausalLM 基準）
- experts-LLL.safetensors     … 層ごとのエキスパート（CPU カーネル形式の int4）
- config.json / tokenizer* / chat_template* / generation_config.json … 元からそのまま複製

エキスパートは 1 個ずつ読み出して量子化するので、変換中のメモリは「エキスパート以外の重み
（数 GB）＋エキスパート数個分」で済む。CPU パック形式は CPU の命令セットに依存しうるので、
変換は実際に動かす PC で行う（形式が違えばロード時に指紋で検出して止める）。
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import int4

FORMAT_VERSION = 1
_SKIP_PARTS = ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector")
_COPY_PATTERNS = ("config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
                  "chat_template*", "added_tokens.json")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def text_key(key: str) -> str | None:
    """チェックポイントの名前を Gemma4ForCausalLM の名前に直す。テキスト以外は None。"""
    if any(part in key for part in _SKIP_PARTS):
        return None
    for prefix in ("model.language_model.", "language_model.model."):
        if key.startswith(prefix):
            return "model." + key[len(prefix):]
    if key.startswith("language_model.lm_head."):
        return key[len("language_model."):]
    if key.startswith(("model.", "lm_head.")):
        return key
    return None


def resolve_source(src: str, log=print) -> str:
    """ローカルのディレクトリならそのまま、そうでなければ HF の repo-id として取得する。"""
    if os.path.isdir(src):
        return src
    from huggingface_hub import snapshot_download

    log(f"[convert] downloading {src} (safetensors + tokenizer only)")
    return snapshot_download(src, allow_patterns=["*.safetensors", "*.json", "*.jinja", "tokenizer*"])


def _weight_files(src: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {src}")
    return files


def text_config_dict(src: str) -> dict:
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def convert(src: str, out: str, *, group: int = 32, scheme: str = "q4_0", log=print) -> str:
    src = resolve_source(src, log)
    tc = text_config_dict(src)
    if not tc.get("enable_moe_block"):
        raise ValueError("this model has no MoE block (enable_moe_block is false)")
    hidden, inter = tc["hidden_size"], tc["moe_intermediate_size"]
    num_experts = tc["num_experts"]
    for name, k in (("hidden_size", hidden), ("moe_intermediate_size", inter)):
        if k % group:
            raise ValueError(f"{name}={k} is not a multiple of group={group}")
    os.makedirs(out, exist_ok=True)

    dense: dict[str, torch.Tensor] = {}
    experts: dict[int, dict[str, torch.Tensor]] = {}
    t0 = time.perf_counter()
    for path in _weight_files(src):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open は keys() しか持たない
                name = text_key(key)
                if name is None:
                    continue
                m = _LAYER_RE.search(name)
                if m and name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                    layer = int(m.group(1))
                    kind = "gate_up" if name.endswith("gate_up_proj") else "down"
                    sl = f.get_slice(key)
                    shape = sl.get_shape()
                    if shape[0] != num_experts:
                        raise ValueError(f"{key}: expected {num_experts} experts, got shape {shape}")
                    packed, szs = [], []
                    for e in range(num_experts):
                        q, sz = int4.quantize(sl[e], group, scheme)
                        packed.append(int4.cpu_pack(q))
                        szs.append(sz)
                    experts.setdefault(layer, {})[kind] = torch.stack(packed)
                    experts[layer][kind + "_sz"] = torch.stack(szs)
                    log(f"[convert] layer {layer:3d} {kind:7s} {shape} ({time.perf_counter() - t0:.0f}s)")
                    if len(experts[layer]) == 4:
                        save_file(experts.pop(layer), os.path.join(out, f"experts-{layer:03d}.safetensors"))
                else:
                    dense[name] = f.get_tensor(key).to(torch.bfloat16).contiguous()
    if experts:
        raise ValueError(f"incomplete expert tensors for layers {sorted(experts)}")
    if "model.embed_tokens.weight" not in dense:
        raise ValueError("embed_tokens not found (unexpected checkpoint layout)")
    save_file(dense, os.path.join(out, "dense.safetensors"))
    for pattern in _COPY_PATTERNS:
        for p in glob.glob(os.path.join(src, pattern)):
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
        "source": src,
        "torch": torch.__version__,
    }
    with open(os.path.join(out, "strata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"[convert] done in {time.perf_counter() - t0:.0f}s -> {out}")
    return out
