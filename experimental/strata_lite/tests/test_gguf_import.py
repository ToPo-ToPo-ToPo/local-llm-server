"""GGUF からの取り込みが、同じ重みを HF safetensors から変換した結果と一致すること。

小さなランダム Gemma 4（conftest の tiny_checkpoint）を gguf-py の GGUFWriter で GGUF に書き、
`convert` に .gguf を渡して取り込む。エキスパートは Q4_0 で書く（本物の QAT 版と同じ）。
dense 側は 2 通り: 全部 F32（厳密一致と logits の一致を見る）と、一部を Q4_0（展開経路を通す）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from strata_lite import int4
from strata_lite.convert import convert, text_key
from strata_lite.gguf_import import (HF_TO_GGUF_BLK, HF_TO_GGUF_EXPERTS, HF_TO_GGUF_TOP, q4_0_to_strata,
                                     q4_0_unpack)

# dense 側で Q4_0 にして書くもの（展開経路の確認用）
_Q4_DENSE = ("self_attn.q_proj.weight", "mlp.down_proj.weight", "model.embed_tokens.weight")


def _gguf_name(hf: str) -> str | None:
    if hf in HF_TO_GGUF_TOP:
        return HF_TO_GGUF_TOP[hf]
    if hf == "lm_head.weight":
        return None  # 共有重み（tie）。llama.cpp も output を書かない
    assert hf.startswith("model.layers."), hf
    _, _, layer, rest = hf.split(".", 3)
    for suffix, gname in HF_TO_GGUF_EXPERTS.items():
        if rest == suffix:
            return f"blk.{layer}.{gname}"
    return f"blk.{layer}.{HF_TO_GGUF_BLK[rest]}"


def _write_gguf(src: str, path: str, q4_dense=()) -> str:
    import gguf
    from gguf import GGMLQuantizationType, quants
    from safetensors import safe_open

    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        tc = json.load(f)["text_config"]
    w = gguf.GGUFWriter(path, "gemma4")
    w.add_block_count(tc["num_hidden_layers"])
    w.add_embedding_length(tc["hidden_size"])
    w.add_expert_count(tc["num_experts"])
    w.add_expert_feed_forward_length(tc["moe_intermediate_size"])
    w.add_feed_forward_length(tc["intermediate_size"])
    # llama.cpp が生成する補助テンソル（HF には無い）。読み飛ばされることを確かめる
    w.add_tensor("rope_freqs.weight", np.ones(8, dtype=np.float32))
    for fn in sorted(p for p in os.listdir(src) if p.endswith(".safetensors")):
        with safe_open(os.path.join(src, fn), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                hf = text_key(key)
                gname = _gguf_name(hf) if hf else None
                if gname is None:
                    continue
                arr = f.get_tensor(key).float().numpy()
                if gname.endswith("_exps.weight") or any(hf.endswith(s) for s in q4_dense):
                    q = quants.quantize(arr, GGMLQuantizationType.Q4_0)
                    # 量子化済みデータはバイト形状のまま渡す（writer が要素数の形に直す）
                    w.add_tensor(gname, q, raw_dtype=GGMLQuantizationType.Q4_0)
                else:
                    w.add_tensor(gname, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture(scope="module")
def tiny_gguf(tiny_checkpoint, tmp_path_factory):
    src, _ = tiny_checkpoint
    return _write_gguf(src, str(tmp_path_factory.mktemp("tiny-gguf") / "tiny.gguf"))


@pytest.fixture(scope="module")
def tiny_gguf_q4(tiny_checkpoint, tmp_path_factory):
    src, _ = tiny_checkpoint
    return _write_gguf(src, str(tmp_path_factory.mktemp("tiny-gguf-q4") / "tiny.gguf"), _Q4_DENSE)


@pytest.fixture(scope="module")
def imported(tiny_gguf, tiny_checkpoint, tmp_path_factory):
    src, _ = tiny_checkpoint
    out = tmp_path_factory.mktemp("tiny-from-gguf")
    convert(tiny_gguf, str(out), base=src, log=lambda *_: None)
    return str(out)


@pytest.fixture(scope="module")
def imported_q4(tiny_gguf_q4, tiny_checkpoint, tmp_path_factory):
    src, _ = tiny_checkpoint
    out = tmp_path_factory.mktemp("tiny-from-gguf-q4")
    convert(tiny_gguf_q4, str(out), base=src, log=lambda *_: None)
    return str(out)


def test_q4_0_unpack_roundtrip():
    from gguf import GGMLQuantizationType, quants

    torch.manual_seed(0)
    w = (torch.randn(6, 64) * 0.1).numpy().astype(np.float32)
    raw = quants.quantize(w, GGMLQuantizationType.Q4_0)
    q, d = q4_0_unpack(raw, 64)
    ref = quants.dequantize(raw, GGMLQuantizationType.Q4_0)
    mine = (q - 8) * np.repeat(d, 32, axis=-1)
    assert np.array_equal(mine.astype(np.float32), ref)
    qt, sz = q4_0_to_strata(q, d)
    back = int4.dequantize(qt, sz, 32)
    # スケールだけ fp16 → bf16 に丸まる（相対 2^-9 以内）。ニブルはそのまま
    assert torch.allclose(back, torch.from_numpy(ref), rtol=4e-3, atol=1e-6)


def test_layout_and_metadata(imported, converted):
    meta = json.load(open(os.path.join(imported, "strata.json"), encoding="utf-8"))
    ref = json.load(open(os.path.join(converted, "strata.json"), encoding="utf-8"))
    assert meta["source_format"] == "gguf"
    assert meta["experts_requantized"] == 0
    assert "Q4_0" in meta["source_quant_types"]
    for k in ("group", "scheme", "hidden_size", "moe_intermediate_size", "num_experts", "moe_layers", "cpu_layout"):
        assert meta[k] == ref[k], k
    assert os.path.isfile(os.path.join(imported, "config.json"))
    assert any(n.startswith("tokenizer") for n in os.listdir(imported))


def test_dense_exact_when_source_is_float(imported, converted):
    a = load_file(os.path.join(imported, "dense.safetensors"))
    b = load_file(os.path.join(converted, "dense.safetensors"))
    assert set(a) == set(b), (set(a) ^ set(b))
    for name in b:
        assert a[name].shape == b[name].shape, name
        assert torch.equal(a[name], b[name]), name


def test_dense_q4_0_is_dequantized(imported_q4, converted):
    a = load_file(os.path.join(imported_q4, "dense.safetensors"))
    b = load_file(os.path.join(converted, "dense.safetensors"))
    assert set(a) == set(b)
    for name in b:
        x, y = a[name].float(), b[name].float()
        if not any(name.endswith(s) for s in _Q4_DENSE):
            assert torch.equal(x, y), name
            continue
        # Q4_0 の格子は (q - 8) * d、d = 絶対値最大 / -8。反対符号の端は 1 刻み（= |d|）までずれうる
        amax = y.reshape(y.shape[0], -1, 32).abs().amax(-1, keepdim=True).expand(-1, -1, 32).reshape(y.shape)
        assert ((x - y).abs() <= amax / 8 * 1.02 + 1e-3).all(), name
        assert not torch.equal(x, y), name  # 本当に展開経路を通った（元の bf16 と同一ではない）


def test_experts_match(imported, converted):
    meta = json.load(open(os.path.join(converted, "strata.json"), encoding="utf-8"))
    h, i, g = meta["hidden_size"], meta["moe_intermediate_size"], meta["group"]
    for lid in meta["moe_layers"]:
        a = load_file(os.path.join(imported, f"experts-{lid:03d}.safetensors"))
        b = load_file(os.path.join(converted, f"experts-{lid:03d}.safetensors"))
        for kind, (n, k) in (("gate_up", (2 * i, h)), ("down", (h, i))):
            perm = int4.cpu_pack_permutation(n, k)
            for e in range(a[kind].shape[0]):
                qa = int4.unpack_cpu_packed(a[kind][e], perm, n, k)
                qb = int4.unpack_cpu_packed(b[kind][e], perm, n, k)
                wa = int4.dequantize(qa, a[kind + "_sz"][e], g)
                wb = int4.dequantize(qb, b[kind + "_sz"][e], g)
                # 同じ格子（Q4_0）なので、違いはスケールの丸め（fp16 と bf16）だけ
                assert torch.allclose(wa, wb, rtol=1e-2, atol=1e-4), (lid, kind, e)


def test_engine_logits_match(imported, converted):
    from strata_lite.engine import Engine

    x = torch.randint(5, 500, (1, 12))
    a = Engine(imported, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0,
               placement="layer", use_profile=False, log=lambda *_: None).logits(x)
    b = Engine(converted, device="cpu", dtype="float32", kernel="dequant", expert_vram_gb=0,
               placement="layer", use_profile=False, log=lambda *_: None).logits(x)
    assert torch.allclose(a, b, rtol=2e-2, atol=2e-2)


def test_cli_convert_gguf(tiny_gguf, tiny_checkpoint, tmp_path):
    from strata_lite.__main__ import main

    src, _ = tiny_checkpoint
    out = tmp_path / "cli-out"
    main(["convert", tiny_gguf, str(out), "--base", src])
    assert (out / "strata.json").is_file() and (out / "dense.safetensors").is_file()


def test_wrong_base_is_rejected(tiny_gguf, tiny_checkpoint, tmp_path):
    src, _ = tiny_checkpoint
    bad = tmp_path / "bad-base"
    bad.mkdir()
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    cfg["text_config"]["num_experts"] += 1
    json.dump(cfg, open(bad / "config.json", "w", encoding="utf-8"))
    with pytest.raises(ValueError, match="expert_count"):
        convert(tiny_gguf, str(tmp_path / "out"), base=str(bad), log=lambda *_: None)
