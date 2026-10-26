"""strata-lite のコマンド。

  python -m strata_lite convert <HF repo-id かディレクトリ> <出力先>
  python -m strata_lite plan    <変換済み> [--vram-gb 8]
  python -m strata_lite bench   <変換済み> [--compare]
  python -m strata_lite chat    <変換済み>
  python -m strata_lite serve   <変換済み> [--port 8090]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BENCH_PROMPTS = [
    "日本の四季について、それぞれの特徴を説明してください。",
    "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring.",
    "ローカルLLMを動かすときに、GPUのメモリが足りない場合の対策を箇条書きで挙げてください。",
    "Explain the difference between TCP and UDP to a beginner.",
    "次の文章を英語に翻訳してください：明日は雨が降るので、傘を持って出かけましょう。",
    "SQL で、注文テーブルから顧客ごとの合計金額を求めるクエリを書いて、説明してください。",
]

# bench --compare で比べる組み合わせ（最初の 1 つが llama.cpp --n-cpu-moe 相当）
COMPARE_MODES = [("layer", "cpu"), ("hot", "cpu"), ("hot", "transfer")]


def _engine_args(p):
    g = p.add_argument_group("engine")
    g.add_argument("--device", default=None, help="cuda / cpu（既定: GPU があれば cuda）")
    g.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    g.add_argument("--placement", default="hot", choices=["hot", "layer"])
    g.add_argument("--miss", default="cpu", choices=["cpu", "transfer"])
    g.add_argument("--expert-vram-gb", type=float, default=None,
                   help="エキスパート用に使う VRAM（既定: 空き VRAM − reserve）")
    g.add_argument("--reserve-gb", type=float, default=1.5, help="KV キャッシュや作業用に残す VRAM")
    g.add_argument("--threads", type=int, default=None, help="CPU スレッド数（既定: 物理コア数）")
    g.add_argument("--no-pin", action="store_true", help="RAM のエキスパートを固定メモリにしない")
    g.add_argument("--kernel", default="auto", choices=["auto", "tinygemm", "dequant", "cpu"])
    g.add_argument("--gpu-min-tokens", type=int, default=4,
                   help="この数以上のトークンが来たエキスパートは GPU へ転送して計算する（プリフィル向け）")
    g.add_argument("--rebalance-every", type=int, default=16, help="hot 配置で入れ替えを見直す間隔（トークン）")
    g.add_argument("--max-swaps", type=int, default=64, help="1 回の見直しで入れ替える上限")
    g.add_argument("--decay", type=float, default=0.95, help="使用回数の減衰（見直しごと）")
    g.add_argument("--no-profile", action="store_true", help="保存済みの使用回数を使わない")


def _engine(a):
    from .engine import Engine

    return Engine(a.model, device=a.device, dtype=a.dtype, placement=a.placement, miss=a.miss,
                  expert_vram_gb=a.expert_vram_gb, reserve_gb=a.reserve_gb, threads=a.threads,
                  pin=not a.no_pin, kernel=a.kernel, gpu_min_tokens=a.gpu_min_tokens,
                  rebalance_every=a.rebalance_every, max_swaps=a.max_swaps, decay=a.decay,
                  use_profile=not a.no_profile, log=lambda *x: print(*x, file=sys.stderr, flush=True))


def cmd_convert(a):
    from .convert import convert

    convert(a.src, a.out, group=a.group, scheme=a.scheme)


def plan(model_dir: str, vram_gb: float, reserve_gb: float, context_gb: float = 0.5) -> dict:
    """ロードせずに、GPU と RAM の使い方を見積もる。"""
    import torch
    import torch.nn as nn
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    with open(os.path.join(model_dir, "strata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = AutoConfig.from_pretrained(model_dir).get_text_config()
    g = meta["group"]
    per_w = 0.5 + 4 / g  # int4 本体 + 群ごとの bf16 の s と z
    h, i, e = meta["hidden_size"], meta["moe_intermediate_size"], meta["num_experts"]
    n_layers = len(meta["moe_layers"])
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg)
    dense = 0.0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name != "lm_head":
            n = mod.in_features * mod.out_features
            dense += n * 2 if name.endswith("router.proj") else n * per_w
    dense += cfg.vocab_size * h * per_w  # lm_head
    dense += sum(p.numel() for n, p in model.named_parameters() if p.dim() == 1) * 2
    slot = (3 * h * i) * per_w
    free = vram_gb * 2**30 - dense - reserve_gb * 2**30 - context_gb * 2**30
    slots = max(0, min(int(free // slot), n_layers * e))
    total = n_layers * e
    return {
        "dense_gpu_gb": dense / 2**30,
        "expert_slot_mb": slot / 2**20,
        "experts_total": total,
        "experts_ram_gb": total * slot / 2**30,
        "embedding_ram_gb": cfg.vocab_size * h * 2 / 2**30,
        "gpu_slots": slots,
        "gpu_fraction": slots / total,
        "layer_mode_gpu_layers": slots // e,
        "active_experts_per_token": cfg.top_k_experts * n_layers,
    }


def cmd_plan(a):
    p = plan(a.model, a.vram_gb, a.reserve_gb)
    print(f"GPU {a.vram_gb:g} GB の見積もり（予備 {a.reserve_gb:g} GB + CUDA 文脈 0.5 GB を除く）")
    print(f"  エキスパート以外（GPU）: {p['dense_gpu_gb']:.2f} GB")
    print(f"  エキスパート 1 個      : {p['expert_slot_mb']:.2f} MB × {p['experts_total']} 個 "
          f"= RAM {p['experts_ram_gb']:.1f} GB（+ 単語埋め込み {p['embedding_ram_gb']:.1f} GB）")
    print(f"  GPU に載るエキスパート : {p['gpu_slots']} 個（{p['gpu_fraction']:.0%}）")
    print(f"  layer 配置なら          : 後ろから {p['layer_mode_gpu_layers']} 層を丸ごと GPU")
    print(f"  1 トークンで使う数      : {p['active_experts_per_token']} 個")


def _run_prompts(eng, prompts, tokens):
    rows = []
    for text in prompts:
        res = eng.chat([{"role": "user", "content": text}], max_new_tokens=tokens)
        rows.append(res)
    return rows


def cmd_bench(a):
    from .experts import CacheStats

    eng = _engine(a)
    prompts = (BENCH_PROMPTS * ((a.prompts // len(BENCH_PROMPTS)) + 1))[: a.prompts]
    modes = COMPARE_MODES if a.compare else [(a.placement, a.miss)]
    report = []
    for placement, miss in modes:
        if (placement, miss) != (eng.cache.placement, eng.cache.miss):
            eng.set_policy(placement, miss)
        if a.warmup:
            _run_prompts(eng, BENCH_PROMPTS[: a.warmup], min(a.tokens, 64))
        eng.cache.stats = CacheStats()
        rows = _run_prompts(eng, prompts, a.tokens)
        st = eng.cache.stats.as_dict()
        new = sum(r.new_tokens - 1 for r in rows)
        dec = sum(r.decode_seconds for r in rows)
        pre_tok = sum(r.prompt_tokens for r in rows)
        pre = sum(r.prefill_seconds for r in rows)
        item = {"placement": placement, "miss": miss, "gpu_slots": eng.cache.slots,
                "gpu_fraction": eng.cache.resident_fraction(),
                "decode_tps": new / dec if dec else 0.0, "prefill_tps": pre_tok / pre if pre else 0.0,
                "decode_gpu_hit_rate": st["decode"]["gpu_hit_rate"], "swaps": st["swaps"],
                "upload_mb": st["upload_mb"]}
        report.append(item)
        print(f"{placement:5s}/{miss:8s} 生成 {item['decode_tps']:6.2f} tok/s | プリフィル "
              f"{item['prefill_tps']:7.1f} tok/s | GPU ヒット率 {item['decode_gpu_hit_rate']:.0%} "
              f"（GPU に {item['gpu_fraction']:.0%}）| 入れ替え {item['swaps']} 回 / "
              f"{item['upload_mb']:.0f} MB", flush=True)
        if a.show_text:
            print("  例:", rows[0].text[:200].replace("\n", " "))
    eng.save_profile()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


def cmd_chat(a):
    eng = _engine(a)
    history = []
    print("終了は Ctrl-D（/stats で統計）", file=sys.stderr)
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if line.strip() == "/stats":
                print(json.dumps(eng.cache.stats.as_dict(), ensure_ascii=False, indent=2))
                continue
            history.append({"role": "user", "content": line})
            res = eng.chat(history, max_new_tokens=a.tokens, temperature=a.temperature)
            history.append({"role": "assistant", "content": res.text})
            print(res.text)
            print(f"[{res.decode_tps:.1f} tok/s, プリフィル {res.prefill_tps:.0f} tok/s]", file=sys.stderr)
    finally:
        eng.save_profile()


def cmd_serve(a):
    from .server import serve

    eng = _engine(a)
    httpd = serve(eng, a.host, a.port, a.model_id)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.save_profile()


def main(argv=None):
    p = argparse.ArgumentParser(prog="strata_lite", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="HF の safetensors を変換する")
    c.add_argument("src")
    c.add_argument("out")
    c.add_argument("--group", type=int, default=32, choices=[32, 64, 128])
    c.add_argument("--scheme", default="q4_0", choices=["q4_0", "asym"])
    c.set_defaults(fn=cmd_convert)

    pl = sub.add_parser("plan", help="GPU と RAM の使い方を見積もる")
    pl.add_argument("model")
    pl.add_argument("--vram-gb", type=float, default=8.0)
    pl.add_argument("--reserve-gb", type=float, default=1.5)
    pl.set_defaults(fn=cmd_plan)

    b = sub.add_parser("bench", help="生成速度と GPU ヒット率を測る")
    b.add_argument("model")
    b.add_argument("--tokens", type=int, default=128)
    b.add_argument("--prompts", type=int, default=6)
    b.add_argument("--warmup", type=int, default=2, help="測る前に流すプロンプト数（hot の学習用）")
    b.add_argument("--compare", action="store_true", help="layer/cpu・hot/cpu・hot/transfer を順に比べる")
    b.add_argument("--show-text", action="store_true")
    b.add_argument("--json", help="結果を JSON で保存する")
    _engine_args(b)
    b.set_defaults(fn=cmd_bench)

    ch = sub.add_parser("chat", help="対話して試す")
    ch.add_argument("model")
    ch.add_argument("--tokens", type=int, default=512)
    ch.add_argument("--temperature", type=float, default=0.0)
    _engine_args(ch)
    ch.set_defaults(fn=cmd_chat)

    s = sub.add_parser("serve", help="OpenAI 互換サーバーを立てる")
    s.add_argument("model")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--model-id", default=None)
    _engine_args(s)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
