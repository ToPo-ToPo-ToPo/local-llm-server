#!/usr/bin/env python
"""公式 Inkling（thinkingmachines/Inkling*）を MLX 量子化モデルへ変換する。

**mlx-vlm 0.6.9 以降が前提**（pyproject でピン済み）。0.6.9 は公式 config をそのまま読むので、
変換は素の `mlx_vlm.convert` と実質同じ——本スクリプトがやるのは tokenizer_config への
pad/eos 補完と、古い mlx-vlm で走らせてしまう事故のガードだけで、config には一切触れない。

    python scripts/convert_inkling.py --hf-path thinkingmachines/Inkling-Small \\
        --mlx-path ~/mlx_models/Inkling-Small-mlx-4bit --q-bits 4

## 0.6.9 未満を使ってはいけない

0.6.7 / 0.6.8 は Inkling を**公開ローダー経路で読めず**（sub-config の再エクスポート漏れ・
inkling_mm_model の別名未登録・prompt_utils.MODEL_CONFIG 未登録）、当時はキー名を翻訳した
config を食わせる回避が要った。さらに深刻なのは、当時の実装が MoE の `mlp.global_scale`
（50 件）と `mlp.gate.bias`（40 件）を持っておらず、**変換時にこれらの重みを黙って落として
いた**こと。0.6.9 はこれらを `switch_mlp.gate_scale` / `out_scale` として扱うので、
0.6.9 で作り直さないと**本来の推論と異なるモデル**になる（形状エラーも出ないので気づけない）。

## tokenizer_config の補完（唯一必要な手当て）

公式は `TokenizersBackend` 独自形式で `pad_token` / `eos_token` をどちらも設定していない。
そのままだと transformers のトークナイザが padding 要求で ValueError を投げるため、変換後の
出力に補う（トークナイザ上の既存 ID を指すだけで、語彙は変えない）。

## MTP ドラフター（speculative decoding）

本体とは別に、公式 bf16 に内蔵された `model.mtp.*` から切り出す:

    python -m mlx_vlm.speculative.drafters.inkling_mtp.split \\
        --model thinkingmachines/Inkling-Small --output ~/mlx_models/Inkling-Small-MTP-bf16

量子化後のリポジトリからは mtp 重みが落ちているので、必ず公式 bf16 から切り出すこと。
（2026-08 時点では、切り出せてもドラフター経路が上流バグで動かない。→ gateway.toml）
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PAD_TOKEN = "<|endoftext|>"                      # 199999
EOS_TOKEN = "<|content_model_end_sampling|>"     # 200006 = config.json の eos_token_id
MIN_MLX_VLM = (0, 6, 9)


def patch_tokenizer_config(path: Path) -> bool:
    """pad_token / eos_token を補う。既に入っていれば何もしない（冪等）。"""
    if not path.exists():
        return False
    tok = json.loads(path.read_text())
    if tok.get("pad_token") == PAD_TOKEN and tok.get("eos_token") == EOS_TOKEN:
        return False
    tok["pad_token"] = PAD_TOKEN
    tok["eos_token"] = EOS_TOKEN
    path.write_text(json.dumps(tok, indent=2))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-path", default="thinkingmachines/Inkling-Small",
                        help="公式リポジトリの HF id かローカルパス（事前に hf download 済みであること）")
    parser.add_argument("--mlx-path", required=True, help="変換先ディレクトリ")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    args = parser.parse_args()

    import mlx_vlm
    from mlx_vlm.convert import convert
    from mlx_vlm.utils import get_model_path

    version = getattr(mlx_vlm, "__version__", "0")
    try:
        parsed = tuple(int(p) for p in version.split(".")[:3])
    except ValueError:
        parsed = (0,)
    if parsed < MIN_MLX_VLM:
        raise SystemExit(
            f"mlx-vlm {version} は Inkling の変換に使えません（0.6.9 以降が必要）。"
            "0.6.8 以前は MoE の global_scale / gate.bias を落とし、"
            "形状エラーも出ないまま本来と異なるモデルになります。"
        )

    source = Path(get_model_path(args.hf_path))
    mlx_path = Path(os.path.expanduser(args.mlx_path))

    # 変換元（HF キャッシュ）は共有物なので触らない。convert が config/tokenizer 類を
    # 出力へコピーするので、補完は**変換後**の出力側に対して行う。
    print(f"[convert] {source} -> {mlx_path}  ({args.q_bits}bit / group {args.q_group_size})")
    convert(
        hf_path=str(source),
        mlx_path=str(mlx_path),
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )
    if patch_tokenizer_config(mlx_path / "tokenizer_config.json"):
        print("[patch] tokenizer_config.json に pad_token / eos_token を補完しました")
    print(f"[done] {mlx_path}")
    print("ゲートウェイには絶対パスで登録すること（~ は subprocess で展開されない）。")


if __name__ == "__main__":
    main()
