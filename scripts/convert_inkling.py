#!/usr/bin/env python
"""公式 Inkling（thinkingmachines/Inkling*）を mlx-vlm が読める MLX 量子化モデルへ変換する。

mlx-vlm 0.6.7 / 0.6.8 の models/inkling は「キー名を変換済みの config」を前提にした実装で、
公式リポジトリの config.json をそのまま `mlx_vlm.convert` へ渡すと失敗する。本スクリプトは
重みをコピーせずに（HF キャッシュへの symlink で）翻訳済み config を持つステージング
ディレクトリを作り、そこから変換する。

必要な翻訳（すべて実機で切り分け済み）:

  model_type          inkling_mm_model → inkling
                      公式は inkling_mm_model。mlx-vlm の MODEL_REMAPPING に別名が無く、
                      get_model_and_args が解決できない。
  text_config         dense_intermediate_size → intermediate_size   （dense MLP 幅）
                      intermediate_size       → moe_intermediate_size（expert 幅）
                      公式と mlx-vlm で意味が入れ替わっている。mlx-vlm の既定値
                      （24576 / 3072）が Inkling-975B の dense/expert 幅と一致することから確認。
                      翻訳しないと layers.0.mlp.gate_proj の形状が合わず ValueError。
  vision/audio_config decoder_dmodel → text_hidden_size、n_channels → num_channels
                      ModelConfig.__post_init__ は text_hidden_size を text_config から
                      配線するが、直後に utils.update_module_configs が VisionConfig /
                      AudioConfig を作り直して配線を捨てる（既定 6144 に戻る）ため明示が要る。
  image_token_id      200054 / audio_token_id 200053（= mlx-vlm の既定。翻訳不要）
                      トークナイザ上の名前は <|unused_200054|> / <|unused_200053|> だが、
                      transformers の InklingProcessor はこれを画像・音声パッチの
                      プレースホルダとして再利用する（実測: 640x480 の画像で 200054 が
                      204 個 = pixel_values の 204 パッチと一致）。種別マーカーの
                      <|content_image|>(200005) は 1 個しか出ないので、そちらを
                      image_token_id にすると特徴が一切差し込まれず、モデルが
                      テキストだけから幻覚を返す（エラーにならないので気付きにくい）。
  tokenizer_config    pad_token / eos_token を追加。公式は TokenizersBackend 独自形式で
                      どちらも未設定のため、transformers のトークナイザが padding 要求で
                      ValueError を投げる。

これとは別に、mlx-vlm の models/inkling/__init__.py が sub-config を再エクスポートしない
問題があり、そちらは local_llm_server/_mlx_vlm_shims.py が起動時に補う（変換時は本
スクリプトが同じパッチを当てる）。

使い方:
    python scripts/convert_inkling.py --hf-path thinkingmachines/Inkling-Small \
        --mlx-path ~/mlx_models/Inkling-Small-mlx-4bit --q-bits 4

MTP ドラフター（speculative decoding）は公式 bf16 に内蔵された model.mtp.* から別途切り出す:
    python -m mlx_vlm.speculative.drafters.inkling_mtp.split \
        --model thinkingmachines/Inkling-Small --output ~/mlx_models/Inkling-Small-MTP-bf16
（量子化後のリポジトリからは mtp 重みが落ちているので、必ず公式 bf16 から切り出すこと。）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PAD_TOKEN = "<|endoftext|>"                      # 199999
EOS_TOKEN = "<|content_model_end_sampling|>"     # 200006 = config.json の eos_token_id


def translate_config(config: dict) -> dict:
    """公式 config.json を mlx-vlm の models/inkling が期待するスキーマへ翻訳する。"""
    c = json.loads(json.dumps(config))  # 深いコピー

    if c.get("model_type") not in ("inkling_mm_model", "inkling"):
        raise SystemExit(f"Inkling の config ではありません: model_type={c.get('model_type')!r}")
    c["model_type"] = "inkling"

    text = c["text_config"]
    if "dense_intermediate_size" in text:
        text["moe_intermediate_size"] = text["intermediate_size"]
        text["intermediate_size"] = text.pop("dense_intermediate_size")

    vision = c["vision_config"]
    if "n_channels" in vision:
        vision["num_channels"] = vision.pop("n_channels")
    if "decoder_dmodel" in vision:
        vision["text_hidden_size"] = vision.pop("decoder_dmodel")

    audio = c["audio_config"]
    if "decoder_dmodel" in audio:
        audio["text_hidden_size"] = audio.pop("decoder_dmodel")

    # image_token_id / audio_token_id は触らない。mlx-vlm の既定（200054 / 200053）が
    # InklingProcessor の出すプレースホルダと一致する（docstring 参照）。
    return c


def stage(source: Path, staging: Path) -> Path:
    """重みは symlink のまま、翻訳した config.json / tokenizer_config.json だけ実体で置く。"""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rewritten = {"config.json", "tokenizer_config.json"}
    for entry in source.iterdir():
        if entry.name in rewritten:
            continue
        (staging / entry.name).symlink_to(entry.resolve())

    with open(source / "config.json") as f:
        config = translate_config(json.load(f))
    with open(staging / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(source / "tokenizer_config.json") as f:
        tok = json.load(f)
    tok["pad_token"] = PAD_TOKEN
    tok["eos_token"] = EOS_TOKEN
    with open(staging / "tokenizer_config.json", "w") as f:
        json.dump(tok, f, indent=2)

    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-path", default="thinkingmachines/Inkling-Small",
                        help="公式リポジトリの HF id かローカルパス（事前に hf download 済みであること）")
    parser.add_argument("--mlx-path", required=True, help="変換先ディレクトリ")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--staging", default=None,
                        help="ステージングディレクトリ（既定: <mlx-path>-src）")
    args = parser.parse_args()

    from mlx_vlm.utils import get_model_path  # 変換環境でだけ import する

    source = Path(get_model_path(args.hf_path))
    mlx_path = Path(os.path.expanduser(args.mlx_path))
    staging = Path(os.path.expanduser(args.staging)) if args.staging \
        else mlx_path.with_name(mlx_path.name + "-src")

    print(f"[stage] {source} -> {staging}")
    stage(source, staging)

    # サーバー起動時と同じシム（sub-config の再エクスポート補完）を変換にも当てる。
    from local_llm_server._mlx_vlm_shims import apply as apply_shims
    from mlx_vlm.convert import convert

    apply_shims()
    print(f"[convert] {staging} -> {mlx_path}  ({args.q_bits}bit / group {args.q_group_size})")
    convert(
        hf_path=str(staging),
        mlx_path=str(mlx_path),
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )

    # 変換後の出力にも pad/eos を残す（convert は元の tokenizer_config.json をコピーする）。
    out_tok = mlx_path / "tokenizer_config.json"
    if out_tok.exists():
        with open(out_tok) as f:
            tok = json.load(f)
        tok["pad_token"] = PAD_TOKEN
        tok["eos_token"] = EOS_TOKEN
        with open(out_tok, "w") as f:
            json.dump(tok, f, indent=2)

    print(f"[done] {mlx_path}")
    print("ゲートウェイには絶対パスで登録すること（~ は subprocess で展開されない）。")


if __name__ == "__main__":
    main()
