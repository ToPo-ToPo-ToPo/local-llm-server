"""小さなランダム初期化の Gemma 4 MoE を、本物と同じ保存形式（Gemma4ForConditionalGeneration）で作る。"""
from __future__ import annotations

import pytest
import torch

VOCAB = 512


def tiny_text_config(**over):
    from transformers import Gemma4TextConfig

    kw = dict(vocab_size=VOCAB, hidden_size=128, intermediate_size=128, num_hidden_layers=4,
              num_attention_heads=2, num_key_value_heads=1, head_dim=32, enable_moe_block=True,
              num_experts=8, top_k_experts=2, moe_intermediate_size=64, hidden_size_per_layer_input=0,
              sliding_window=16, vocab_size_per_layer_input=VOCAB, max_position_embeddings=256,
              pad_token_id=0, eos_token_id=1, bos_token_id=2)
    kw.update(over)
    return Gemma4TextConfig(**kw)


def _tokenizer():
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    words = [f"w{i}" for i in range(VOCAB - len(specials) - 3)] + ["user", "model", "hello"]
    vocab = {t: i for i, t in enumerate(specials + words)}
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                   bos_token="<bos>", unk_token="<unk>")
    fast.chat_template = (
        "{{ bos_token }}{% for m in messages %}<start_of_turn> {{ m['role'] }} {{ m['content'] }} "
        "<end_of_turn> {% endfor %}{% if add_generation_prompt %}<start_of_turn> model {% endif %}")
    return fast


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """(ディレクトリ, 参照用 Gemma4ForCausalLM)。重みはランダム（専門家ごとに値を散らす）。"""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    torch.manual_seed(0)
    tc = tiny_text_config()
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=tc.to_dict(), vision_config=None,
                                                        audio_config=None)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.08)
            elif "per_expert_scale" in name:
                p.uniform_(0.5, 1.5)
        for layer in model.model.language_model.layers:
            # ルーターが偏るようにして、hot 配置で効き目の差が出るようにする
            layer.router.proj.weight[:2] *= 4.0
    model = model.to(torch.bfloat16)  # 本物の Gemma 4 と同じく bf16 で保存する
    d = tmp_path_factory.mktemp("tiny-gemma4")
    model.save_pretrained(d)
    _tokenizer().save_pretrained(d)
    return str(d), model


@pytest.fixture(scope="session")
def converted(tiny_checkpoint, tmp_path_factory):
    from strata_lite.convert import convert

    src, _ = tiny_checkpoint
    out = tmp_path_factory.mktemp("tiny-strata")
    convert(src, str(out), group=32, scheme="q4_0", log=lambda *_: None)
    return str(out)
