"""scripts/check_mtp.py の出力解析と判定（モデルは動かさない）。"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# mlx-vlm 0.7.10 の `mlx_vlm.generate --verbose` の実出力（MTP あり）。
MTP_OUT = """Loading drafter (mtp): ToPo-ToPo/Qwen3.6-27B-MTP-bf16
==========
Files: []

Prompt: <|im_start|>user
Say hi<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?
==========
Prompt: 14 tokens, 63.552 tokens-per-sec
Generation: 10 tokens, 45.068 tokens-per-sec
Peak memory: 17.706 GB
Speculative decoding: 2.50 accepted tokens/round (1.50 accepted drafts/round, 75.0% of drafted, avg draft 2.00) over 4 rounds
"""


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_mtp", ROOT / "scripts/check_mtp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(text="same", tps=30.0, acceptance=None):
    return {"text": text, "tokens": 10, "tps": tps, "accepted_per_round": None,
            "acceptance": acceptance}


def test_parse_mtp_output(check):
    r = check.parse_output(MTP_OUT)
    assert r["tokens"] == 10 and r["tps"] == pytest.approx(45.068)
    assert r["acceptance"] == pytest.approx(75.0)
    assert r["accepted_per_round"] == pytest.approx(2.5)
    assert r["text"].endswith("Hi! How can I help you today?")


def test_parse_without_speculative_line(check):
    r = check.parse_output(MTP_OUT.split("Speculative")[0])
    assert r["acceptance"] is None and r["tps"] == pytest.approx(45.068)


def test_ok_when_faster_and_identical(check):
    assert check.judge(result(), result(tps=55.0, acceptance=90.0), None, 10) == ("OK", [])


def test_ng_when_mtp_not_engaged(check):
    level, problems = check.judge(result(), result(tps=55.0), None, 10)
    assert level == "NG" and "効いていない" in problems[0]


def test_ng_when_unparseable(check):
    level, _ = check.judge(result(tps=None), result(tps=55.0, acceptance=90.0), None, 10)
    assert level == "NG"


def test_ng_when_slower(check):
    level, problems = check.judge(result(tps=30.0), result(tps=25.0, acceptance=40.0), None, 10)
    assert level == "NG" and "遅い" in problems[0]


def test_ng_on_acceptance_drop_against_baseline(check):
    prev = {"acceptance": 90.8, "mlx_vlm": "0.7.10"}
    level, problems = check.judge(result(), result(tps=40.0, acceptance=70.0), prev, 10)
    assert level == "NG" and "0.7.10" in problems[0]
    # 閾値未満の揺れは許容する。
    assert check.judge(result(), result(tps=40.0, acceptance=85.0), prev, 10)[0] == "OK"


def test_warn_on_output_divergence(check):
    level, problems = check.judge(result("abcdef"), result("abcxyz", 55.0, 90.0), None, 10)
    assert level == "WARN" and "3 文字目" in problems[0]
    assert "本体: def" in problems[0] and "MTP : xyz" in problems[0]


def test_first_divergence(check):
    assert check.first_divergence("abc", "abc") is None
    assert check.first_divergence("abc", "abcd") == 3
    assert check.first_divergence("abc", "axc") == 1
