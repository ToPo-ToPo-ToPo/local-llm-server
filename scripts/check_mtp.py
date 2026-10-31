"""mlx-vlm を更新したあとに MTP（speculative decoding）が劣化していないかを実測で確かめる。

Run with uv run --frozen python scripts/check_mtp.py [model ...]

ドラフターの重みは変換時の mlx-vlm の形式（重み名・ノルムの +1.0 補正・config 項目）で
焼き込まれており、実行時の mlx-vlm がその解釈を変えると**黙って**採択率が落ちる（本体が
検証するので出力は変わらず、遅くなるだけ）。検証ループ自体のバグなら出力が壊れる。
どちらもエラーにならないので、本体単独と MTP ありを同じ条件（greedy）で生成して比べる:

- NG: 推測デコードの統計が出ない（MTP が効いていない）／MTP ありの方が遅い／
      採択率が前回保存したベースラインより --max-drop ポイント以上落ちた。
- WARN: MTP あり・なしで greedy 出力が一致しない。4bit 本体では一括検証の数値誤差で
        序盤から言い回しが入れ替わることがある（Qwen3.6-27B の日本語で実際に起きる）ので、
        ずれた先の文面を並べて表示する。MTP 側が崩れていれば検証ループの破損を疑う。

model 省略時は対応表（MTP_DRAFTERS）のうち本体とドラフターが両方取得済みのものを全部測る。
既定では導入済み gw の実行環境の Python（＝デーモンが使う mlx-vlm）で生成する。
モデルを本体単独・MTP ありで計 4 回ロードするので、1 モデルあたり数分かかる。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 date string and returns a "
            "datetime, with docstring and unit tests.",
    "ja": "日本の四季について、それぞれの特徴を300字程度で説明してください。",
}

_GEN_RE = re.compile(r"^Generation:\s*(\d+) tokens,\s*([\d.]+) tokens-per-sec", re.M)
_SPEC_RE = re.compile(
    r"^Speculative decoding:\s*([\d.]+) accepted tokens/round.*?([\d.]+)% of drafted", re.M
)


def default_baseline() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "local-llm-server" / "mtp_baseline.json"


def gw_python() -> Path | None:
    """導入済み gw（~/.local/bin/gw）の venv の Python を返す。無ければ None。"""
    launcher = Path.home() / ".local/bin/gw"
    if not launcher.exists():
        return None
    target = launcher.resolve()
    sibling = target.parent / "python"
    return sibling if sibling.is_file() else None


def parse_output(out: str) -> dict:
    """mlx_vlm.generate --verbose の出力から本文・速度・採択率を取り出す。"""
    gen = _GEN_RE.search(out)
    spec = _SPEC_RE.search(out)
    # 本文は最初の "==========" から次の "==========" まで（プロンプトの echo を含むが、
    # MTP あり・なしで同じなので一致判定には影響しない）。
    parts = out.split("==========")
    return {
        "text": parts[1].strip() if len(parts) >= 3 else "",
        "tokens": int(gen.group(1)) if gen else None,
        "tps": float(gen.group(2)) if gen else None,
        "accepted_per_round": float(spec.group(1)) if spec else None,
        "acceptance": float(spec.group(2)) if spec else None,
    }


def first_divergence(a: str, b: str) -> int | None:
    """a と b が最初に食い違う文字位置。完全一致なら None。"""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def judge(base: dict, mtp: dict, previous: dict | None, max_drop: float) -> tuple[str, list[str]]:
    """1 組（本体単独／MTP あり）の結果を OK / WARN / NG に判定する。"""
    problems: list[str] = []
    level = "OK"
    if base["tps"] is None or mtp["tps"] is None:
        return "NG", ["生成の統計（Generation: …）が読めない。mlx-vlm の出力形式が変わった可能性"]
    if mtp["acceptance"] is None:
        return "NG", ["推測デコードの統計が出ない（MTP が効いていない）"]
    if mtp["tps"] < base["tps"]:
        level = "NG"
        problems.append(f"MTP ありの方が遅い（{mtp['tps']:.1f} < {base['tps']:.1f} tok/s）")
    if previous and previous.get("acceptance") is not None:
        drop = previous["acceptance"] - mtp["acceptance"]
        if drop >= max_drop:
            level = "NG"
            problems.append(
                f"採択率が前回より {drop:.1f} pt 低下（{previous['acceptance']:.1f}% → "
                f"{mtp['acceptance']:.1f}%、前回 mlx-vlm {previous.get('mlx_vlm', '?')}）"
            )
    pos = first_divergence(base["text"], mtp["text"])
    if pos is not None:
        if level == "OK":
            level = "WARN"
        # ずれる位置では壊れたかどうか判断できない（4bit 本体は一括検証の数値誤差で序盤から
        # 言い回しが入れ替わることがある）ので、ずれた先の文面を並べて人が読んで判断する。
        def tail(s: str) -> str:
            return s[pos:pos + 60].replace("\n", " ")

        problems.append(
            f"greedy 出力が {pos} 文字目から不一致。MTP 側が自然な文なら数値誤差、"
            f"崩れていれば検証ループの破損を疑う\n"
            f"           本体: {tail(base['text'])}\n"
            f"           MTP : {tail(mtp['text'])}"
        )
    return level, problems


def run_generate(python: str, model: str, prompt: str, max_tokens: int,
                 drafter: str | None) -> dict:
    cmd = [python, "-m", "mlx_vlm.generate", "--verbose", "--model", model,
           "--prompt", prompt, "--max-tokens", str(max_tokens), "--temperature", "0"]
    if drafter:
        cmd += ["--draft-model", drafter, "--draft-kind", "mtp"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_vlm.generate が失敗しました（{model}）:\n{proc.stderr[-1500:]}")
    return parse_output(proc.stdout)


def mlx_vlm_version(python: str) -> str:
    code = "from importlib.metadata import version;print(version('mlx-vlm'))"
    return subprocess.run([python, "-I", "-c", code], capture_output=True, text=True,
                          check=True).stdout.strip()


def ready_models() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from local_llm_server.model_catalog import MTP_DRAFTERS, ensure_cached

    out = []
    for model, drafter in MTP_DRAFTERS.items():
        try:
            ensure_cached(model)
            ensure_cached(drafter)
        except ValueError:
            continue
        out.append(model)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="本体の HF id（省略時は取得済みの対応モデル全部）")
    parser.add_argument("--python", help="mlx-vlm を動かす Python（既定: 導入済み gw の実行環境）")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--max-drop", type=float, default=10.0,
                        help="ベースラインからの採択率低下（pt）がこれ以上なら NG（既定 10）")
    parser.add_argument("--baseline", type=Path, default=default_baseline())
    parser.add_argument("--save-baseline", action="store_true",
                        help="今回の結果をベースラインとして保存する（NG があっても保存する）")
    args = parser.parse_args()

    python = args.python or str(gw_python() or sys.executable)
    sys.path.insert(0, str(ROOT))
    from local_llm_server.model_catalog import MTP_DRAFTERS

    models = args.models or ready_models()
    if not models:
        print("測れるモデルがありません（本体とドラフターが両方取得済みのものが対応表に無い）。")
        return 1
    version = mlx_vlm_version(python)
    print(f"python: {python}\nmlx-vlm: {version}\nbaseline: {args.baseline}\n")

    try:
        baseline = json.loads(args.baseline.read_text())
    except (OSError, ValueError):
        baseline = {}

    worst = "OK"
    results: dict = {}
    for model in models:
        drafter = MTP_DRAFTERS.get(model)
        if not drafter:
            print(f"{model}: 対応表に無いのでスキップ（MTP_DRAFTERS に登録されていない）")
            worst = "NG"
            continue
        print(f"{model}\n  drafter: {drafter}")
        for name, prompt in PROMPTS.items():
            base = run_generate(python, model, prompt, args.max_tokens, None)
            mtp = run_generate(python, model, prompt, args.max_tokens, drafter)
            previous = baseline.get(model, {}).get(name)
            level, problems = judge(base, mtp, previous, args.max_drop)
            speedup = (mtp["tps"] / base["tps"]) if base["tps"] and mtp["tps"] else None
            print(f"  [{level}] {name:4s} 採択率 {mtp['acceptance'] or 0:.1f}%  "
                  f"{base['tps'] or 0:.1f} → {mtp['tps'] or 0:.1f} tok/s"
                  + (f"（{speedup:.2f}x）" if speedup else ""))
            for p in problems:
                print(f"         - {p}")
            results.setdefault(model, {})[name] = {
                "acceptance": mtp["acceptance"], "speedup": speedup, "mlx_vlm": version,
                "date": datetime.date.today().isoformat(),
            }
            order = ["OK", "WARN", "NG"]
            worst = max(worst, level, key=order.index)

    if args.save_baseline:
        baseline.update(results)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
        print(f"\nベースラインを保存しました: {args.baseline}")
    elif not baseline:
        print("\nベースライン未保存のため前回比は見ていません（--save-baseline で保存）。")
    print(f"\n結果: {worst}")
    return 1 if worst == "NG" else 0


if __name__ == "__main__":
    sys.exit(main())
