# /// script
# requires-python = ">=3.11"
# dependencies = ["local-llm-server>=0.6"]
# ///
"""ゲートウェイ経由で生成し、速度（tok/s）を測るサンプル（接続専用）。

MTP（投機的デコード）は gateway.toml の `draft_model` で設定する（クライアントからは透過）。
`draft_model = "auto"` を設定したモデルは本体の出力を変えずに ~2倍速になる（→ docs/mtp.md）。

事前に別ターミナルでゲートウェイを起動しておくこと（gateway.toml に下記モデルと
`draft_model = "auto"` を登録）:

    uv run local-llm-server

そのうえで実行:

    uv run examples/generate_with_mtp.py
"""
from __future__ import annotations

import sys
import time

from local_llm_server import LLMClient, is_ready

BASE_URL = "http://127.0.0.1:8799/v1"
MODEL = "mlx-community/Qwen3.6-27B-4bit"
PROMPT = "ローカルLLMを自宅で動かす利点を、初心者にもわかるように3つ挙げてください。"


def main() -> None:
    if not is_ready(BASE_URL):
        sys.exit(
            f"ゲートウェイが {BASE_URL} で応答していません。別ターミナルで "
            f"`uv run local-llm-server`（gateway.toml に {MODEL} ＋ draft_model=\"auto\" を登録）"
            "を起動してから実行してください。"
        )
    llm = LLMClient(model=MODEL, base_url=BASE_URL)

    start = time.perf_counter()
    # usage（生成トークン数）が欲しいので非ストリームで土台クライアントを直接使う。
    resp = llm.openai.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": PROMPT}]
    )
    elapsed = time.perf_counter() - start

    print(resp.choices[0].message.content)
    usage = getattr(resp, "usage", None)
    if usage and getattr(usage, "completion_tokens", None):
        tps = usage.completion_tokens / elapsed
        print(
            f"\n{usage.completion_tokens} tokens / {elapsed:.1f}s = {tps:.1f} tok/s "
            "（gateway.toml で draft_model を設定していれば MTP で高速化）"
        )


if __name__ == "__main__":
    main()
