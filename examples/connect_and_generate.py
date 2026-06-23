# /// script
# requires-python = ">=3.11"
# dependencies = ["local-llm-server[mlx,client]"]
# ///
"""高レベル API（connect）だけで生成する最小サンプル。

connect() が「既存サーバーがあれば相乗り、無ければ MTP 付きで自動起動」してから、
繋がった LLMClient を返す。generate_with_mtp.py の LocalServer + openai 手書きを
1 呼び出しに畳んだ版。

    uv run examples/connect_and_generate.py
"""
from __future__ import annotations

from local_llm_server import connect

MODEL = "mlx-community/Qwen3.6-27B-4bit"


def main() -> None:
    # backend は省略時 OS 自動判定（mac arm64 → mlx-vlm）。draft_model="auto" で MTP。
    llm = connect(model=MODEL, draft_model="auto", log=print)
    try:
        # ストリーミングで逐次表示。
        for piece in llm.respond(
            "ローカルLLMを自宅で動かす利点を、初心者にもわかるように3つ。",
            stream=True,
        ):
            print(piece, end="", flush=True)
        print()
    finally:
        llm.stop()  # connect が自動起動した場合のみ停止（相乗りなら無害）


if __name__ == "__main__":
    main()
