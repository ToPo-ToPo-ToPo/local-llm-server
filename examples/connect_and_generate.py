# /// script
# requires-python = ">=3.11"
# dependencies = ["local-llm-server"]
# ///
"""ゲートウェイに接続して生成する最小サンプル（接続専用。サーバーは起動しない）。

事前に別ターミナルでゲートウェイを起動しておくこと（gateway.toml に下記モデルを登録）:

    uv run local-llm-server

そのうえで実行:

    uv run examples/connect_and_generate.py
"""
from __future__ import annotations

import sys

from local_llm_server import LLMClient, is_ready

BASE_URL = "http://127.0.0.1:8799/v1"
MODEL = "mlx-community/Qwen3.6-27B-4bit"


def main() -> None:
    if not is_ready(BASE_URL):
        sys.exit(
            f"ゲートウェイが {BASE_URL} で応答していません。別ターミナルで "
            f"`uv run local-llm-server`（gateway.toml に {MODEL} を登録）を起動してから実行してください。"
        )
    llm = LLMClient(model=MODEL, base_url=BASE_URL)
    # ストリーミングで逐次表示（モデルは初回リクエストでゲートウェイが遅延ロードする）。
    for piece in llm.respond("ローカルLLMの利点を3つ。", stream=True):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
