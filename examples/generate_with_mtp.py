# /// script
# requires-python = ">=3.11"
# dependencies = ["local-llm-server[mlx]", "openai>=1.0"]
# ///
"""実LLM + MTP（Multi-Token Prediction）で生成する end-to-end サンプル。

local-automata が既定で使っていた構成をそのまま単体で再現する:

  - バックエンド : mlx-vlm（vision 対応。MTP は mlx-vlm でのみ動く）
  - 本体モデル   : mlx-community/Qwen3.6-27B-4bit
  - ドラフター   : draft_model="auto" → mlx-community/Qwen3.6-27B-MTP-4bit
                   （投機的デコード。本体の出力を変えずに実測 ~2倍速）

サーバーの起動・待機・停止は LocalServer（context manager）に任せ、生成は
標準的な openai クライアントで行う。

実行（Apple Silicon。初回は本体＋ドラフターの2モデルを自動ダウンロード）:

    uv run examples/generate_with_mtp.py

PEP 723 のインライン依存により、uv が local-llm-server[mlx] と openai を
含む一時環境を用意して実行する（追加の準備は不要）。
"""
from __future__ import annotations

import time

import local_llm_server as srv
from openai import OpenAI

MODEL = "mlx-community/Qwen3.6-27B-4bit"
PROMPT = "ローカルLLMを自宅で動かす利点を、初心者にもわかるように3つ挙げてください。"


def main() -> None:
    # local-automata の auto_start と同じ ServerConfig（MTP は draft_model="auto"）。
    config = srv.ServerConfig(
        backend="mlx-vlm",
        model=MODEL,
        draft_model="auto",  # → mlx-community/Qwen3.6-27B-MTP-4bit に自動解決
    )

    print(f"起動中: {config.model}  (+MTP drafter, backend={config.backend})")
    with srv.LocalServer(config) as server:
        server.wait_until_ready(timeout=600)  # 初回はモデルDLがあるため長めに待つ
        print(f"準備完了: {server.base_url}")
        print("読み込み済みモデル:", srv.list_models(server.base_url))

        client = OpenAI(base_url=server.base_url, api_key="not-needed")

        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.7,
        )
        elapsed = time.perf_counter() - start

        print("\n--- 生成結果 ---")
        print(resp.choices[0].message.content)

        # MTP の効果を tok/s で確認（usage を返すバックエンドのみ）。
        usage = getattr(resp, "usage", None)
        if usage and getattr(usage, "completion_tokens", None):
            tps = usage.completion_tokens / elapsed
            print(
                f"\n--- 速度 ---\n"
                f"{usage.completion_tokens} tokens / {elapsed:.1f}s = {tps:.1f} tok/s"
                f"  (MTP 投機的デコードによる高速化が効いています)"
            )
    # with を抜けるとサーバーは自動停止する。


if __name__ == "__main__":
    main()
