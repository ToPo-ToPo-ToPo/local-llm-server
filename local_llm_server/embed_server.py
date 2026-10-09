"""テキスト埋め込みモデルを OpenAI 互換の Embeddings サーバとして 1 モデル 1 プロセスで公開する。

ゲートウェイの `backend = "embed"` がこのモジュールを

    python -m local_llm_server.embed_server --model <repo> --host <h> --port <p>

の形で起動する。既存の LLM / STT / TTS バックエンドと同じく「単一モデルの OpenAI 互換サーバ」として
振る舞うので、ゲートウェイの遅延ロード・LRU 退避・idle アンロード・在席即時解放がそのまま効く。
狙いは **RAG を持つアプリ（agent-corporation の search-tool 等）からモデルの常駐を剥がすこと** ——
アプリはジョブごとにモデルを読み直さず（import と読み込みで 10 秒前後かかる）、公開ポートへ文を
POST するだけでよい。

公開するのは最小限:
  - GET  /v1/models       … ロード中モデルの id を 1 件返す（is_ready 判定用）
  - POST /v1/embeddings   … {"model", "input": str | [str], "dimensions"?, "encoding_format"?: "float"|"base64"}

実装は transformers + torch（Apple Silicon では bfloat16 / mps、他は float32 / cpu）。重みは HF の
`AutoModel` で読み、`last_hidden_state` を平均プーリング（モデルが sentence-transformers 形式の
`1_Pooling/config.json` を同梱していれば、その指定＝CLS / 平均に従う）して L2 正規化する。
EmbeddingGemma 2（google/embeddinggemma-2）は画像・音声の符号化器を外してテキスト部分だけを読む
（config の vision_config / audio_config を None に。270M 分で済む）。この経路の出力は
sentence-transformers のものと一致する（cos ≈ 1.0、bf16 の丸めの範囲）。`dimensions` は Matryoshka の
切り詰め（先頭 N 次元を取って再正規化）。

接頭辞（"task: search result | query: …" など）は OpenAI API に無い概念なので付けない——クライアントの
責任（モデルの作法に従って input に含める）。mlx 版（mlx-vlm の embedding_loader）はリリース版に
EmbeddingGemma 2 の読み込みが入った時点で `_Backend` の差し替えだけで移れる。
モデルの重みは事前に `hf download` 済みであること（ゲートウェイが HF_HUB_OFFLINE=1 で起動するため）。
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

from .proxy import reject_overloaded_connection

_MAX_BODY_BYTES = 16 * 1024 * 1024
_MAX_INPUTS = 2048          # 1 リクエストで受ける文の数の上限（索引づくりのバッチを想定）
_BATCH = 32                 # 1 回の forward に入れる文の数
_MAX_TOKENS = 2048          # 1 文のトークン数の上限（超えた分は切る。8K 文脈のモデルでもここで切る）
_ENCODING_FORMATS = {"float", "base64"}


class _Backend:
    """1 モデルの読み込みと埋め込み（torch）。import は重いので遅延させ、1 プロセスに 1 つだけ作る。"""

    def __init__(self, model: str) -> None:
        self.model_id = model
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._pooling = "mean"
        self.dim = 0

    # ---- 読み込み ----
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32  # float16 は使わない（Gemma 系で NaN / 劣化）
        config = AutoConfig.from_pretrained(self.model_id)
        kwargs: dict = {"dtype": dtype}
        # マルチモーダルの埋め込みモデル（EmbeddingGemma 2）はテキスト部分だけを読む。
        # 無い設定を None にすると transformers が怒るので、ある属性だけ外す。
        for name in ("vision_config", "audio_config"):
            if getattr(config, name, None) is not None:
                kwargs[name] = None
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id, **kwargs).to(device).eval()
        self._device = device
        self._pooling = _pooling_mode(self.model_id)
        self.dim = int(getattr(config, "hidden_size", 0) or 0)

    # ---- 埋め込み ----
    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """文の列 → (L2 正規化した float のベクトル列, 使ったトークン数)。"""
        import torch

        self.load()
        assert self._model is not None and self._tokenizer is not None
        out: list[list[float]] = []
        tokens = 0
        with torch.no_grad():
            for i in range(0, len(texts), _BATCH):
                batch = texts[i : i + _BATCH]
                inputs = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=_MAX_TOKENS, return_tensors="pt"
                ).to(self._device)
                mask = inputs["attention_mask"]
                tokens += int(mask.sum().item())
                hidden = self._model(**inputs).last_hidden_state
                if self._pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    m = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
                vec = torch.nn.functional.normalize(pooled.float(), dim=-1).cpu()
                out.extend(vec.tolist())
        if out and not self.dim:
            self.dim = len(out[0])
        return out, tokens


def _pooling_mode(model_id: str) -> str:
    """sentence-transformers 形式の 1_Pooling/config.json があればそれに従う（無ければ平均）。"""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "1_Pooling/config.json")
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.get("pooling_mode_cls_token"):
            return "cls"
    except Exception:  # noqa: BLE001 - 無い・読めない → 平均（埋め込みモデルの大多数）
        pass
    return "mean"


def _truncate(vec: list[float], dim: int) -> list[float]:
    """Matryoshka の切り詰め: 先頭 dim 次元を取り、L2 正規化し直す（切った後に正規化し直さないと順位が黙って崩れる）。"""
    if dim <= 0 or dim >= len(vec):
        return vec
    head = vec[:dim]
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


def _b64(vec: list[float]) -> str:
    import array

    return base64.b64encode(array.array("f", vec).tobytes()).decode("ascii")


def parse_request(payload: dict) -> tuple[list[str], int, str]:
    """リクエスト本文 → (文の列, dimensions（0 なら切らない）, encoding_format)。誤りは ValueError（400）。"""
    raw = payload.get("input")
    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, list) and raw and all(isinstance(t, str) for t in raw):
        texts = list(raw)
    else:
        raise ValueError("'input' must be a non-empty string or a list of strings (token ids are not supported)")
    if len(texts) > _MAX_INPUTS:
        raise ValueError(f"too many inputs ({len(texts)} > {_MAX_INPUTS}); split the request")
    if any(not t.strip() for t in texts):
        raise ValueError("'input' contains an empty string")
    dims = payload.get("dimensions", 0)
    if dims is None:
        dims = 0
    if isinstance(dims, bool) or not isinstance(dims, int) or dims < 0:
        raise ValueError("'dimensions' must be a positive integer")
    fmt = str(payload.get("encoding_format") or "float").lower()
    if fmt not in _ENCODING_FORMATS:
        raise ValueError("'encoding_format' must be 'float' or 'base64'")
    return texts, int(dims), fmt


def build_response(model: str, vectors: list[list[float]], tokens: int, dims: int, fmt: str) -> dict:
    data = []
    for i, vec in enumerate(vectors):
        v = _truncate(vec, dims)
        data.append({"object": "embedding", "index": i, "embedding": _b64(v) if fmt == "base64" else v})
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # アクセスログは出さない（親と同様）
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/models"):
            server = cast("_Server", self.server)
            self._send_json(200, {"object": "list", "data": [{"id": server.model, "object": "model"}]})
            return
        self._send_json(404, {"error": f"GET {self.path} not supported"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if not path.endswith("/embeddings"):
            self._send_json(404, {"error": f"POST {self.path} not supported"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if length < 0 or length > _MAX_BODY_BYTES:
            self._send_json(413, {"error": "request body is too large"})
            return
        body = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(body or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            texts, dims, fmt = parse_request(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        server = cast("_Server", self.server)
        try:
            # 同一プロセス内でモデルの呼び出しを直列化する（ゲートウェイは並列 acquire を許す）。
            with server.lock:
                vectors, tokens = server.backend.embed(texts)
        except Exception as exc:  # noqa: BLE001 - 内部パス等はリモートへ返さない
            print(f"[embed_server] embedding failed: {exc}", file=sys.stderr, flush=True)
            self._send_json(500, {"error": "embedding failed"})
            return
        if dims and server.backend.dim and dims > server.backend.dim:
            self._send_json(400, {"error": f"'dimensions' exceeds the model's size ({server.backend.dim})"})
            return
        self._send_json(200, build_response(server.model, vectors, tokens, dims, fmt))


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, model: str, max_workers: int = 4, backend: _Backend | None = None) -> None:
        super().__init__(addr, _Handler)
        self.model = model
        self.backend = backend or _Backend(model)
        self.lock = threading.Lock()  # モデル呼び出しの直列化用
        self._request_slots = threading.BoundedSemaphore(max_workers)

    def process_request(self, request, client_address):
        if not self._request_slots.acquire(blocking=False):
            reject_overloaded_connection(request, self.close_request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            self._request_slots.release()


def _warm(server: _Server) -> None:
    """バックグラウンドでモデルを事前ロードする（初回リクエストの待ち時間を減らす）。失敗は実リクエスト時に 500 で表面化する。"""
    try:
        with server.lock:
            server.backend.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[embed_server] warm-up skipped: {exc}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local_llm_server.embed_server")
    parser.add_argument("--model", required=True, help="HF repo-id（テキスト埋め込みモデル）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    # tokenizers のフォーク警告・並列を抑える（サーバはスレッドで直列に呼ぶ）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    server = _Server((args.host, args.port), args.model)
    threading.Thread(target=_warm, args=(server,), daemon=True).start()
    print(f"[embed_server] serving {args.model} on {args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
