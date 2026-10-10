"""リランカー（クロスエンコーダ）を Cohere / Jina 互換の Rerank サーバとして 1 モデル 1 プロセスで公開する。

ゲートウェイの `backend = "rerank"` がこのモジュールを

    python -m local_llm_server.rerank_server --model <repo> --host <h> --port <p>

の形で起動する。埋め込み（embed_server）と同じく「単一モデルの HTTP サーバ」として振る舞うので、
ゲートウェイの遅延ロード・LRU 退避・idle アンロード・在席即時解放がそのまま効く。RAG の 2 段目——
埋め込みで広く拾った候補（数十件）を、問いと文の組ごとに精査して並べ直し、較正された関連度（0〜1）を
付ける——をアプリから剥がし、ゲートウェイに常駐させるのが狙い。生成はしない（分類モデル）。

公開するのは最小限:
  - GET  /v1/models   … ロード中モデルの id を 1 件返す（is_ready 判定用）
  - POST /v1/rerank   … {"model", "query", "documents": [str | {"text"}], "top_n"?, "return_documents"?, "instruction"?}
                        → {"model", "results": [{"index", "relevance_score", "document"?}], "usage": {"total_tokens"}}
                        （results は関連度の高い順。OpenAI に rerank は無いので Cohere / Jina の形に合わせる）

実装は transformers + torch（Apple Silicon では bfloat16 / mps、他は float32 / cpu）。モデルは 2 系統を見分ける:
  - 系列分類のクロスエンコーダ（ModernBERT / XLM-R 系: ruri-v3-reranker、bge-reranker-v2-m3、japanese-reranker-*）
    … (query, document) の組を 1 本の入力にして logit → sigmoid（2 クラスなら softmax の正例側）
  - 生成モデル型（Qwen3-Reranker）… 公式の chat 形式の prompt で「yes / no」の次トークン logit を取り、
    softmax の yes 側を関連度にする（モデルカードの作法どおり）
モデルの重みは事前に `hf download` 済みであること（ゲートウェイが HF_HUB_OFFLINE=1 で起動するため）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from .proxy import reject_overloaded_connection

_MAX_BODY_BYTES = 16 * 1024 * 1024
_MAX_DOCUMENTS = 512        # 1 リクエストで受ける文の数の上限（RAG の候補は数十件を想定）
_BATCH = 16                 # 1 回の forward に入れる組の数
_MAX_TOKENS = 2048          # 1 組（問い＋文）のトークン数の上限（超えた分は文の側を切る）
_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
# Qwen3-Reranker の公式 prompt（モデルカードどおり。変えると較正が崩れる）
_QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class _Backend:
    """1 モデルの読み込みと採点（torch）。import は重いので遅延させ、1 プロセスに 1 つだけ作る。"""

    def __init__(self, model: str) -> None:
        self.model_id = model
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"
        self.kind = ""          # "seqcls" | "yesno"
        self._yes_no: tuple[int, int] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32
        config = AutoConfig.from_pretrained(self.model_id)
        archs = [str(a) for a in (getattr(config, "architectures", None) or [])]
        if any(a.endswith("ForCausalLM") for a in archs):
            # 生成モデル型（Qwen3-Reranker）: yes / no の次トークンで採点する
            self.kind = "yesno"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
            tok = self._tokenizer
            self._yes_no = (int(tok.convert_tokens_to_ids("yes")), int(tok.convert_tokens_to_ids("no")))
        else:
            self.kind = "seqcls"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
        self._device = device

    def score(self, query: str, documents: list[str], instruction: str | None = None) -> tuple[list[float], int]:
        """(問い, 文) の組ごとの関連度（0〜1）と、使ったトークン数。"""
        import torch

        self.load()
        assert self._model is not None and self._tokenizer is not None
        scores: list[float] = []
        tokens = 0
        with torch.no_grad():
            for i in range(0, len(documents), _BATCH):
                batch = documents[i : i + _BATCH]
                if self.kind == "yesno":
                    inst = instruction or _DEFAULT_INSTRUCTION
                    texts = [
                        _QWEN_PREFIX + f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {d}" + _QWEN_SUFFIX
                        for d in batch
                    ]
                    inputs = self._tokenizer(
                        texts, padding=True, truncation=True, max_length=_MAX_TOKENS, return_tensors="pt"
                    ).to(self._device)
                    tokens += int(inputs["attention_mask"].sum().item())
                    logits = self._model(**inputs).logits[:, -1, :]
                    assert self._yes_no is not None
                    yes_id, no_id = self._yes_no
                    pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
                    scores.extend(torch.softmax(pair, dim=1)[:, 1].cpu().tolist())
                else:
                    inputs = self._tokenizer(
                        [query] * len(batch), batch, padding=True, truncation="only_second",
                        max_length=_MAX_TOKENS, return_tensors="pt",
                    ).to(self._device)
                    tokens += int(inputs["attention_mask"].sum().item())
                    logits = self._model(**inputs).logits.float()
                    if logits.shape[-1] == 1:
                        probs = torch.sigmoid(logits[:, 0])
                    else:
                        probs = torch.softmax(logits, dim=-1)[:, -1]
                    scores.extend(probs.cpu().tolist())
        return scores, tokens


def parse_request(payload: dict) -> tuple[str, list[str], int, bool, str | None]:
    """リクエスト本文 → (問い, 文の列, top_n（0 なら全部）, return_documents, instruction)。誤りは ValueError（400）。"""
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    raw = payload.get("documents")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'documents' must be a non-empty list of strings (or objects with 'text')")
    docs: list[str] = []
    for d in raw:
        if isinstance(d, dict):
            d = d.get("text")
        if not isinstance(d, str) or not d.strip():
            raise ValueError("'documents' must contain non-empty strings (or objects with a non-empty 'text')")
        docs.append(d)
    if len(docs) > _MAX_DOCUMENTS:
        raise ValueError(f"too many documents ({len(docs)} > {_MAX_DOCUMENTS}); narrow the candidates first")
    top_n = payload.get("top_n", 0)
    if top_n is None:
        top_n = 0
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
        raise ValueError("'top_n' must be a positive integer")
    return_documents = bool(payload.get("return_documents", False))
    instruction = payload.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError("'instruction' must be a string")
    return query.strip(), docs, int(top_n), return_documents, instruction


def build_response(model: str, docs: list[str], scores: list[float], tokens: int, top_n: int, return_documents: bool) -> dict:
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    if top_n:
        order = order[:top_n]
    results = []
    for i in order:
        item: dict = {"index": i, "relevance_score": round(float(scores[i]), 6)}
        if return_documents:
            item["document"] = {"text": docs[i]}
        results.append(item)
    return {"model": model, "results": results, "usage": {"total_tokens": tokens}}


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
        if not path.endswith("/rerank"):
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
            query, docs, top_n, return_documents, instruction = parse_request(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        server = cast("_Server", self.server)
        try:
            with server.lock:  # 同一プロセス内でモデルの呼び出しを直列化する
                scores, tokens = server.backend.score(query, docs, instruction)
        except Exception as exc:  # noqa: BLE001 - 内部パス等はリモートへ返さない
            print(f"[rerank_server] rerank failed: {exc}", file=sys.stderr, flush=True)
            self._send_json(500, {"error": "rerank failed"})
            return
        self._send_json(200, build_response(server.model, docs, scores, tokens, top_n, return_documents))


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, model: str, max_workers: int = 4, backend: _Backend | None = None) -> None:
        super().__init__(addr, _Handler)
        self.model = model
        self.backend = backend or _Backend(model)
        self.lock = threading.Lock()
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
    """バックグラウンドでモデルを事前ロードする。失敗は実リクエスト時に 500 で表面化する。"""
    try:
        with server.lock:
            server.backend.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[rerank_server] warm-up skipped: {exc}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local_llm_server.rerank_server")
    parser.add_argument("--model", required=True, help="HF repo-id（リランカー）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    server = _Server((args.host, args.port), args.model)
    threading.Thread(target=_warm, args=(server,), daemon=True).start()
    print(f"[rerank_server] serving {args.model} on {args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
