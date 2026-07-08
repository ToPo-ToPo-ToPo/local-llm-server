"""mlx-whisper を OpenAI 互換の STT サーバとして 1 モデル 1 プロセスで公開する。

ゲートウェイの `backend = "whisper"` がこのモジュールを

    python -m local_llm_server.stt_server --model <repo> --host <h> --port <p>

の形で起動する。既存の LLM バックエンド（mlx_lm.server / mlx_vlm.server /
llama-server）と同じく「単一モデルの OpenAI 互換サーバ」として振る舞うので、
ゲートウェイの遅延ロード・LRU 退避・idle アンロード・在席即時解放がそのまま効く。

公開するのは最小限:
  - GET  /v1/models                 … ロード中モデルの id を 1 件返す（is_ready 判定用）
  - POST /v1/audio/transcriptions   … 文字起こし（task=transcribe）
  - POST /v1/audio/translations     … 英訳（task=translate）

音声デコードに ffmpeg CLI（PATH 上）が要る（mlx_whisper.load_audio の仕様）。
モデルの重みは事前に `hf download` 済みであること（ゲートウェイが HF_HUB_OFFLINE=1 で
起動するため、未取得ならロード時にエラーになる）。
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import multipart

# mlx_whisper.transcribe は毎回 load_model() でディスクからモデルを読み直す（キャッシュ無し）。
# 単一モデルを常駐させる本サーバでは無駄なので、モジュール属性を lru_cache で包んで
# 「初回だけロード、以降は再利用」にする（本サーバは 1 モデル固定なので maxsize=1 で十分）。
_mod = None  # mlx_whisper.transcribe サブモジュール（遅延 import。import 自体が重い＝mlx/torch）


def _backend_module():
    """mlx_whisper.transcribe サブモジュールを返す（load_model を lru_cache 済みにして）。

    注意: `mlx_whisper.__init__` が `transcribe` 属性を関数で上書きするため、
    `import mlx_whisper.transcribe as t` では関数が束縛されてしまう。サブモジュール
    実体は importlib で取得する。
    """
    global _mod
    if _mod is None:
        import importlib
        t = importlib.import_module("mlx_whisper.transcribe")
        t.load_model = functools.lru_cache(maxsize=1)(t.load_model)
        _mod = t
    return _mod


def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_timestamp(seconds: float) -> str:
    return _srt_timestamp(seconds).replace(",", ".")


def _format_result(result: dict, response_format: str) -> tuple[str, str]:
    """whisper の結果を response_format に応じた本文と Content-Type にする。"""
    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    if response_format == "text":
        return text + "\n", "text/plain; charset=utf-8"
    if response_format == "verbose_json":
        body = {
            "task": result.get("task", "transcribe"),
            "language": result.get("language"),
            "duration": segments[-1].get("end") if segments else None,
            "text": text,
            "segments": segments,
        }
        return json.dumps(body, ensure_ascii=False), "application/json"
    if response_format in ("srt", "vtt"):
        lines: list[str] = []
        if response_format == "vtt":
            lines.append("WEBVTT\n")
        stamp = _vtt_timestamp if response_format == "vtt" else _srt_timestamp
        for i, seg in enumerate(segments, 1):
            if response_format == "srt":
                lines.append(str(i))
            lines.append(f"{stamp(seg.get('start', 0.0))} --> {stamp(seg.get('end', 0.0))}")
            lines.append((seg.get("text") or "").strip())
            lines.append("")
        ctype = "text/vtt" if response_format == "vtt" else "application/x-subrip"
        return "\n".join(lines) + "\n", f"{ctype}; charset=utf-8"
    # 既定: OpenAI の json（{"text": ...} のみ）
    return json.dumps({"text": text}, ensure_ascii=False), "application/json"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # server 側で設定する。
    model: str = ""

    def log_message(self, *_args) -> None:  # アクセスログは出さない（親と同様）
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
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
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{"id": self.server.model, "object": "model"}],
            })
            return
        self._send_json(404, {"error": f"GET {self.path} not supported"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/audio/transcriptions"):
            self._transcribe("transcribe")
        elif path.endswith("/audio/translations"):
            self._transcribe("translate")
        else:
            self._send_json(404, {"error": f"POST {self.path} not supported"})

    def _transcribe(self, task: str) -> None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        body = self.rfile.read(length) if length > 0 else b""
        ctype = self.headers.get("Content-Type", "")
        parts = multipart.parse(body, ctype)
        audio = next((p for p in parts if p.filename is not None), None)
        if audio is None:
            self._send_json(400, {"error": "no audio 'file' field in multipart body"})
            return
        fields = {p.name: p.text() for p in parts if p.filename is None}
        response_format = (fields.get("response_format") or "json").lower()
        options: dict = {"task": task}
        if fields.get("language"):
            options["language"] = fields["language"]
        if fields.get("prompt"):
            options["initial_prompt"] = fields["prompt"]
        if fields.get("temperature"):
            try:
                options["temperature"] = float(fields["temperature"])
            except ValueError:
                pass

        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                fh.write(audio.value)
                tmp_path = fh.name
            mod = _backend_module()
            # 同一プロセス内で mlx の呼び出しを直列化する（Metal コンテキストを複数
            # スレッドから同時に叩かない。ゲートウェイは並列 acquire を許すため）。
            with self.server.lock:
                result = mod.transcribe(
                    tmp_path, path_or_hf_repo=self.server.model, **options
                )
        except Exception as exc:  # noqa: BLE001 モデル/デコード失敗をそのまま 500 で返す
            self._send_json(500, {"error": f"transcription failed: {exc}"})
            return
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        text_body, out_ctype = _format_result(result, response_format)
        self._send(200, text_body.encode("utf-8"), out_ctype)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, model: str) -> None:
        super().__init__(addr, _Handler)
        self.model = model
        self.lock = threading.Lock()  # mlx 呼び出しの直列化用


def _warm(model: str) -> None:
    """バックグラウンドでモデルを事前ロードする（初回リクエストの待ち時間を減らす）。

    失敗しても握りつぶす（未 DL・重み不整合などは実リクエスト時に 500 で表面化する）。
    サーバの bind/受付はこれを待たない。
    """
    try:
        mod = _backend_module()
        mod.load_model(model)  # lru_cache 済み。以降の transcribe が即使う
    except Exception as exc:  # noqa: BLE001
        print(f"[stt_server] warm-up skipped: {exc}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local_llm_server.stt_server")
    parser.add_argument("--model", required=True, help="HF repo-id（mlx whisper モデル）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    server = _Server((args.host, args.port), args.model)
    threading.Thread(target=_warm, args=(args.model,), daemon=True).start()
    print(f"[stt_server] serving {args.model} on {args.host}:{args.port}",
          file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
