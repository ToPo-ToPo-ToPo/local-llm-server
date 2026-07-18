"""OpenAI 互換プロキシの共有ヘルパー（gateway が利用）。

`BaseHTTPRequestHandler` を受け取り、現在のリクエストを上流へ中継したり、
JSON でエラー/結果を返したりする。HTTP/1.0（接続クローズ区切り）前提で、
ストリーミング/非ストリーミングを一律に「上流を読み切るまで中継して閉じる」形で扱う。

クライアント切断の伝播: クライアント（エージェント）が生成を見限って接続を
閉じたら、上流への接続も速やかに閉じて inflight 枠を解放する。ストリーミング
生成中の上流（mlx_lm.server 等）は次のトークン書き込みの Broken pipe で生成を
打ち切るので、無駄な GPU 時間もそこで止まる。これが無いと、見捨てられた
リクエストが生成完了（または request_timeout）まで枠を握り続け、逐次処理の
バックエンドでは後続リクエストの開始も遅らせてしまう。
"""
from __future__ import annotations

import http.client
import json
import select
import socket
import threading
from typing import Any

# クライアント切断をこの間隔（秒）で監視する。
_CLIENT_POLL_S = 0.25


def _client_disconnected(sock: Any) -> bool:
    """クライアントソケットが閉じられていれば True。

    HTTP/1.0（接続クローズ区切り）ではリクエスト送信後にクライアントから届く
    データは無いので、「読める」のに MSG_PEEK が空 = FIN 受信（正常切断）。
    RST 等で recv が失敗した場合も切断とみなす。"""
    if sock is None:
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


def forward(handler: Any, addr: tuple[str, int], body: bytes, timeout_s: float | None) -> None:
    """handler の現在のリクエスト（command/path/headers）を上流 addr へ中継する。

    上流応答（ストリーミング含む）をそのままクライアントへ書き戻す。上流へ到達
    できなければ 502 を返す。
    `timeout_s`（gateway.toml の request_timeout）を渡すとソケットの無応答上限になり、
    上流が沈黙したまま `timeout_s` 秒を超えたら中継を打ち切る（枠を握り続けさせない保険）。
    トークンが届く限りタイムアウトは都度リセットされるので長時間生成は妨げない。

    クライアント切断は監視スレッドが ~_CLIENT_POLL_S 秒以内に検知し、上流接続を
    閉じてブロック中の待ち（応答ヘッダ待ち・トークン待ち）を解除する。読み取り
    経路そのものには手を入れない（chunked デコードは途中のソケットタイムアウトで
    壊れるため、ポーリング読みにはできない）。非ストリーミングでは応答ヘッダ自体
    が生成完了まで届かないので、ヘッダ待ちの間の切断検知が特に効く。
    """
    host, port = addr
    headers = {"Content-Type": handler.headers.get("Content-Type", "application/json")}
    auth = handler.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    if handler.command == "POST":
        headers["Content-Length"] = str(len(body))

    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request(handler.command, handler.path, body=body or None, headers=headers)
    except OSError as exc:
        send_error(handler, 502, f"upstream {host}:{port} unreachable: {exc}")
        return

    # クライアント切断の監視係: 切断を検知したら上流接続を閉じる。これで
    # ブロック中の getresponse()/read1() が例外で解け、下の except が拾って
    # 速やかに return する（呼び出し側の release で inflight 枠が戻る）。
    stop_watch = threading.Event()
    client_gone = threading.Event()

    def _watch_client() -> None:
        while not stop_watch.wait(_CLIENT_POLL_S):
            if _client_disconnected(handler.connection):
                client_gone.set()
                # shutdown のみ行う: FIN/EOF でブロック中の recv を確実に解除しつつ、
                # http.client オブジェクトの内部状態には触れない（別スレッドから
                # conn.close() すると本スレッドの内部クローズと競合して壊れる）。
                # conn.close() は本スレッドの finally に一本化してある。
                sock = conn.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return

    watcher = threading.Thread(target=_watch_client, daemon=True, name="proxy-client-watch")
    watcher.start()

    try:
        try:
            resp = conn.getresponse()
        except Exception as exc:  # noqa: BLE001 - 切断による中断と上流障害をここで振り分ける
            if client_gone.is_set():
                return  # クライアントが見限った。上流はクローズ済み
            if isinstance(exc, OSError):
                send_error(handler, 502, f"upstream {host}:{port} unreachable: {exc}")
                return
            raise
        try:
            handler.send_response(resp.status)
            ctype = resp.getheader("Content-Type")
            if ctype:
                handler.send_header("Content-Type", ctype)
            handler.end_headers()
            while True:
                # read1: ソケットに届いた分だけ即返す（read(8192) は 8192 バイト溜まるまで
                # ブロックするため、SSE のトークン逐次配信が全バッファリングされてしまう）。
                try:
                    chunk = resp.read1(8192)
                except (OSError, ValueError, http.client.HTTPException):
                    # TimeoutError: 上流が timeout_s 秒沈黙した（従来挙動）。
                    # それ以外は監視係が切断検知で conn を閉じた際の中断。どちらも
                    # 応答ヘッダ送出済みでエラー本文は返せないため、中継を打ち切って
                    # conn.close()（finally）で上流へ切断を伝える。
                    break
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # クライアント切断（書き込み時に検知）
    finally:
        stop_watch.set()
        conn.close()


def send_error(handler: Any, status: int, message: str) -> None:
    """{"error": message} を JSON で返す。"""
    send_json(handler, status, {"error": message})


def send_json(handler: Any, status: int, obj: Any) -> None:
    """obj を JSON 本文として status で返す（Content-Length 付き）。"""
    payload = json.dumps(obj).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass
