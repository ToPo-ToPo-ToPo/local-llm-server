"""server↔client の結線テスト（対パッケージの契約を server 側で常時検証する）。

local-llm-server と local-llm-client は別リポジトリ・別 PyPI パッケージで開発する
（能力の隔離＝client 環境からゲートウェイを起動できない構成の強制、および
ゲートウェイの自動更新運用のため。統合しない決定 2026-06-28 / 再確認 2026-07-09）。
その代償である「対の変更の検出」をこのテストが担う: PyPI の local-llm-client（dev 依存）を
in-process のゲートウェイへ実 HTTP で接続し、client の公開 API が server の応答と噛み合う
ことを検証する。実 LLM は使わない（上流は OpenAI 互換のフェイク）。

ここが赤くなったら「server の変更が公開クライアントとの契約を壊した」ことを意味する。
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import local_llm_client as llc

from local_llm_server import ServerConfig
from local_llm_server import daemon as gw


def _openai_upstream():
    """OpenAI 互換の chat.completion を返すフェイク上流（実モデルの代役）。"""

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            data = json.dumps({
                "id": "cmpl-itest", "object": "chat.completion", "created": 0,
                "model": "itest-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "pong"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class _NoopServer:
    """LocalServer 差し替え（プロセスを起動せず、フェイク上流のポートをそのまま使う）。"""

    def __init__(self, config, log_path=None):
        self.config = config

    def start(self):
        pass

    def wait_until_ready(self, *a, **k):
        pass

    def stop(self, grace: float = 10.0):
        pass


def _start_gateway(monkeypatch, api_key=None):
    upstream = _openai_upstream()
    monkeypatch.setattr(gw, "LocalServer",
                        lambda config, log_path=None: _NoopServer(config, log_path))
    configs = [ServerConfig(backend="mlx-vlm", model="itest-model", host="127.0.0.1",
                            port=upstream.server_address[1])]
    mgr = gw.ModelManager(configs, dynamic=False)
    server = gw.GatewayServer(("127.0.0.1", 0), mgr, catalog=["itest-model"],
                              api_key=api_key)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return server, mgr, upstream, base


def _teardown(server, mgr, upstream):
    server.shutdown()
    server.server_close()
    mgr.shutdown()
    upstream.shutdown()
    upstream.server_close()


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def test_client_is_ready_and_list_models(monkeypatch):
    # 死活確認とカタログ取得: client の is_ready / list_models が gateway の
    # /v1/models 応答と噛み合う（エージェントの接続前チェックで使う基本 API）。
    server, mgr, upstream, base = _start_gateway(monkeypatch)
    try:
        assert llc.is_ready(base) is True
        assert "itest-model" in llc.list_models(base)
    finally:
        _teardown(server, mgr, upstream)


def test_client_respond_round_trip_and_presence_unload(monkeypatch):
    # 本丸: respond() が gateway 経由で応答を得る（OpenAI 互換の契約）＋
    # 在席セッション（register/heartbeat/release）が効き、close() で在席 0 に
    # なった瞬間モデルが即アンロードされる（対パッケージ間の独自プロトコル）。
    server, mgr, upstream, base = _start_gateway(monkeypatch)
    client = llc.LLMClient(model="itest-model", base_url=base, stream=False)
    try:
        assert client.respond("ping") == "pong"
        assert mgr._models["itest-model"].instances  # 応答後はロード済み
        client.close()  # 在席 release → 他に在席が無いので即アンロード
        assert _wait_until(lambda: not mgr._models["itest-model"].instances)
    finally:
        client.close()
        _teardown(server, mgr, upstream)


def test_client_api_key_auth(monkeypatch):
    # 認証ありゲートウェイの契約: client が api_key を Bearer で送り、正キーは通り
    # 誤キーは 401 で弾かれる（client 0.4.0 で server の api_key 対応に追随した機能）。
    server, mgr, upstream, base = _start_gateway(monkeypatch, api_key="itest-key")
    good = llc.LLMClient(model="itest-model", base_url=base, api_key="itest-key",
                         stream=False)
    bad = llc.LLMClient(model="itest-model", base_url=base, api_key="wrong",
                        stream=False, session=False)
    try:
        assert good.respond("ping") == "pong"
        try:
            bad.respond("ping")
            raise AssertionError("wrong api_key must be rejected")
        except Exception as exc:  # openai SDK は 401 を AuthenticationError にする
            assert "401" in str(exc) or "unauthorized" in str(exc).lower()
    finally:
        good.close()
        bad.close()
        _teardown(server, mgr, upstream)
