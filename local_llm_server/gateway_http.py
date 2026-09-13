# mypy: disable-error-code="attr-defined,index,dict-item"
"""HTTP request boundary for the multi-model gateway.

The server lifecycle and model scheduler live in ``daemon``.  This module owns
only protocol validation, request transformation, routing, and HTTP responses.
"""

from __future__ import annotations

import hmac
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler

from . import image, multipart, video
from .backend_runtime import (
    llama_provision_info,
    sglang_provision_info,
    vllm_provision_info,
)
from .gateway_errors import CapacityError, GatewayDraining
from .gateway_runtime import primary_lan_ip
from .model_catalog import discover_cached_models
from .proxy import forward, send_error, send_json


# 受け付けるリクエストボディの上限（バイト）。vision の base64 画像を見込んでも十分大きく、
# かつ「巨大 Content-Length を申告してメモリを食い潰す」DoS は防ぐ。
_MAX_BODY_BYTES = 100 * 1024 * 1024


class GatewayRequestHandler(BaseHTTPRequestHandler):
    server_version = "local-llm-gateway"
    # HTTP/1.0: 応答ボディは接続クローズ区切り（router と同じ）。
    protocol_version = "HTTP/1.0"
    # リクエスト受信（ヘッダ・ボディ）のソケットタイムアウト。ネットワーク公開時に
    # 「ヘッダを送り切らない接続」がハンドラスレッドを永久に塞がないようにする（Slowloris 対策）。
    # 応答の書き出しは生成中ほぼブロックしないので、長時間生成の妨げにはならない。
    timeout = 60

    def log_message(self, *_args) -> None:  # アクセスログは出さない
        pass

    def _route_path(self) -> str:
        """ルーティング用のパス（クエリ文字列を除き、末尾の '/' を落とす）。

        `GET /v1/models?limit=10` のようにクエリが付いても正しくマッチさせる
        （上流への転送には self.path をそのまま使う）。
        """
        return urllib.parse.urlsplit(self.path).path.rstrip("/")

    def _read_body(self) -> bytes | None:
        """Content-Length を検証して本文を読む。不正・過大は応答を返して None。"""
        raw = self.headers.get("Content-Length") or "0"
        try:
            length = int(raw)
        except ValueError:
            send_error(self, 400, "invalid Content-Length header")
            return None
        if length < 0:
            send_error(self, 400, "invalid Content-Length header")
            return None
        if length > _MAX_BODY_BYTES:
            send_error(self, 413, f"request body too large (> {_MAX_BODY_BYTES} bytes)")
            return None
        return self.rfile.read(length) if length else b""

    def _client_is_loopback(self) -> bool:
        """接続元が同一マシン（ループバック、または bind 先そのもの）か。

        特定 IP に bind した場合（host = "192.168.x.y"）、同一マシンの TUI/CLI も
        その IP 経由で接続し、接続元アドレスは bind 先と同じになる（TCP のハンドシェイクを
        通るため他マシンからは名乗れない）。それも「同一マシン」と扱わないと、管理系
        エンドポイントが自分の TUI からも 403 になってしまう。
        """
        host = self.client_address[0]
        if host in ("127.0.0.1", "::1", "::ffff:127.0.0.1") or host.startswith("127."):
            return True
        bind_host = self.server.server_address[0]
        return bind_host not in ("0.0.0.0", "::", "") and host == bind_host

    def _require_loopback(self) -> bool:
        """管理系（状態・設定）はローカルからのみ許可。非ループバックなら 403 を返して False。"""
        if self._client_is_loopback():
            return True
        send_error(self, 403, "this endpoint is restricted to localhost")
        return False

    def _require_api_key(self) -> bool:
        """api_key が設定されていれば Authorization: Bearer <key> を要求する。

        未設定なら誰でも可（True）。設定済みでキーが無い/一致しなければ 401 を返して False。
        比較は hmac.compare_digest（タイミング安全）。
        """
        key = getattr(self.server, "api_key", None)
        if not key:
            return True  # 認証なし運用
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        # bytes で比較する（str の compare_digest は非 ASCII で TypeError → 500 になる）。
        if token and hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8")):
            return True
        send_error(self, 401, "missing or invalid API key")
        return False

    def _require_safe_browser_context(self) -> bool:
        """cross-site POST と DNS rebinding を、認証や本文読み込みより前に拒否する。"""
        host_header = self.headers.get("Host", "")
        try:
            request_url = urllib.parse.urlsplit(f"//{host_header}")
            host = request_url.hostname
            request_port = request_url.port or 80
        except ValueError:
            host = None
            request_port = -1
        bound = str(self.server.server_address[0]).split("%", 1)[0]  # type: ignore[attr-defined]
        allowed = {"127.0.0.1", "localhost", "::1", bound}
        if bound == "0.0.0.0":
            allowed.update({"127.0.0.1", "localhost"})
            lan = primary_lan_ip()
            if lan:
                allowed.add(lan)
        fetch_site = self.headers.get("Sec-Fetch-Site", "").lower()
        origin = self.headers.get("Origin")
        # HTTP/1.0 のネイティブクライアントは Host を省略できる。ブラウザ由来ヘッダが無い
        # 場合だけ互換性のため許可する（ブラウザは Host を必ず送る）。
        if not host and not origin and not fetch_site:
            return True
        if not host or host.split("%", 1)[0].lower() not in {
            h.lower() for h in allowed
        }:
            send_error(self, 403, "untrusted Host header")
            return False
        if fetch_site in {"cross-site", "same-site"}:
            send_error(self, 403, "cross-site browser requests are not allowed")
            return False
        if origin:
            try:
                parsed = urllib.parse.urlsplit(origin)
                origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            except ValueError:
                parsed = urllib.parse.SplitResult("", "", "", "", "")
                origin_port = -2
            if (
                parsed.scheme != "http"
                or parsed.hostname != host
                or origin_port != request_port
            ):
                send_error(self, 403, "cross-origin browser requests are not allowed")
                return False
        return True

    def do_GET(self) -> None:
        if not self._require_safe_browser_context():
            return
        path = self._route_path()
        srv = self.server  # type: ignore[assignment]
        # /v1/models は設定済みカタログを合成して返す（is_ready 判定・モデル取り違え
        # 警告に対応）。実モデルは起動していなくてもカタログとして列挙する。
        if path.endswith("/models"):
            if not self._require_api_key():
                return
            # 事前登録カタログ＋現在管理中（動的ロード分）を重複なく列挙する（標準どおり）。
            # DL 済みモデルの「発見一覧」は TUI 専用（/admin/status の available）に集約する。
            ids = list(dict.fromkeys(srv.catalog + srv.manager.model_ids))
            data = {
                "object": "list",
                "data": [{"id": m, "object": "model"} for m in ids],
            }
            send_json(self, 200, data)
            return
        # /admin/status は常駐モデルのライブ状態（loaded/inflight）＋運用ポリシーを返す。
        # TUI が詳しい状態（server_status より細かいライブ状態）を出すための読み取り口。
        if path.endswith("/admin/status"):
            if not self._require_loopback():
                return
            # 更新チェックをオンデマンドで温める。トレイがメニューを開くたびにここへ来るので、
            # **リスタート無しで**「更新の有無」を最新化できる（Ollama と同じく、確認は
            # 動いたまま・適用のときだけ再起動）。適用はしない（それは watcher と手動更新の
            # 役目）。タグ照会しすぎないよう _UPDATE_ONDEMAND_THROTTLE 秒のスロットル付き。
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            # トレイの取得自体は既に UI 外のスレッドで動く。ここでもう一段バックグラウンド化
            # すると、最初の応答は古い state → 次の取得で新 state となり、更新項目を二度
            # 操作しないと反映されないように見える。明示指定時だけ同じ要求内で確認を終える。
            wait_for_update = query.get("refresh_updates") == ["1"]
            refresh = getattr(srv, "refresh_update_state", None)
            if refresh is not None:
                refresh(wait=wait_for_update)
            host, port = srv.server_address[0], srv.server_address[1]
            models = srv.manager.status()
            data = {
                "object": "gateway.status",
                "host": host,
                "port": port,
                "max_resident": srv.max_resident,
                "idle_timeout": srv.idle_timeout,
                "load_timeout": srv.load_timeout,
                "default_model": srv.default_model,
                "uptime": round(srv.manager.uptime(), 1),
                "requests": sum(m.get("requests", 0) for m in models),
                # 起動元情報: いつ・どこから立ったゲートウェイかを示す（起動経路は
                # gw start の 1 本だけなので経路の識別は無い）。
                "pid": srv.pid,
                "started_at": srv.started_at,
                "cwd": srv.start_cwd,
                # 導入した llama.cpp / vLLM / SGLang の素性。未導入は None。
                "llama": llama_provision_info(),
                "vllm": vllm_provision_info(),
                "sglang": sglang_provision_info(),
                "models": models,
                # キャッシュにある DL 済みモデル（TUI が未ロード候補として一覧する）。
                "available": discover_cached_models(),
                # 新版の検知状態（update watcher が更新。トレイの更新マーク・gw status 用）。
                # fetched=true はソース追従済みで再起動待ちだけが残っている状態。
                "update": dict(getattr(srv, "update_state", None) or {}),
            }
            send_json(self, 200, data)
            return
        send_error(self, 404, f"GET {self.path} is not supported by the gateway")

    def do_POST(self) -> None:
        srv = self.server  # type: ignore[assignment]
        path = self._route_path()
        if not self._require_safe_browser_context():
            return
        # 認可はボディを読む前に判定する（未認証のリモートに巨大ボディを読み込まされない）。
        # 管理系はローカル接続 + 上の Host/Origin 検査で保護する。これによりトレイへ
        # api_key をコマンドラインで渡さずに済む。クライアント向けは API キーを要求する。
        if path.endswith(("/admin/config", "/admin/drain", "/admin/update")):
            if not self._require_loopback():
                return
        elif not self._require_api_key():
            return
        body = self._read_body()
        if body is None:
            return
        # 音声（STT）は OpenAI 仕様で multipart/form-data。本文は JSON ではないので、
        # multipart（または query）から model を取り出して振り分ける（chat と別処理）。
        if path.endswith(("/audio/transcriptions", "/audio/translations")):
            self._handle_audio(srv, body)
            return
        content_type = (
            self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        )
        if content_type != "application/json":
            send_error(
                self, 415, "JSON endpoints require Content-Type: application/json"
            )
            return
        try:
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, ValueError):
            send_error(self, 400, "invalid JSON body")
            return
        if not isinstance(payload, dict):
            # [1] や "x" など dict 以外の JSON は .get で落ちる前に弾く（400 を返す）。
            send_error(self, 400, "JSON body must be an object")
            return
        # エージェント在席（セッション）管理エンドポイント。チャット転送とは別系統で、
        # 「使う人が居なくなったモデルを即アンロードする」ための登録/心拍/解除を受ける。
        if path.endswith("/admin/config"):
            self._handle_config_update(srv, payload)
            return
        # 再起動準備（drain）。管理クライアンが「アイドル確認＋新規受付停止」を
        # 原子的に行うために使う（→ ModelManager.begin_drain）。ローカルの管理操作なので loopback 限定。
        if path.endswith("/admin/drain"):
            if not self._require_loopback():
                return
            if payload.get("enable", True):
                res = srv.manager.begin_drain()
                send_json(
                    self, 200, {"object": "gateway.drain", "draining": res["ok"], **res}
                )
            else:
                srv.manager.end_drain()
                send_json(
                    self,
                    200,
                    {"object": "gateway.drain", "draining": False, "ok": True},
                )
            return
        # 「今すぐ更新して再起動」（トレイの更新メニュー / Ollama の Restart to update 相当）。
        # ローカルの管理操作なので loopback 限定。
        if path.endswith("/admin/update"):
            if not self._require_loopback():
                return
            self._handle_update_now(srv)
            return
        if path.endswith("/admin/sessions/register"):
            self._handle_session_register(srv, payload)
            return
        if path.endswith("/admin/sessions/heartbeat"):
            self._handle_session_heartbeat(srv, payload)
            return
        if path.endswith("/admin/sessions/release"):
            self._handle_session_release(srv, payload)
            return
        model = payload.get("model") or srv.default_model
        if not model:
            send_error(
                self,
                400,
                "no 'model' in the request and no default_model is configured",
            )
            return
        # 動画入力: video_url をフレーム画像（image_url）列に展開してから先へ進む。展開した
        # フレームは以降の画像扱い。抽出失敗は 400。
        if video.request_has_video(payload):
            if not srv._media_slots.acquire(blocking=False):
                send_error(self, 503, "video processing capacity is busy; retry later")
                return
            try:
                video.expand_video_parts(
                    payload,
                    srv.video_frames,
                    srv.video_max_edge,
                    allow_local=self._client_is_loopback()
                    and not bool(self.headers.get("Origin")),
                )
            except video.VideoError as exc:
                send_error(self, 400, f"video input could not be processed: {exc}")
                return
            finally:
                srv._media_slots.release()
            body = json.dumps(payload).encode("utf-8")
        # 画像縮小: 長辺が image_max_edge を超える画像は上流へ渡す前に縮める。解像度上限の無い
        # VLM（Qwen3.6 等の qwen3_5 系）に巨大画像を渡すと vision トークンが数千に膨れ、Dense
        # モデルの prefill 速度がそのまま効いて数十秒かかる（実測: 1400px で 1,960 トークン・
        # 49 秒 → 768px なら 599 トークン・8.8 秒）。動画フレームの video_max_edge と同じ発想。
        # 動画展開の**後**に置くので、抽出済みフレーム（≤ video_max_edge）は無変更で素通りする。
        if getattr(srv, "image_max_edge", 0) and image.request_has_image(payload):
            if not srv._media_slots.acquire(blocking=False):
                send_error(self, 503, "image processing capacity is busy; retry later")
                return
            try:
                if image.downscale_image_parts(payload, srv.image_max_edge):
                    body = json.dumps(payload).encode("utf-8")
            except image.ImageError as exc:
                send_error(self, 413, str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - 安全化処理を素通ししない
                print(f"Image preprocessing failed: {exc}", file=sys.stderr)
                send_error(self, 400, "image input could not be processed")
                return
            finally:
                srv._media_slots.release()
        # 繰り返しループ抑制: mlx 系宛の生成リクエストに repetition_penalty を既定注入する
        # （chat/text completions のみ。クライアント明示は尊重。設定で無効化可）。
        if path.endswith(("/chat/completions", "/completions")):
            body = self._maybe_inject_repetition(srv, model, payload, body)
            body = self._maybe_disable_thinking(srv, model, payload, body)
        self._acquire_and_forward(srv, model, body)

    def _handle_update_now(self, srv) -> None:
        """POST /admin/update: 新版を適用して再起動する（Ollama の「再起動して更新」相当）。

        別の手動経路が既にソースを追従済み（update_state.fetched）なら再起動だけを要求する。
        未取得なら、その場で check → apply（git pull + 依存同期。数十秒かかることがある）
        してから再起動を要求する。drain（アイドル待ち）は**しない**——ユーザーが明示的に
        「今すぐ」を選んだ操作なので、処理中のリクエストより更新を優先する。
        応答を返し切ってから再起動する（応答が途中で切れないよう少しだけ遅らせる）。

        取ってくるものが無くても、**走っているコードがディスク上のソースより古ければ
        再起動する**（restart_required。editable 運用で別経路の `git pull` が入った後の
        状態）。ここで up-to-date と答えて何もしないと、`gw update` なら直る状態が
        トレイからは直せず、更新マークを押しても消えないままになる（→ cmd_update と同じ挙動）。
        """
        request_restart = getattr(srv, "request_restart", None)
        if request_restart is None:
            send_error(
                self, 503, "restart is not available (gateway not fully started)"
            )
            return
        state = getattr(srv, "update_state", None)
        if not (state and state.get("fetched")):
            from . import update

            try:
                st = update.check(timeout=5.0)
            except Exception as exc:  # noqa: BLE001 - ネットワーク不調は 502 で返す
                send_error(self, 502, f"update check failed: {exc}")
                return
            if not st.available and not st.restart_required:
                send_json(
                    self,
                    200,
                    {
                        "object": "gateway.update",
                        "status": "up-to-date",
                        "current": st.current,
                        "latest": st.latest,
                    },
                )
                return
            if st.available:
                if not st.can_apply:
                    send_error(
                        self,
                        409,
                        f"update available but cannot apply: {st.reason}",
                    )
                    return
                try:
                    ok, msg = update.apply_update()
                except Exception as exc:  # noqa: BLE001
                    ok, msg = False, str(exc)
                if not ok:
                    send_error(self, 500, f"update failed: {msg}")
                    return
            # 新版が無くても restart_required ならここへ落ちる＝**再起動だけ**する。
            if state is not None:
                # 見せる版は「再起動後に走る版」。取得した直後はそれが最新リリース、
                # 再起動だけのときは（既に pull 済みの）ソース版。
                state.update(
                    {
                        "fetched": True,
                        "latest": st.latest if st.available else st.current,
                    }
                )
        send_json(
            self,
            200,
            {
                "object": "gateway.update",
                "status": "restarting",
                "latest": (state or {}).get("latest"),
            },
        )
        threading.Timer(0.5, request_restart).start()

    def _maybe_inject_repetition(self, srv, model, payload: dict, body: bytes) -> bytes:
        """mlx / mlx-vlm 宛のリクエストに repetition_penalty（+任意で context_size）を付与する。

        - サーバー設定が無効（None）なら何もしない（＝設定しない選択）。
        - クライアントが自分で repetition_penalty を指定していれば尊重して上書きしない。
        - バックエンドが mlx 系でなければ何もしない（llama-cpp は名前が repeat_penalty で別物）。
        戻り値は（必要なら差し替えた）リクエストボディ。
        """
        rp = getattr(srv, "repetition_penalty", None)
        if rp is None or not isinstance(model, str):
            return body
        if "repetition_penalty" in payload:
            return body
        # 構造化リクエスト保護（既定オフ）: tools（native ツールコール）や response_format
        # （構造化出力）を含むリクエストには注入しない。JSON 構文の必須の繰り返しを減点しうる
        # のを避ける保険（有効化は gateway.toml の repetition_penalty_skip_structured = true）。
        if getattr(srv, "repetition_penalty_skip_structured", False) and (
            "tools" in payload or "response_format" in payload
        ):
            return body
        try:
            backend = srv.manager.backend_for(model)
        except Exception:  # noqa: BLE001 - 判定不能なら注入しない（安全側）
            return body
        if backend not in ("mlx", "mlx-vlm"):
            return body
        payload["repetition_penalty"] = rp
        rcs = getattr(srv, "repetition_context_size", None)
        if rcs is not None and "repetition_context_size" not in payload:
            payload["repetition_context_size"] = rcs
        return json.dumps(payload).encode("utf-8")

    def _maybe_disable_thinking(self, srv, model, payload: dict, body: bytes) -> bytes:
        """mlx-vlm 宛のリクエストに reasoning_effort="none" を注入して思考を止める。

        `disable_thinking = true` を指定した [[models]] のみが対象。mlx-vlm には
        build_command 側で思考を止める手段が無い（--chat-template-args を渡すのは
        mlx / llama-cpp 経路だけ）ので、リクエスト側で落とす。

        背景: mlx-vlm サーバの既定は思考 OFF だが、それは chat template が
        `enable_thinking` を見るモデルに限った話。Inkling は **常に**
        「Thinking effort level: 0.9」をテンプレートで注入する作りで、
        enable_thinking では止まらず、OpenAI 互換の reasoning_effort でしか制御できない
        （"none"/"minimal"/"low"/"medium"/"high"/"max" または 0.0〜0.99 の float）。

        クライアントが自分で reasoning_effort / reasoning を指定していれば尊重する。
        """
        if not isinstance(model, str):
            return body
        if "reasoning_effort" in payload or "reasoning" in payload:
            return body
        try:
            if srv.manager.backend_for(model) != "mlx-vlm":
                return body
            if not srv.manager.disable_thinking_for(model):
                return body
        except Exception:  # noqa: BLE001 - 判定不能なら注入しない（安全側）
            return body
        payload["reasoning_effort"] = "none"
        return json.dumps(payload).encode("utf-8")

    def _handle_audio(self, srv, body: bytes) -> None:
        """STT（/v1/audio/transcriptions・/translations）を振り分ける。

        chat と違い body は multipart/form-data。model はフォームフィールドから拾う
        （OpenAI クライアントはここに載せる）。取れなければ query の ?model=、最後に
        default_model にフォールバックする。振り分け後は chat と同じ acquire→forward。
        """
        ctype = self.headers.get("Content-Type", "")
        model = None
        if "multipart/form-data" in ctype.lower():
            model = multipart.field(body, ctype, "model")
        if not model:
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            vals = q.get("model")
            model = vals[0] if vals else None
        model = model or srv.default_model
        if not model:
            send_error(
                self,
                400,
                "no 'model' field in the audio request and no default_model is configured",
            )
            return
        self._acquire_and_forward(srv, model, body)

    def _acquire_and_forward(self, srv, model, body: bytes) -> None:
        """model を acquire し、現在のリクエストを担当インスタンスへ中継する。

        chat（JSON）と STT（multipart）の共通処理。model 検証・容量/起動エラーの
        HTTP 変換・在席解放（release）をここに集約する。
        """
        if not isinstance(model, str):
            send_error(self, 400, "'model' must be a string")
            return
        # 動的ロードはローカルの任意パスも受け付ける（開発向け）。リモートのクライアントには
        # 許さない（ファイルシステムの探索・存在確認オラクルにさせない）。
        if model.startswith(("/", ".", "~", "\\")) and not self._client_is_loopback():
            send_error(
                self, 400, "path-like model ids are not allowed from remote clients"
            )
            return
        try:
            addr, handle = srv.manager.acquire(model)
        except KeyError:
            send_error(self, 404, f"model '{model}' is not configured in the gateway")
            return
        except ValueError as exc:
            # モデル指定/解決の不正（未キャッシュの repo-id 等）。
            send_error(self, 400, f"cannot load model '{model}': {exc}")
            return
        except GatewayDraining as exc:
            # 再起動準備中 → 一時的な 503（openai SDK は自動リトライし、新プロセスへ繋ぎ直る）。
            send_error(self, 503, str(exc))
            return
        except CapacityError as exc:
            # 全枠が処理中で空かなかった → 混雑（後で再試行を促す）。
            send_error(self, 503, f"gateway busy: {exc}")
            return
        except (RuntimeError, TimeoutError) as exc:
            send_error(self, 502, f"failed to start model '{model}': {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - 起動系の OS/バックエンド例外を HTTP 境界で閉じる
            print(f"Model startup failed for {model!r}: {exc}", file=sys.stderr)
            send_error(self, 502, f"failed to start model '{model}'")
            return
        try:
            forward(self, addr, body, srv.timeout_s)
        finally:
            srv.manager.release(handle)

    def do_DELETE(self) -> None:
        """DELETE /admin/sessions … エージェント停止時の解除（POST .../release と等価）。"""
        if not self._require_safe_browser_context():
            return
        path = self._route_path()
        if not path.endswith("/admin/sessions"):
            send_error(self, 404, f"DELETE {self.path} is not supported by the gateway")
            return
        if not self._require_api_key():  # 認可はボディを読む前に判定する
            return
        body = self._read_body()
        if body is None:
            return
        content_type = (
            self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        )
        if content_type != "application/json":
            send_error(
                self, 415, "JSON endpoints require Content-Type: application/json"
            )
            return
        try:
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, ValueError):
            send_error(self, 400, "invalid JSON body")
            return
        if not isinstance(payload, dict):
            send_error(self, 400, "JSON body must be an object")
            return
        self._handle_session_release(self.server, payload)  # type: ignore[arg-type]

    def _handle_config_update(self, srv, payload: dict) -> None:
        """POST /admin/config … 実行中の運用ポリシーを変更する（今は max_resident のみ）。

        `{"max_resident": N}` で常駐上限を変更。N は 1 以上の整数、または null / 0 /
        "off" / "unlimited" で無制限。稼働中（busy）のモデルは止めず、超過分はアイドルから
        順に非同期退避する（set_max_resident 参照）。再起動すると gateway.toml の値に戻る。
        """
        if "max_resident" not in payload:
            send_error(self, 400, "no 'max_resident' in the request")
            return
        raw = payload.get("max_resident")
        if raw in (None, 0, "", "off", "none", "unlimited"):
            value: int | None = None
        else:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                send_error(
                    self,
                    400,
                    "max_resident must be an integer >= 1 (or null/0/off for unlimited)",
                )
                return
            if value < 1:
                send_error(
                    self,
                    400,
                    "max_resident must be 1 or greater (or null/0/off for unlimited)",
                )
                return
        srv.manager.set_max_resident(value)
        srv.max_resident = value  # GET /admin/status の表示にも即反映する
        send_json(self, 200, {"object": "gateway.config", "max_resident": value})

    def _handle_session_register(self, srv, payload: dict) -> None:
        agent_id = payload.get("agent_id")
        model = payload.get("model") or srv.default_model
        if not agent_id:
            send_error(self, 400, "no 'agent_id' in the request")
            return
        if not model:
            send_error(
                self,
                400,
                "no 'model' in the request and no default_model is configured",
            )
            return
        srv.manager.register_session(str(agent_id), str(model))
        send_json(
            self,
            200,
            {
                "object": "gateway.session",
                "agent_id": agent_id,
                "model": model,
                "registered": True,
            },
        )

    def _handle_session_heartbeat(self, srv, payload: dict) -> None:
        """旧クライアント互換。生存推定には使わない（受けても何も更新しない）。

        既知の agent_id には 200、未知には 404 を返す。404 は旧 local-llm-client の
        **自己修復経路**——heartbeat 失敗で再 register する実装なので、ゲートウェイ再起動や
        アンロード時の在席掃除でセッションが消えても、次の heartbeat で登録が復元される。
        （常時 200 にすると旧クライアントが再 register せず、在席ゼロ扱いのモデルが
        他エージェントの release で使用中に落とされ得る。）
        """
        agent_id = payload.get("agent_id")
        if not agent_id:
            send_error(self, 400, "no 'agent_id' in the request")
            return
        if not srv.manager.session_known(str(agent_id)):
            send_error(self, 404, f"unknown session '{agent_id}'; register first")
            return
        send_json(
            self,
            200,
            {"object": "gateway.session", "agent_id": agent_id, "alive": True},
        )

    def _handle_session_release(self, srv, payload: dict) -> None:
        agent_id = payload.get("agent_id")
        if not agent_id:
            send_error(self, 400, "no 'agent_id' in the request")
            return
        existed = srv.manager.unregister_session(str(agent_id))
        send_json(
            self,
            200,
            {"object": "gateway.session", "agent_id": agent_id, "released": existed},
        )
