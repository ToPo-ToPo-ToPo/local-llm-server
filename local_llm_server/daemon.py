"""複数モデルを 1 つの OpenAI 互換エンドポイントで束ねる「モデル振り分けゲートウェイ」。

`local-llm-server` で起動する（カレントディレクトリの `./gateway.toml` を読む）。公開ポート（例
http://127.0.0.1:8799/v1）を 1 つだけ立て、受信した `/v1/chat/completions` の
`model` フィールドを見て、そのモデルのローカルサーバー（`LocalServer`）へ転送する。
モデルは**初回リクエスト時に遅延起動**し、`max_resident` を超えると最終利用が古い
ものから LRU で停止する。外部アプリ（Ollama / LM Studio 等）に依存せず、既存の
`LocalServer`（サブプロセス管理）と `proxy.forward`（中継）だけで完結する。

クライアント（各エージェントの agent.toml）は **base_url を公開ポート共通**にし、
各自の `model` を指定して接続する（エージェントはサーバーを起動しない）。管理者は
ゲートウェイ 1 プロセスだけを起動/停止すればよい（停止時に配下のモデルサーバーも全て止める）。
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .proxy import forward, send_error, send_json
from .server import (
    BACKENDS,
    DEFAULT_BACKEND,
    LocalServer,
    ServerConfig,
    daemon_log_path,
    ignore_shutdown_signals,
    resolve_drafter,
)

# MTP（speculative decoding）が効くバックエンド。draft_model は mlx-vlm のみ build_command が
# 反映する（他は無視）。ゲートウェイはこれを使って継承・検証する。
_MTP_BACKEND = "mlx-vlm"
# draft_model を無効化する文字列（ゲートウェイ既定を個別に打ち消すため）。
_DRAFT_OFF = ("", "off", "none")

__all__ = [
    "CapacityError",
    "GatewayConfig",
    "GatewayServer",
    "ModelManager",
    "load_gateway_config",
    "run_gateway",
]


class CapacityError(RuntimeError):
    """常駐枠が全て処理中で、待っても空かず新モデルをロードできなかった（→ 503）。"""


@dataclass
class _Model:
    """ゲートウェイが管理する 1 モデルの状態。"""

    config: ServerConfig  # host/port は内部用に割当済み
    server: LocalServer | None = None
    ready: bool = False
    inflight: int = 0  # 処理中リクエスト数（>0 の間は退避しない）
    last_used: float = 0.0  # time.monotonic()。LRU の基準
    requests: int = 0  # このモデルに振り分けた累計リクエスト数（表示用）


class ModelManager:
    """model 名 → ローカルサーバーの遅延起動・LRU 退避をするスレッドセーフな管理。

    - 既ロードかつ ready のモデルは state ロックのみの高速パスで内部アドレスを返す。
    - 未ロードは control ロックで直列化し（巨大モデルの同時ロードを防ぐ）、必要なら
      LRU で退避してから起動する。ロード済みモデルへのリクエストは高速パスで素通りする。
    - **max_resident はハードな上限**。空き枠が無く、退避できるアイドルモデルも無い
      （全て処理中）場合は、いずれかの処理が終わって枠が空くまで**待つ**（OOM を避ける）。
      `load_timeout` 秒以内に空かなければ `CapacityError`（→ 503）。
    - inflight>0 のモデルは退避対象から除外する（処理中は止めない）。
    """

    def __init__(
        self,
        configs: list[ServerConfig],
        max_resident: int | None = None,
        load_timeout: float | None = None,
    ) -> None:
        self._models: dict[str, _Model] = {c.model: _Model(config=c) for c in configs}
        self._max_resident = max_resident
        self._load_timeout = load_timeout   # 枠が空くのを待つ最大秒数（None で無期限）
        self._started = time.monotonic()    # 起動経過時間（uptime 表示用）の基準
        # registry 保護＋「枠が空いた」通知用。release で inflight→0 のとき notify する。
        self._state = threading.Condition()
        self._control = threading.Lock()    # 起動/退避（control plane）の直列化

    @property
    def model_ids(self) -> list[str]:
        return list(self._models)

    def acquire(self, model_id: str) -> tuple[tuple[str, int], _Model]:
        """model_id のサーバーを（必要なら起動して）確保し、(内部アドレス, ハンドル) を返す。

        呼び出し側は転送後に必ず release(ハンドル) すること（inflight を戻すため）。
        未知のモデルは KeyError、起動失敗は RuntimeError/TimeoutError、空き枠が
        `load_timeout` 内に得られなければ CapacityError（→ 503）を投げる。
        """
        mm = self._models.get(model_id)
        if mm is None:
            raise KeyError(model_id)
        # 高速パス: 既にロード済みなら state ロックのみ。
        with self._state:
            if mm.ready and mm.server is not None:
                mm.inflight += 1
                mm.requests += 1
                mm.last_used = time.monotonic()
                return (mm.config.host, mm.config.port), mm
        # 低速パス: ロードが要る。control で直列化する。
        with self._control:
            with self._state:
                if mm.ready and mm.server is not None:  # 待っている間に他スレッドがロード済み
                    mm.inflight += 1
                    mm.requests += 1
                    mm.last_used = time.monotonic()
                    return (mm.config.host, mm.config.port), mm
            self._evict_if_needed(keep=model_id)
            server = LocalServer(mm.config, log_path=daemon_log_path(mm.config.port))
            try:
                server.start()
                server.wait_until_ready()
            except (RuntimeError, TimeoutError):
                server.stop()
                raise
            with self._state:
                mm.server = server
                mm.ready = True
                mm.inflight += 1
                mm.requests += 1
                mm.last_used = time.monotonic()
            return (mm.config.host, mm.config.port), mm

    def release(self, mm: _Model) -> None:
        with self._state:
            if mm.inflight > 0:
                mm.inflight -= 1
                if mm.inflight == 0:
                    # 枠が空いた可能性。_evict_if_needed で待っているスレッドを起こす。
                    self._state.notify_all()

    def _evict_if_needed(self, keep: str) -> None:
        """control ロック保持下で呼ぶ。常駐数が上限なら LRU で 1 枠空ける（ハード上限）。

        退避候補は「ロード済み・処理中でない（inflight==0）・keep 以外」。候補が無い
        （全て処理中）ときは、いずれかが release されて枠が空くまで**待つ**（OOM を避ける）。
        `load_timeout` 秒以内に空かなければ `CapacityError` を投げる（呼び出し側で 503）。
        待っている間も `control` は握ったまま（他のロードは直列化）だが、`state` 条件変数は
        手放すので、ロード済みモデルへの高速パス（acquire/release）は進められる。
        """
        if self._max_resident is None:
            return
        deadline = (
            time.monotonic() + self._load_timeout if self._load_timeout else None
        )
        while True:
            victim_srv = None
            with self._state:
                resident = sum(1 for m in self._models.values() if m.server is not None)
                if resident < self._max_resident:
                    return
                candidates = [
                    m for m in self._models.values()
                    if m.server is not None and m.ready and m.inflight == 0
                    and m.config.model != keep
                ]
                if candidates:
                    victim = min(candidates, key=lambda m: m.last_used)
                    victim_srv = victim.server
                    victim.server = None
                    victim.ready = False
                else:
                    # 全て処理中 → 枠が空く（release の notify）まで待つ。
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise CapacityError(
                            f"all {self._max_resident} model slot(s) busy; could not free one "
                            f"within {self._load_timeout:g}s"
                        )
                    self._state.wait(timeout=remaining)
                    continue  # 起きたら再判定
            if victim_srv is not None:
                victim_srv.stop()  # state ロックの外で（最長 ~10s）。停止後にループ再確認。

    def evict_idle(self, timeout: float) -> int:
        """最終利用から `timeout` 秒を超えて使われていないモデルを停止する（idle TTL）。

        処理中（inflight>0）のモデルは対象外。停止した数を返す。control ロックを握って
        起動（slow path）と直列化するので、停止直後に同じモデルを再ロードする際の
        ポート再利用衝突を避けられる（fast path＝ロード済みへのリクエストは妨げない）。
        """
        if timeout <= 0:
            return 0
        now = time.monotonic()
        with self._control:
            with self._state:
                victims = []
                for m in self._models.values():
                    if (
                        m.server is not None and m.ready and m.inflight == 0
                        and (now - m.last_used) > timeout
                    ):
                        victims.append(m.server)
                        m.server = None
                        m.ready = False
            for srv in victims:
                srv.stop()  # state ロックの外で（最長 ~10s かかるため）
        return len(victims)

    def uptime(self) -> float:
        """起動からの経過秒数（表示用）。"""
        return time.monotonic() - self._started

    def status(self) -> list[dict]:
        now = time.monotonic()
        with self._state:
            out = []
            for m in self._models.values():
                loaded = m.server is not None and m.ready
                # アイドル経過は「ロード済みかつ処理中でない」ときだけ意味がある
                # （idle_timeout までの残り表示に使う）。それ以外は None。
                idle_for = round(now - m.last_used, 1) if (loaded and m.inflight == 0) else None
                out.append({
                    "model": m.config.model,
                    "backend": m.config.backend,
                    "port": m.config.port,
                    "loaded": loaded,
                    "inflight": m.inflight,
                    "requests": m.requests,
                    "idle_for": idle_for,
                })
            return out

    def shutdown(self) -> None:
        """全モデルサーバーを並列に停止する（ゲートウェイ終了時）。

        並列にするのは、外部 `--stop`（SIGTERM→10s→SIGKILL）の猶予内に収めるため。
        """
        with self._state:
            servers = []
            for m in self._models.values():
                if m.server is not None:
                    servers.append(m.server)
                    m.server = None
                    m.ready = False
        threads = [threading.Thread(target=s.stop) for s in servers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class GatewayServer(ThreadingHTTPServer):
    """model 振り分けゲートウェイの HTTP サーバー。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        addr: tuple[str, int],
        manager: ModelManager,
        catalog: list[str],
        default_model: str | None = None,
        timeout_s: float | None = None,
        max_resident: int | None = None,
        idle_timeout: float | None = None,
        load_timeout: float | None = None,
    ) -> None:
        super().__init__(addr, _GatewayHandler)
        self.manager = manager
        self.catalog = catalog            # /v1/models で返すモデル一覧
        self.default_model = default_model
        self.timeout_s = timeout_s        # None なら無制限（長時間生成に備える）
        # GET /admin/status（TUI 等の監視用）で返すゲートウェイ設定。運用ポリシーを
        # 添えることで、常駐モデルのライブ状態と一緒に「上限/退避方針」も読み取れる。
        self.max_resident = max_resident
        self.idle_timeout = idle_timeout
        self.load_timeout = load_timeout


class _GatewayHandler(BaseHTTPRequestHandler):
    server_version = "local-llm-gateway"
    # HTTP/1.0: 応答ボディは接続クローズ区切り（router と同じ）。
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args) -> None:  # アクセスログは出さない
        pass

    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        srv = self.server  # type: ignore[assignment]
        # /v1/models は設定済みカタログを合成して返す（is_ready 判定・モデル取り違え
        # 警告に対応）。実モデルは起動していなくてもカタログとして列挙する。
        if path.endswith("/models"):
            data = {
                "object": "list",
                "data": [{"id": m, "object": "model"} for m in srv.catalog],
            }
            send_json(self, 200, data)
            return
        # /admin/status は常駐モデルのライブ状態（loaded/inflight）＋運用ポリシーを返す。
        # TUI が CLI の --status より詳しい状態を出すための読み取り口。
        if path.endswith("/admin/status"):
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
                "models": models,
            }
            send_json(self, 200, data)
            return
        send_error(self, 404, f"GET {self.path} is not supported by the gateway")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        srv = self.server  # type: ignore[assignment]
        try:
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, ValueError):
            send_error(self, 400, "invalid JSON body")
            return
        model = payload.get("model") or srv.default_model
        if not model:
            send_error(self, 400, "no 'model' in the request and no default_model is configured")
            return
        try:
            addr, handle = srv.manager.acquire(model)
        except KeyError:
            send_error(self, 404, f"model '{model}' is not configured in the gateway")
            return
        except CapacityError as exc:
            # 全枠が処理中で空かなかった → 混雑（後で再試行を促す）。
            send_error(self, 503, f"gateway busy: {exc}")
            return
        except (RuntimeError, TimeoutError) as exc:
            send_error(self, 502, f"failed to start model '{model}': {exc}")
            return
        try:
            forward(self, addr, body, srv.timeout_s)
        finally:
            srv.manager.release(handle)


@dataclass
class GatewayConfig:
    host: str
    port: int
    max_resident: int | None
    default_model: str | None
    models: list[ServerConfig] = field(default_factory=list)
    idle_timeout: float | None = 1200.0  # 秒。これだけ使われないモデルを自動アンロード（既定 1200=20分。None/0 で無効）
    load_timeout: float = 300.0        # 秒。全枠処理中のとき、空くのを待つ最大時間（超過で 503）


def _resolve_model_draft(
    entry: dict, default_draft, backend: str, model: str
) -> str | None:
    """1 モデルの MTP ドラフターを解決する（個別指定 > ゲートウェイ既定）。

    - 個別の `draft_model` があればそれを、無ければゲートウェイ既定を継承する。
    - `""` / `"off"` / `"none"` で無効化（継承既定の打ち消しに使える）。
    - mlx-vlm のみ MTP が効くので、その場合だけ `resolve_drafter` で解決・検証する
      （`"auto"` が本体名から引けなければ ValueError＝起動時に即エラー）。他バックエンドでは
      無視するが、**個別に明示**されていた場合だけ「無視される」旨を警告する。
    """
    has_own = "draft_model" in entry
    raw = entry.get("draft_model", default_draft)
    if isinstance(raw, str) and raw.strip().lower() in _DRAFT_OFF:
        raw = None
    if not raw:
        return None
    if backend == _MTP_BACKEND:
        return resolve_drafter(model, raw)  # auto を解決＆未対応なら ValueError
    if backend == "llama-cpp":
        # llama.cpp のspeculative decodingはドラフト GGUF のパスを直接指定する（-md）。
        # MTP ヘッドのファイル名は build_command 側で検出して --spec-type draft-mtp を付ける。
        # "auto" の自動解決表は llama.cpp には無いので、明示パス以外は無効扱い。
        if raw == "auto":
            return None
        return raw
    if has_own:
        print(
            f"Warning: draft_model (MTP) is ignored for backend '{backend}' "
            f"(MTP needs {_MTP_BACKEND}); model {model}",
            file=sys.stderr,
        )
    return None


def load_gateway_config(path: str) -> GatewayConfig:
    """ゲートウェイ設定 TOML を読み込んで検証する。

    形式（例）:
        host = "127.0.0.1"          # 公開ホスト（省略時 127.0.0.1）
        port = 8799                 # 公開ポート（省略時 8799）
        max_resident = 2            # 同時常駐モデル数の上限（ハード。省略時 無制限）
        load_timeout = 300          # 全枠処理中のとき空くのを待つ最大秒数（超過で 503。省略時 300）
        idle_timeout = 1200         # この秒数使われないモデルを自動アンロード（省略時 1200=20分。0 で無効）
        internal_base_port = 9001   # 内部サーバーの割当開始ポート（省略時 9001）
        default_model = "..."       # model 省略リクエスト時のモデル（省略可）
        draft_model = "auto"        # 全モデルの MTP ドラフター既定（mlx-vlm のみ有効。省略可）

        [[models]]
        model = "mlx-community/Qwen3.6-27B-4bit"
        backend = "mlx-vlm"
        # draft_model 省略 → 上の既定 "auto" を継承（Qwen3.6 の MTP）

        [[models]]
        model = "mlx-community/gemma-4-31b-it-4bit"
        backend = "mlx"
        draft_model = "off"         # このモデルだけ MTP を無効化（既定の打ち消し）
    """
    import tomllib

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    host = str(data.get("host", "127.0.0.1"))
    port = int(data.get("port", 8799))
    internal_base = int(data.get("internal_base_port", 9001))
    max_resident = data.get("max_resident")
    if max_resident is not None:
        max_resident = int(max_resident)
        if max_resident < 1:
            raise ValueError("max_resident must be 1 or greater")
    default_model = data.get("default_model")
    # 一定時間使われないモデルを自動アンロードする秒数（idle TTL）。省略時 1200（=20分）、0 で無効。
    idle_timeout = data.get("idle_timeout", 1200)
    if idle_timeout is not None:
        idle_timeout = float(idle_timeout)
        if idle_timeout < 0:
            raise ValueError("idle_timeout must be 0 or greater (0 disables)")
        if idle_timeout == 0:
            idle_timeout = None
    # 全枠が処理中のとき、空くのを待つ最大秒数（超過で 503）。
    load_timeout = float(data.get("load_timeout", 300.0))
    if load_timeout < 1:
        raise ValueError("load_timeout must be 1 or greater")
    # ゲートウェイ全体の MTP ドラフター既定。各 [[models]] が draft_model を持たなければ
    # これを継承する（"auto" で本体名から自動選択）。個別に "" / "off" / "none" で無効化。
    default_draft = data.get("draft_model")

    entries = data.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("gateway config needs a non-empty [[models]] array")

    configs: list[ServerConfig] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("model"):
            raise ValueError("each [[models]] entry needs a 'model'")
        model = str(entry["model"])
        if model in seen:
            raise ValueError(f"duplicate model in gateway config: {model}")
        seen.add(model)
        backend = str(entry.get("backend", DEFAULT_BACKEND))
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS} (model {model})")
        internal_port = internal_base + i
        if internal_port == port:
            raise ValueError(
                f"internal port {internal_port} collides with the public port {port}; "
                "raise internal_base_port"
            )
        parallel = entry.get("parallel")
        if parallel is not None and int(parallel) < 1:
            raise ValueError(f"parallel must be 1 or greater (model {model})")
        draft = _resolve_model_draft(entry, default_draft, backend, model)
        configs.append(
            ServerConfig(
                backend=backend,
                model=model,
                host="127.0.0.1",
                port=internal_port,
                parallel=parallel,
                disable_thinking=bool(entry.get("disable_thinking", False)),
                draft_model=draft,
                extra_args=list(entry.get("extra_args", [])),
            )
        )

    if default_model is not None and default_model not in seen:
        raise ValueError(f"default_model '{default_model}' is not listed in [[models]]")

    return GatewayConfig(
        host, port, max_resident, default_model, configs, idle_timeout, load_timeout
    )


def run_gateway(cfg: GatewayConfig) -> int:
    """ゲートウェイを起動し、割り込み（Ctrl+C / SIGTERM）まで動かす。

    終了時に配下のモデルサーバーを全て停止する。SIGTERM/SIGHUP を
    KeyboardInterrupt に変換する install_shutdown_handlers() が呼ばれていれば、
    `kill` や `--stop`、端末クローズでも下の finally を通って後始末する。
    """
    manager = ModelManager(
        cfg.models, max_resident=cfg.max_resident, load_timeout=cfg.load_timeout
    )
    server = GatewayServer(
        (cfg.host, cfg.port),
        manager,
        catalog=[c.model for c in cfg.models],
        default_model=cfg.default_model,
        max_resident=cfg.max_resident,
        idle_timeout=cfg.idle_timeout,
        load_timeout=cfg.load_timeout,
    )
    public = f"http://{cfg.host}:{cfg.port}/v1"
    print("Gateway ready (lazy multi-model):", file=sys.stderr)
    print(f"  public: {public}", file=sys.stderr)
    for c in cfg.models:
        print(f"    {c.model}  ->  127.0.0.1:{c.port} ({c.backend})", file=sys.stderr)
    cap = "unlimited" if cfg.max_resident is None else (
        f"{cfg.max_resident} (hard; waits up to {cfg.load_timeout:g}s for a slot, else 503)"
    )
    print(f"  max resident models: {cap}", file=sys.stderr)
    print(
        f"  idle unload: {f'{cfg.idle_timeout:g}s' if cfg.idle_timeout else 'off'}",
        file=sys.stderr,
    )
    print(
        f'Point each agent.toml at base_url = "{public}" and set its own `model`. '
        "Agents only connect; models load on first request.",
        file=sys.stderr,
    )

    # 一定時間使われないモデルを自動アンロードする掃除スレッド（idle_timeout 指定時のみ）。
    stop_reaper = threading.Event()
    if cfg.idle_timeout:
        interval = min(max(cfg.idle_timeout / 2, 1.0), 30.0)  # チェック間隔（最大 30s）

        def _reaper() -> None:
            while not stop_reaper.wait(interval):
                try:
                    freed = manager.evict_idle(cfg.idle_timeout)
                    if freed:
                        print(f"Idle unload: stopped {freed} model(s).", file=sys.stderr)
                except Exception:  # noqa: BLE001 - 掃除スレッドは落とさない
                    pass

        threading.Thread(target=_reaper, daemon=True).start()

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        threading.Event().wait()  # 割り込みまでブロック（メインスレッドで受ける）
    except KeyboardInterrupt:
        pass
    finally:
        # 後始末中に再度シグナル（--stop の killpg 等で連続して届く）が来ても中断されず、
        # 配下のモデルサーバーを必ず止め切るため、まず以降のシグナルを無視にする。
        ignore_shutdown_signals()
        stop_reaper.set()
        print("\nShutting down the gateway and its model servers...", file=sys.stderr)
        server.shutdown()
        server.server_close()
        manager.shutdown()
    return 0
