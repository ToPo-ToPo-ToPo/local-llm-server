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

**在席ベースの即時アンロード（任意）**: エージェントは「このモデルを使う」ことを宣言でき、
停止時に解除できる。あるモデルの在席エージェントが 0 になったら、処理中（inflight>0）で
なければ短い猶予（_RELEASE_LINGER_S）ののち **idle_timeout を待たずにアンロード**して
メモリを解放する。チャット転送（/v1/...）とは別系統の管理エンドポイント:

  - `POST /admin/sessions/register`   `{"agent_id", "model"}`  … 利用開始（在席を宣言）
  - `POST /admin/sessions/release`    `{"agent_id"}`           … 利用終了（= `DELETE /admin/sessions`）
  - `POST /admin/sessions/heartbeat`  `{"agent_id"}`           … 旧クライアント互換の no-op

**生存推定はしない**。ハートビート途絶からエージェントの死を推定してモデルを落とすことはない
（旧実装はこれで、生成中のクライアントの足元から巨大なモデルを外す事故を起こした）。
release を送れずに落ちたエージェントの置き去りセッションは、モデルが idle_timeout で解放される
ときに一緒に掃除される。したがって**在席は解放を早めるだけで、遅らせる力を持たない**。
在席はメモリをピン留めもしない（枠が要れば従来どおり LRU 退避が優先される）。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, fields
from http.server import ThreadingHTTPServer
from typing import Callable

from . import gateway_config as _gateway_config
from . import gateway_updates as _gateway_updates
from . import provisioner, sglang_provisioner, vllm_provisioner
from . import video as video
from .gateway_config import GatewayConfig
from .gateway_errors import CapacityError
from .gateway_errors import GatewayDraining as GatewayDraining
from .gateway_http import GatewayRequestHandler as _GatewayHandler
from .gateway_manager import (
    ModelManager as _GatewayModelManager,
)
from .gateway_manager import (
    _Instance as _Instance,
)
from .gateway_manager import (
    _Model as _Model,
)
from .gateway_manager import (
    _Session as _Session,
)
from .proxy import reject_overloaded_connection
from .server import (
    DEFAULT_BACKEND,
    GatewayAlreadyRunning,
    GatewayLock,
    LocalServer,
    backend_spec,
    clear_gateway_runtime,
    enable_child_tethering,
    estimate_model_bytes,
    ignore_shutdown_signals,
    local_connect_host,
    primary_lan_ip,
    reap_orphan_workers,
    reclaim_stale_workers,
    set_llama_server_binary,
    set_sglang_python,
    set_vllm_python,
    write_gateway_runtime,
)

# 複製インスタンス起動前の猶予秒数。ストリーミングのクライアントは [DONE] を受けた
# 直後に次のリクエストを送るが、ゲートウェイ側の inflight 解放は転送スレッドが上流の
# 終端を読み切る数 ms 後になる。この隙間に届いた**逐次**リクエストが「満杯」に見えて
# 複製が誤発動しないよう、猶予を置いてから持続的な競合かを再確認する。
_REPLICA_GRACE_S = 1.0

__all__ = [
    "CapacityError",
    "GatewayConfig",
    "GatewayServer",
    "ModelManager",
    "load_gateway_config",
    "run_gateway",
]


def _total_ram() -> int | None:
    """物理メモリの総バイト数。psutil が無い／取得不可なら None。"""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - psutil 不在・取得失敗はメモリガード無効として扱う
        return None


# release 後にモデルを保持する猶予。互換用に daemon 側の設定点を維持する。
_RELEASE_LINGER_S = 60.0


class ModelManager(_GatewayModelManager):
    """互換用ファサード。実装は gateway_manager に分離している。"""

    def __init__(self, *args, **kwargs) -> None:
        # daemon の差し替え点を構築時に注入し、既存のテスト・埋め込み利用を保つ。
        kwargs["_server_factory"] = lambda config, log_path=None: LocalServer(
            config, log_path=log_path
        )
        kwargs["_estimate_model_bytes"] = estimate_model_bytes
        kwargs["_reclaim_stale"] = reclaim_stale_workers
        kwargs["_replica_grace_s"] = _REPLICA_GRACE_S
        kwargs["_release_linger_s"] = _RELEASE_LINGER_S
        super().__init__(*args, **kwargs)


# zero-drop restart で Listen ソケット fd を新イメージへ渡す環境変数。
_LISTEN_FD_ENV = "GW_LISTEN_FD"


class GatewayServer(ThreadingHTTPServer):
    """model 振り分けゲートウェイの HTTP サーバー。"""

    daemon_threads = True
    allow_reuse_address = True
    # 再起動の受け渡し窓（accept 停止〜新イメージの accept 再開。十数秒）に到着した接続は
    # カーネルの accept キューで待たせる。既定の 5 では窓の間に溢れて接続拒否になり得る。
    request_queue_size = 128

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
        api_key: str | None = None,
        video_frames: int = 8,
        video_max_edge: int = 768,
        image_max_edge: int = 1024,
        repetition_penalty: float | None = None,
        repetition_context_size: int | None = None,
        repetition_penalty_skip_structured: bool = False,
        max_request_workers: int = 32,
        max_media_workers: int = 2,
        listen_fd: int | None = None,
    ) -> None:
        if listen_fd is None:
            super().__init__(addr, _GatewayHandler)
        else:
            # zero-drop restart: 前イメージから Listen ソケットを引き継ぐ。bind/listen は
            # 行わない——受け渡し窓の間にカーネルの accept キューへ並んだ接続をそのまま引き取る。
            super().__init__(addr, _GatewayHandler, bind_and_activate=False)
            inherited = socket.socket(fileno=listen_fd)
            self.socket.close()
            self.socket = inherited
            host, port = inherited.getsockname()[:2]
            self.server_address = (host, port)
            self.server_name = socket.getfqdn(host)
            self.server_port = port
        # 生きている接続数（accept 済み〜応答完了）。quiesce_for_restart の判定に使う。
        # inflight（モデル振り分け後のカウント）より外側の数なので、「accept 済みだが
        # ボディ受信中で inflight 計上前」のリクエストも取りこぼさない。
        self._active_conns = 0
        self._conns_cv = threading.Condition()
        self._request_slots = threading.BoundedSemaphore(max_request_workers)
        self._media_slots = threading.BoundedSemaphore(max_media_workers)
        self.manager = manager
        # 繰り返しループ抑制の既定注入（mlx 系のみ。None で無効）。do_POST が chat リクエストに
        # 付与する（クライアントが自分で指定していれば尊重して上書きしない）。
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size
        # true なら tools / response_format を含む structured リクエストには注入しない（既定 false）。
        self.repetition_penalty_skip_structured = repetition_penalty_skip_structured
        self.catalog = catalog  # /v1/models で返すモデル一覧
        self.default_model = default_model
        self.timeout_s = timeout_s  # None なら無制限（長時間生成に備える）
        # 動画入力: video_url をゲートウェイでフレーム画像列に展開する設定（バックエンド非依存）。
        self.video_frames = video_frames
        self.video_max_edge = video_max_edge
        # 画像入力: 長辺がこの px を超える画像は上流へ渡す前に縮小する（0 で無効）。解像度上限の
        # 無い VLM に巨大画像を渡したときの vision トークン爆発（＝異常に遅い）を防ぐ。
        self.image_max_edge = image_max_edge
        # ネットワーク公開時の API キー（None/空 で認証なし）。chat（/v1/*）と在席セッション
        # （/admin/sessions/*）に Authorization: Bearer <key> を要求する。管理操作はループバック
        # + Host/Origin 検査で制限し、トレイ/CLI がキーを別経路で持つ必要を無くす。
        self.api_key = api_key
        # GET /admin/status（TUI 等の監視用）で返すゲートウェイ設定。運用ポリシーを
        # 添えることで、常駐モデルのライブ状態と一緒に「上限/退避方針」も読み取れる。
        self.max_resident = max_resident
        self.idle_timeout = idle_timeout
        self.load_timeout = load_timeout
        # 起動元情報（provenance）。「いつ・どこから立ったゲートウェイか」を /admin/status で
        # 見えるようにする。起動経路は `gw start` の 1 本だけ（__main__ が spawn マークの無い
        # 直接起動を拒否する）ので、経路の識別（旧 launcher フィールド）は無い。
        self.pid = os.getpid()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.start_cwd = os.getcwd()
        # HTTP 境界へ更新サービスを注入する。gateway_http から daemon を逆参照させない。
        self.refresh_update_state = lambda *, wait=False: maybe_refresh_update_state(
            self, wait=wait
        )
        # Update control is part of the server's real interface.  Initialising it
        # here avoids a partially constructed object whose attributes depend on
        # _run_gateway_locked having reached a later phase.
        self.update_state: dict[str, object] = {
            "available": False,
            "current": None,
            "latest": None,
            "fetched": False,
            "reason": None,
        }
        self.request_restart: Callable[[], None] | None = None
        self._last_update_check = 0.0
        self._update_check_inflight = False
        self._update_check_done: threading.Event | None = None

    # --- zero-drop restart（Listen ソケット引き継ぎ）------------------------------
    #
    # 再起動でリクエストを 1 つも落とさないための 3 点セット:
    #   ① quiesce_for_restart: accept を止め、処理中の接続が掃けるのを待つ。Listen ソケットは
    #      開いたままなので、以後の新規接続は拒否されずカーネルの accept キューで待つ。
    #   ② detach_listen_fd: fd を所有権ごと取り出して execv を生き延びさせる。
    #   ③ __init__(listen_fd=...): 新イメージが fd を引き継ぎ、キューの接続を順に処理する。
    # クライアントから見ると「再起動の窓に投げた 1 発」は失敗せず、少し待たされるだけになる。

    def process_request(self, request, client_address):
        # accept スレッド側で数える（ワーカースレッド開始後に数えると、開始前の隙間が
        # quiesce の判定から漏れる）。減算は shutdown_request（全経路で 1 回呼ばれる）。
        if not self._request_slots.acquire(blocking=False):
            reject_overloaded_connection(request, self.close_request)
            return
        with self._conns_cv:
            self._active_conns += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._conns_cv:
                self._active_conns = max(0, self._active_conns - 1)
                self._conns_cv.notify_all()
            self._request_slots.release()
            raise

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            with self._conns_cv:
                self._active_conns = max(0, self._active_conns - 1)
                self._conns_cv.notify_all()
            self._request_slots.release()

    def quiesce_for_restart(self, timeout: float = 5.0) -> bool:
        """accept を止め、処理中の接続が掃けるのを待つ。成功なら True。

        成功後も Listen ソケットは開いたままなので、以後に来た接続は接続拒否にも 503 にも
        ならず accept キューに並ぶ（request_queue_size 分）。新イメージが引き継いだ時点で
        順に処理される。timeout 内に掃けなければ accept を再開して False（受信中・生成中の
        接続は切らない。呼び出し側は次周期で再試行する）。

        旧実装（begin_drain の inflight 判定＋503）と違い、接続数で見るので「accept 済みだが
        ボディ受信中で inflight 計上前」のリクエストも取りこぼさない。
        """
        self.shutdown()  # serve_forever を止める（Listen ソケットは閉じない）
        with self._conns_cv:
            deadline = time.monotonic() + timeout
            while self._active_conns > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._conns_cv.wait(remaining)
            idle = self._active_conns == 0
        if idle:
            return True
        threading.Thread(target=self.serve_forever, daemon=True).start()  # 再開
        return False

    def detach_listen_fd(self) -> int | None:
        """Listen ソケットの fd を所有権ごと取り出し、execv を生き延びるよう継承可能にする。

        detach 後のソケットオブジェクトは fd を閉じない（GC で閉じられると accept キューの
        接続ごと失われるため、所有権を外すことが本質）。

        引き継ぎが成立しないプラットフォーム（Windows のソケットハンドルは CRT の fd では
        ないため os.set_inheritable が Errno 9 で落ちる）では None を返し、呼び出し側は
        通常のクローズにフォールバックする。ここで例外を投げると、呼び出し元の finally が
        途中で抜けて**モデルサーバーを止め切らずにゲートウェイが落ちる**。zero-drop を
        諦めるだけなら再起動は成立するので、失敗は握って落とさない。
        """
        fd = self.socket.detach()
        try:
            os.set_inheritable(fd, True)
        except OSError:
            try:
                socket.socket(fileno=fd).close()  # detach した所有権を回収して閉じる
            except OSError:
                pass
            return None
        return fd


def load_gateway_config(path: str) -> GatewayConfig:
    """互換用ファサード。設定解析は gateway_config に分離している。"""
    return _gateway_config.load_gateway_config(path, default_backend=DEFAULT_BACKEND)


# gateway.toml を保存した瞬間に反映するホットリロードの監視周期（秒）。mtime ポーリング。
_CONFIG_POLL_INTERVAL = 1.0
# 稼働中には変えられない構造設定（ソケットは bind 済み、内部ポート割当は起動時に固定）。
# 変更を検知したら「要再起動」を警告するだけで、サーバーは止めず旧値のまま動かし続ける。
_RESTART_ONLY_FIELDS = (
    "host",
    "port",
    "internal_base_port",
    "models",
    "max_request_workers",
    "max_media_workers",
    "tray",
    # llama-server バイナリ・vLLM/SGLang venv は起動時に導入・解決するため、変更は再起動が要る。
    "llama_accel",
    "llama_build",
)


def apply_live_config(
    server: GatewayServer,
    manager: ModelManager,
    cfg: GatewayConfig,
    new: GatewayConfig,
) -> tuple[list[str], list[str]]:
    """読み直した設定 `new` を稼働中の server / manager / cfg へ無停止で反映する。

    ポリシー設定（各 timeout・max_resident・api_key・動的ロード既定）は
    その場で差し替える。動的ロード既定（draft_model / parallel / disable_thinking /
    max_memory_fraction / dynamic / start_timeout）は**次回ロードから**有効。
    host / port / internal_base_port / [[models]] は稼働中に変えられない（ソケット bind 済み・
    ポート割当の一貫性）ので、変更検知時は「要再起動」として警告用リストに積むだけで適用しない。

    戻り値 `(changed, restart_needed)`: それぞれ変更フィールドの人間向け説明。副作用として
    server / manager と、掃除スレッドが毎周期読む `cfg` を書き換える（末尾で cfg を new に揃える）。
    """
    changed: list[str] = []
    restart_needed: list[str] = []

    def note(label: str, old, newv) -> None:
        changed.append(f"{label}: {old!r} → {newv!r}")

    # --- 稼働中に変えられない構造設定は警告のみ（適用しない） ---
    for fld in _RESTART_ONLY_FIELDS:
        if getattr(cfg, fld) != getattr(new, fld):
            restart_needed.append(fld)

    # --- max_resident: 退避を伴うので専用セッター経由（超過分は非同期 LRU 退避） ---
    if cfg.max_resident != new.max_resident:
        note("max_resident", cfg.max_resident, new.max_resident)
        manager.set_max_resident(new.max_resident)
        server.max_resident = new.max_resident

    # --- サーバーがリクエスト毎に読むポリシー ---
    if cfg.video_frames != new.video_frames:
        note("video_frames", cfg.video_frames, new.video_frames)
        server.video_frames = new.video_frames
    if cfg.video_max_edge != new.video_max_edge:
        note("video_max_edge", cfg.video_max_edge, new.video_max_edge)
        server.video_max_edge = new.video_max_edge
    if cfg.image_max_edge != new.image_max_edge:
        note("image_max_edge", cfg.image_max_edge, new.image_max_edge)
        server.image_max_edge = new.image_max_edge
    if cfg.repetition_penalty != new.repetition_penalty:
        note("repetition_penalty", cfg.repetition_penalty, new.repetition_penalty)
        server.repetition_penalty = new.repetition_penalty
    if cfg.repetition_context_size != new.repetition_context_size:
        note(
            "repetition_context_size",
            cfg.repetition_context_size,
            new.repetition_context_size,
        )
        server.repetition_context_size = new.repetition_context_size
    if cfg.repetition_penalty_skip_structured != new.repetition_penalty_skip_structured:
        note(
            "repetition_penalty_skip_structured",
            cfg.repetition_penalty_skip_structured,
            new.repetition_penalty_skip_structured,
        )
        server.repetition_penalty_skip_structured = (
            new.repetition_penalty_skip_structured
        )
    if cfg.default_model != new.default_model:
        note("default_model", cfg.default_model, new.default_model)
        server.default_model = new.default_model
    if cfg.request_timeout != new.request_timeout:
        note("request_timeout", cfg.request_timeout, new.request_timeout)
        server.timeout_s = new.request_timeout
    if cfg.api_key != new.api_key:
        # キー実体はログに出さない（設定の有無だけ示す）。
        note("api_key", "set" if cfg.api_key else None, "set" if new.api_key else None)
        server.api_key = new.api_key

    # --- 掃除スレッドが cfg から毎周期読む閾値（server にも監視表示用のコピーを持つ） ---
    if cfg.idle_timeout != new.idle_timeout:
        note("idle_timeout", cfg.idle_timeout, new.idle_timeout)
        server.idle_timeout = new.idle_timeout
    if cfg.load_timeout != new.load_timeout:
        note("load_timeout", cfg.load_timeout, new.load_timeout)
        server.load_timeout = new.load_timeout
        manager._load_timeout = new.load_timeout

    # --- 動的ロードの既定（次回ロードから有効） ---
    if cfg.start_timeout != new.start_timeout:
        note("start_timeout", cfg.start_timeout, new.start_timeout)
        manager._start_timeout = new.start_timeout
    if cfg.dynamic != new.dynamic:
        note("dynamic", cfg.dynamic, new.dynamic)
        manager._dynamic = new.dynamic
    if cfg.disable_thinking != new.disable_thinking:
        note("disable_thinking", cfg.disable_thinking, new.disable_thinking)
        manager._default_disable_thinking = new.disable_thinking
    if cfg.stream_tool_calls != new.stream_tool_calls:
        note("stream_tool_calls", cfg.stream_tool_calls, new.stream_tool_calls)
        manager._default_stream_tool_calls = new.stream_tool_calls
    if cfg.draft_model != new.draft_model:
        note("draft_model", cfg.draft_model, new.draft_model)
        manager._default_draft = new.draft_model
    if cfg.parallel != new.parallel:
        note("parallel", cfg.parallel, new.parallel)
        manager._default_parallel = new.parallel
    if cfg.max_memory_fraction != new.max_memory_fraction:
        # 有効化するには総RAMが要る。取得できていなければ適用を見送って警告に回す。
        if new.max_memory_fraction and manager._mem_total is None:
            total = _total_ram()
            if not total:
                restart_needed.append(
                    "max_memory_fraction (total RAM を取得できず未適用)"
                )
            else:
                manager._mem_total = total
                manager._mem_fraction = new.max_memory_fraction
                note(
                    "max_memory_fraction",
                    cfg.max_memory_fraction,
                    new.max_memory_fraction,
                )
        else:
            manager._mem_fraction = new.max_memory_fraction
            note(
                "max_memory_fraction", cfg.max_memory_fraction, new.max_memory_fraction
            )

    # cfg を new に揃える: ①掃除スレッドが cfg.idle_timeout を毎周期読む
    # ②次回リロードの比較基準を「今の設定」にして、未適用の構造設定を毎回警告し続けないため。
    # 構造設定(host/port/...)も cfg 上は new に寄せる（稼働中の bind 済みソケットは旧値のまま
    # だが、cfg のこれらは起動時以外に参照されない）。
    for f in fields(GatewayConfig):
        setattr(cfg, f.name, getattr(new, f.name))

    return changed, restart_needed


def watch_config_file(
    server: GatewayServer,
    manager: ModelManager,
    cfg: GatewayConfig,
    config_path: str,
    stop_event: threading.Event,
    poll_interval: float = _CONFIG_POLL_INTERVAL,
) -> None:
    """gateway.toml の mtime を監視し、保存された瞬間に apply_live_config で無停止反映する。

    `stop_event` がセットされるまでポーリングし続ける（掃除スレッドと同じ停止イベントを共有）。
    編集途中の不正な TOML は握りつぶして旧設定のまま動かし続け、同じ mtime では再警告しない。
    """
    try:
        last_mtime = os.path.getmtime(config_path)
    except OSError:
        last_mtime = None
    skip_mtime = (
        None  # 直近に読み込み失敗した mtime（同一内容の再警告・再試行を避ける）
    )
    while not stop_event.wait(poll_interval):
        try:
            mtime = os.path.getmtime(config_path)
        except OSError:
            continue  # 一時的に消えた（エディタの原子的保存の隙間など）。次周期で拾う。
        if mtime == last_mtime or mtime == skip_mtime:
            continue
        try:
            new_cfg = load_gateway_config(config_path)
        except (
            OSError,
            ValueError,
        ) as exc:  # TOMLDecodeError も ValueError の subclass
            skip_mtime = mtime
            print(
                f"Config reload skipped (invalid gateway.toml, keeping current "
                f"settings): {exc}",
                file=sys.stderr,
            )
            continue
        try:
            changed, restart_needed = apply_live_config(server, manager, cfg, new_cfg)
        except Exception as exc:  # noqa: BLE001 - 監視スレッドは落とさない
            skip_mtime = mtime
            print(f"Config reload failed to apply: {exc}", file=sys.stderr)
            continue
        last_mtime = mtime
        skip_mtime = None
        if changed:
            print(
                "Config reloaded (applied live): " + "; ".join(changed), file=sys.stderr
            )
        if restart_needed:
            print(
                "Config reloaded: these changes need a restart to take effect "
                "(still running with the old values): " + ", ".join(restart_needed),
                file=sys.stderr,
            )
        if not changed and not restart_needed:
            print("Config reloaded: no effective change.", file=sys.stderr)


# 手動更新を適用したので新コードで再起動したい、を表す内部終了コード（run_gateway が execv）。
# 通常終了(0)・既に起動済み(3)と衝突しない値。
_RESTART_CODE = 7

# 更新確認ウォッチャーの周期（秒）。モジュール定数にして差し替え可能にする。
_UPDATE_WARMUP_INTERVAL = (
    60.0  # 起動直後は 1 分だけ待ってから初回チェック（起動処理と競合させない）
)
_UPDATE_CHECK_INTERVAL = 3600.0  # 以降、新版が未検知のあいだの確認周期
# オンデマンド確認（トレイのメニューを開くたび = /admin/status GET）のスロットル。
# タグ照会しすぎないための最短間隔。定期チェック（1時間）より短く、確認をほぼ即時にする。
_UPDATE_ONDEMAND_THROTTLE = 30.0


def refresh_update_state(state: dict) -> None:
    """互換用ファサード。"""
    _gateway_updates.refresh_update_state(state)


def maybe_refresh_update_state(srv, *, wait: bool = False) -> None:
    """互換用ファサード。実行時のスロットル値と差し替え関数を渡す。"""
    _gateway_updates.maybe_refresh_update_state(
        srv,
        wait=wait,
        throttle=_UPDATE_ONDEMAND_THROTTLE,
        refresh=refresh_update_state,
    )


def _update_watcher(
    stop: threading.Event,
    *,
    state: dict | None = None,
    notify=None,
) -> None:
    """互換用ファサード。テスト時に変更可能な周期を明示的に注入する。"""
    _gateway_updates.update_watcher(
        stop,
        state=state,
        notify=notify,
        warmup_interval=_UPDATE_WARMUP_INTERVAL,
        check_interval=_UPDATE_CHECK_INTERVAL,
    )


def run_gateway(cfg: GatewayConfig, config_path: str | None = None) -> int:
    """ゲートウェイを起動し、割り込み（Ctrl+C / SIGTERM）まで動かす。

    終了時に配下のモデルサーバーを全て停止する。SIGTERM/SIGHUP を
    KeyboardInterrupt に変換する install_shutdown_handlers() が呼ばれていれば、
    `kill` や TUI からの停止、端末クローズでも下の finally を通って後始末する。

    起動時にマシン単位の単一起動ロック（GatewayLock）を取る。既に別のゲートウェイが
    起動していれば、2 個目を立てずに明示エラー（戻り値 3）で終わる。これで開発ツール等が
    別ディレクトリから勝手に起動してもゲートウェイが乱立しない（1 マシン 1 ゲートウェイ）。

    `config_path`（gateway.toml のパス）を渡すと、そのファイルを保存した瞬間にポリシー設定を
    無停止で反映するホットリロード監視を有効にする（→ apply_live_config）。
    """
    # 単一起動ガード: サーバー本体（ポート bind やモデル起動）に入る前に取る。
    try:
        lock = GatewayLock().acquire()
    except GatewayAlreadyRunning as exc:
        print(f"Refusing to start: {exc}", file=sys.stderr)
        return 3
    # 起動時の孤児掃除（crash-only: 起動処理 = 復旧処理）。前回のゲートウェイが kill -9 や
    # クラッシュで死んでいた場合、ワーカー台帳に残る「生きていて自分由来の」プロセスだけを
    # ここで回収する。ロック取得後なので、稼働中ゲートウェイの現役ワーカーを誤射しない。
    try:
        orphans = reap_orphan_workers()
    except Exception:  # noqa: BLE001 - 掃除失敗（psutil 不在等）で起動を止めない
        orphans = []
    if orphans:
        print(
            f"Startup reconciliation: reclaimed {len(orphans)} orphaned worker(s) "
            f"{orphans} left behind by a previous gateway.",
            file=sys.stderr,
        )
    # 以後に起動するワーカーをこのプロセスへ繋留する（デーモンが死ねばワーカーも死ぬ）。
    enable_child_tethering()
    try:
        rc = _run_gateway_locked(cfg, config_path)
    finally:
        lock.release()
    # 手動更新を適用したら、ロックとポートを解放し切った **後** で自分自身を
    # 新コードに置き換える（execv は fd を引き継ぐので、ロック保持中に再取得すると自分と
    # 衝突する。必ず lock.release() を通してから exec する）。exec は戻らない。
    if rc == _RESTART_CODE:
        from . import update

        # 依存の入れ直しは再起動の直前（全ワーカー停止済み・自分は exec 目前）に行う——
        # tool venv（make install 導入）は uv sync では更新されず、これを怠るとコードだけ
        # 新しく依存が古い「静かな機能欠け」になる（例: pyobjc 不在でトレイが出ない）。
        ok, msg = update.refresh_tool_env(update.repo_root())
        print(f"Manual update: dependencies — {msg}", file=sys.stderr)
        if not ok:
            print(
                "Manual update: 依存の入れ直しに失敗しました。挙動がおかしい場合は "
                "`make install` を実行してください。",
                file=sys.stderr,
            )
        update.reexec_daemon()
    return rc


def _llama_cpp_in_use(cfg: GatewayConfig) -> bool:
    """この構成で llama-server（llama-cpp）が使われ得るか。

    - 事前登録に llama-cpp モデルがあれば True。
    - 動的ロード有効で OS 既定バックエンドが llama-cpp（＝非 Apple Silicon）なら True。
    Apple Silicon で llama-cpp モデルの登録が無い場合は、既定が mlx-vlm なので False
    （GGUF を明示要求したときだけ llama-server が要る。その場合は PATH / system で賄う）。
    """
    if any(backend_spec(c.backend).provisioner == "llama" for c in cfg.models):
        return True
    return cfg.dynamic and DEFAULT_BACKEND == "llama-cpp"


def provision_llama_if_needed(cfg: GatewayConfig) -> None:
    """必要なら起動時に llama-server を自動導入し、build_command に使わせる。

    ダウンロード（初回のみ）を初回推論のレイテンシに混ぜないよう起動時に済ませる。
    導入に失敗してもゲートウェイは起動する（mlx 等は動く。llama-cpp モデルの要求時に
    分かりやすいエラーになる）。llama-cpp を使わない構成では何もしない（macOS で不要な
    ダウンロードをしないため）。
    """
    if not _llama_cpp_in_use(cfg):
        return
    try:
        binary, info = provisioner.ensure_llama_server(
            accel=cfg.llama_accel,
            build=cfg.llama_build,
        )
    except Exception as exc:  # noqa: BLE001 - 導入失敗で起動を止めない（オフライン・未知アーキ等も含む）
        print(
            f"llama.cpp provisioning failed (continuing without it): {exc}",
            file=sys.stderr,
        )
        return
    # 実際に解決された素性（実ビルド番号・accel）ごと登録する。
    set_llama_server_binary(binary, build=info.get("build"), accel=info.get("accel"))
    print(
        f"llama.cpp ready: {binary} "
        f"(build={info.get('build') or '-'}, accel={info.get('accel') or '-'})",
        file=sys.stderr,
    )


def _vllm_in_use(cfg: GatewayConfig) -> bool:
    """事前登録に backend="vllm" のモデルがあるか（vLLM は明示 opt-in 専用）。"""
    return any(backend_spec(c.backend).provisioner == "vllm" for c in cfg.models)


def provision_vllm_if_needed(cfg: GatewayConfig) -> None:
    """vllm モデルが登録された構成のときだけ、起動時に vLLM を隔離 venv へ導入する。

    導入は数 GB・数分かかる（初回のみ）。失敗してもゲートウェイは起動を続ける
    （他バックエンドは動く。vllm モデルの要求時に分かりやすいエラーになる）。
    """
    if not _vllm_in_use(cfg):
        return
    try:
        py = vllm_provisioner.ensure_vllm()
    except Exception as exc:  # noqa: BLE001 - 導入失敗で起動を止めない（GPU 非検出・pip 失敗等）
        print(
            f"vLLM provisioning failed (continuing without it): {exc}", file=sys.stderr
        )
        return
    set_vllm_python(py)
    print(f"vLLM ready: {py}", file=sys.stderr)


def _sglang_in_use(cfg: GatewayConfig) -> bool:
    """事前登録に backend="sglang" のモデルがあるか（SGLang は明示 opt-in 専用）。"""
    return any(backend_spec(c.backend).provisioner == "sglang" for c in cfg.models)


def provision_sglang_if_needed(cfg: GatewayConfig) -> None:
    """sglang モデルが登録された構成のときだけ、起動時に SGLang を隔離 venv へ導入する。

    導入は数 GB・数分かかる（初回のみ）。失敗してもゲートウェイは起動を続ける
    （他バックエンドは動く。sglang モデルの要求時に分かりやすいエラーになる）。
    """
    if not _sglang_in_use(cfg):
        return
    try:
        py = sglang_provisioner.ensure_sglang()
    except Exception as exc:  # noqa: BLE001 - 導入失敗で起動を止めない（GPU 非検出・pip 失敗等）
        print(
            f"SGLang provisioning failed (continuing without it): {exc}",
            file=sys.stderr,
        )
        return
    set_sglang_python(py)
    print(f"SGLang ready: {py}", file=sys.stderr)


def _maybe_spawn_tray(cfg: GatewayConfig) -> tuple[subprocess.Popen | None, int | None]:
    """メニューバーアイコン（tray.py）を随伴プロセスとして起動する（macOS・tray=true のみ）。

    デーモンと同じプロセスグループに置く（`gw stop` の killpg で一緒に止まる）うえ、
    トレイ**専用の**パイプを渡す——EOF（デーモンの死。kill -9 でも OS が閉じる）で
    アイコンが自分から消えるのはワーカーの繋留と同じで、加えてこのパイプは
    **更新通知の下り線**を兼ねる（update watcher が `update-available <ver>` を書くと
    トレイが更新マークを出す）。ワーカーの繋留パイプと分けるのは、パイプは放送ではなく
    早い者勝ちの読み取りなので、通知がワーカー側ラッパーに食われないため。
    起動失敗（rumps 不在等）は無視する——アイコンは飾りで、ゲートウェイの本体機能ではない。
    戻り値は (プロセス, 書き込み端 fd)。起動しなかったときは (None, None)。
    """
    if sys.platform != "darwin" or not cfg.tray:
        return None, None
    try:
        rfd, wfd = os.pipe()
    except OSError:
        return None, None
    cmd = [
        sys.executable,
        "-m",
        "local_llm_server.tray",
        "--host",
        local_connect_host(cfg.host),
        "--port",
        str(cfg.port),
        "--fd",
        str(rfd),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            pass_fds=(rfd,),  # stdout/err はデーモンのログへ
        )
    except OSError as exc:
        print(f"tray icon not started (continuing without it): {exc}", file=sys.stderr)
        os.close(rfd)
        os.close(wfd)
        return None, None
    finally:
        # 読み取り端は子（トレイ）だけが持つ。親が持ち続けると EOF が永遠に来ない。
        try:
            os.close(rfd)
        except OSError:
            pass
    return proc, wfd


@dataclass
class _GatewayRunResources:
    """Own every resource acquired by one gateway run transaction."""

    tray_proc: subprocess.Popen | None = None
    tray_fd: int | None = None
    manager: ModelManager | None = None
    server: GatewayServer | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    threads: list[threading.Thread] = field(default_factory=list)
    server_thread: threading.Thread | None = None
    runtime_written: bool = False
    restart: bool = False
    closed: bool = False

    def start_thread(
        self,
        *,
        target,
        name: str,
        args: tuple = (),
        kwargs: dict | None = None,
        server: bool = False,
    ) -> threading.Thread:
        thread = threading.Thread(
            target=target,
            name=name,
            args=args,
            kwargs=kwargs or {},
            daemon=True,
        )
        thread.start()
        self.threads.append(thread)
        if server:
            self.server_thread = thread
        return thread

    def close(self) -> None:
        """Stop resources in dependency order; safe after partial startup."""
        if self.closed:
            return
        self.closed = True
        ignore_shutdown_signals()
        self.stop_event.set()

        if self.server is not None and self.server_thread is not None:
            if self.server_thread.is_alive():
                try:
                    self.server.shutdown()
                except Exception as exc:  # noqa: BLE001 - continue full cleanup
                    print(f"gateway shutdown failed: {exc}", file=sys.stderr)
            self.server_thread.join(timeout=10.0)

        # Watchers use stop_event.wait(), so they normally exit immediately.  A
        # bounded join prevents a failed network check from hanging shutdown.
        for thread in self.threads:
            if thread is self.server_thread or thread is threading.current_thread():
                continue
            thread.join(timeout=10.0)

        if self.server is not None:
            done = self.server._update_check_done
            if done is not None:
                done.wait(4.0)
            try:
                if self.restart:
                    fd = self.server.detach_listen_fd()
                    if fd is None:
                        os.environ.pop(_LISTEN_FD_ENV, None)
                    else:
                        os.environ[_LISTEN_FD_ENV] = str(fd)
                else:
                    self.server.server_close()
            except Exception as exc:  # noqa: BLE001 - continue worker cleanup
                print(f"gateway socket cleanup failed: {exc}", file=sys.stderr)

        if self.manager is not None:
            try:
                self.manager.shutdown()
            except Exception as exc:  # noqa: BLE001 - continue tray/record cleanup
                print(f"model manager shutdown failed: {exc}", file=sys.stderr)

        if self.tray_fd is not None:
            try:
                os.close(self.tray_fd)
            except OSError:
                pass
            self.tray_fd = None
        if self.tray_proc is not None:
            try:
                if self.tray_proc.poll() is None:
                    try:
                        self.tray_proc.wait(timeout=1.0)  # pipe EOF normally exits
                    except subprocess.TimeoutExpired:
                        self.tray_proc.terminate()
                        try:
                            self.tray_proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            self.tray_proc.kill()
                            self.tray_proc.wait(timeout=2.0)
            except (OSError, subprocess.SubprocessError):
                pass

        if self.runtime_written and not self.restart:
            clear_gateway_runtime()


def _run_gateway_locked(cfg: GatewayConfig, config_path: str | None = None) -> int:
    """単一起動ロック取得済みで実際にゲートウェイを回す本体（run_gateway が呼ぶ）。

    `config_path` が渡されれば、gateway.toml を保存した瞬間に設定を無停止で反映する
    ホットリロード監視スレッドを起動する（→ apply_live_config）。
    """
    from . import update

    update.mark_running_source()
    resources = _GatewayRunResources()
    resources.tray_proc, resources.tray_fd = _maybe_spawn_tray(cfg)
    try:
        provision_llama_if_needed(cfg)
        provision_vllm_if_needed(cfg)
        provision_sglang_if_needed(cfg)
        resources.manager = ModelManager(
            cfg.models,
            max_resident=cfg.max_resident,
            load_timeout=cfg.load_timeout,
            start_timeout=cfg.start_timeout,
            dynamic=cfg.dynamic,
            default_disable_thinking=cfg.disable_thinking,
            default_stream_tool_calls=cfg.stream_tool_calls,
            default_draft=cfg.draft_model,
            default_parallel=cfg.parallel,
            max_memory_fraction=cfg.max_memory_fraction,
            internal_base_port=cfg.internal_base_port,
            public_port=cfg.port,
        )
        return _run_gateway_session(cfg, config_path, resources)
    finally:
        resources.close()


def _run_gateway_session(
    cfg: GatewayConfig,
    config_path: str | None,
    resources: _GatewayRunResources,
) -> int:
    """Run an acquired gateway session; resources owns every cleanup path."""
    tray_fd = resources.tray_fd
    manager = resources.manager
    assert manager is not None

    def _tray_notify(line: str) -> None:
        """トレイへ 1 行通知する（トレイ無し・死亡済みは黙って無視）。"""
        if tray_fd is None:
            return
        try:
            os.write(tray_fd, (line + "\n").encode("utf-8"))
        except OSError:
            pass

    def _make_server(listen_fd: int | None) -> GatewayServer:
        return GatewayServer(
            (cfg.host, cfg.port),
            manager,
            catalog=[c.model for c in cfg.models],
            default_model=cfg.default_model,
            timeout_s=cfg.request_timeout,
            max_resident=cfg.max_resident,
            idle_timeout=cfg.idle_timeout,
            load_timeout=cfg.load_timeout,
            api_key=cfg.api_key,
            video_frames=cfg.video_frames,
            video_max_edge=cfg.video_max_edge,
            image_max_edge=cfg.image_max_edge,
            repetition_penalty=cfg.repetition_penalty,
            repetition_context_size=cfg.repetition_context_size,
            repetition_penalty_skip_structured=cfg.repetition_penalty_skip_structured,
            max_request_workers=cfg.max_request_workers,
            max_media_workers=cfg.max_media_workers,
            listen_fd=listen_fd,
        )

    # zero-drop restart: 前イメージが detach した Listen ソケットが在れば引き継ぐ
    # （bind し直さない＝再起動の窓に accept キューへ並んだ接続をそのまま処理する）。
    # pop するので、この後に起動する子プロセスへ fd 番号が漏れることは無い。
    inherited_fd: int | None = None
    raw_fd = os.environ.pop(_LISTEN_FD_ENV, None)
    if raw_fd:
        try:
            inherited_fd = int(raw_fd)
        except ValueError:
            inherited_fd = None
    try:
        server = _make_server(inherited_fd)
        if inherited_fd is not None:
            print(
                "Zero-drop restart: adopted the listening socket from the previous "
                "image; connections that arrived during the restart are being served.",
                file=sys.stderr,
            )
    except Exception:
        if inherited_fd is None:
            raise
        # 引き継ぎに失敗（fd が壊れている等）。fd を閉じてから通常 bind へフォールバック
        # （閉じないと同ポートの bind が EADDRINUSE で失敗する。キューの接続は失われるが、
        # このパスは fd 破損という異常時のみ）。
        try:
            os.close(inherited_fd)
        except OSError:
            pass
        print(
            "Zero-drop restart: failed to adopt the inherited socket; "
            "falling back to a fresh bind.",
            file=sys.stderr,
        )
        server = _make_server(None)
    resources.server = server
    public = f"http://{cfg.host}:{cfg.port}/v1"
    wildcard = cfg.host in ("0.0.0.0", "")
    # ループバック以外へ bind したら「公開」扱い（特定 LAN IP への bind も外から届く）。
    exposed = wildcard or cfg.host not in ("127.0.0.1", "localhost", "::1")
    print("Gateway ready (lazy multi-model):", file=sys.stderr)
    print(f"  public: {public}", file=sys.stderr)
    # 全インターフェース公開時は、リモートのクライアントが指す LAN URL を案内する。
    if wildcard:
        lan = primary_lan_ip()
        if lan:
            print(f"  reachable from LAN: http://{lan}:{cfg.port}/v1", file=sys.stderr)
    # ネットワーク公開の認証状態。未認証公開は明示 opt-in 済みだが、常に目立つ警告を出す。
    if cfg.api_key:
        print("  auth: API key required (Authorization: Bearer <key>)", file=sys.stderr)
    elif exposed:
        print(
            "  WARNING: bound to a network interface WITHOUT an api_key — anyone who can "
            "reach this host:port can use the models (explicit unsafe opt-in).",
            file=sys.stderr,
        )
    print(
        "  admin (/admin/status, /admin/config, /admin/update): localhost only",
        file=sys.stderr,
    )
    for c in cfg.models:
        print(f"    {c.model}  ->  127.0.0.1:{c.port} ({c.backend})", file=sys.stderr)
    cap = (
        "unlimited"
        if cfg.max_resident is None
        else (
            f"{cfg.max_resident} (hard; waits up to {cfg.load_timeout:g}s for a slot, else 503)"
        )
    )
    print(f"  max resident models: {cap}", file=sys.stderr)
    if cfg.max_memory_fraction:
        total = _total_ram()
        budget = f"{total * cfg.max_memory_fraction / 1e9:.1f}GB" if total else "?"
        print(
            f"  memory cap: {cfg.max_memory_fraction:g} of RAM (~{budget}); "
            "refuses a load that would exceed it (evicts idle first, else 503)",
            file=sys.stderr,
        )
    print(
        f"  idle unload: {f'{cfg.idle_timeout:g}s' if cfg.idle_timeout else 'off'}",
        file=sys.stderr,
    )
    print(
        f"  session unload: {_RELEASE_LINGER_S:g}s after the last agent releases "
        "(no heartbeat/liveness guessing; strays are swept on idle unload)",
        file=sys.stderr,
    )
    if cfg.image_max_edge:
        print(
            f"  image downscale: longest edge -> {cfg.image_max_edge}px "
            "(data URLs only; keeps vision tokens from exploding)",
            file=sys.stderr,
        )
    print(
        f'Point each agent.toml at base_url = "{public}" and set its own `model`. '
        "Agents only connect; models load on first request.",
        file=sys.stderr,
    )

    # ランタイム記録: 稼働中ゲートウェイの接続先を固定パスに残す。gateway.toml の無い
    # ディレクトリからでも `gw status` / `gw stop` がこの 1 ファイルで唯一のデーモンを見つける。
    write_gateway_runtime(
        cfg.host, cfg.port, server.pid, server.start_cwd, server.started_at
    )
    resources.runtime_written = True

    # 掃除スレッド: ①クラッシュした内部ワーカーの健全性チェック（常時）②idle TTL 超過モデルの
    # アンロード。健全性チェックは常に走らせる（死んだワーカーへ流し続けて 502 を返す事態を
    # 防ぐ）。チェック間隔は有効な閾値と健全性チェック周期の短い方。
    # 在席の「ハートビート途絶の掃除」は持たない（生存推定をしない設計。→ _Session）。
    stop_reaper = resources.stop_event
    _HEALTH_INTERVAL = 15.0  # 死んだワーカーの検知周期（idle が無効でもこの周期で回す）
    bounds = [t / 2 for t in (cfg.idle_timeout,) if t]
    bounds.append(_HEALTH_INTERVAL)
    interval = min(max(min(bounds), 1.0), 30.0)  # チェック間隔（最大 30s）
    last_reaper_error_at = 0.0

    def _reaper() -> None:
        nonlocal last_reaper_error_at
        while not stop_reaper.wait(interval):
            try:
                dead = manager.reap_dead_instances()
                if dead:
                    print(
                        f"Health check: removed {dead} dead worker instance(s) "
                        "(crashed); the slot is free to reload on the next request.",
                        file=sys.stderr,
                    )
                if cfg.idle_timeout:
                    freed = manager.evict_idle(cfg.idle_timeout)
                    if freed:
                        print(
                            f"Idle unload: stopped {freed} model(s).", file=sys.stderr
                        )
            except Exception as exc:  # noqa: BLE001 - 掃除スレッドは落とさない
                now = time.monotonic()
                if now - last_reaper_error_at >= 300.0:
                    print(f"Health/idle reaper failed: {exc}", file=sys.stderr)
                    last_reaper_error_at = now

    resources.start_thread(target=_reaper, name="gateway-health-reaper")

    # ホットリロード監視: gateway.toml を保存した瞬間に、ポリシー設定を無停止で反映する。
    # 構造設定（host/port/internal_base_port/[[models]]）の変更は「要再起動」を警告するだけ。
    if config_path:
        resources.start_thread(
            target=watch_config_file,
            name="gateway-config-watcher",
            args=(server, manager, cfg, config_path, stop_reaper),
        )

    # 更新監視は確認と通知だけを行う。ソース取得・依存同期・再起動は、ユーザーが明示的に
    # 「今すぐ更新」または `gw update` を実行したときだけ行う。
    restart_requested = threading.Event()
    # 検知状態と再起動要求を HTTP ハンドラ（/admin/status・/admin/update）から使えるようにする。
    server.update_state = {
        "available": False,
        "current": None,
        "latest": None,
        "fetched": False,
        "reason": None,
    }
    server.request_restart = restart_requested.set
    # オンデマンド確認（/admin/status GET から）のスロットル用。0.0 = 未確認なので、
    # 最初のメニューオープンで即チェックが走る（起動直後から「更新の有無」が正しく出る）。
    server._last_update_check = 0.0
    server._update_check_inflight = False
    server._update_check_done = None
    resources.start_thread(
        target=_update_watcher,
        name="gateway-update-watcher",
        args=(stop_reaper,),
        kwargs={
            "state": server.update_state,
            "notify": _tray_notify,
        },
    )

    resources.start_thread(
        target=server.serve_forever,
        name="gateway-http-server",
        server=True,
    )
    restart = False
    try:
        # 割り込み（Ctrl+C / SIGTERM）または手動更新の再起動要求までブロックする。
        restart = restart_requested.wait()
    except KeyboardInterrupt:
        pass
    finally:
        resources.restart = restart
        print("\nShutting down the gateway and its model servers...", file=sys.stderr)
    return _RESTART_CODE if restart else 0
