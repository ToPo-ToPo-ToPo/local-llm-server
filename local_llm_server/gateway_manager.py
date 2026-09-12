"""Model residency, replication, LRU eviction, and client presence."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Protocol

from .backend_core import ServerConfig, backend_spec, infer_backend, parallel_supported
from .gateway_errors import CapacityError, GatewayDraining
from .gateway_runtime import daemon_log_path
from .model_catalog import MTP_DRAFTERS, resolve_drafter

_DRAFT_OFF = ("", "off", "none")
_REPLICA_GRACE_S = 1.0


class ManagedServer(Protocol):
    """Process handle required by the scheduler, independent of LocalServer."""

    @property
    def base_url(self) -> str: ...

    def start(self) -> None: ...

    def stop(self, grace: float = 10.0) -> None: ...

    def wait_until_ready(
        self, timeout: float = 120.0, interval: float = 1.0
    ) -> None: ...

    def is_alive(self) -> bool: ...


def _total_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - optional resource guard
        return None


@dataclass
class _Instance:
    """1 モデルの 1 起動インスタンス（独立プロセス・独立ポート）。

    同一モデルへリクエストが集中したとき、負荷ベースでこのインスタンスを複数起動して
    並列化する（→ ModelManager.acquire）。各インスタンスは自前の inflight（処理中数）と
    last_used（LRU 基準）を持ち、LRU 退避・idle 解放はインスタンス単位で行う。
    """

    config: ServerConfig  # このインスタンス専用の host/port（同一モデルでもポートは別）
    server: ManagedServer | None = None
    ready: bool = False
    inflight: int = 0  # このインスタンスの処理中リクエスト数（>0 の間は退避しない）
    last_used: float = 0.0  # time.monotonic()。LRU の基準


@dataclass
class _Model:
    """1 model_id の共通設定と、その起動インスタンス群。

    instances は現在起動中（ready）のインスタンス。0 個ならモデルは未ロード。負荷に応じて
    max_resident とメモリの範囲で複製・退避され増減する。requests は表示用の累計で、
    インスタンスが退避されても失われないようモデル側に持つ。
    """

    config: ServerConfig  # テンプレート（backend/parallel/draft 等）。単一運用時の既定ポートも保持
    instances: list[_Instance] = field(default_factory=list)
    requests: int = (
        0  # このモデルに振り分けた累計リクエスト数（表示用。退避で消えない）
    )
    dynamic: bool = (
        False  # 未登録モデルを動的登録したもの（全インスタンス消滅時に登録ごと消す）
    )
    footprint: int | None = (
        None  # 1 インスタンスの概算占有メモリ（バイト）。0=見積もり不能
    )


# release 後にモデルを保持する猶予（秒）。設定にはしない——値の精度に意味が無く
# 「プロセスの入れ替わりを吸収できれば十分」だからで、ノブを増やさない方針。
#
# 即時アンロードだった頃は、タスクごとに子プロセスを起動する構成（cad-agent の MCP）で
# release の数秒後に次のタスクが register し、その都度アンロード→再ロードが起きていた
# （巨大なモデルではロードのたびに無視できない待ちが発生し、クライアントの入口プローブまで食い潰した）。
_RELEASE_LINGER_S = 60.0


@dataclass
class _Session:
    """1 エージェントの在席（このモデルを使うと宣言したクライアント）。

    inflight（処理中リクエスト数）とは別の軸で「接続中のエージェント」を数えるための
    もの。register で増え、release で減る。あるモデルのセッションが 0 になったら、
    _RELEASE_LINGER_S の猶予後に（その間に誰も来なければ）アンロードする。

    **生存推定はしない**。ハートビートによる死活監視は持たず、在席したまま応答が無い
    エージェントを「死んだ」と推定してモデルを落とすことは無い（旧実装はこれで、生成中の
    クライアントの足元から巨大なモデルを外す事故を起こした）。release を送れずに落ちた
    エージェントの置き去りセッションは、モデルが idle_timeout で解放される際に一緒に掃除
    される（セッションは解放を**早める**だけで、遅らせる力を持たない）。
    """

    model_id: str


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
        *,
        start_timeout: float = 120.0,
        dynamic: bool = False,
        default_disable_thinking: bool = False,
        default_stream_tool_calls: bool = False,
        default_draft: str | None = None,
        default_parallel: int | None = None,
        max_memory_fraction: float | None = None,
        internal_base_port: int = 9001,
        public_port: int | None = None,
        _server_factory=None,
        _estimate_model_bytes=lambda _config: None,
        _reclaim_stale=lambda _port: [],
        _replica_grace_s: float = _REPLICA_GRACE_S,
        _release_linger_s: float = _RELEASE_LINGER_S,
    ) -> None:
        self._server_factory = _server_factory
        self._estimate_model_bytes = _estimate_model_bytes
        self._reclaim_stale_workers = _reclaim_stale
        self._replica_grace_s = _replica_grace_s
        self._release_linger_s = _release_linger_s
        self._models: dict[str, _Model] = {c.model: _Model(config=c) for c in configs}
        self._max_resident = max_resident
        self._load_timeout = load_timeout  # 枠が空くのを待つ最大秒数（None で無期限）
        self._start_timeout = (
            start_timeout  # 1 インスタンスの起動完了（ready）を待つ最大秒数
        )
        self._started = time.monotonic()  # 起動経過時間（uptime 表示用）の基準
        # 未登録モデルの動的ロード（IDからバックエンド推論。ロード時に表示へ追加・アンロードで消す）。
        self._dynamic = dynamic
        self._default_disable_thinking = default_disable_thinking
        self._default_stream_tool_calls = default_stream_tool_calls
        # 動的ロード時の MTP ドラフター既定。None なら mlx-vlm は "auto"（対応表 MTP_DRAFTERS
        # から本体名で自動選択）を試みる。"off"/"none"/"" で無効化、明示 id でその指定を使う。
        self._default_draft = default_draft
        # 動的ロード時の並列スロット既定（llama-cpp のみ。他バックエンドでは無視）。
        self._default_parallel = default_parallel
        # メモリガード: 常駐モデルの推定占有量の合計が「総RAM × この割合」を超えるロードを
        # 拒否する（None で無効）。総RAM は psutil から起動時に 1 度だけ取得。
        self._mem_fraction = max_memory_fraction
        self._mem_total = _total_ram() if max_memory_fraction else None
        if max_memory_fraction and not self._mem_total:
            raise ValueError(
                "max_memory_fraction is set but total RAM could not be read "
                "(psutil unavailable?). Install psutil or unset max_memory_fraction."
            )
        self._public_port = public_port
        # 動的モデルの内部ポート割当カーソル（事前登録分の次から）。
        self._next_port = internal_base_port + len(configs)
        # registry 保護＋「枠が空いた」通知用。release で inflight→0 のとき notify する。
        self._state = threading.Condition()
        self._control = threading.Lock()  # 起動/退避（control plane）の直列化
        # エージェント在席トラッキング（agent_id → セッション）と、その逆引き
        # （model_id → 在席エージェントの集合）。あるモデルの集合が空になった瞬間に
        # 即アンロードする判定に使う。_state ロック下で操作する。
        self._sessions: dict[str, _Session] = {}
        self._model_sessions: dict[str, set[str]] = {}
        # release 猶予タイマー。モデルごとに最大1本とし、再 release 時は古いものを cancel する。
        self._release_timers: dict[str, threading.Timer] = {}
        # drain（再起動準備）の期限。monotonic 時刻がこれ未満の間は新規 acquire を
        # GatewayDraining で拒否する。0.0 で無効。再起動側が死んでも TTL で自動復帰する。
        self._drain_deadline: float = 0.0
        # 同一モデルの複製インスタンスを裏で起動中の model_id 集合（多重起動を防ぐ）。
        self._spawning: set[str] = set()
        self._background_threads: set[threading.Thread] = set()
        self._shutdown_event = threading.Event()
        # shutdown 済みフラグと、起動処理中（start〜instances 登録前）のサーバー集合。
        # shutdown はこの集合も止めることで、「ロード中に Ctrl+C → 起動しかけの巨大モデルが
        # 孤児プロセスとしてメモリとポートを掴んだまま残る」のを防ぐ。_state ロック下で操作する。
        self._closing = False
        self._starting: set[ManagedServer] = set()

    def _alloc_port_locked(self) -> int:
        """内部ポートを1つ払い出す（state ロック保持下で呼ぶ）。

        各モデルの既定ポート・起動中の全インスタンスのポート・公開ポートを避けて連番で割り当てる。
        動的モデルの初回インスタンスにも、複製インスタンスにも使う。
        """
        used = {m.config.port for m in self._models.values()}
        for m in self._models.values():
            used.update(i.config.port for i in m.instances)
        if self._public_port is not None:
            used.add(self._public_port)
        p = self._next_port
        while p in used:
            p += 1
        self._next_port = p + 1
        return p

    def _register_dynamic_locked(self, model_id: str) -> _Model:
        """未登録モデルを動的登録する（control＋state ロック保持下で呼ぶ）。

        ID からバックエンドを推論し、内部ポートを割り当てて _Model を作る。MTP（mlx-vlm）は
        対応表から本体名で自動選択できるため、事前登録なしでも有効化する（下記
        `_dynamic_draft`）。parallel やマルチモーダルの mmproj 自動付与など他のオプションは
        付かない（個別チューニングが要るものだけ gateway.toml に事前登録する）。
        """
        backend = infer_backend(model_id)
        cfg = ServerConfig(
            backend=backend,
            model=model_id,
            host="127.0.0.1",
            port=self._alloc_port_locked(),
            # 並列スロットは llama-cpp のみ有効（他は逐次処理なので付けない）。
            parallel=self._default_parallel if parallel_supported(backend) else None,
            disable_thinking=self._default_disable_thinking,
            stream_tool_calls=self._default_stream_tool_calls,
            draft_model=self._dynamic_draft(model_id, backend),
        )
        mm = _Model(config=cfg, dynamic=True)
        self._models[model_id] = mm
        return mm

    def _dynamic_draft(self, model_id: str, backend: str) -> str | None:
        """動的ロードするモデルの MTP ドラフターを解決する（事前登録なしでも有効化）。

        - 既定（`_default_draft` が None）では mlx-vlm のみ `"auto"` を試みる。本体名が対応表
          `MTP_DRAFTERS` にあればそのドラフターを返し、無ければ静かに None（MTP なし）にする。
          動的ロードを未対応モデルで失敗させないための graceful な解決。
        - `_default_draft` を明示していればそれを尊重する（`"off"`/`"none"`/`""` で無効化、
          HF id で明示指定）。
        - llama.cpp の MTP は埋め込みヘッドの有無を repo-id から確実に判定できず、未対応 GGUF に
          `--spec-type draft-mtp` を付けると起動失敗するため、動的ロードでは付けない（要事前登録）。
        """
        raw = self._default_draft if self._default_draft is not None else "auto"
        if isinstance(raw, str) and raw.strip().lower() in _DRAFT_OFF:
            return None
        if backend_spec(backend).draft_style != "mtp":
            return None
        if raw == "auto" and model_id not in MTP_DRAFTERS:
            return None  # 対応表に無い → MTP なしで普通にロード
        return resolve_drafter(model_id, raw)

    @property
    def model_ids(self) -> list[str]:
        with self._state:
            return list(self._models)

    def disable_thinking_for(self, model_id: str) -> bool:
        """model_id に disable_thinking が指定されているか（未登録は False）。

        do_POST が「mlx-vlm 宛に reasoning_effort=none を注入するか」の判定に使う。
        """
        with self._state:
            mm = self._models.get(model_id)
        return bool(mm.config.disable_thinking) if mm is not None else False

    def backend_for(self, model_id: str) -> str:
        """model_id のバックエンドを返す（登録済みは config 値、未登録は ID から推論）。

        do_POST が「mlx 系のみ repetition_penalty を注入する」判定に使う。acquire 前でも
        判定できるよう、まだ登録されていない動的モデルは ID から推論する。
        """
        with self._state:
            mm = self._models.get(model_id)
        if mm is not None:
            return mm.config.backend
        return infer_backend(model_id)

    def _capacity(self, config: ServerConfig) -> int:
        """1 インスタンスが同時に捌けるリクエスト数。llama-cpp は parallel スロット、他は 1。

        この本数に達したインスタンスを「満杯」とみなし、負荷ベースで複製インスタンスを増やす
        判断に使う（mlx 系は 1 なので、2 本目の同時リクエストで複製が検討される。llama-cpp は
        まずプロセス内の parallel スロットを使い切ってから複製する）。
        """
        if parallel_supported(config.backend) and config.parallel:
            return int(config.parallel)
        return 1

    def _route_locked(self, inst: _Instance) -> tuple[str, int]:
        """インスタンス inst にリクエストを1つ割り当てる（_state 保持下で呼ぶ）。"""
        inst.inflight += 1
        inst.last_used = time.monotonic()
        return (inst.config.host, inst.config.port)

    def _running_instances_locked(self) -> list[tuple[_Model, _Instance]]:
        """起動中（server がある）の (model, instance) を全モデル横断で列挙（_state 保持下）。"""
        return [
            (m, i)
            for m in self._models.values()
            for i in m.instances
            if i.server is not None
        ]

    def _port_in_use_locked(self, port: int) -> bool:
        """port を現在いずれかの起動中インスタンスが使っているか（_state 保持下）。"""
        return any(
            i.config.port == port
            for m in self._models.values()
            for i in m.instances
            if i.server is not None
        )

    def _make_instance_config_locked(self, mm: _Model) -> ServerConfig:
        """mm の新規インスタンス用に、専用ポートを与えた ServerConfig を作る（_state 保持下）。

        既定ポート（mm.config.port）が空いていれば単一運用の予測性のためそれを使い、既に別
        インスタンスが使っていれば連番で新ポートを払い出す（複製インスタンス用）。
        """
        base = mm.config.port
        port = base if not self._port_in_use_locked(base) else self._alloc_port_locked()
        return replace(mm.config, port=port)

    def _reclaim_stale_port(self, port: int) -> None:
        """ワーカー起動の直前、対象ポートに残る自分由来の孤児ワーカーを回収する。

        前回のクラッシュ / `kill -9` で取り残されたモデルサーバーが同じ内部ポートを掴んで
        いると、新ワーカーが bind できず起動失敗になり、加えて GPU メモリを無駄に占有する。
        起動する側（このゲートウェイ）が握っている枠は追跡済みポートを避けて割り当てられる
        ので、そこに居る our-worker は必ず未追跡＝孤児。回収失敗で起動自体は止めない。
        """
        try:
            stale = self._reclaim_stale_workers(port)
        except Exception:  # noqa: BLE001 - 回収失敗（lsof 不在等）は起動を妨げない
            return
        if stale:
            print(
                f"Reclaimed orphaned worker(s) {stale} on internal port {port} "
                "before starting a fresh one.",
                file=sys.stderr,
            )

    def acquire(self, model_id: str) -> tuple[tuple[str, int], _Instance]:
        """model_id のインスタンスを（必要なら起動して）確保し、(内部アドレス, ハンドル) を返す。

        呼び出し側は転送後に必ず release(ハンドル) すること（inflight を戻すため）。ready な
        インスタンスが複数あれば**最も空いているもの**へ振り分ける。最も空いているものすら満杯
        （inflight >= capacity）だった＝リクエストが競合しているときは、max_resident とメモリの
        範囲で**バックグラウンドで複製インスタンスを1つ増やす**（現在のリクエストは待たせず、その
        まま最少負荷のインスタンスへ転送する）。
        未登録モデルは、dynamic 有効なら ID からバックエンドを推論して初回インスタンスを起動する
        （無効なら KeyError）。起動失敗は RuntimeError/TimeoutError、初回起動の空き枠が
        `load_timeout` 内に得られなければ CapacityError（→ 503）を投げる。
        """
        # 高速パス: ready なインスタンスがあれば、最も空いているものへ割り当てる（state のみ）。
        spawn = False
        with self._state:
            # drain（再起動準備）中は新規を受けない。inflight の増加と同一ロックなので、
            # begin_drain の「アイドル確認」とここが競合しても取りこぼしが起きない。
            if self._draining_locked():
                raise GatewayDraining(
                    "gateway is restarting to apply an update; retry in a few seconds"
                )
            mm = self._models.get(model_id)
            ready = (
                [i for i in mm.instances if i.ready and i.server is not None]
                if mm
                else []
            )
            if ready:
                assert mm is not None  # ready は mm から作るため必ず存在する
                inst = min(ready, key=lambda i: i.inflight)
                # 「最少負荷のインスタンスすら満杯」なら競合中 → 複製を検討（割当は前の値で判定）。
                spawn = inst.inflight >= self._capacity(mm.config)
                addr = self._route_locked(inst)
                mm.requests += 1
        if ready:
            if spawn:
                self._maybe_spawn_replica_async(model_id)
            return addr, inst
        # 低速パス: ready インスタンスが1つも無い → 初回インスタンスを起動（control で直列化）。
        with self._control:
            with self._state:
                if self._draining_locked():
                    raise GatewayDraining(
                        "gateway is restarting to apply an update; "
                        "retry in a few seconds"
                    )
                if self._closing:
                    raise RuntimeError("gateway is shutting down")
                mm = self._models.get(model_id)
                if mm is None:
                    if not self._dynamic:
                        raise KeyError(model_id)
                    mm = self._register_dynamic_locked(model_id)
                else:
                    ready = [
                        i for i in mm.instances if i.ready and i.server is not None
                    ]
                    if ready:  # 待つ間に他スレッドが用意した
                        inst = min(ready, key=lambda i: i.inflight)
                        addr = self._route_locked(inst)
                        mm.requests += 1
                        return addr, inst
            try:
                self._evict_if_needed(keep=model_id)
            except Exception:
                # 枠・メモリ不足（CapacityError 等）で起動を諦めたとき、動的登録だけが
                # 幽霊としてカタログに残らないよう取り消す（起動失敗パスと同じ扱い）。
                if mm.dynamic:
                    with self._state:
                        if not mm.instances:
                            self._models.pop(model_id, None)
                raise
            with self._state:
                cfg = self._make_instance_config_locked(mm)
            inst = _Instance(config=cfg)
            if self._server_factory is None:
                raise RuntimeError("a model server factory is required")
            server = self._server_factory(cfg, log_path=daemon_log_path(cfg.port))
            with self._state:
                if self._closing:
                    raise RuntimeError("gateway is shutting down")
                self._starting.add(
                    server
                )  # shutdown が起動途中のサーバーも止められるように
            try:
                self._reclaim_stale_port(
                    cfg.port
                )  # 同ポートに残る孤児ワーカーを先に掃除
                server.start()
                server.wait_until_ready(timeout=self._start_timeout)
            except Exception:
                # OSError/設定型エラーを含む全ての通常例外で起動途中状態と子を回収する。
                try:
                    server.stop(grace=0.0)
                finally:
                    with self._state:
                        self._starting.discard(server)
                # 動的登録の初回起動失敗は、他に生きたインスタンスが無ければ登録ごと取り消す。
                if mm.dynamic:
                    with self._state:
                        if not mm.instances:
                            self._models.pop(model_id, None)
                raise
            with self._state:
                self._starting.discard(server)
                if self._closing:  # 起動完了と同時に shutdown が走った → 登録せず止める
                    threading.Thread(target=server.stop, daemon=True).start()
                    raise RuntimeError("gateway is shutting down")
                inst.server = server
                inst.ready = True
                addr = self._route_locked(inst)
                mm.requests += 1
                mm.instances.append(inst)
            return addr, inst

    def release(self, inst: _Instance) -> None:
        with self._state:
            if inst.inflight > 0:
                inst.inflight -= 1
                if inst.inflight == 0:
                    # 枠が空いた可能性。_evict_if_needed で待っているスレッドを起こす。
                    self._state.notify_all()

    def _maybe_spawn_replica_async(self, model_id: str) -> None:
        """満杯モデルの複製インスタンスを1つ、バックグラウンドで起動する（多重起動を防ぐ）。

        既に同モデルの複製を起動中なら何もしない（1 モデルにつき同時 1 本だけウォームアップ）。
        HTTP 応答を待たせないよう別スレッドで行う。
        """
        with self._state:
            if self._closing or model_id in self._spawning:
                return
            self._spawning.add(model_id)
            thread = threading.Thread(
                target=self._spawn_replica,
                args=(model_id,),
                daemon=True,
                name=f"model-replica-{model_id}",
            )
            self._background_threads.add(thread)
            # Start while holding _state so shutdown cannot snapshot an unstarted
            # Thread and then fail to join it.
            thread.start()

    def _spawn_replica(self, model_id: str) -> None:
        """複製インスタンスを1つ起動する。枠が取れない/もう満杯でなければ黙って諦める。

        現在のリクエストは既に別インスタンスへ流れているので、これは将来の負荷に備えた
        best-effort なウォームアップ。枠確保は**非ブロッキング**（処理中のインスタンスは止めず、
        退避できるアイドルが無ければ複製しない）。起動失敗も本流に影響させない。
        """
        try:
            # 逐次クライアントのフェーズ境界レース（[DONE] 受信〜release の数 ms 差）に
            # よる誤発動を除外する猶予（_REPLICA_GRACE_S 参照）。この間 _spawning に
            # 登録済みなので同モデルの再トリガーは重複しない
            if self._shutdown_event.wait(self._replica_grace_s):
                return
            with self._control:
                with self._state:
                    mm = self._models.get(model_id)
                    if mm is None:
                        return
                    ready = [
                        i for i in mm.instances if i.ready and i.server is not None
                    ]
                    cap = self._capacity(mm.config)
                    # 猶予後も「全インスタンスに容量+1 以上積まれている」＝複数リクエスト
                    # が実際に同時へ載っている場合のみ複製する。単なる処理中 (inflight==cap)
                    # はトリガー時のレース痕跡と区別できないため複製しない（真の並行負荷では
                    # 追い越したリクエストも同じインスタンスへ載るので inflight が cap を超える）
                    if not ready or min(i.inflight for i in ready) <= cap:
                        return
                if not self._make_room_for_replica(keep=model_id):
                    return  # 上限・メモリで枠が取れない（アイドル退避もできない）→ 複製しない
                with self._state:
                    mm = self._models.get(model_id)
                    if mm is None or self._closing:
                        return
                    cfg = self._make_instance_config_locked(mm)
                inst = _Instance(config=cfg)
                if self._server_factory is None:
                    raise RuntimeError("a model server factory is required")
                server = self._server_factory(cfg, log_path=daemon_log_path(cfg.port))
                with self._state:
                    if self._closing:
                        return
                    self._starting.add(
                        server
                    )  # shutdown が起動途中の複製も止められるように
                try:
                    self._reclaim_stale_port(
                        cfg.port
                    )  # 同ポートに残る孤児ワーカーを先に掃除
                    server.start()
                    server.wait_until_ready(timeout=self._start_timeout)
                except Exception:
                    try:
                        server.stop(grace=0.0)
                    finally:
                        with self._state:
                            self._starting.discard(server)
                    return
                with self._state:
                    self._starting.discard(server)
                    mm = self._models.get(model_id)
                    if (
                        mm is None or self._closing
                    ):  # 起動中にモデルが消えた/終了中 → 止める
                        threading.Thread(target=server.stop, daemon=True).start()
                        return
                    inst.server = server
                    inst.ready = True
                    inst.last_used = time.monotonic()
                    mm.instances.append(inst)
                    self._state.notify_all()
        finally:
            with self._state:
                self._spawning.discard(model_id)
                self._background_threads.discard(threading.current_thread())
                self._state.notify_all()

    def _make_room_for_replica(self, keep: str) -> bool:
        """複製用に枠を確保する（非ブロッキング）。確保できたら True（control 保持下で呼ぶ）。

        上限・メモリに余裕があればそのまま True。超過していても、アイドルなインスタンス
        （keep 以外・処理中でない）を LRU 退避して空けられれば True。処理中しか無く空けられない
        なら False（複製を諦める＝busy は止めない）。_evict_if_needed と違い**待たない**。

        max_resident もメモリ上限（max_memory_fraction）も無い構成では**複製しない**（False）。
        際限なく重みのコピーが増えて OOM する事故を防ぐため、負荷ベースの並列化を使うには
        どちらかで総量の範囲を決めることを要求する。
        """
        budget = self._mem_budget()
        if self._max_resident is None and budget is None:
            return False
        while True:
            with self._state:
                over, _over_mem, running, _need = self._over_capacity_locked(
                    keep, budget
                )
                if not over:
                    return True
                victim_model, victim_srv = self._pop_lru_idle_locked(running, keep)
                if victim_model is None:
                    return False  # 退避できるアイドルが無い → 複製しない
            victim_srv.stop()  # state ロックの外で（最長 ~10s）
            self._drop_if_empty_dynamic(victim_model)

    def _over_capacity_locked(self, keep: str, budget: int | None):
        """常駐数・メモリ予算の超過判定（state ロック保持下で呼ぶ）。

        _make_room_for_replica と _evict_if_needed が共有する LRU 退避の判定部。
        戻り値: (over, over_mem, running, need)。over は数・メモリいずれかの超過、
        over_mem はメモリ超過のみ、need は keep の概算占有バイト（メモリ予算が無効なら 0。
        _evict_if_needed の CapacityError 文面が使う）。
        """
        running = self._running_instances_locked()
        over_count = (
            self._max_resident is not None and len(running) >= self._max_resident
        )
        over_mem = False
        need = 0
        if budget is not None:
            keep_mm = self._models.get(keep)
            need = self._footprint_locked(keep_mm) if keep_mm else 0
            used = sum(self._footprint_locked(m) for (m, _i) in running)
            over_mem = (used + need) > budget
        return over_count or over_mem, over_mem, running, need

    def _pop_lru_idle_locked(self, running, keep: str):
        """LRU 退避の対象（ready・inflight==0・keep 以外の最古）を選んで instances から外す。

        戻り値: (victim_model, victim_srv)。候補が無ければ (None, None)。
        stop() はロック時間が長い（最長 ~10s）ので呼び出し側が state ロックの外で行う。
        """
        candidates = [
            (m, i)
            for (m, i) in running
            if i.ready and i.inflight == 0 and m.config.model != keep
        ]
        if not candidates:
            return None, None
        victim_model, victim_inst = min(candidates, key=lambda mi: mi[1].last_used)
        victim_model.instances.remove(victim_inst)
        return victim_model, victim_inst.server

    def _drop_if_empty_dynamic(self, victim_model) -> None:
        """動的登録モデルのインスタンスが空になったら登録ごと消す（stop 後の後始末）。"""
        if victim_model.dynamic and not victim_model.instances:
            with self._state:
                self._models.pop(victim_model.config.model, None)

    def _footprint_locked(self, mm: _Model) -> int:
        """モデルの概算占有メモリ（バイト）。一度計算したらキャッシュする。0=見積もり不能。"""
        if mm.footprint is None:
            mm.footprint = self._estimate_model_bytes(mm.config) or 0
        return mm.footprint

    def _mem_budget(self) -> int | None:
        """メモリガードの上限バイト数（総RAM × max_memory_fraction）。無効なら None。"""
        if self._mem_fraction is None or not self._mem_total:
            return None
        return int(self._mem_total * self._mem_fraction)

    def _evict_if_needed(self, keep: str) -> None:
        """control ロック保持下で呼ぶ。常駐数の上限（max_resident）と推定メモリ占有量の上限
        （max_memory_fraction）のどちらかを超えるなら、LRU でアイドルモデルを退避して空ける。

        退避候補は「ロード済み・処理中でない（inflight==0）・keep 以外」。候補が無い
        （全て処理中）ときは、いずれかが release されて枠が空くまで**待つ**（OOM を避ける）。
        メモリ上限の場合、退避できるモデルが無く（=keep 単体で予算超過）なら待っても無駄なので
        即 `CapacityError`。`load_timeout` 秒以内に空かなくても同様（呼び出し側で 503）。
        待っている間も `control` は握ったまま（他のロードは直列化）だが、`state` 条件変数は
        手放すので、ロード済みモデルへの高速パス（acquire/release）は進められる。
        """
        budget = self._mem_budget()
        if self._max_resident is None and budget is None:
            return
        deadline = time.monotonic() + self._load_timeout if self._load_timeout else None
        while True:
            with self._state:
                if self._closing:
                    raise CapacityError("gateway is shutting down")
                over, over_mem, running, need = self._over_capacity_locked(keep, budget)
                resident = len(running)
                if not over:
                    return
                victim_model, victim_srv = self._pop_lru_idle_locked(running, keep)
                if victim_model is not None:
                    pass  # 選べた → ループ末尾で stop する
                elif over_mem and resident == 0:
                    # 退避できる常駐インスタンスが無く、keep 単体で予算超過 → 待っても無駄。
                    assert (
                        need is not None
                        and budget is not None
                        and self._mem_total is not None
                    )
                    raise CapacityError(
                        f"model '{keep}' needs ~{need / 1e9:.1f}GB but the memory budget is "
                        f"{budget / 1e9:.1f}GB (max_memory_fraction={self._mem_fraction:g} of "
                        f"{self._mem_total / 1e9:.1f}GB); not loading to avoid OOM"
                    )
                else:
                    # 全て処理中 → 枠が空く（release の notify）まで待つ。
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        why = (
                            "memory budget exceeded"
                            if over_mem
                            else (f"all {self._max_resident} instance slot(s) busy")
                        )
                        raise CapacityError(
                            f"{why}; could not free room within {self._load_timeout:g}s"
                        )
                    self._state.wait(timeout=remaining)
                    continue  # 起きたら再判定
            if victim_srv is not None:
                victim_srv.stop()  # state ロックの外で（最長 ~10s）。停止後にループ再確認。
                self._drop_if_empty_dynamic(victim_model)

    def begin_drain(self, ttl: float = 120.0) -> dict:
        """再起動準備（drain）を試みる。アイドル確認と新規受付停止を**原子的に**行う。

        `_state` ロック下で「処理中リクエスト 0」を確認し、満たすときだけ drain を開始する
        （以後 acquire は GatewayDraining → 503）。busy なら開始せず現状を返す（呼び出し側は
        空くのを待って再試行する）。再起動側が死んで drain だけ残っても、ttl 秒で自動解除され
        通常運転へ戻る。

        **在席セッションは見ない**。在席は「解放を早める」だけの存在で、drain を塞ぐ権限を
        持たない（release を送れずに落ちたエージェントの置き去りが、常時使用中の共有モデルに
        残ると drain が永久に通らなくなるため。sessions は情報として返すだけ）。

        戻り値: {"ok": bool, "inflight": n, "sessions": n}
        """
        with self._state:
            inflight = sum(
                i.inflight for m in self._models.values() for i in m.instances
            )
            sessions = len(self._sessions)
            if inflight:
                return {"ok": False, "inflight": inflight, "sessions": sessions}
            self._drain_deadline = time.monotonic() + ttl
            return {"ok": True, "inflight": 0, "sessions": sessions}

    def end_drain(self) -> None:
        """drain を解除して通常受付に戻す（更新の見送り・失敗時）。"""
        with self._state:
            self._drain_deadline = 0.0

    def _draining_locked(self) -> bool:
        """drain 中か（_state ロック下で呼ぶ）。期限切れは自動的に False。"""
        return time.monotonic() < self._drain_deadline

    def set_max_resident(self, value: int | None) -> None:
        """常駐上限（max_resident）を実行中に変更する。処理中（busy）のモデルは止めない。

        value は 1 以上の整数、または None（無制限）。上限を上げる／無制限にするときは、
        枠が空くのを待って止まっていたロードを起こすだけ。下げるときは、超過している常駐
        モデルをアイドルなものから LRU で **非同期に** 退避する（inflight>0 のモデルには
        一切触れないので、生成中のリクエストは止まらない）。退避しきれなかった超過分は、
        次の release/acquire もしくは idle_timeout で片付く。
        """
        with self._state:
            self._max_resident = value
            # 枠が広がった可能性 → _evict_if_needed で待っているロードを起こす。
            self._state.notify_all()
        if value is not None:
            # 縮小時の超過分を裏で片付ける（busy は残すので HTTP 応答を待たせない）。
            threading.Thread(target=self._trim_to_limit, daemon=True).start()

    def _trim_to_limit(self) -> None:
        """現在の max_resident を超える常駐モデルを、アイドルなものから LRU で退避する。

        処理中（inflight>0）のモデルには一切触れない（＝更新で稼働中の生成を止めない）。
        上限内に収まるか、退避できるアイドルモデルが尽きたら終わる。control を握って起動
        （slow path）／idle 退避と直列化し、stop 自体（最長 ~10s）は state ロックの外で行う。
        """
        with self._control:
            while True:
                with self._state:
                    limit = self._max_resident
                    if limit is None:
                        return
                    running = self._running_instances_locked()
                    if len(running) <= limit:
                        return
                    idle = [(m, i) for (m, i) in running if i.ready and i.inflight == 0]
                    if not idle:
                        return  # 残りは全て処理中 → 止めない（後で片付く）
                    victim_model, victim_inst = min(
                        idle, key=lambda mi: mi[1].last_used
                    )
                    victim_srv = victim_inst.server
                    assert (
                        victim_srv is not None
                    )  # running は server 付きインスタンスだけを返す
                    victim_model.instances.remove(victim_inst)
                victim_srv.stop()  # state ロックの外で（最長 ~10s）
                if (
                    victim_model.dynamic and not victim_model.instances
                ):  # 空なら登録ごと消す
                    with self._state:
                        self._models.pop(victim_model.config.model, None)

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
                    for i in list(m.instances):
                        if (
                            i.server is not None
                            and i.ready
                            and i.inflight == 0
                            and (now - i.last_used) > timeout
                        ):
                            victims.append((m, i.server))
                            m.instances.remove(i)
            for _m, srv in victims:
                srv.stop()  # state ロックの外で（最長 ~10s かかるため）
            # アンロードしたモデルの在席登録を捨てる。release を送れずに落ちたエージェントの
            # 置き去りはここで回収されるので、ハートビートによる死活監視は要らない。
            for m, _srv in victims:
                if not m.instances:
                    self.drop_sessions_for(m.config.model)
            with self._state:
                for (
                    m,
                    _srv,
                ) in victims:  # 全インスタンスが消えた動的モデルは登録ごと消す
                    if m.dynamic and not m.instances:
                        self._models.pop(m.config.model, None)
        return len(victims)

    def reap_dead_instances(self) -> int:
        """ワーカープロセスが死んだインスタンスを登録から外す（健全性チェック）。

        クラッシュや `kill -9` で内部ワーカーが落ちると、ゲートウェイはそれを ready と信じた
        まま新規リクエストをその内部ポートへ流し、502 を返し続ける（かつ枠を占有し続ける）。
        掃除スレッドから定期的に呼び、死んだインスタンスを外して枠を戻す（次リクエストで新規
        ロードし直せる）。停止した数を返す。inflight>0 でもプロセスが死んでいればもう進まない
        ので外す（担当ハンドラは上流の接続断で forward が返り、finally の release で整合する）。
        control を握って起動（slow path）/退避と直列化し、ポート再利用衝突を避ける。
        """
        with self._control:
            with self._state:
                victims = []
                for m in self._models.values():
                    for i in list(m.instances):
                        if i.server is not None and i.ready and not i.server.is_alive():
                            victims.append((m, i.server))
                            m.instances.remove(i)
            for _m, srv in victims:
                srv.stop()  # ログ fd を閉じ、死んだプロセスグループを掃除する
            with self._state:
                emptied = [m.config.model for m, _srv in victims if not m.instances]
                for (
                    m,
                    _srv,
                ) in victims:  # 全インスタンスが消えた動的モデルは登録ごと消す
                    if m.dynamic and not m.instances:
                        self._models.pop(m.config.model, None)
        # ワーカーがクラッシュしたモデルの在席登録を捨てる（エージェントごと巻き込まれた
        # クラッシュの置き去り対策。生きているエージェントは heartbeat 404 → 再 register で
        # 自己修復する）。
        for model_id in emptied:
            self.drop_sessions_for(model_id)
        return len(victims)

    # --- エージェント在席（セッション）管理 -----------------------------------
    #
    # idle_timeout / LRU とは別の「即時解放」経路。エージェントが register で在席を宣言し、
    # 停止時に release を呼ぶ（or ハートビート途絶を reap が検出する）。あるモデルの在席が
    # 0 になった瞬間、そのモデルが処理中（inflight>0）でなければ即アンロードする。
    # 在席はメモリをピン留めしない（max_resident の LRU 退避は従来どおり優先される）—
    # あくまで「使う人が居なくなったら早く片付ける」ための仕組み。

    def register_session(self, agent_id: str, model_id: str) -> None:
        """エージェントの利用開始を記録する（モデルは従来どおり初回リクエストで遅延ロード）。

        既に別モデルに在席していた agent_id は、まず旧モデルから外す（乗り換え）。旧モデルが
        それで無人かつ処理中でなくなれば、猶予後にアンロードする。
        """
        freed: str | None = None
        with self._state:
            prev = self._sessions.get(agent_id)
            if prev is not None and prev.model_id != model_id:
                freed = self._detach_locked(agent_id, prev.model_id)
            self._sessions[agent_id] = _Session(model_id=model_id)
            self._model_sessions.setdefault(model_id, set()).add(agent_id)
        if freed is not None:
            self._free_model_async(freed)

    def unregister_session(self, agent_id: str) -> bool:
        """エージェントの利用終了を記録する（停止時に呼ぶ）。

        対象モデルがそれで無人になったら、_RELEASE_LINGER_S の猶予後にアンロードする
        （バックグラウンド）。猶予中に誰かが register すれば解放は取り消される。
        登録の有無に関わらず冪等。実際に登録が在ったときだけ True。
        """
        with self._state:
            sess = self._sessions.get(agent_id)
            if sess is None:
                return False
            freed = self._detach_locked(agent_id, sess.model_id)
            self._sessions.pop(agent_id, None)
        if freed is not None:
            self._free_model_async(freed)
        return True

    def _detach_locked(self, agent_id: str, model_id: str) -> str | None:
        """agent_id を model_id の在席集合から外す（_state 保持下で呼ぶ）。

        その結果モデルが無人になったら model_id を返す（呼び出し側が解放判定する）。
        まだ他のエージェントが居れば None（＝「他に同じモデルに接続しているエージェントが
        居る」ので解放しない）。_sessions 自体の削除は呼び出し側が行う。
        """
        members = self._model_sessions.get(model_id)
        if members is None:
            return None
        members.discard(agent_id)
        if members:
            return None
        self._model_sessions.pop(model_id, None)
        return model_id

    def _free_model_async(self, model_id: str) -> None:
        """無人になったモデルを、猶予をおいて別スレッドで解放する。

        HTTP 応答を stop の 10s 待たせないためにスレッドへ逃がすのは従来どおり。加えて
        _RELEASE_LINGER_S 待ってから解放する: タスクごとに子プロセスを起動する構成では
        release の直後に次のタスクが register するため、即時解放するとアンロード→再ロードを
        毎回繰り返してしまう。猶予中に再 register されれば _free_idle_model 側の在席チェックで
        解放は自然に取り消される。

        猶予はモデルごとに1本の threading.Timer で延長する。新しい release は古い
        タイマーを cancel し、最後のタイマーだけが実際に解放する。これが無いと、60 秒以内に
        release が連続したとき古い予約が後続の猶予を侵食する:
          T=0 A release（予約T=60）→ T=50 B release（予約T=110）→ T=60 A の古い予約が発火
          → B の release から 10 秒しか経っていないのに解放される。

        callback は辞書に現在登録されている Timer 自身かも再確認するため、cancel と発火が
        競合しても古い予約がモデルを解放しない。モデル数を超える待機スレッドも残さない。
        """

        def _delayed() -> None:
            with self._state:
                if self._release_timers.get(model_id) is not timer:
                    return
                self._release_timers.pop(model_id, None)
            self._free_idle_model(model_id)

        timer = threading.Timer(self._release_linger_s, _delayed)
        timer.daemon = True
        with self._state:
            if self._closing:
                return
            previous = self._release_timers.get(model_id)
            if previous is not None:
                previous.cancel()
            self._release_timers[model_id] = timer
        timer.start()

    def _free_idle_model(self, model_id: str) -> bool:
        """無人かつ処理中でないモデルを即停止してメモリを解放する。停止したら True。

        control を握って起動（slow path）/idle 退避と直列化し、stop 直後の再ロードに伴う
        ポート再利用衝突を避ける。停止判断後にもう一度 state 下で「まだ無人か・処理中で
        ないか・ロード済みか」を確認してから止める（解放手前で再 register された等の競合に
        備える）。停止自体（最長 ~10s）は state ロックの外で行う。
        """
        with self._control:
            with self._state:
                mm = self._models.get(model_id)
                if mm is None or not mm.instances:
                    return False
                if any(i.inflight > 0 for i in mm.instances):
                    return False  # まだ処理中のインスタンスがある → 残す
                if self._model_sessions.get(model_id):
                    return False  # 解放手前で誰かが再登録した → 残す
                victims = [i.server for i in mm.instances if i.server is not None]
                mm.instances.clear()  # このモデルの全インスタンスを解放する
                dyn = mm.dynamic
            for srv in victims:
                srv.stop()  # state ロックの外で（最長 ~10s）
            if dyn:  # 全インスタンスを落とした動的モデルは登録ごと消す（表示から外す）
                with self._state:
                    mm2 = self._models.get(model_id)
                    if (
                        mm2 is not None
                        and not mm2.instances
                        and not self._model_sessions.get(model_id)
                    ):
                        self._models.pop(model_id, None)
        return bool(victims)

    def session_known(self, agent_id: str) -> bool:
        """agent_id の在席登録が存在するか（heartbeat 互換応答の判定用。生存推定には使わない）。"""
        with self._state:
            return agent_id in self._sessions

    def drop_sessions_for(self, model_id: str) -> int:
        """model_id の在席登録を全て捨てる（_state 保持下では呼ばない）。

        モデルがアンロードされた時点で呼ぶ。release を送れずに落ちたエージェントの
        置き去りセッションはここで回収されるので、ハートビートによる死活監視は要らない。
        「セッションは解放を早めるだけで、遅らせる力を持たない」という不変条件をこれが担保する
        （置き去りが残っても idle_timeout の解放は在席を見ないので、必ず解放される）。
        """
        with self._state:
            members = self._model_sessions.pop(model_id, set())
            for aid in members:
                self._sessions.pop(aid, None)
        return len(members)

    def uptime(self) -> float:
        """起動からの経過秒数（表示用）。"""
        return time.monotonic() - self._started

    def status(self) -> list[dict]:
        now = time.monotonic()
        with self._state:
            out = []
            for m in self._models.values():
                ready = [i for i in m.instances if i.server is not None and i.ready]
                loaded = bool(ready)
                inflight = sum(i.inflight for i in ready)
                # アイドル経過は「ロード済みかつ処理中でない」ときだけ意味がある
                # （idle_timeout までの残り表示に使う）。最後に使ったインスタンス基準。それ以外は None。
                idle_for = (
                    round(now - max(i.last_used for i in ready), 1)
                    if (loaded and inflight == 0)
                    else None
                )
                out.append(
                    {
                        "model": m.config.model,
                        "backend": m.config.backend,
                        "port": ready[0].config.port if ready else m.config.port,
                        "loaded": loaded,
                        # 起動中インスタンス数（負荷ベースの複製で >1 になる。並列度の目安）。
                        "instances": len(ready),
                        # 各インスタンスのワーカー PID（健全性の確認・孤児との突き合わせ用）。
                        "pids": [
                            pid
                            for i in ready
                            if (pid := getattr(i.server, "pid", None)) is not None
                        ],
                        "inflight": inflight,
                        "requests": m.requests,
                        "idle_for": idle_for,
                        # このモデルに在席宣言しているエージェント数（0 で即アンロード対象）。
                        "sessions": len(self._model_sessions.get(m.config.model, ())),
                    }
                )
            return out

    def shutdown(self) -> None:
        """全モデルサーバー（全インスタンス）を並列に停止する（ゲートウェイ終了時）。

        全体を畳むので graceful は不要 —— `grace=0` で各モデルを即 SIGKILL する。SIGTERM で
        待つと mlx/Metal の終了時クリーンアップに数秒かかり、それが TUI の quit 待ち時間として
        表面化するため。カーネルがメモリを回収するので即 kill でも取りこぼしはない。並列に
        するのは、外部からの停止（stop_pid の猶予）内に確実に収めるため。起動途中（_starting）の
        サーバーも止める（ロード中の Ctrl+C で巨大モデルのプロセスが孤児として残らないように）。
        以降の起動は _closing で拒否する。
        """
        with self._state:
            self._closing = True
            self._shutdown_event.set()
            timers = list(self._release_timers.values())
            self._release_timers.clear()
            background = list(self._background_threads)
            servers = list(self._starting)
            for m in self._models.values():
                for i in m.instances:
                    if i.server is not None:
                        servers.append(i.server)
                m.instances.clear()
            # _evict_if_needed で枠待ちしているロードを起こす（closing を見て中断させる）。
            self._state.notify_all()
        for timer in timers:
            timer.cancel()
        threads = [
            threading.Thread(target=s.stop, kwargs={"grace": 0.0}) for s in servers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Replica warm-up threads may have been between their closing check and
        # server registration. LocalServer's start/stop hand-off prevents a late
        # Popen leak; joining here makes their ownership explicit.
        for t in background:
            t.join(timeout=10.0)
        for timer in timers:
            try:
                if timer.is_alive():
                    timer.join(timeout=1.0)
            except RuntimeError:
                # shutdown can race with timer.start(); cancelled unstarted timers
                # have no thread or resource to reap.
                pass
        # A start transaction can enter _starting immediately before the first
        # snapshot. Sweep once more after background joins to cover that boundary.
        with self._state:
            late_servers = [s for s in self._starting if s not in servers]
        late_threads = [
            threading.Thread(target=s.stop, kwargs={"grace": 0.0})
            for s in late_servers
        ]
        for t in late_threads:
            t.start()
        for t in late_threads:
            t.join()
