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
停止時に解除できる。あるモデルの在席エージェントが 0 になった瞬間（＝他に同じモデルへ
接続しているエージェントが居ない）、処理中（inflight>0）でなければ **idle_timeout を待たずに
即アンロード**してメモリを解放する。チャット転送（/v1/...）とは別系統の管理エンドポイント:

  - `POST /admin/sessions/register`   `{"agent_id", "model"}`  … 利用開始（在席を宣言）
  - `POST /admin/sessions/heartbeat`  `{"agent_id"}`           … 生存通知（session_ttl 内に定期送信）
  - `POST /admin/sessions/release`    `{"agent_id"}`           … 利用終了（= `DELETE /admin/sessions`）

明示の release を送れずに落ちたエージェントは、ハートビートが `session_ttl` 秒途絶した時点で
掃除スレッドが無人扱いし、同じく即アンロードする。在席はメモリをピン留めしない（枠が要れば
従来どおり LRU 退避が優先される）。あくまで「使う人が居なくなったら早く片付ける」仕組み。
"""
from __future__ import annotations

import hmac
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field, fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import multipart, provisioner, sglang_provisioner, video, vllm_provisioner
from .proxy import forward, send_error, send_json
from .server import (
    BACKENDS,
    DEFAULT_BACKEND,
    MTP_DRAFTERS,
    GatewayAlreadyRunning,
    GatewayLock,
    LocalServer,
    ServerConfig,
    clear_gateway_runtime,
    daemon_log_path,
    discover_cached_models,
    enable_child_tethering,
    estimate_model_bytes,
    ignore_shutdown_signals,
    infer_backend,
    llama_provision_info,
    local_connect_host,
    parallel_supported,
    primary_lan_ip,
    reap_orphan_workers,
    reclaim_stale_workers,
    resolve_drafter,
    set_llama_server_binary,
    set_sglang_python,
    set_vllm_python,
    sglang_provision_info,
    vllm_provision_info,
    write_gateway_runtime,
)

# 複製インスタンス起動前の猶予秒数。ストリーミングのクライアントは [DONE] を受けた
# 直後に次のリクエストを送るが、ゲートウェイ側の inflight 解放は転送スレッドが上流の
# 終端を読み切る数 ms 後になる。この隙間に届いた**逐次**リクエストが「満杯」に見えて
# 複製が誤発動しないよう、猶予を置いてから持続的な競合かを再確認する。
_REPLICA_GRACE_S = 1.0

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


def _total_ram() -> int | None:
    """物理メモリの総バイト数。psutil が無い／取得不可なら None。"""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - psutil 不在・取得失敗はメモリガード無効として扱う
        return None


def _request_has_images(payload: dict) -> bool:
    """chat リクエストに画像入力が含まれるか（OpenAI vision 形式を検出）。

    OpenAI 互換の vision は `messages[].content` が配列で、その要素に `{"type": "image_url", ...}`
    （mlx_vlm は `image`/`input_image` も受ける）が混ざる。一部クライアントはトップレベル
    `images=[...]` を渡すのでそれも見る。type に "image" を含むかで緩く判定する。動画等の他
    モダリティは対象外（画像固有）。`vision_model` への振り分け判定に使う。
    """
    if payload.get("images"):
        return True
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                t = part.get("type")
                if isinstance(t, str) and "image" in t:
                    return True
    return False


class CapacityError(RuntimeError):
    """常駐枠が全て処理中で、待っても空かず新モデルをロードできなかった（→ 503）。"""


class GatewayDraining(RuntimeError):
    """ゲートウェイが再起動準備（drain）中で、新規リクエストを受け付けない（→ 503）。

    自動更新の再起動前に「実行中の処理・在席が無いことの確認」と「新規受付の停止」を
    同一ロック内で原子的に行うための状態。確認と kill の間に新しい生成が滑り込んで
    強制終了される事故（作業中の処理が落ちる）を防ぐ。クライアント（openai SDK）は
    503 を自動リトライするので、数秒後に上がる新ゲートウェイへ繋ぎ直される。
    """


@dataclass
class _Instance:
    """1 モデルの 1 起動インスタンス（独立プロセス・独立ポート）。

    同一モデルへリクエストが集中したとき、負荷ベースでこのインスタンスを複数起動して
    並列化する（→ ModelManager.acquire）。各インスタンスは自前の inflight（処理中数）と
    last_used（LRU 基準）を持ち、LRU 退避・idle 解放はインスタンス単位で行う。
    """

    config: ServerConfig  # このインスタンス専用の host/port（同一モデルでもポートは別）
    server: LocalServer | None = None
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
    requests: int = 0  # このモデルに振り分けた累計リクエスト数（表示用。退避で消えない）
    dynamic: bool = False  # 未登録モデルを動的登録したもの（全インスタンス消滅時に登録ごと消す）
    footprint: int | None = None  # 1 インスタンスの概算占有メモリ（バイト）。0=見積もり不能


@dataclass
class _Session:
    """1 エージェントの在席（このモデルを使うと宣言したクライアント）。

    inflight（処理中リクエスト数）とは別の軸で「接続中のエージェント」を数えるための
    もの。register で増え、release / ハートビート途絶（reap）で減る。あるモデルの
    セッションが 0 になった瞬間（＝そのモデルを使うエージェントが誰も居なくなった）に、
    処理中でなければ即座にアンロードしてメモリを解放する（idle_timeout を待たない）。
    """

    model_id: str
    last_seen: float  # time.monotonic()。最後のハートビート/登録時刻


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
        default_draft: str | None = None,
        default_parallel: int | None = None,
        max_memory_fraction: float | None = None,
        internal_base_port: int = 9001,
        public_port: int | None = None,
    ) -> None:
        self._models: dict[str, _Model] = {c.model: _Model(config=c) for c in configs}
        self._max_resident = max_resident
        self._load_timeout = load_timeout   # 枠が空くのを待つ最大秒数（None で無期限）
        self._start_timeout = start_timeout  # 1 インスタンスの起動完了（ready）を待つ最大秒数
        self._started = time.monotonic()    # 起動経過時間（uptime 表示用）の基準
        # 未登録モデルの動的ロード（IDからバックエンド推論。ロード時に表示へ追加・アンロードで消す）。
        self._dynamic = dynamic
        self._default_disable_thinking = default_disable_thinking
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
        self._control = threading.Lock()    # 起動/退避（control plane）の直列化
        # エージェント在席トラッキング（agent_id → セッション）と、その逆引き
        # （model_id → 在席エージェントの集合）。あるモデルの集合が空になった瞬間に
        # 即アンロードする判定に使う。_state ロック下で操作する。
        self._sessions: dict[str, _Session] = {}
        self._model_sessions: dict[str, set[str]] = {}
        # drain（再起動準備）の期限。monotonic 時刻がこれ未満の間は新規 acquire を
        # GatewayDraining で拒否する。0.0 で無効。再起動側が死んでも TTL で自動復帰する。
        self._drain_deadline: float = 0.0
        # 同一モデルの複製インスタンスを裏で起動中の model_id 集合（多重起動を防ぐ）。
        self._spawning: set[str] = set()
        # shutdown 済みフラグと、起動処理中（start〜instances 登録前）のサーバー集合。
        # shutdown はこの集合も止めることで、「ロード中に Ctrl+C → 起動しかけの巨大モデルが
        # 孤児プロセスとしてメモリとポートを掴んだまま残る」のを防ぐ。_state ロック下で操作する。
        self._closing = False
        self._starting: set[LocalServer] = set()

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
        if backend != _MTP_BACKEND:
            return None
        if raw == "auto" and model_id not in MTP_DRAFTERS:
            return None  # 対応表に無い → MTP なしで普通にロード
        return resolve_drafter(model_id, raw)

    @property
    def model_ids(self) -> list[str]:
        return list(self._models)

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
            stale = reclaim_stale_workers(port)
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
                [i for i in mm.instances if i.ready and i.server is not None] if mm else []
            )
            if ready:
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
                    ready = [i for i in mm.instances if i.ready and i.server is not None]
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
            server = LocalServer(cfg, log_path=daemon_log_path(cfg.port))
            with self._state:
                if self._closing:
                    raise RuntimeError("gateway is shutting down")
                self._starting.add(server)  # shutdown が起動途中のサーバーも止められるように
            self._reclaim_stale_port(cfg.port)  # 同ポートに残る孤児ワーカーを先に掃除
            try:
                server.start()
                server.wait_until_ready(timeout=self._start_timeout)
            except (RuntimeError, TimeoutError, ValueError):
                # ValueError は build_command の解決失敗（未キャッシュの repo-id 等）。
                server.stop()
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
            if model_id in self._spawning:
                return
            self._spawning.add(model_id)
        threading.Thread(
            target=self._spawn_replica, args=(model_id,), daemon=True
        ).start()

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
            time.sleep(_REPLICA_GRACE_S)
            with self._control:
                with self._state:
                    mm = self._models.get(model_id)
                    if mm is None:
                        return
                    ready = [i for i in mm.instances if i.ready and i.server is not None]
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
                server = LocalServer(cfg, log_path=daemon_log_path(cfg.port))
                with self._state:
                    if self._closing:
                        return
                    self._starting.add(server)  # shutdown が起動途中の複製も止められるように
                self._reclaim_stale_port(cfg.port)  # 同ポートに残る孤児ワーカーを先に掃除
                try:
                    server.start()
                    server.wait_until_ready(timeout=self._start_timeout)
                except (RuntimeError, TimeoutError, ValueError):
                    server.stop()
                    with self._state:
                        self._starting.discard(server)
                    return
                with self._state:
                    self._starting.discard(server)
                    mm = self._models.get(model_id)
                    if mm is None or self._closing:  # 起動中にモデルが消えた/終了中 → 止める
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
            victim_srv = None
            victim_model = None
            with self._state:
                running = self._running_instances_locked()
                resident = len(running)
                over_count = (
                    self._max_resident is not None and resident >= self._max_resident
                )
                over_mem = False
                if budget is not None:
                    keep_mm = self._models.get(keep)
                    need = self._footprint_locked(keep_mm) if keep_mm else 0
                    used = sum(self._footprint_locked(m) for (m, _i) in running)
                    over_mem = (used + need) > budget
                if not over_count and not over_mem:
                    return True
                candidates = [
                    (m, i) for (m, i) in running
                    if i.ready and i.inflight == 0 and m.config.model != keep
                ]
                if not candidates:
                    return False  # 退避できるアイドルが無い → 複製しない
                victim_model, victim_inst = min(candidates, key=lambda mi: mi[1].last_used)
                victim_srv = victim_inst.server
                victim_model.instances.remove(victim_inst)
            victim_srv.stop()  # state ロックの外で（最長 ~10s）
            if victim_model.dynamic and not victim_model.instances:
                with self._state:
                    self._models.pop(victim_model.config.model, None)

    def _footprint_locked(self, mm: _Model) -> int:
        """モデルの概算占有メモリ（バイト）。一度計算したらキャッシュする。0=見積もり不能。"""
        if mm.footprint is None:
            mm.footprint = estimate_model_bytes(mm.config) or 0
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
        deadline = (
            time.monotonic() + self._load_timeout if self._load_timeout else None
        )
        while True:
            victim_srv = None
            victim_model = None
            with self._state:
                if self._closing:
                    raise CapacityError("gateway is shutting down")
                running = self._running_instances_locked()
                resident = len(running)
                over_count = (
                    self._max_resident is not None and resident >= self._max_resident
                )
                over_mem = False
                if budget is not None:
                    keep_mm = self._models.get(keep)
                    need = self._footprint_locked(keep_mm) if keep_mm else 0
                    used = sum(self._footprint_locked(m) for (m, _i) in running)
                    over_mem = (used + need) > budget
                if not over_count and not over_mem:
                    return
                candidates = [
                    (m, i) for (m, i) in running
                    if i.ready and i.inflight == 0 and m.config.model != keep
                ]
                if candidates:
                    victim_model, victim_inst = min(candidates, key=lambda mi: mi[1].last_used)
                    victim_srv = victim_inst.server
                    victim_model.instances.remove(victim_inst)
                elif over_mem and resident == 0:
                    # 退避できる常駐インスタンスが無く、keep 単体で予算超過 → 待っても無駄。
                    raise CapacityError(
                        f"model '{keep}' needs ~{need / 1e9:.1f}GB but the memory budget is "
                        f"{budget / 1e9:.1f}GB (max_memory_fraction={self._mem_fraction:g} of "
                        f"{self._mem_total / 1e9:.1f}GB); not loading to avoid OOM"
                    )
                else:
                    # 全て処理中 → 枠が空く（release の notify）まで待つ。
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        why = "memory budget exceeded" if over_mem else (
                            f"all {self._max_resident} instance slot(s) busy"
                        )
                        raise CapacityError(
                            f"{why}; could not free room within {self._load_timeout:g}s"
                        )
                    self._state.wait(timeout=remaining)
                    continue  # 起きたら再判定
            if victim_srv is not None:
                victim_srv.stop()  # state ロックの外で（最長 ~10s）。停止後にループ再確認。
                if victim_model.dynamic and not victim_model.instances:  # 空なら登録ごと消す
                    with self._state:
                        self._models.pop(victim_model.config.model, None)

    def begin_drain(self, ttl: float = 120.0) -> dict:
        """再起動準備（drain）を試みる。アイドル確認と新規受付停止を**原子的に**行う。

        `_state` ロック下で「処理中リクエスト 0 かつ 在席エージェント 0」を確認し、
        満たすときだけ drain を開始する（以後 acquire は GatewayDraining → 503）。
        busy なら開始せず現状を返す（呼び出し側は空くのを待って再試行する）。
        再起動側が死んで drain だけ残っても、ttl 秒で自動解除され通常運転へ戻る。

        戻り値: {"ok": bool, "inflight": n, "sessions": n}
        """
        with self._state:
            inflight = sum(
                i.inflight for m in self._models.values() for i in m.instances
            )
            sessions = len(self._sessions)
            if inflight or sessions:
                return {"ok": False, "inflight": inflight, "sessions": sessions}
            self._drain_deadline = time.monotonic() + ttl
            return {"ok": True, "inflight": 0, "sessions": 0}

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
                    victim_model, victim_inst = min(idle, key=lambda mi: mi[1].last_used)
                    victim_srv = victim_inst.server
                    victim_model.instances.remove(victim_inst)
                victim_srv.stop()  # state ロックの外で（最長 ~10s）
                if victim_model.dynamic and not victim_model.instances:  # 空なら登録ごと消す
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
                            i.server is not None and i.ready and i.inflight == 0
                            and (now - i.last_used) > timeout
                        ):
                            victims.append((m, i.server))
                            m.instances.remove(i)
            for _m, srv in victims:
                srv.stop()  # state ロックの外で（最長 ~10s かかるため）
            with self._state:
                for m, _srv in victims:  # 全インスタンスが消えた動的モデルは登録ごと消す
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
                for m, _srv in victims:  # 全インスタンスが消えた動的モデルは登録ごと消す
                    if m.dynamic and not m.instances:
                        self._models.pop(m.config.model, None)
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
        それで無人かつ処理中でなくなれば、バックグラウンドで即アンロードする。
        """
        now = time.monotonic()
        freed: str | None = None
        with self._state:
            prev = self._sessions.get(agent_id)
            if prev is not None and prev.model_id != model_id:
                freed = self._detach_locked(agent_id, prev.model_id)
            self._sessions[agent_id] = _Session(model_id=model_id, last_seen=now)
            self._model_sessions.setdefault(model_id, set()).add(agent_id)
        if freed is not None:
            self._free_model_async(freed)

    def heartbeat(self, agent_id: str) -> bool:
        """在席エージェントの生存を更新する。未知の agent_id なら False（要 register）。"""
        with self._state:
            sess = self._sessions.get(agent_id)
            if sess is None:
                return False
            sess.last_seen = time.monotonic()
            return True

    def unregister_session(self, agent_id: str) -> bool:
        """エージェントの利用終了を記録する（停止時に呼ぶ）。

        対象モデルがそれで無人になり、かつ処理中でなければ即アンロードする（バックグラウンド）。
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
        """無人になったモデルを別スレッドで即アンロードする（HTTP 応答を 10s 待たせない）。"""
        threading.Thread(
            target=self._free_idle_model, args=(model_id,), daemon=True
        ).start()

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
                    if mm2 is not None and not mm2.instances and not self._model_sessions.get(model_id):
                        self._models.pop(model_id, None)
        return bool(victims)

    def reap_sessions(self, ttl: float) -> int:
        """ハートビートが ttl 秒途絶えた在席を掃除する（異常終了したエージェント対策）。

        途絶検出で無人になったモデルは即アンロードする。解放したモデル数を返す。掃除
        スレッドから定期的に呼ぶ（明示の release を呼べずに落ちたエージェントの保険）。
        """
        if ttl <= 0:
            return 0
        now = time.monotonic()
        freed: list[str] = []
        with self._state:
            dead = [
                aid for aid, s in self._sessions.items() if (now - s.last_seen) > ttl
            ]
            for aid in dead:
                sess = self._sessions.pop(aid)
                gone = self._detach_locked(aid, sess.model_id)
                if gone is not None:
                    freed.append(gone)
        count = 0
        for model_id in freed:
            if self._free_idle_model(model_id):
                count += 1
        return count

    def session_counts(self) -> dict[str, int]:
        """model_id → 在席エージェント数（status 表示用）。"""
        with self._state:
            return {mid: len(members) for mid, members in self._model_sessions.items()}

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
                    if (loaded and inflight == 0) else None
                )
                out.append({
                    "model": m.config.model,
                    "backend": m.config.backend,
                    "port": ready[0].config.port if ready else m.config.port,
                    "loaded": loaded,
                    # 起動中インスタンス数（負荷ベースの複製で >1 になる。並列度の目安）。
                    "instances": len(ready),
                    # 各インスタンスのワーカー PID（健全性の確認・孤児との突き合わせ用）。
                    "pids": [
                        pid for i in ready
                        if (pid := getattr(i.server, "pid", None)) is not None
                    ],
                    "inflight": inflight,
                    "requests": m.requests,
                    "idle_for": idle_for,
                    # このモデルに在席宣言しているエージェント数（0 で即アンロード対象）。
                    "sessions": len(self._model_sessions.get(m.config.model, ())),
                })
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
            servers = list(self._starting)
            for m in self._models.values():
                for i in m.instances:
                    if i.server is not None:
                        servers.append(i.server)
                m.instances.clear()
            # _evict_if_needed で枠待ちしているロードを起こす（closing を見て中断させる）。
            self._state.notify_all()
        threads = [threading.Thread(target=s.stop, kwargs={"grace": 0.0}) for s in servers]
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
        session_ttl: float | None = None,
        api_key: str | None = None,
        vision_model: str | None = None,
        video_frames: int = 8,
        video_max_edge: int = 768,
        repetition_penalty: float | None = None,
        repetition_context_size: int | None = None,
        repetition_penalty_skip_structured: bool = False,
    ) -> None:
        super().__init__(addr, _GatewayHandler)
        self.manager = manager
        # 繰り返しループ抑制の既定注入（mlx 系のみ。None で無効）。do_POST が chat リクエストに
        # 付与する（クライアントが自分で指定していれば尊重して上書きしない）。
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size
        # true なら tools / response_format を含む structured リクエストには注入しない（既定 false）。
        self.repetition_penalty_skip_structured = repetition_penalty_skip_structured
        self.catalog = catalog            # /v1/models で返すモデル一覧
        self.default_model = default_model
        self.timeout_s = timeout_s        # None なら無制限（長時間生成に備える）
        # 画像入りリクエストの振り分け先モデル（None で無効）。画像が壊れている vision モデルを
        # 避け、画像だけを確実に動くモデル（gemma-4 系など）へ流すための任意設定。
        self.vision_model = vision_model
        # 動画入力: video_url をゲートウェイでフレーム画像列に展開する設定（バックエンド非依存）。
        self.video_frames = video_frames
        self.video_max_edge = video_max_edge
        # ネットワーク公開時の API キー（None/空 で認証なし）。chat（/v1/*）と在席セッション
        # （/admin/sessions/*）に Authorization: Bearer <key> を要求する。/admin/status と
        # /admin/config はループバック限定（キーではなく接続元で制限）。
        self.api_key = api_key
        # GET /admin/status（TUI 等の監視用）で返すゲートウェイ設定。運用ポリシーを
        # 添えることで、常駐モデルのライブ状態と一緒に「上限/退避方針」も読み取れる。
        self.max_resident = max_resident
        self.idle_timeout = idle_timeout
        self.load_timeout = load_timeout
        self.session_ttl = session_ttl
        # 起動元情報（provenance）。「いつ・どこから立ったゲートウェイか」を /admin/status で
        # 見えるようにする。起動経路は `gw start` の 1 本だけ（__main__ が spawn マークの無い
        # 直接起動を拒否する）ので、経路の識別（旧 launcher フィールド）は無い。
        self.pid = os.getpid()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.start_cwd = os.getcwd()


# 受け付けるリクエストボディの上限（バイト）。vision の base64 画像を見込んでも十分大きく、
# かつ「巨大 Content-Length を申告してメモリを食い潰す」DoS は防ぐ。
_MAX_BODY_BYTES = 100 * 1024 * 1024


class _GatewayHandler(BaseHTTPRequestHandler):
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

    def do_GET(self) -> None:
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
            # 役目）。PyPI を叩きすぎないよう _UPDATE_ONDEMAND_THROTTLE 秒のスロットル付き。
            maybe_refresh_update_state(srv)
            host, port = srv.server_address[0], srv.server_address[1]
            models = srv.manager.status()
            data = {
                "object": "gateway.status",
                "host": host,
                "port": port,
                "max_resident": srv.max_resident,
                "idle_timeout": srv.idle_timeout,
                "load_timeout": srv.load_timeout,
                "session_ttl": srv.session_ttl,
                "default_model": srv.default_model,
                "vision_model": srv.vision_model,
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
        # 認可はボディを読む前に判定する（未認証のリモートに巨大ボディを読み込まされない）。
        # 管理系（設定変更）はローカルからのみ。以降（在席セッション・chat 転送）は
        # クライアント向けで、API キーが設定されていれば要求する。
        if path.endswith("/admin/config"):
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
        # 再起動準備（drain）。自動更新が「アイドル確認＋新規受付停止」を原子的に行うために
        # 使う（→ ModelManager.begin_drain）。ローカルの管理操作なので loopback 限定。
        if path.endswith("/admin/drain"):
            if not self._require_loopback():
                return
            if payload.get("enable", True):
                res = srv.manager.begin_drain()
                send_json(self, 200, {"object": "gateway.drain",
                                      "draining": res["ok"], **res})
            else:
                srv.manager.end_drain()
                send_json(self, 200, {"object": "gateway.drain",
                                      "draining": False, "ok": True})
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
            send_error(self, 400, "no 'model' in the request and no default_model is configured")
            return
        # 動画入力: video_url をフレーム画像（image_url）列に展開してから先へ進む。展開した
        # フレームは以降の画像扱い（vision_model 振り分けの対象にもなる）。抽出失敗は 400。
        if video.request_has_video(payload):
            try:
                video.expand_video_parts(
                    payload, srv.video_frames, srv.video_max_edge)
            except video.VideoError as exc:
                send_error(self, 400, f"video input could not be processed: {exc}")
                return
            body = json.dumps(payload).encode("utf-8")
        # vision_model が設定されていれば、**画像を含むリクエストはそのモデルへ振り分ける**。
        # 一部の vision モデル（例: Qwen3.6-27B / qwen3_5）は現行 mlx_vlm で画像入力が壊れて
        # いる（get_rope_index のスレッド/ストリーム不具合。MTP の有無に関係なくハング/エラー）。
        # 画像だけを「画像が確実に動くモデル」（gemma-4 系など）へ流すことで、テキストは元モデルの
        # まま・画像は読める、を両立する。既に vision_model 宛ならそのまま。body の model も書き換える。
        vmodel = getattr(srv, "vision_model", None)
        if (
            vmodel and isinstance(model, str) and model != vmodel
            and _request_has_images(payload)
        ):
            model = vmodel
            payload["model"] = vmodel
            body = json.dumps(payload).encode("utf-8")
        # 繰り返しループ抑制: mlx 系宛の生成リクエストに repetition_penalty を既定注入する
        # （chat/text completions のみ。クライアント明示は尊重。設定で無効化可）。
        if path.endswith(("/chat/completions", "/completions")):
            body = self._maybe_inject_repetition(srv, model, payload, body)
        self._acquire_and_forward(srv, model, body)

    def _handle_update_now(self, srv) -> None:
        """POST /admin/update: 新版を適用して再起動する（Ollama の「再起動して更新」相当）。

        自動更新が既にソースを追従済み（update_state.fetched）なら再起動だけを要求する。
        未取得なら、その場で check → apply（git pull + 依存同期。数十秒かかることがある）
        してから再起動を要求する。drain（アイドル待ち）は**しない**——ユーザーが明示的に
        「今すぐ」を選んだ操作なので、処理中のリクエストより更新を優先する。
        応答を返し切ってから再起動する（応答が途中で切れないよう少しだけ遅らせる）。
        """
        request_restart = getattr(srv, "request_restart", None)
        if request_restart is None:
            send_error(self, 503, "restart is not available (gateway not fully started)")
            return
        state = getattr(srv, "update_state", None)
        if not (state and state.get("fetched")):
            from . import update
            try:
                st = update.check(timeout=5.0)
            except Exception as exc:  # noqa: BLE001 - ネットワーク不調は 502 で返す
                send_error(self, 502, f"update check failed: {exc}")
                return
            if not st.available:
                send_json(self, 200, {"object": "gateway.update", "status": "up-to-date",
                                      "current": st.current, "latest": st.latest})
                return
            if not st.can_apply:
                send_error(self, 409,
                           f"update available but cannot auto-apply: {st.reason}")
                return
            try:
                ok, msg = update.apply_update()
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, str(exc)
            if not ok:
                send_error(self, 500, f"update failed: {msg}")
                return
            if state is not None:
                state.update({"fetched": True, "latest": st.latest})
        send_json(self, 200, {"object": "gateway.update", "status": "restarting",
                              "latest": (state or {}).get("latest")})
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
                self, 400,
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
            send_error(self, 400, "path-like model ids are not allowed from remote clients")
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
        try:
            forward(self, addr, body, srv.timeout_s)
        finally:
            srv.manager.release(handle)

    def do_DELETE(self) -> None:
        """DELETE /admin/sessions … エージェント停止時の解除（POST .../release と等価）。"""
        path = self._route_path()
        if not path.endswith("/admin/sessions"):
            send_error(self, 404, f"DELETE {self.path} is not supported by the gateway")
            return
        if not self._require_api_key():  # 認可はボディを読む前に判定する
            return
        body = self._read_body()
        if body is None:
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
                    self, 400,
                    "max_resident must be an integer >= 1 (or null/0/off for unlimited)",
                )
                return
            if value < 1:
                send_error(
                    self, 400,
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
            send_error(self, 400, "no 'model' in the request and no default_model is configured")
            return
        srv.manager.register_session(str(agent_id), str(model))
        send_json(self, 200, {"object": "gateway.session", "agent_id": agent_id,
                              "model": model, "registered": True})

    def _handle_session_heartbeat(self, srv, payload: dict) -> None:
        agent_id = payload.get("agent_id")
        if not agent_id:
            send_error(self, 400, "no 'agent_id' in the request")
            return
        alive = srv.manager.heartbeat(str(agent_id))
        if not alive:
            # 未知のセッション（reap 済み or 未 register）。クライアントに再登録を促す。
            send_error(self, 404, f"unknown session '{agent_id}'; register first")
            return
        send_json(self, 200, {"object": "gateway.session", "agent_id": agent_id, "alive": True})

    def _handle_session_release(self, srv, payload: dict) -> None:
        agent_id = payload.get("agent_id")
        if not agent_id:
            send_error(self, 400, "no 'agent_id' in the request")
            return
        existed = srv.manager.unregister_session(str(agent_id))
        send_json(self, 200, {"object": "gateway.session", "agent_id": agent_id,
                              "released": existed})


@dataclass
class GatewayConfig:
    host: str
    port: int
    max_resident: int | None
    default_model: str | None
    models: list[ServerConfig] = field(default_factory=list)
    idle_timeout: float | None = 1200.0  # 秒。これだけ使われないモデルを自動アンロード（既定 1200=20分。None/0 で無効）
    load_timeout: float = 300.0        # 秒。全枠処理中のとき、空くのを待つ最大時間（超過で 503）
    start_timeout: float = 120.0       # 秒。モデルサーバー1つの起動完了（ready）を待つ最大時間（巨大モデルは延ばす）
    request_timeout: float | None = 600.0  # 秒。上流との通信が無応答のとき打ち切る（0 で無制限）。ハングした／
                                       # 沈黙した上流が inflight を握ったまま枠を塞ぎ続けるのを防ぐ保険。トークンが
                                       # 流れている限り切れないので、長時間ストリーミング生成は妨げない（既定 600=10分）
    session_ttl: float | None = 90.0   # 秒。在席エージェントのハートビートがこれだけ途絶えたら無人扱いで掃除（既定 90。None/0 で無効）
    dynamic: bool = True               # 未登録モデルを ID 推論で動的ロードする（false で事前登録のみ）
    disable_thinking: bool = False     # 動的ロード時の既定（思考抑制）。事前登録は各 [[models]] が優先
    draft_model: str | None = None     # 動的ロード時の MTP 既定。None で mlx-vlm は "auto"（対応表から自動）。"off" で無効
    parallel: int | None = None        # 動的ロード時の並列スロット既定（llama-cpp のみ。他は無視）
    max_memory_fraction: float | None = None  # 常駐モデルの推定占有量の合計を総RAMのこの割合に制限（None で無効）
    internal_base_port: int = 9001     # 内部サーバーの割当開始ポート（動的モデルもこの続きから割り当て）
    api_key: str | None = None         # ネットワーク公開時の API キー（None/空 で認証なし）。chat と在席セッションに要求
    auto_update: bool = True           # 常駐デーモンが PyPI 新版を検知したら git pull で自動追従する（既定 true。false で無効）
    tray: bool = True                  # 稼働中メニューバーにアイコンを出す（macOS のみ。false で非表示 → tray.py）
    vision_model: str | None = None    # 画像入りリクエストの振り分け先モデル（None で無効）。画像が壊れている
                                       # vision モデルを避け、画像だけを確実に動くモデル（gemma-4 系等）へ流す
    # --- llama.cpp（llama-server）バイナリの自動導入。[llama_cpp] テーブルで設定 ---
    # 導入方法は選ばせない（管理dirの導入済みを再利用→無ければプリビルト自動DL の一本道）。
    llama_accel: str = "auto"          # auto=検出（GPU なら vulkan、mac は metal、無ければ cpu）/ cuda / vulkan / metal / cpu
    llama_build: str | None = None     # ビルド番号の固定（例 "b9946"）。省略で最新を取得し導入済みを使い続ける
    # vLLM / SGLang（Linux/NVIDIA・Windows は WSL2）も一本道: 現在の環境に有ればそれを、
    # 無ければ隔離 venv へ自動導入（backend='vllm'/'sglang' のモデルを使ったときだけ動く）。
    # --- 動画入力: ゲートウェイが video_url をフレーム画像列へ展開して上流へ渡す ---
    video_frames: int = 8              # 1 本の動画から等間隔で抜くフレーム数
    video_max_edge: int = 768          # 各フレームの縮小サイズ（長辺ピクセル）
    # --- 繰り返しループ抑制: mlx 系バックエンド（mlx / mlx-vlm）宛の chat リクエストに
    #     repetition_penalty を既定注入する（mlx-lm/mlx-vlm 拡張パラメータ）。低温・量子化の
    #     ローカル LLM が「同じ内容を繰り返して終わらない」degeneration の緩和。llama-cpp は
    #     パラメータ名が異なる（repeat_penalty）ので対象外＝mlx 系だけに付ける（ユーザー方針）。
    #     クライアントが自分で repetition_penalty を指定していれば尊重する（上書きしない）。---
    repetition_penalty: float | None = 1.1   # 既定 1.1（llama.cpp 既定と同値の穏当な値）。
                                             # 0 / false / "off" で無効化（注入しない）＝設定しない選択
    repetition_context_size: int | None = None  # 併せて注入する参照窓（mlx 既定 20。研究推奨 64）。
                                                # None なら注入しない（repetition_penalty だけ付ける）
    # 構造化リクエスト（`tools`＝native ツールコール / `response_format`＝構造化出力）には
    # repetition_penalty を注入しないオプション。既定 false（＝従来どおり全 chat に注入）。
    # true にすると、必須の繰り返し記号（JSON 構文・フィールド名）を減点しうる structured 生成を
    # 保護できる（実測では 1.1 で実害は無いが、保険として明示的に切れるようにする）。
    repetition_penalty_skip_structured: bool = False


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
        session_ttl = 90            # 在席エージェントのハートビート猶予秒数（省略時 90。0 で無効）
        internal_base_port = 9001   # 内部サーバーの割当開始ポート（省略時 9001）
        default_model = "..."       # model 省略リクエスト時のモデル（省略可）
        draft_model = "auto"        # 全モデルの MTP ドラフター既定（mlx-vlm のみ有効。省略可）

        [[models]]
        model = "ToPo-ToPo/Qwen3.6-27B-mlx-4bit"
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
    # ゲートウェイは AF_INET（IPv4）で bind する。"::" / "*" は bind 時に分かりにくい
    # OSError で落ちるので、設定読み込みの時点で明確に断る。
    if host in ("::", "*"):
        raise ValueError(
            f'host = "{host}" is not supported; use "0.0.0.0" to listen on all '
            "IPv4 interfaces (or a specific IPv4 address)"
        )
    port = int(data.get("port", 8799))
    internal_base = int(data.get("internal_base_port", 9001))
    # ネットワーク公開時の API キー（省略/空 で認証なし）。chat（/v1/*）と在席セッション
    # （/admin/sessions/*）に Authorization: Bearer <key> を要求する。
    api_key = data.get("api_key")
    if api_key is not None:
        api_key = str(api_key).strip() or None
    max_resident = data.get("max_resident")
    if max_resident is not None:
        max_resident = int(max_resident)
        if max_resident < 1:
            raise ValueError("max_resident must be 1 or greater")
    default_model = data.get("default_model")
    # 画像入りリクエストの振り分け先モデル（省略で無効）。空文字は None 扱い。
    vision_model = data.get("vision_model")
    if vision_model is not None:
        vision_model = str(vision_model).strip() or None
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
    # モデルサーバー1つの起動完了（ready）を待つ最大秒数。巨大モデル・コールドディスクでは
    # 120 秒を超えることがあるので設定可能にする。
    start_timeout = float(data.get("start_timeout", 120.0))
    if start_timeout < 1:
        raise ValueError("start_timeout must be 1 or greater")
    # 上流モデルサーバーとの通信タイムアウト（ソケット単位の無応答秒数）。省略時 600（=10分）、0 で無制限。
    # ハングした／沈黙したサーバーが inflight を握り続けて枠を塞ぐ事故の保険（トークンが流れている
    # 限り切れないので、ストリーミングの長時間生成は妨げない）。正当な長時間生成を切らないよう高め。
    request_timeout = data.get("request_timeout", 600.0)
    if request_timeout is not None:
        request_timeout = float(request_timeout)
        if request_timeout < 0:
            raise ValueError("request_timeout must be 0 or greater (0 disables)")
        if request_timeout == 0:
            request_timeout = None
    # 在席エージェントのハートビート猶予秒数。途絶でそのエージェントを無人扱いし、モデルが
    # 無人になれば即アンロード（明示 release を呼べずに落ちたエージェントの保険）。0 で無効。
    session_ttl = data.get("session_ttl", 90)
    if session_ttl is not None:
        session_ttl = float(session_ttl)
        if session_ttl < 0:
            raise ValueError("session_ttl must be 0 or greater (0 disables)")
        if session_ttl == 0:
            session_ttl = None
    # TUI が PyPI 新版を検知したら git pull で自動追従するか（既定 true。false で無効）。
    auto_update = bool(data.get("auto_update", True))
    tray = bool(data.get("tray", True))
    # 未登録モデルを ID 推論で動的ロードするか（既定 true）。false なら事前登録のみ（旧挙動）。
    dynamic = bool(data.get("dynamic", True))
    # 動的ロード時の既定 disable_thinking（事前登録の [[models]] は各自の値が優先）。
    dyn_disable_thinking = bool(data.get("disable_thinking", False))
    # ゲートウェイ全体の MTP ドラフター既定。各 [[models]] が draft_model を持たなければ
    # これを継承する（"auto" で本体名から自動選択）。個別に "" / "off" / "none" で無効化。
    default_draft = data.get("draft_model")
    # 動的ロード時の並列スロット既定（llama-cpp のみ。他バックエンドは逐次処理なので無視）。
    default_parallel = data.get("parallel")
    if default_parallel is not None:
        default_parallel = int(default_parallel)
        if default_parallel < 1:
            raise ValueError("parallel must be 1 or greater")
    # メモリガード（→ docs/llama-cpp.md）。常駐モデルの推定占有量の合計を総RAMのこの割合に
    # 制限する。0 < x <= 1。省略で無効。
    max_memory_fraction = data.get("max_memory_fraction")
    if max_memory_fraction is not None:
        max_memory_fraction = float(max_memory_fraction)
        if not (0.0 < max_memory_fraction <= 1.0):
            raise ValueError("max_memory_fraction must be in (0, 1]")

    # 繰り返しループ抑制の既定注入（mlx 系のみ）。既定 1.1。0 / false / "off" / "none" で
    # 無効化（＝注入しない＝「設定しない」選択）。< 1.0 は繰り返しを助長するので拒否する
    # （1.0 は中立＝無効相当だが受け付ける）。
    rp_raw = data.get("repetition_penalty", 1.1)
    if (rp_raw is None or rp_raw is False
            or (isinstance(rp_raw, str) and rp_raw.strip().lower() in ("off", "none", "false", ""))):
        repetition_penalty = None
    else:
        repetition_penalty = float(rp_raw)
        if repetition_penalty == 0.0:
            repetition_penalty = None  # 0 も無効化として扱う
        elif repetition_penalty < 1.0:
            raise ValueError(
                "repetition_penalty must be >= 1.0 (1.0 = neutral; < 1.0 encourages "
                "repetition). Use 0 / false / \"off\" to disable injection."
            )
    # 併せて注入する参照窓（省略時は付けない＝上流の既定 20 に任せる）。
    rcs_raw = data.get("repetition_context_size")
    repetition_context_size = None if rcs_raw is None else int(rcs_raw)
    if repetition_context_size is not None and repetition_context_size < 1:
        raise ValueError("repetition_context_size must be 1 or greater")
    # 構造化リクエスト（tools / response_format）を注入対象から外すか（既定 false）。
    repetition_penalty_skip_structured = bool(data.get("repetition_penalty_skip_structured", False))

    # [llama_cpp] テーブル: llama-server バイナリの自動導入設定（すべて省略可＝全自動）。
    # 導入方法の選択肢（旧 provision）は無い——一本道なので accel / pin だけ。
    llama = data.get("llama_cpp") or {}
    if not isinstance(llama, dict):
        raise ValueError("[llama_cpp] must be a table")
    llama_accel = str(llama.get("accel", "auto"))
    if llama_accel not in ("auto", "cuda", "vulkan", "metal", "cpu"):
        raise ValueError("llama_cpp.accel must be auto / cuda / vulkan / metal / cpu")
    llama_build = llama.get("pin")
    if llama_build is not None:
        llama_build = str(llama_build).strip() or None

    # 動画入力のフレーム展開設定。省略で 8 フレーム / 長辺 768px。
    video_frames = int(data.get("video_frames", 8))
    if video_frames < 1:
        raise ValueError("video_frames must be 1 or greater")
    video_max_edge = int(data.get("video_max_edge", 768))
    if video_max_edge < 64:
        raise ValueError("video_max_edge must be 64 or greater")

    entries = data.get("models") or []
    if not isinstance(entries, list):
        raise ValueError("[[models]] must be an array")
    if not entries and not dynamic:
        raise ValueError(
            "gateway config needs a non-empty [[models]] array (or set dynamic = true)"
        )

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

    # dynamic 無効のときだけ default_model が事前登録に在ることを要求する
    # （dynamic 有効なら未登録でも動的ロードされる）。
    if default_model is not None and not dynamic and default_model not in seen:
        raise ValueError(f"default_model '{default_model}' is not listed in [[models]]")

    return GatewayConfig(
        host, port, max_resident, default_model, configs, idle_timeout, load_timeout,
        start_timeout=start_timeout,
        request_timeout=request_timeout,
        session_ttl=session_ttl,
        dynamic=dynamic, disable_thinking=dyn_disable_thinking,
        draft_model=default_draft, parallel=default_parallel,
        max_memory_fraction=max_memory_fraction,
        internal_base_port=internal_base,
        api_key=api_key,
        auto_update=auto_update,
        tray=tray,
        vision_model=vision_model,
        llama_accel=llama_accel,
        llama_build=llama_build,
        video_frames=video_frames,
        video_max_edge=video_max_edge,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        repetition_penalty_skip_structured=repetition_penalty_skip_structured,
    )


# gateway.toml を保存した瞬間に反映するホットリロードの監視周期（秒）。mtime ポーリング。
_CONFIG_POLL_INTERVAL = 1.0
# 稼働中には変えられない構造設定（ソケットは bind 済み、内部ポート割当は起動時に固定）。
# 変更を検知したら「要再起動」を警告するだけで、サーバーは止めず旧値のまま動かし続ける。
_RESTART_ONLY_FIELDS = (
    "host", "port", "internal_base_port", "models",
    # llama-server バイナリ・vLLM/SGLang venv は起動時に導入・解決するため、変更は再起動が要る。
    "llama_accel", "llama_build",
)


def apply_live_config(
    server: GatewayServer,
    manager: ModelManager,
    cfg: GatewayConfig,
    new: GatewayConfig,
) -> tuple[list[str], list[str]]:
    """読み直した設定 `new` を稼働中の server / manager / cfg へ無停止で反映する。

    ポリシー設定（vision_model・各 timeout・max_resident・api_key・動的ロード既定）は
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
    if cfg.vision_model != new.vision_model:
        note("vision_model", cfg.vision_model, new.vision_model)
        server.vision_model = new.vision_model
    if cfg.video_frames != new.video_frames:
        note("video_frames", cfg.video_frames, new.video_frames)
        server.video_frames = new.video_frames
    if cfg.video_max_edge != new.video_max_edge:
        note("video_max_edge", cfg.video_max_edge, new.video_max_edge)
        server.video_max_edge = new.video_max_edge
    if cfg.repetition_penalty != new.repetition_penalty:
        note("repetition_penalty", cfg.repetition_penalty, new.repetition_penalty)
        server.repetition_penalty = new.repetition_penalty
    if cfg.repetition_context_size != new.repetition_context_size:
        note("repetition_context_size", cfg.repetition_context_size, new.repetition_context_size)
        server.repetition_context_size = new.repetition_context_size
    if cfg.repetition_penalty_skip_structured != new.repetition_penalty_skip_structured:
        note("repetition_penalty_skip_structured",
             cfg.repetition_penalty_skip_structured, new.repetition_penalty_skip_structured)
        server.repetition_penalty_skip_structured = new.repetition_penalty_skip_structured
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
    if cfg.session_ttl != new.session_ttl:
        note("session_ttl", cfg.session_ttl, new.session_ttl)
        server.session_ttl = new.session_ttl
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
                note("max_memory_fraction", cfg.max_memory_fraction,
                     new.max_memory_fraction)
        else:
            manager._mem_fraction = new.max_memory_fraction
            note("max_memory_fraction", cfg.max_memory_fraction,
                 new.max_memory_fraction)

    # cfg を new に揃える: ①掃除スレッドが cfg.idle_timeout / cfg.session_ttl を毎周期読む
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
    skip_mtime = None  # 直近に読み込み失敗した mtime（同一内容の再警告・再試行を避ける）
    while not stop_event.wait(poll_interval):
        try:
            mtime = os.path.getmtime(config_path)
        except OSError:
            continue  # 一時的に消えた（エディタの原子的保存の隙間など）。次周期で拾う。
        if mtime == last_mtime or mtime == skip_mtime:
            continue
        try:
            new_cfg = load_gateway_config(config_path)
        except (OSError, ValueError) as exc:  # TOMLDecodeError も ValueError の subclass
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
            print("Config reloaded (applied live): " + "; ".join(changed),
                  file=sys.stderr)
        if restart_needed:
            print(
                "Config reloaded: these changes need a restart to take effect "
                "(still running with the old values): " + ", ".join(restart_needed),
                file=sys.stderr,
            )
        if not changed and not restart_needed:
            print("Config reloaded: no effective change.", file=sys.stderr)


# 自動更新を適用したので新コードで再起動したい、を表す内部終了コード（run_gateway が execv）。
# 通常終了(0)・既に起動済み(3)と衝突しない値。
_RESTART_CODE = 7

# 自動更新ウォッチャーの周期（秒）。モジュール定数にして差し替え可能にする。
_UPDATE_WARMUP_INTERVAL = 60.0     # 起動直後は 1 分だけ待ってから初回チェック（起動処理と競合させない）
_UPDATE_CHECK_INTERVAL = 3600.0    # 以降、新版が未検知のあいだの確認周期
_UPDATE_DRAIN_POLL_INTERVAL = 30.0  # 取得済み・再起動待ちのあいだ、空くのを待つ周期
# オンデマンド確認（トレイのメニューを開くたび = /admin/status GET）のスロットル。
# PyPI を叩きすぎないための最短間隔。定期チェック（1時間）より短く、確認をほぼ即時にする。
_UPDATE_ONDEMAND_THROTTLE = 30.0


def refresh_update_state(state: dict) -> None:
    """update.check() を 1 回だけ実行して update_state を更新する（**適用はしない**）。

    「更新の有無」を最新化する純粋な確認。オンデマンド（トレイのメニューを開いたとき）に
    リスタート無しで呼ぶための小片。取得や再起動は一切しない——それは _update_watcher と
    手動更新（/admin/update）の役目。ネットワーク I/O は失敗しても握りつぶす。
    fetched（取得済み・再起動待ち）フラグは watcher が立てたものを消さない（触らない）。
    """
    from . import update
    try:
        st = update.check(timeout=3.0)
    except Exception:  # noqa: BLE001 - 確認失敗（オフライン等）は状態を変えず黙って戻る
        return
    state["available"] = bool(st.available)
    state["current"] = st.current
    state["latest"] = st.latest
    state["reason"] = st.reason


def maybe_refresh_update_state(srv) -> None:
    """スロットル付きで、バックグラウンドに 1 本だけオンデマンド確認を走らせる。

    /admin/status GET のたびに呼ばれる（トレイがメニューを開くたび）。前回から
    _UPDATE_ONDEMAND_THROTTLE 秒未満・確認中・状態が無いときは何もしない。レスポンスは
    ブロックしない（結果は次回の GET で反映される＝トレイは次に開いたとき最新になる）。
    """
    state = getattr(srv, "update_state", None)
    if state is None:
        return
    now = time.monotonic()
    if getattr(srv, "_update_check_inflight", False):
        return
    if now - getattr(srv, "_last_update_check", 0.0) < _UPDATE_ONDEMAND_THROTTLE:
        return
    srv._last_update_check = now
    srv._update_check_inflight = True

    def _work() -> None:
        try:
            refresh_update_state(state)
        finally:
            srv._update_check_inflight = False

    threading.Thread(target=_work, daemon=True).start()


def _update_watcher(
    manager: "ModelManager",
    stop: threading.Event,
    restart_requested: threading.Event,
    *,
    auto_apply: bool = True,
    state: dict | None = None,
    notify=None,
) -> None:
    """PyPI 新版を検知し、（auto_apply なら）作業ツリーがクリーンな時 git pull で追従する常駐スレッド。

    旧 TUI が担っていた自動更新（clone 運用で PyPI 新版を git で追従）をデーモン本体へ移したもの。
    安全側の 2 段構え —— ①**取得は稼働中に先に済ませる**（`git pull`＋`uv sync`。プロセスには
    触れず、この間も通常どおりリクエストを受ける）②**再起動は drain が通ったときだけ**行う。
    `manager.begin_drain()` が「処理中 0・在席 0」の確認と新規受付停止を**原子的に**行うので、
    確認と再起動の隙に生成が滑り込んで強制終了される余地が無い。busy なら何も止めずに保留し、
    空いた瞬間に再起動する。ネットワーク I/O・git は失敗しても握りつぶす（稼働は妨げない）。

    未検知のあいだは 1 時間おき、取得済みで再起動待ちのあいだは 30 秒おきに drain を再試行する。

    **チェック自体は auto_apply=false でも行う**（適用はしない）——Ollama と同じく
    「更新がある」ことをトレイの更新マークで見せるため。検知状態は `state`
    （server.update_state。/admin/status に載る）へ書き、`notify`（トレイへの通知線）に
    `update-available <ver>` / `update-ready <ver>` を 1 版につき 1 回だけ流す。
    """
    from . import update

    fetched = False  # ソースは新版へ追従済みで、あとは drain が通れば再起動するだけ
    notified: str | None = None  # この版は通知済み（毎時間チカチカ再通知しない）
    first = True

    def _tell(kind: str, latest: str) -> None:
        nonlocal notified
        if notify is None or notified == f"{kind}:{latest}":
            return
        notified = f"{kind}:{latest}"
        try:
            notify(f"{kind} {latest}")
        except Exception:  # noqa: BLE001 - 通知はおまけ（トレイ不在等で失敗しても続行）
            pass

    while not stop.wait(
        _UPDATE_WARMUP_INTERVAL if first
        else (_UPDATE_DRAIN_POLL_INTERVAL if fetched else _UPDATE_CHECK_INTERVAL)
    ):
        first = False
        if not fetched:
            try:
                st = update.check(timeout=3.0)
            except Exception:  # noqa: BLE001 - 監視スレッドは落とさない
                continue
            if state is not None:
                state.update({
                    "available": bool(st.available), "current": st.current,
                    "latest": st.latest, "reason": st.reason,
                })
            if not st.available:
                continue  # オフライン・最新
            if not (auto_apply and st.can_apply):
                # 自動適用しない（auto_update=false）／できない（dirty で WIP を守る等）。
                # 更新マークだけ出して、適用はユーザーの「今すぐ更新」（/admin/update）に任せる。
                _tell("update-available", st.latest)
                continue
            # 取得は稼働中に先に済ませる（プロセスには触れない。ここでは再起動しない）。
            try:
                ok, msg = update.apply_update()
            except Exception as exc:  # noqa: BLE001
                print(f"Auto-update: fetch skipped ({exc}).", file=sys.stderr)
                continue
            if not ok:
                print(f"Auto-update: not applied ({msg}).", file=sys.stderr)
                continue
            fetched = True
            if state is not None:
                state["fetched"] = True
            _tell("update-ready", st.latest)
            print(
                f"Auto-update: fetched ({msg}); will restart on new code when idle.",
                file=sys.stderr,
            )
        # ソース追従済み。処理中/在席が 0 を原子的に確認できた（drain 成功）ときだけ再起動する。
        # 成功後は新規受付が止まる（503→クライアントが新プロセスへリトライ）ので、下の
        # メインループが finally でクリーン停止 → run_gateway が execv で新コードに置き換える。
        if manager.begin_drain()["ok"]:
            print("Auto-update: idle; restarting the gateway on new code...", file=sys.stderr)
            restart_requested.set()
            return
        # busy → 何も止めずに保留（次周期で再試行）。


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
    # 自動更新を idle 時に適用したら、ロックとポートを解放し切った **後** で自分自身を
    # 新コードに置き換える（execv は fd を引き継ぐので、ロック保持中に再取得すると自分と
    # 衝突する。必ず lock.release() を通してから exec する）。exec は戻らない。
    if rc == _RESTART_CODE:
        from . import update
        # 依存の入れ直しは再起動の直前（全ワーカー停止済み・自分は exec 目前）に行う——
        # tool venv（make install 導入）は uv sync では更新されず、これを怠るとコードだけ
        # 新しく依存が古い「静かな機能欠け」になる（例: pyobjc 不在でトレイが出ない）。
        ok, msg = update.refresh_tool_env(update.repo_root())
        print(f"Auto-update: dependencies — {msg}", file=sys.stderr)
        if not ok:
            print("Auto-update: 依存の入れ直しに失敗しました。挙動がおかしい場合は "
                  "`make install` を実行してください。", file=sys.stderr)
        update.reexec_daemon()
    return rc


def _llama_cpp_in_use(cfg: GatewayConfig) -> bool:
    """この構成で llama-server（llama-cpp）が使われ得るか。

    - 事前登録に llama-cpp モデルがあれば True。
    - 動的ロード有効で OS 既定バックエンドが llama-cpp（＝非 Apple Silicon）なら True。
    Apple Silicon で llama-cpp モデルの登録が無い場合は、既定が mlx-vlm なので False
    （GGUF を明示要求したときだけ llama-server が要る。その場合は PATH / system で賄う）。
    """
    if any(c.backend == "llama-cpp" for c in cfg.models):
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
        binary = provisioner.ensure_llama_server(
            accel=cfg.llama_accel,
            build=cfg.llama_build,
        )
    except Exception as exc:  # noqa: BLE001 - 導入失敗で起動を止めない（オフライン・未知アーキ等も含む）
        print(f"llama.cpp provisioning failed (continuing without it): {exc}",
              file=sys.stderr)
        return
    # 実際に解決された素性（実ビルド番号・accel）はプロビジョナが記録している。
    info = provisioner.last_info() or {}
    set_llama_server_binary(binary, build=info.get("build"), accel=info.get("accel"))
    print(f"llama.cpp ready: {binary} "
          f"(build={info.get('build') or '-'}, accel={info.get('accel') or '-'})",
          file=sys.stderr)


def _vllm_in_use(cfg: GatewayConfig) -> bool:
    """事前登録に backend="vllm" のモデルがあるか（vLLM は明示 opt-in 専用）。"""
    return any(c.backend == "vllm" for c in cfg.models)


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
        print(f"vLLM provisioning failed (continuing without it): {exc}",
              file=sys.stderr)
        return
    set_vllm_python(py)
    print(f"vLLM ready: {py}", file=sys.stderr)


def _sglang_in_use(cfg: GatewayConfig) -> bool:
    """事前登録に backend="sglang" のモデルがあるか（SGLang は明示 opt-in 専用）。"""
    return any(c.backend == "sglang" for c in cfg.models)


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
        print(f"SGLang provisioning failed (continuing without it): {exc}",
              file=sys.stderr)
        return
    set_sglang_python(py)
    print(f"SGLang ready: {py}", file=sys.stderr)


def _maybe_spawn_tray(cfg: GatewayConfig) -> tuple[subprocess.Popen | None, int | None]:
    """メニューバーアイコン（tray.py）を随伴プロセスとして起動する（macOS・tray=true のみ）。

    デーモンと同じプロセスグループに置く（`gw stop` の killpg で一緒に止まる）うえ、
    トレイ**専用の**パイプを渡す——EOF（デーモンの死。kill -9 でも OS が閉じる）で
    アイコンが自分から消えるのはワーカーの繋留と同じで、加えてこのパイプは
    **更新通知の下り線**を兼ねる（update watcher が `update-ready <ver>` 等を書くと
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
        sys.executable, "-m", "local_llm_server.tray",
        "--host", local_connect_host(cfg.host), "--port", str(cfg.port),
        "--fd", str(rfd),
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, pass_fds=(rfd,),  # stdout/err はデーモンのログへ
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


def _run_gateway_locked(cfg: GatewayConfig, config_path: str | None = None) -> int:
    """単一起動ロック取得済みで実際にゲートウェイを回す本体（run_gateway が呼ぶ）。

    `config_path` が渡されれば、gateway.toml を保存した瞬間に設定を無停止で反映する
    ホットリロード監視スレッドを起動する（→ apply_live_config）。
    """
    # メニューバーアイコン（macOS・tray=true）は**最初に**出す——この後の自動導入
    # （llama.cpp 等のダウンロード。初回は数十秒〜数分）を待たせない。アイコンは
    # 「デーモンが生きている」の表示であり、準備完了の表示ではない（メニューは
    # ゲートウェイが応答するまで「状態を取得中…」を出す）。専用パイプの EOF で消える
    # 「アイコンの存在＝デーモンの生存」はそのまま。書き込み端（tray_fd）は
    # update watcher が更新通知を流すのに使う。
    tray_proc, tray_fd = _maybe_spawn_tray(cfg)

    def _tray_notify(line: str) -> None:
        """トレイへ 1 行通知する（トレイ無し・死亡済みは黙って無視）。"""
        if tray_fd is None:
            return
        try:
            os.write(tray_fd, (line + "\n").encode("utf-8"))
        except OSError:
            pass

    provision_llama_if_needed(cfg)
    provision_vllm_if_needed(cfg)
    provision_sglang_if_needed(cfg)
    manager = ModelManager(
        cfg.models, max_resident=cfg.max_resident, load_timeout=cfg.load_timeout,
        start_timeout=cfg.start_timeout,
        dynamic=cfg.dynamic, default_disable_thinking=cfg.disable_thinking,
        default_draft=cfg.draft_model, default_parallel=cfg.parallel,
        max_memory_fraction=cfg.max_memory_fraction,
        internal_base_port=cfg.internal_base_port, public_port=cfg.port,
    )
    server = GatewayServer(
        (cfg.host, cfg.port),
        manager,
        catalog=[c.model for c in cfg.models],
        default_model=cfg.default_model,
        timeout_s=cfg.request_timeout,
        max_resident=cfg.max_resident,
        idle_timeout=cfg.idle_timeout,
        load_timeout=cfg.load_timeout,
        session_ttl=cfg.session_ttl,
        api_key=cfg.api_key,
        vision_model=cfg.vision_model,
        video_frames=cfg.video_frames,
        video_max_edge=cfg.video_max_edge,
        repetition_penalty=cfg.repetition_penalty,
        repetition_context_size=cfg.repetition_context_size,
        repetition_penalty_skip_structured=cfg.repetition_penalty_skip_structured,
    )
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
    # ネットワーク公開の認証状態。公開かつ未認証は目立つ警告を出す。
    if cfg.api_key:
        print("  auth: API key required (Authorization: Bearer <key>)", file=sys.stderr)
    elif exposed:
        print(
            "  WARNING: bound to a network interface WITHOUT an api_key — anyone who can "
            "reach this host:port can use the models. Set api_key in gateway.toml.",
            file=sys.stderr,
        )
    print("  admin (/admin/status, /admin/config): localhost only", file=sys.stderr)
    for c in cfg.models:
        print(f"    {c.model}  ->  127.0.0.1:{c.port} ({c.backend})", file=sys.stderr)
    cap = "unlimited" if cfg.max_resident is None else (
        f"{cfg.max_resident} (hard; waits up to {cfg.load_timeout:g}s for a slot, else 503)"
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
        "  session unload: immediate when no agent is registered"
        + (f" (heartbeat TTL {cfg.session_ttl:g}s)" if cfg.session_ttl else " (release only)"),
        file=sys.stderr,
    )
    if cfg.vision_model:
        print(
            f"  vision routing: image requests -> {cfg.vision_model} "
            "(routes any request that includes an image)",
            file=sys.stderr,
        )
    print(
        f'Point each agent.toml at base_url = "{public}" and set its own `model`. '
        "Agents only connect; models load on first request.",
        file=sys.stderr,
    )

    # ランタイム記録: 稼働中ゲートウェイの接続先を固定パスに残す。gateway.toml の無い
    # ディレクトリからでも `gw status` / `gw stop` がこの 1 ファイルで唯一のデーモンを見つける。
    write_gateway_runtime(cfg.host, cfg.port, server.pid, server.start_cwd, server.started_at)

    # 掃除スレッド: ①クラッシュした内部ワーカーの健全性チェック（常時）②idle TTL 超過モデルの
    # アンロード ③在席ハートビート途絶の掃除。健全性チェックは常に走らせる（死んだワーカーへ
    # 流し続けて 502 を返す事態を防ぐ）。チェック間隔は有効な閾値と健全性チェック周期の短い方。
    stop_reaper = threading.Event()
    _HEALTH_INTERVAL = 15.0  # 死んだワーカーの検知周期（idle/session が無効でもこの周期で回す）
    bounds = [t / 2 for t in (cfg.idle_timeout, cfg.session_ttl) if t]
    bounds.append(_HEALTH_INTERVAL)
    interval = min(max(min(bounds), 1.0), 30.0)  # チェック間隔（最大 30s）

    def _reaper() -> None:
        while not stop_reaper.wait(interval):
            try:
                dead = manager.reap_dead_instances()
                if dead:
                    print(
                        f"Health check: removed {dead} dead worker instance(s) "
                        "(crashed); the slot is free to reload on the next request.",
                        file=sys.stderr,
                    )
                if cfg.session_ttl:
                    gone = manager.reap_sessions(cfg.session_ttl)
                    if gone:
                        print(
                            f"Session unload: stopped {gone} model(s) "
                            "(agent heartbeat timed out).",
                            file=sys.stderr,
                        )
                if cfg.idle_timeout:
                    freed = manager.evict_idle(cfg.idle_timeout)
                    if freed:
                        print(f"Idle unload: stopped {freed} model(s).", file=sys.stderr)
            except Exception:  # noqa: BLE001 - 掃除スレッドは落とさない
                pass

    threading.Thread(target=_reaper, daemon=True).start()

    # ホットリロード監視: gateway.toml を保存した瞬間に、ポリシー設定を無停止で反映する。
    # 構造設定（host/port/internal_base_port/[[models]]）の変更は「要再起動」を警告するだけ。
    if config_path:
        threading.Thread(
            target=watch_config_file,
            args=(server, manager, cfg, config_path, stop_reaper),
            daemon=True,
        ).start()

    # 自動更新監視: PyPI 新版を検知し、作業ツリーがクリーンかつ処理中/在席が 0（idle）の
    # 瞬間に git pull で追従する。適用できたら restart_requested を立てて下のメインループを
    # 抜け、finally でクリーン停止 → run_gateway が execv で新コードに置き換える。
    # TUI 廃止に伴い、旧 TUI が担っていた「PyPI 新版を git で追従」をデーモン本体へ移した。
    # gateway.toml の auto_update=false で**適用**は無効化できるが、**チェックは常に行う**
    # ——Ollama と同じく、更新があることをトレイの更新マークで見せるため（適用はユーザーの
    # 「今すぐ更新」= /admin/update か `gw update` に任せる）。
    restart_requested = threading.Event()
    # 検知状態と再起動要求を HTTP ハンドラ（/admin/status・/admin/update）から使えるようにする。
    server.update_state = {"available": False, "current": None, "latest": None,
                           "fetched": False, "reason": None}
    server.request_restart = restart_requested.set
    # オンデマンド確認（/admin/status GET から）のスロットル用。0.0 = 未確認なので、
    # 最初のメニューオープンで即チェックが走る（起動直後から「更新の有無」が正しく出る）。
    server._last_update_check = 0.0
    server._update_check_inflight = False
    threading.Thread(
        target=_update_watcher,
        args=(manager, stop_reaper, restart_requested),
        kwargs={"auto_apply": cfg.auto_update, "state": server.update_state,
                "notify": _tray_notify},
        daemon=True,
    ).start()

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    restart = False
    try:
        # 割り込み（Ctrl+C / SIGTERM）または自動更新の再起動要求までブロックする。
        restart = restart_requested.wait()
    except KeyboardInterrupt:
        pass
    finally:
        # 後始末中に再度シグナル（停止時の killpg 等で連続して届く）が来ても中断されず、
        # 配下のモデルサーバーを必ず止め切るため、まず以降のシグナルを無視にする。
        ignore_shutdown_signals()
        stop_reaper.set()
        print("\nShutting down the gateway and its model servers...", file=sys.stderr)
        server.shutdown()
        server.server_close()
        manager.shutdown()
        # メニューバーアイコンを畳む（パイプ EOF でも消えるが、明示終了の方が即時）。
        if tray_proc is not None and tray_proc.poll() is None:
            tray_proc.terminate()
        # 正常停止のときだけランタイム記録を消す。自動更新の再起動（execv）では消さない
        # ——同じ pid/port で立ち直す新イメージが上書きするので、その隙に `gw status` が
        # 「ゲートウェイ無し」と誤認しないため。
        if not restart:
            clear_gateway_runtime()
    return _RESTART_CODE if restart else 0
