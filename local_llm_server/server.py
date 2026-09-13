from __future__ import annotations

import os
import platform  # noqa: F401 - compatibility seam for callers/tests
import subprocess
import sys
import tempfile  # noqa: F401 - compatibility seam used by runtime-dir callers/tests
import threading
import time
import urllib.request  # noqa: F401 - compatibility seam for health-query tests
from typing import TextIO

from . import gateway_runtime as _gateway_runtime
from . import model_catalog as _model_catalog
from . import process_control as _process_control
from . import server_health as _server_health
from .backend_core import (  # noqa: F401 - public compatibility exports
    DEFAULT_BACKEND,
    BackendSpec,
    ServerConfig,
    BACKEND_SPECS as _CORE_BACKEND_SPECS,
    default_backend,
    infer_backend,
    parallel_supported,
)
from .backend_runtime import (  # noqa: F401 - public compatibility exports
    _PROVISIONED,
    llama_provision_info,
    llama_server_binary,
    provisioned,
    set_llama_server_binary,
    set_provisioned,
    set_sglang_python,
    set_vllm_python,
    sglang_provision_info,
    sglang_python,
    vllm_provision_info,
    vllm_python,
)

# 起動可能なバックエンド一覧は同梱の constants から取得（OpenAI互換APIの公開値）。
from .constants import BACKENDS, log_dir  # noqa: F401
from .gateway_runtime import (  # noqa: F401 - compatibility re-exports
    _WORKERS_FILE_LOCK,
    LOG_ROTATION_COUNT,
    GatewayAlreadyRunning,
    GatewayLock,
    _admin_request,
    _atomic_write_json,
    _flock_exclusive_nb,
    _flock_unlock,
    _load_workers_unlocked,
    _read_lock_pid,
    _save_workers_unlocked,
    bench_model,
    clear_gateway_runtime,
    daemon_log_path,
    enable_child_tethering,
    gateway_admin_status,
    gateway_drain,
    gateway_lock_path,
    gateway_log_path,
    gateway_runtime_path,
    gateway_set_max_resident,
    ignore_shutdown_signals,
    local_connect_host,
    owned_worker_pids_on_ports,
    pid_is_alive,
    primary_lan_ip,
    prune_server_logs,
    read_gateway_runtime,
    reap_orphan_workers,
    register_worker,
    rotate_log,
    runtime_dir,
    server_status,
    unregister_worker,
    worker_pid_is_owned,
    workers_state_path,
    write_gateway_runtime,
)
from .model_catalog import (  # noqa: F401 - compatibility re-exports
    _DISCOVER_CACHE,
    _DRAFTER_REPOS,
    _EXTRA_DRAFTER_REPOS,
    MTP_DRAFTERS,
    _blocking_incomplete,
    _dir_weight_bytes,
    _hf_hub_cache,
    _is_generative_repo,
    _snapshot_weights_complete,
    looks_like_local_path,
    resolve_drafter,
    thinking_markers,
)


def start_gateway_background(
    cwd: str,
    host: str = "127.0.0.1",
    port: int = 8799,
    *,
    start_timeout: float = 120.0,
) -> int:
    """Compatibility facade that forwards replaceable runtime dependencies."""
    return _gateway_runtime.start_gateway_background(
        cwd,
        host,
        port,
        start_timeout=start_timeout,
        _find_pids=find_pids_on_port,
        _admin_status=gateway_admin_status,
        _connect_host=local_connect_host,
        _looks_like_gateway=pid_looks_like_gateway,
        _log_path=gateway_log_path,
        _rotate=rotate_log,
        _prune=prune_server_logs,
        _ready=is_ready,
    )


def warn(message: str) -> None:
    """運用者向けの警告を 1 行で出す（ゲートウェイの stderr → デーモンログ）。

    「起動は続けるが期待どおりではない」ことを伝えるための経路。握りつぶすと気づけず、
    例外にすると動くはずのものが止まる、という中間の事象に使う。
    """
    print(f"warning: {message}", file=sys.stderr, flush=True)


def _physical_cores() -> int:
    """物理コア数（ハイパースレッド/E コアを除く。取れなければ論理コア数）。"""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001 - psutil 不在・取得失敗はフォールバック
        pass
    return os.cpu_count() or 4


# 既にユーザーが extra_args で指定していれば自動付与しないフラグ群（等価表記も含む）。
_NGL_FLAGS = ("-ngl", "--n-gpu-layers", "--gpu-layers")
_THREAD_FLAGS = ("-t", "--threads")


def auto_llama_flags(config: "ServerConfig") -> list[str]:
    """自動導入した llama.cpp の accel に応じた計算効率フラグ（ユーザー未指定時のみ）。

    - GPU（accel が cpu 以外）: `-ngl 999`（全層 GPU オフロード。llama.cpp が実層数に丸める）。
    - CPU（accel == cpu）: `--threads <物理コア数>`（既定の論理コア数より CPU 推論で速いことが多い）。
    自動導入していない（プロビジョナ未実行＝素性不明）ときは何もしない。
    """
    info = llama_provision_info()
    if not info:
        return []
    accel = info.get("accel")
    if not accel:
        return []
    extra = config.extra_args
    if accel != "cpu":
        if not any(a in _NGL_FLAGS for a in extra):
            return ["-ngl", "999"]
        return []
    if not any(a in _THREAD_FLAGS for a in extra):
        return ["--threads", str(_physical_cores())]
    return []


# Process policy has one canonical implementation.  The names remain exported
# here so existing callers keep the same API while gateway layers import the
# dependency-free module directly.
_POSIX = _process_control.POSIX
install_shutdown_handlers = _process_control.install_shutdown_handlers
_signal_process_tree = _process_control.signal_process_tree
_stop_process_tree = _process_control.stop_process_tree
find_pids_on_port = _process_control.find_pids_on_port
pid_looks_like_ours = _process_control.pid_looks_like_ours
pid_looks_like_gateway = _process_control.pid_looks_like_gateway
_process_fingerprint = _process_control.process_fingerprint
pid_matches_record = _process_control.pid_matches_record
stop_pid = _process_control.stop_pid
_stop_pid_windows = _process_control._stop_pid_windows


def reclaim_stale_workers(port: int, timeout: float = 6.0) -> list[int]:
    """port を LISTEN している「このパッケージ由来の」孤児ワーカーを止めて回収する。

    ゲートウェイがワーカーを起動する直前に呼ぶ。前回のクラッシュや `kill -9` で取り残された
    モデルサーバー（台帳指紋と正規コマンドが一致）がそのポートを掴んでいると、新しいワーカーが
    bind できず起動失敗→502 になり、加えて GPU メモリを無駄に占有し続ける。ここで止めてから
    起動することで衝突を防ぎメモリを解放する。**無関係な別プロセスには手を出さない**
    （`pid_looks_like_ours` で選別し、判定不能なものは残す）。停止した PID の一覧を返す。
    """
    with _WORKERS_FILE_LOCK:
        records = [e for e in _load_workers_unlocked() if e.get("port") == port]
    reclaimed: list[int] = []
    for pid in find_pids_on_port(port):
        # 自分自身（ゲートウェイ本体）は絶対に殺さない。内部ワーカーは別プロセスなので、
        # ここに現れる our-worker は孤児だけ。万一 self が現れても手を出さない安全弁。
        if pid == os.getpid():
            continue
        record = next((e for e in records if e.get("pid") == pid), None)
        if (
            record is not None
            and pid_matches_record(pid, record)
            and pid_looks_like_ours(pid)
            and stop_pid(pid, timeout=timeout)
        ):
            reclaimed.append(pid)
            unregister_worker(pid)
    return reclaimed


def resolve_gguf(model: str) -> str:
    """互換用ファサード。キャッシュ探索の実装は model_catalog に置く。"""
    return _model_catalog.resolve_gguf(model, cache_root=_hf_hub_cache())


def ensure_cached(repo: str, *, what: str = "モデル") -> str:
    """互換用ファサード。従来どおり server 側のキャッシュ設定を尊重する。"""
    return _model_catalog.ensure_cached(repo, what=what, cache_root=_hf_hub_cache())


def mtp_status(model: str, drafter: str | None = None) -> str | None:
    """互換用ファサード。"""
    return _model_catalog.mtp_status(model, drafter, cache_root=_hf_hub_cache())


def discover_cached_models(ttl: float = 10.0) -> list[dict]:
    """互換用ファサード。"""
    return _model_catalog.discover_cached_models(ttl, cache_root=_hf_hub_cache())


def estimate_model_bytes(config: ServerConfig) -> int | None:
    """常駐に要するメモリの**概算**（バイト）。重みファイルのサイズを基準にする。

    - llama-cpp: 本体 GGUF（＋自動付与される mmproj、＋ドラフト GGUF）のファイルサイズ合計。
    - mlx / mlx-vlm: HF キャッシュのスナップショットに在るファイルサイズ合計（blob 実体で重複排除）。

    取得できない（未キャッシュ・未DL 等）ときは `None` を返す（メモリガードはそのモデルを
    スキップする）。KVキャッシュ・ランタイムバッファは含まない**下限寄りの見積もり**なので、
    呼び出し側で余裕係数を掛ける前提（→ docs/llama-cpp.md のメモリガード）。
    """
    try:
        if backend_spec(config.backend).gguf:
            path = resolve_gguf(config.model)
            total = os.path.getsize(path)
            mmproj = find_sibling_mmproj(path)
            if mmproj and not any(a in ("--no-mmproj",) for a in config.extra_args):
                total += os.path.getsize(mmproj)
            if config.draft_model:
                try:
                    total += os.path.getsize(resolve_gguf(config.draft_model))
                except (ValueError, OSError):
                    pass  # ドラフトが解決できなくても本体分は数える
            return total
        # mlx / mlx-vlm: ローカルパス指定なら、そのディレクトリの重みサイズ合計。
        # repo-id ではないので下の HF キャッシュ探索には載らず、そのままだと見積もり不能
        # （＝メモリガードがそのモデルを素通しする）。巨大なモデルをローカル変換物として
        # 登録する運用（→ gateway.toml の Inkling / DeepSeek）では素通しは危険なので数える。
        # ドラフター（MTP）も常駐するので合算する。
        spec = config.model.strip()
        if looks_like_local_path(spec):
            total = _dir_weight_bytes(os.path.expanduser(spec))
            if config.draft_model and looks_like_local_path(config.draft_model):
                total += _dir_weight_bytes(
                    os.path.expanduser(config.draft_model.strip())
                )
            return total or None
        # HF repo-id: models--org--name/snapshots/<hash>/ の合計（blob 実体で重複排除）
        repo = spec
        if repo.count("/") != 1:
            return None
        org, name = repo.split("/", 1)
        snap_root = os.path.join(_hf_hub_cache(), f"models--{org}--{name}", "snapshots")
        if not os.path.isdir(snap_root):
            return None  # 未DL（mlx はロード時に自動取得）→ 見積もり不能
        by_blob: dict[str, int] = {}
        for root, _dirs, files in os.walk(snap_root):
            for f in files:
                p = os.path.join(root, f)
                try:
                    by_blob.setdefault(os.path.realpath(p), os.path.getsize(p))
                except OSError:
                    pass
        return sum(by_blob.values()) or None
    except (ValueError, OSError):
        return None


def find_sibling_mmproj(model_path: str) -> str | None:
    """GGUF 本体と同じディレクトリにある vision projector（mmproj）を探す。

    llama.cpp のマルチモーダルは本体 GGUF とは別に mmproj(.gguf) が要る。HF の GGUF
    リポジトリは慣例的に `mmproj-*.gguf` / `*mmproj*.gguf` を本体と同梱するため、本体の
    隣を探して見つかれば自動で `--mmproj` に渡す（テキストのみ入力でも速度・精度に影響は
    無く、画像が来たときだけ使われる）。本体がローカルファイルでない／隣に無ければ None。
    """
    directory = os.path.dirname(model_path)
    if not directory or not os.path.isdir(directory):
        return None
    candidates = sorted(
        name
        for name in os.listdir(directory)
        if "mmproj" in name.lower() and name.lower().endswith(".gguf")
    )
    if not candidates:
        return None
    return os.path.abspath(os.path.join(directory, candidates[0]))


def _build_mlx(config: ServerConfig) -> list[str]:
    """mlx_lm.server（テキスト専用。逐次処理。並列スロットの概念なし）。"""
    # 自動ダウンロードは行わない。事前に `hf download` 済みであることを起動前に確認する
    # （未取得ならここで案内付き ValueError）。
    ensure_cached(config.model)
    command = [
        "mlx_lm.server",
        "--model",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    if config.disable_thinking:
        command += ["--chat-template-args", '{"enable_thinking": false}']
    return command


def _build_mlx_vlm(config: ServerConfig) -> list[str]:
    """mlx_vlm.server（vision 対応。画像入力 image_url を受けられる）。

    コンソールスクリプトが無い環境もあるため `python -m` で確実に起動する。逐次処理で
    並列スロットの概念はないため --parallel は渡さない。thinking はサーバー既定が OFF
    （--enable-thinking を渡さなければ無効）であり、リクエスト毎の明示制御は
    クライアントがトップレベル enable_thinking で行う（llm.py 参照）。
    """
    # 自動ダウンロードは行わない。本体は事前に `hf download` 済みであることを確認する。
    ensure_cached(config.model)
    # 直接 mlx_vlm.server を起動せず、シム経由で起動する（_mlx_vlm_shims が上流の
    # 未修正部分を当ててから同じ引数で mlx_vlm.server を __main__ 実行する）。
    # site-packages を書き換えると手動更新の `uv sync` で消えるため。
    command = [
        sys.executable,
        "-m",
        "local_llm_server._mlx_vlm_shims",
        "--model",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    # Gemma 4 の MTP ドラフターによる speculative decoding。draft_kind は mtp に固定する
    # （他種別＝dflash / eagle3 は今回は対象外）。draft_model="auto" は本体名から
    # 対応ドラフターを自動選択する。自動ダウンロードはしない（事前 DL 必須）。
    #
    # ただしドラフターが未取得でも**本体の起動は止めない**。MTP は高速化の手段でしかなく、
    # 無くても本体は完全に動くので、ここで中止すると「速くならない」で済むはずの状況が
    # 「モデルが使えない」に化ける。対応表を更新して配ると、まだドラフターを取得していない
    # PC が全滅する——これは実際に 0.38.5 で起きうる（#55 の切り替え）。
    # 警告を出して MTP 無しで起動し、`gw pull <drafter>` で後から有効化できるようにする。
    # 取得状況は `gw list` / `gw mtp` の MTP 列（mtp_status）でいつでも確認できる。
    drafter = resolve_drafter(config.model, config.draft_model)
    if drafter:
        try:
            ensure_cached(drafter, what="ドラフター")
        except ValueError as exc:
            warn(
                f"MTP ドラフター {drafter!r} が未取得のため、{config.model!r} を "
                f"MTP 無効で起動します（本体は通常どおり動作します）。"
                f" 有効化するには `gw pull {drafter}`。詳細: {exc}"
            )
            drafter = None
    if drafter:
        command += ["--draft-model", drafter, "--draft-kind", "mtp"]
    return command


def _build_llama_cpp(config: ServerConfig) -> list[str]:
    """llama-server（--parallel で並列スロットを確保）。"""
    # model は HF repo-id（org/repo[:selector]）。DL 済みキャッシュから実 GGUF を解決する
    # （キャッシュに無ければ ValueError。クライアントに見せる ID は repo-id のまま）。
    model_path = resolve_gguf(config.model)
    command = [
        llama_server_binary(),  # プロビジョナが導入した絶対パス、無ければ PATH の "llama-server"
        "-m",
        model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    # 埋め込み MTP（Qwen3.6 等、本体 GGUF に MTP ヘッドが内蔵）。draft_model="self"/"mtp"
    # で有効化＝別ドラフトファイル不要（--spec-type draft-mtp のみ）。この方式は llama.cpp 側で
    # --mmproj（vision）と --parallel>1 が未対応なので、両者は付けない（付けると起動失敗する）。
    draft_model = config.draft_model
    embedded_mtp = (
        bool(draft_model)
        and draft_model is not None
        and draft_model.strip().lower() in ("self", "mtp")
    )
    if config.parallel is not None and not embedded_mtp:
        command += ["--parallel", str(config.parallel)]
    if config.disable_thinking:
        command += ["--chat-template-kwargs", '{"enable_thinking": false}']
    if embedded_mtp:
        command += ["--spec-type", "draft-mtp"]
    else:
        # マルチモーダル: 本体の隣に mmproj があれば自動で渡す（手動設定不要）。
        # ユーザーが extra_args で明示制御していれば（--mmproj / --no-mmproj）尊重する。
        if not any(a in ("--mmproj", "-mm", "--no-mmproj") for a in config.extra_args):
            mmproj = find_sibling_mmproj(model_path)
            if mmproj:
                command += ["--mmproj", mmproj]
        # 別ヘッド方式の speculative decoding（gemma4 等）。draft_model に MTP ヘッドの
        # HF repo-id（org/repo:F16-MTP 等）を指定すると有効化（-md）。ファイル名に
        # "mtp" を含めば MTP ヘッドとみなし --spec-type draft-mtp を付ける（それ以外は
        # llama.cpp 既定の draft-simple）。
        # mlx-vlm 側と同じく、ドラフトが未取得でも本体の起動は止めない（警告のみ）。
        if config.draft_model and "-md" not in config.extra_args:
            try:
                draft_path = resolve_gguf(config.draft_model)
            except ValueError as exc:
                warn(
                    f"ドラフトモデル {config.draft_model!r} を解決できないため、"
                    f"{config.model!r} を speculative decoding 無効で起動します"
                    f"（本体は通常どおり動作します）。詳細: {exc}"
                )
                draft_path = None
            if draft_path:
                command += ["-md", draft_path]
                if "mtp" in os.path.basename(draft_path).lower():
                    command += ["--spec-type", "draft-mtp"]
    # 計算効率の自動チューニング（自動導入バイナリの accel に合わせる。extra_args 優先）。
    return command + auto_llama_flags(config)


def _build_whisper(config: ServerConfig) -> list[str]:
    """mlx-whisper を OpenAI 互換の STT サーバ（1 モデル 1 プロセス）として起動する。

    専用サーバは同梱の local_llm_server.stt_server（標準ライブラリのみ）。
    本体は事前 DL 必須（未取得なら案内付き ValueError）。音声デコードに ffmpeg CLI が要る。
    """
    ensure_cached(config.model)
    return [
        sys.executable,
        "-m",
        "local_llm_server.stt_server",
        "--model",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]


def _build_vllm(config: ServerConfig) -> list[str]:
    """vLLM の OpenAI 互換 API サーバを、隔離 venv の python から起動する（→ vllm_provisioner）。

    model は HF repo-id。事前 DL 済みを前提に確認する（未取得は案内付き ValueError）。
    クライアントに見せる id を repo-id に固定するため --served-model-name も同じ id にする。
    逐次でなく連続バッチングで多人数同時に強い（並列は vLLM が内部で捌く）。
    """
    ensure_cached(config.model)
    return [
        vllm_python(),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model,
        "--served-model-name",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]


def _build_sglang(config: ServerConfig) -> list[str]:
    """SGLang の OpenAI 互換 API サーバを、隔離 venv の python から起動する（→ sglang_provisioner）。

    SGLang は引数が vLLM と違い、モデルは --model-path で渡す。RadixAttention で
    共有プレフィックス（システムプロンプト/ツール定義）の多い用途に強い。
    """
    ensure_cached(config.model)
    return [
        sglang_python(),
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model,
        "--served-model-name",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]


_COMMAND_BUILDERS = {
    "mlx": _build_mlx,
    "mlx-vlm": _build_mlx_vlm,
    "llama-cpp": _build_llama_cpp,
    "whisper": _build_whisper,
    "vllm": _build_vllm,
    "sglang": _build_sglang,
}

# Public compatibility descriptors retain their historical ``build`` strategy,
# while lower gateway layers consume dependency-free metadata from backend_core.
BACKEND_SPECS: dict[str, BackendSpec] = {
    name: BackendSpec(
        name=spec.name,
        build=_COMMAND_BUILDERS[name],
        draft_style=spec.draft_style,
        parallel=spec.parallel,
        gguf=spec.gguf,
        provisioner=spec.provisioner,
    )
    for name, spec in _CORE_BACKEND_SPECS.items()
}


def backend_spec(name: str) -> BackendSpec:
    spec = BACKEND_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown backend: {name!r} (choose from {BACKENDS})")
    return spec


def build_command(config: ServerConfig) -> list[str]:
    """バックエンドに応じた起動コマンドを組み立てる（実体は BACKEND_SPECS の build）。

    いずれも OpenAI 互換サーバーを立ち上げる。extra_args は全バックエンド共通で末尾に付く
    （ユーザーの明示指定が自動付与より後＝優先になる）。
    """
    builder = backend_spec(config.backend).build
    assert builder is not None
    return builder(config) + config.extra_args


# Health queries are likewise implemented once and re-exported for compatibility.
is_ready = _server_health.is_ready
list_models = _server_health.list_models
running_model = _server_health.running_model
model_available = _server_health.model_available
models_match = _server_health.models_match
parse_host_port = _server_health.parse_host_port


class LocalServer:
    """ローカルLLMサーバーをサブプロセスとして管理する。"""

    def __init__(self, config: ServerConfig, log_path: str) -> None:
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._log_file: TextIO | None = None
        self._lifecycle = threading.Condition()
        self._stop_lock = threading.Lock()
        self._starting = False
        self._stop_requested = False
        # サーバーの大量ログ（INFO/Stream finished 等）で対話画面が乱れないよう、
        # 標準出力・標準エラーはこのログファイルへ逃がす（端末には流さない）。
        # 必須引数——省略時に一時ファイルへ逃がすフォールバックは、誰にも掃除されない
        # ゴミを temp に落とすだけなので廃止した（実運用は常に daemon_log_path を渡す）。
        self.log_path = log_path

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def pid(self) -> int | None:
        """起動済みワーカーの PID（未起動・停止後は None）。"""
        return self._proc.pid if self._proc is not None else None

    def is_alive(self) -> bool:
        """ワーカーのサブプロセスが起動済みでまだ生きているか（poll が None）。

        クラッシュ等で落ちていれば False。ゲートウェイの健全性チェックが、ready と信じている
        インスタンスの内部ワーカーが実際に生存しているかを確認するのに使う（安価な poll のみ）。
        """
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        with self._lifecycle:
            if self._proc is not None or self._starting:
                raise RuntimeError("server already started")
            self._starting = True
            self._stop_requested = False
        cancelled = False
        try:
            # ログパス（ゲートウェイの daemon_log_path 等）は親ディレクトリが
            # 無いことがあるので作る（log_dir は呼び出し側が作る設計）。
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._log_file = open(self.log_path, "a", encoding="utf-8")
            # 自動ダウンロードを完全に無効化する hard guard。バックエンド（mlx_lm / mlx_vlm /
            # transformers / tokenizers / ドラフター）はキャッシュのみを参照し、未取得ファイルが
            # あればその場でエラーになる（ネットワークへ取りに行かない＝DL 停滞も起きない）。
            # build_command 側の ensure_cached 事前チェックと二重で「事前 DL 必須」を担保する。
            env = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            # 思考チャネルの分離を明示設定する（mlx-vlm 経路）。mlx-vlm サーバは
            # reasoning を content から切り出して reasoning_content へ回すが、その開始/終了
            # マーカーはモデルごとに違う。gemma-4-A4B 系は「開 <|channel>thought / 閉
            # <channel|>」を使う（他モデルの慣習 <think>…</think> とは別形式）。mlx-vlm
            # 0.6.3 の既定マーカーには両形式が含まれるが、内部既定に依存せず将来のバージョン
            # でも確実に分離させるため env で明示する（未設定時のみ。ユーザー上書きは尊重）。
            # 設定したペアが最優先で試され、既定（<think> 等）も後段で効くので副作用はない。
            if self.config.backend == "mlx-vlm":
                start_marker, end_marker = thinking_markers(self.config.model)
                env.setdefault("MLX_VLM_THINKING_START_TOKEN", start_marker)
                env.setdefault("MLX_VLM_THINKING_END_TOKEN", end_marker)
                # プロンプトキャッシュ（APC）を既定で有効にする。mlx-vlm は
                # APC_ENABLED が無いとキャッシュ管理そのものを作らない
                # （apc.from_env が None を返す）ので、既定のままだと毎回フルに
                # プリフィルしていた。実測: 同一プロンプトの再送で 1623ms → 60ms。
                # メモリ増は観測されず（26B-4bit で RSS 16.1GB のまま）、外して
                # おく理由が無いため既定を有効側にする。ブロック数などの調整は
                # APC_* を環境変数で渡す（ここは未設定時のみ＝ユーザー指定が優先）。
                env.setdefault("APC_ENABLED", "1")
                # ツール呼び出しの生成中トークンを流す(シム側が読む。→ ServerConfig.stream_tool_calls)
                if self.config.stream_tool_calls:
                    env["LOCAL_LLM_STREAM_TOOL_CALLS"] = "1"
                # exact モード（gemma4 系などハイブリッド注意機構のモデルが該当。
                # RotatingKVCache 層はブロック分割で再構成できないため、mlx-vlm は
                # プレフィックス丸ごとのスナップショット方式に落とす）向けの既定。
                #
                # guard: スナップショットを「プロンプト末尾から何トークン手前」で
                # 切るか。上流既定の 16 は同一プロンプト再送にしか効かない。
                # エージェント用途のプロンプトは「共通の前方（system+履歴）+
                # 毎回変わる末尾（状態通知・ユーザー発言）」の形で、切れ目が共通部分に
                # 入るまで広げて初めて前方一致が働く（実測: guard=512 で別末尾の
                # プリフィル 1901〜3110ms → 409〜524ms）。1024 は実キャラクター
                # アプリの動的な末尾（数百トークン）を覆う値。
                # entries: スナップショットの保持数。上流既定の 2 は 1 ターンに複数回
                # LLM を呼ぶエージェントで玉突き追い出しを起こす。8 でも RSS 増は
                # 観測されなかった（26B-4bit で 16.1GB のまま）。
                env.setdefault("APC_EXACT_PREFIX_GUARD_TOKENS", "1024")
                env.setdefault("APC_EXACT_CACHE_ENTRIES", "8")
            cmd = build_command(self.config)
            extra: dict = {}
            # 繋留が有効（デーモン内）なら、ワーカーを tether ラッパー越しに起動する。
            # ラッパーはデーモンの死（パイプ EOF）を検知して自分のグループごと終了する
            # ので、デーモンが kill -9 で死んでもモデルサーバーが孤児として残らない。
            if _POSIX and _gateway_runtime._TETHER_READ_FD is not None:
                cmd = [
                    sys.executable,
                    "-m",
                    "local_llm_server.tether",
                    "--fd",
                    str(_gateway_runtime._TETHER_READ_FD),
                    "--",
                    *cmd,
                ]
                extra["pass_fds"] = (_gateway_runtime._TETHER_READ_FD,)
            self._proc = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                # 子を独立したプロセスグループにして、停止時に孫まで一括終了できるようにする
                # （POSIX のみ。Windows では無視される）。stop() の killpg と対になる。
                start_new_session=_POSIX,
                **extra,
            )
            # ワーカー台帳へ記録（デーモンごと死んだときの、次回起動時の孤児掃除の手掛かり）。
            register_worker(self._proc.pid, self.config.port, self.config.model)
        except FileNotFoundError as exc:
            self._close_log()
            raise RuntimeError(
                f"バックエンド実行ファイルが見つかりません: {exc.filename}。"
                " mlx_lm / mlx_vlm / llama.cpp がインストール・PATH 上にあるか確認してください。"
            ) from exc
        except Exception:
            # build_command/Popen/台帳更新のどこで失敗しても、開いたログと起動済みの子を残さない。
            with self._lifecycle:
                self._starting = False
                self._lifecycle.notify_all()
            try:
                self.stop(grace=0.0)
            except Exception as cleanup_exc:  # noqa: BLE001 - 元の起動例外を保持する
                warn(f"failed to clean up a partially started worker: {cleanup_exc}")
            raise
        finally:
            with self._lifecycle:
                if self._starting:
                    self._starting = False
                    cancelled = self._stop_requested
                    self._lifecycle.notify_all()
        if cancelled:
            # A concurrent shutdown waited for the spawn transaction to finish.
            # Participate in cleanup as well; _stop_lock makes duplicate callers
            # harmless and start() never reports success after cancellation.
            self.stop(grace=0.0)
            raise RuntimeError("server start cancelled by shutdown")

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def wait_until_ready(self, timeout: float = 120.0, interval: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {self._proc.returncode})"
                )
            if is_ready(self.config.base_url):
                return
            time.sleep(interval)
        raise TimeoutError(
            f"server not ready within {timeout}s at {self.config.base_url}"
        )

    def wait(self) -> int:
        """サーバープロセスが終了するまでブロックする。"""
        if self._proc is None:
            raise RuntimeError("server not started")
        return self._proc.wait()

    def stop(self, grace: float = 10.0) -> None:
        """モデルサーバー（とプロセスグループ）を止める。

        grace > 0 は graceful: プロセスグループ全体へ SIGTERM を送り、grace 秒だけ自発終了を
        待ってから SIGKILL でとどめを刺す（LRU 退避など、単体を丁寧に止めたいとき用）。
        grace <= 0 は最初から SIGKILL する。ゲートウェイの全体終了時はこれを使う —— どうせ
        全プロセスを畳むので graceful は不要で、mlx/Metal の終了時クリーンアップ（数秒かかる
        ことがある）を待たずカーネルに即回収させた方が、TUI の quit が目に見えて速くなる。
        """
        with self._lifecycle:
            if self._starting:
                self._stop_requested = True
                while self._starting:
                    self._lifecycle.wait()
        with self._stop_lock:
            proc = self._proc
            if proc is None:
                self._close_log()
                return
            try:
                stopped = _stop_process_tree(
                    proc,
                    grace=max(grace, 0.0),
                    kill_timeout=5.0,
                )
            except Exception as exc:  # noqa: BLE001 - preserve ownership for retry
                stopped = False
                warn(f"failed to stop worker pid {proc.pid}: {exc}")
            finally:
                # The child inherited its own fd. Closing the parent's copy is safe
                # even when termination failed and prevents a descriptor leak here.
                self._close_log()
            if stopped:
                with self._lifecycle:
                    if self._proc is proc:
                        self._proc = None
                unregister_worker(proc.pid)
            else:
                warn(
                    f"worker pid {proc.pid} did not exit after termination; "
                    "keeping its ownership ledger entry for startup reconciliation"
                )

    def __enter__(self) -> "LocalServer":
        self.start()
        self.wait_until_ready()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
