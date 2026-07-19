from __future__ import annotations

import glob
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

# 起動可能なバックエンド一覧は同梱の constants から取得（OpenAI互換APIの公開値）。
from .constants import BACKENDS, project_cache_dir  # noqa: F401


def default_backend() -> str:
    """OS に応じた既定バックエンド。

    Apple Silicon の macOS なら mlx-vlm（vision 対応）を既定にする。既定モデルの
    Qwen3.6 はマルチモーダルなので、1プロセスでテキストも画像も扱え、画像・動画
    入力がそのまま動く。テキスト専用で軽くしたい場合は backend="mlx" を選ぶ。
    それ以外（Linux / Windows / Intel Mac）は llama.cpp を既定にする。
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx-vlm"
    return "llama-cpp"


# サーバー未起動・バックエンド未指定のときに使う既定バックエンド（OSで自動判定）
DEFAULT_BACKEND = default_backend()


# llama-server 実行ファイルのパス。ゲートウェイ起動時にプロビジョナ（provisioner）が解決して
# set_llama_server_binary() で差し込む。未設定なら PATH の "llama-server"（従来挙動 / system）。
_LLAMA_SERVER_BIN: str | None = None
# 導入した llama.cpp の素性（build/accel/binary/provision）。/admin/status・TUI 表示用。
_LLAMA_INFO: dict | None = None


def set_llama_server_binary(
    path: str | None, *, build: str | None = None, accel: str | None = None,
    provision: str | None = None,
) -> None:
    """起動時にプロビジョナが解決した llama-server の絶対パス（と素性）を登録する。"""
    global _LLAMA_SERVER_BIN, _LLAMA_INFO
    _LLAMA_SERVER_BIN = path
    _LLAMA_INFO = None if path is None else {
        "binary": path, "build": build, "accel": accel, "provision": provision,
    }


def llama_server_binary() -> str:
    """build_command が使う llama-server コマンド（未プロビジョン時は PATH 探索の名前）。"""
    return _LLAMA_SERVER_BIN or "llama-server"


def llama_provision_info() -> dict | None:
    """導入済み llama.cpp の素性（未導入は None）。/admin/status・TUI が表示に使う。"""
    return _LLAMA_INFO


# vLLM を起動する python（隔離 venv）のパス。起動時に vllm_provisioner が解決して差し込む。
# 未設定なら sys.executable（provision=system 相当）。build_command の vllm 分岐が使う。
_VLLM_PYTHON: str | None = None
_VLLM_INFO: dict | None = None


def set_vllm_python(path: str | None, *, provision: str | None = None) -> None:
    """起動時にプロビジョナが解決した vLLM 用 python のパス（と素性）を登録する。"""
    global _VLLM_PYTHON, _VLLM_INFO
    _VLLM_PYTHON = path
    _VLLM_INFO = None if path is None else {"python": path, "provision": provision}


def vllm_python() -> str:
    """build_command が使う vLLM 用 python（未プロビジョン時は現在の python）。"""
    return _VLLM_PYTHON or sys.executable


def vllm_provision_info() -> dict | None:
    """導入済み vLLM の素性（未導入は None）。/admin/status・TUI が表示に使う。"""
    return _VLLM_INFO


# SGLang を起動する python（隔離 venv）のパス。起動時に sglang_provisioner が解決して差し込む。
_SGLANG_PYTHON: str | None = None
_SGLANG_INFO: dict | None = None


def set_sglang_python(path: str | None, *, provision: str | None = None) -> None:
    """起動時にプロビジョナが解決した SGLang 用 python のパス（と素性）を登録する。"""
    global _SGLANG_PYTHON, _SGLANG_INFO
    _SGLANG_PYTHON = path
    _SGLANG_INFO = None if path is None else {"python": path, "provision": provision}


def sglang_python() -> str:
    """build_command が使う SGLang 用 python（未プロビジョン時は現在の python）。"""
    return _SGLANG_PYTHON or sys.executable


def sglang_provision_info() -> dict | None:
    """導入済み SGLang の素性（未導入は None）。/admin/status・TUI が表示に使う。"""
    return _SGLANG_INFO


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
    自動導入していない（provision=system 等で素性不明）ときは何もしない
    （利用者が自分でフラグ管理している前提。既存挙動を壊さない）。
    """
    info = llama_provision_info()
    if not info:
        return []
    accel = info.get("accel")
    if not accel:
        # provision=system（素性不明のユーザー管理バイナリ）は accel=None → 何も足さない。
        return []
    extra = config.extra_args
    if accel != "cpu":
        if not any(a in _NGL_FLAGS for a in extra):
            return ["-ngl", "999"]
        return []
    if not any(a in _THREAD_FLAGS for a in extra):
        return ["--threads", str(_physical_cores())]
    return []


# STT（音声→テキスト）モデルの id 判定に使う語。whisper 系は id に "mlx" を含む
# （例 mlx-community/whisper-large-v3-mlx）ため、mlx-vlm 判定より先に見る必要がある。
_STT_HINTS = ("whisper", "parakeet")


def infer_backend(model: str) -> str:
    """登録の無いモデル ID からバックエンドを推論する（動的ロード用）。

    - STT（id に 'whisper'/'parakeet' を含む）→ whisper（音声→テキスト）
    - GGUF（id に 'gguf' を含む）→ llama-cpp
    - mlx（id に 'mlx' を含む。例 `mlx-community/...`・`*-MLX-*`）→ mlx-vlm（vision 兼テキスト）
    - それ以外 → OS 既定（Apple Silicon: mlx-vlm / 他: llama-cpp）
    """
    low = model.lower()
    # whisper 系は "...-mlx" を含むので、mlx-vlm より先に STT へ振り分ける。
    if any(h in low for h in _STT_HINTS):
        return "whisper"
    if "gguf" in low:
        return "llama-cpp"
    if "mlx" in low:
        return "mlx-vlm"
    return default_backend()

# POSIX（macOS / Linux）か。プロセスグループ操作（killpg / setsid）の可否に使う。
_POSIX = os.name == "posix"


def install_shutdown_handlers() -> None:
    """SIGTERM / SIGHUP を Ctrl+C と同じ KeyboardInterrupt に変換する。

    既定では Python は SIGTERM を受け取ると finally を実行せずに即終了するため、
    `kill <pid>` やターミナルを閉じた（SIGHUP）ときに、自動起動した LLM サーバーが
    孫プロセスとして置き去りになる。これらのシグナルを KeyboardInterrupt として
    送出することで、各エントリポイントの既存 try/finally（= server.stop()）を必ず通す。
    シグナルハンドラはメインスレッドからのみ登録できる（それ以外では黙って無視）。
    """
    def _raise_keyboard_interrupt(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)  # SIGHUP は Windows に無い
        if sig is None:
            continue
        try:
            signal.signal(sig, _raise_keyboard_interrupt)
        except (ValueError, OSError):
            pass  # メインスレッド以外などでは登録できない


def _signal_process_tree(proc: subprocess.Popen, *, kill: bool) -> None:
    """proc とその子孫（プロセスグループ）へ終了シグナルを送る。

    POSIX では start() が start_new_session=True で子を独立したプロセスグループに
    しているため、killpg でグループ全体へ一括送信できる。これによりバックエンドが
    内部で起こすワーカー等の孫プロセスも取りこぼさない。Windows には killpg が無い
    ので proc 自身を terminate / kill する。
    """
    if proc.poll() is not None:
        return  # 既に終了している
    if _POSIX:
        sig = signal.SIGKILL if kill else signal.SIGTERM
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # グループ送信に失敗したら単体送信にフォールバック
    if kill:
        proc.kill()
    else:
        proc.terminate()


def find_pids_on_port(port: int) -> list[int]:
    """指定ポートを LISTEN しているプロセスの PID 一覧を返す（macOS / Linux / Windows）。

    POSIX は lsof、Windows は netstat を使う。該当ツールが無い等で特定できなければ
    空リストを返す（呼び出し側で案内する）。
    """
    if _POSIX:
        return _find_pids_lsof(port)
    if os.name == "nt":
        return _find_pids_netstat(port)
    return []


def _find_pids_lsof(port: int) -> list[int]:
    """lsof で LISTEN 中の PID を引く（macOS / Linux）。"""
    try:
        # プロトコルとポートは 1 つの -i セレクタにまとめる。-iTCP と -i:PORT を
        # 分けると lsof は両者を OR 解釈し、全 TCP プロセスにマッチしてしまう。
        result = subprocess.run(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for token in result.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            pass
    return pids


def _find_pids_netstat(port: int) -> list[int]:
    """netstat -ano で LISTENING 中の PID を引く（Windows）。

    出力の各行は「Proto  ローカルアドレス  外部アドレス  状態  PID」。ローカルアドレスの
    ポート（末尾 `:<port>`）が一致し、状態が LISTENING の行から PID 列を集める
    （0.0.0.0 / [::] / 127.0.0.1 などホスト表記の違いはポート一致で吸収する）。
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local, state, pid = parts[1], parts[3], parts[4]
        if state.upper() != "LISTENING":
            continue
        if local.rsplit(":", 1)[-1] != str(port):
            continue
        try:
            p = int(pid)
        except ValueError:
            continue
        if p and p not in pids:
            pids.append(p)
    return pids


# このパッケージが起動するプロセスのコマンドラインに現れる目印。TUI の停止/終了処理が
# 「ポートを LISTEN しているだけの無関係なプロセス」を巻き添えにしないための判定に使う。
_OUR_CMD_MARKERS = (
    "local_llm_server", "local-llm-server", "llama-server", "mlx_lm", "mlx_vlm",
)


def pid_looks_like_ours(pid: int) -> bool:
    """PID がこのパッケージ由来（ゲートウェイ / モデルサーバー）のプロセスに見えるか。

    ポート番号だけを頼りに stop すると、たまたま同じポートで動いている別プロジェクトの
    サーバーを殺してしまう。コマンドラインに目印が含まれるものだけ「ours」と判定する。
    判定不能（プロセス消滅・権限なし・psutil 不在）は False（手を出さない）。
    """
    try:
        import psutil
        cmd = " ".join(psutil.Process(pid).cmdline())
    except Exception:  # noqa: BLE001 - 判定できないものは殺さない側に倒す
        return False
    return any(m in cmd for m in _OUR_CMD_MARKERS)


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    """PID（とその子/プロセスグループ）を停止する（macOS / Linux / Windows）。

    POSIX はプロセスグループへ SIGTERM→（猶予後）SIGKILL、Windows は taskkill /T /F で
    ツリーごと止める。停止を試みたら True、対象が既にいなければ False。
    """
    if os.name == "nt":
        return _stop_pid_windows(pid)
    if not _POSIX:
        return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False

    def _alive() -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False  # 権限が無い（他ユーザーの）プロセスには手を出さない
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive():
            return True
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)  # 猶予内に終わらなければ強制終了
    except (ProcessLookupError, PermissionError):
        pass
    return True


def _stop_pid_windows(pid: int) -> bool:
    """taskkill でプロセスツリー（/T）を強制終了する（Windows）。

    モデルサーバーはゲートウェイの子プロセスなので /T で一緒に止まる。対象が既に
    いない / taskkill が無いときは False。
    """
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def reclaim_stale_workers(port: int, timeout: float = 6.0) -> list[int]:
    """port を LISTEN している「このパッケージ由来の」孤児ワーカーを止めて回収する。

    ゲートウェイがワーカーを起動する直前に呼ぶ。前回のクラッシュや `kill -9` で取り残された
    モデルサーバー（`_OUR_CMD_MARKERS` にマッチ）がそのポートを掴んでいると、新しいワーカーが
    bind できず起動失敗→502 になり、加えて GPU メモリを無駄に占有し続ける。ここで止めてから
    起動することで衝突を防ぎメモリを解放する。**無関係な別プロセスには手を出さない**
    （`pid_looks_like_ours` で選別し、判定不能なものは残す）。停止した PID の一覧を返す。
    """
    reclaimed: list[int] = []
    for pid in find_pids_on_port(port):
        # 自分自身（ゲートウェイ本体）は絶対に殺さない。内部ワーカーは別プロセスなので、
        # ここに現れる our-worker は孤児だけ。万一 self が現れても手を出さない安全弁。
        if pid == os.getpid():
            continue
        if pid_looks_like_ours(pid) and stop_pid(pid, timeout=timeout):
            reclaimed.append(pid)
    return reclaimed


@dataclass
class ServerConfig:
    """起動するローカルLLMサーバーの設定。"""

    backend: str  # "mlx" | "llama-cpp"
    model: str
    host: str = "127.0.0.1"
    port: int = 8080
    parallel: int | None = None  # 同時処理スロット数（llama.cpp のみ）
    disable_thinking: bool = False  # Qwen3 系の思考モードを無効化して起動
    # speculative decoding 用ドラフター。今回は公式対応する
    # Gemma 4 の MTP（Multi-Token Prediction）ドラフターに限定する。
    # draft_model にドラフターの HF id / パス（例
    # mlx-community/gemma-4-E4B-it-qat-assistant-bf16）を指定すると、本体の出力を
    # 変えずに高速化する。"auto" にすると本体名から対応ドラフターを自動選択する
    # （MTP_DRAFTERS の対応表）。本体・ドラフターとも事前に `hf download` 済みである必要が
    # ある（自動ダウンロードはしない）。MTP は vision 対応の mlx-vlm バックエンドのみ対応。
    draft_model: str | None = None
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def parallel_supported(backend: str) -> bool:
    """そのバックエンドが並列スロット指定に対応するか。"""
    return backend == "llama-cpp"


# 本体（target）→ 対応する MTP ドラフター（assistant）の内蔵対応表。
# mlx-community のペアで、いずれも実機で検証済み。draft_model = "auto" のときに
# 本体名から対応ドラフターを引く（明示指定すればここを介さない）。未収載のモデルを
# auto にした場合はエラーで明示指定を促す（MTP 自体は非収載でも明示すれば使える）。
# Gemma 4 が中心だが、Qwen3.6 も MTP 方式で動作確認済み（mlx_vlm --draft-kind mtp）。
MTP_DRAFTERS = {
    "mlx-community/gemma-4-E4B-it-qat-4bit":
        "mlx-community/gemma-4-E4B-it-qat-assistant-bf16",
    "mlx-community/gemma-4-12B-it-qat-4bit":
        "mlx-community/gemma-4-12B-it-qat-assistant-4bit",
    "mlx-community/gemma-4-26B-A4B-it-qat-4bit":
        "mlx-community/gemma-4-26B-A4B-it-qat-assistant-nvfp4",
    "mlx-community/gemma-4-31B-it-qat-4bit":
        "mlx-community/gemma-4-31B-it-qat-assistant-bf16",
    # 非QAT 8bit（26B-A4B）。ドラフターは非QAT の assistant-bf16。
    "mlx-community/gemma-4-26b-a4b-it-8bit":
        "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
    # Qwen3.6-27B（既定モデル）の MTP ドラフター。実測 ~2倍速（38→75 tok/s, 採択93%）。
    "mlx-community/Qwen3.6-27B-4bit":
        "mlx-community/Qwen3.6-27B-MTP-4bit",
    # 自作 ToPo-ToPo 版の Qwen3.6-27B（既定運用）。本体は同じ Qwen3.6-27B ベースなので、
    # ドラフターは mlx-community の MTP ヘッドを共用できる（量子化違いも同一ドラフターで可）。
    "ToPo-ToPo/Qwen3.6-27B-mlx-4bit":
        "mlx-community/Qwen3.6-27B-MTP-4bit",
    "ToPo-ToPo/Qwen3.6-27B-mlx-8bit":
        "mlx-community/Qwen3.6-27B-MTP-4bit",
    "ToPo-ToPo/Qwen3.6-27B-mlx-bf16":
        "mlx-community/Qwen3.6-27B-MTP-4bit",
    # 自作 ToPo-ToPo 版 gemma 4。各 model card が推奨する Google 公式 MTP ドラフター
    # google/gemma-4-<size>-it-assistant を使う（mlx-vlm で変換不要・サイズ固有で量子化に依らず共通。
    # mlx-vlm >= 0.6.3 が必要）。
    "ToPo-ToPo/gemma-4-31b-it-mlx-4bit": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-31b-it-mlx-8bit": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-31b-it-mlx-bf16": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-31b-it-qat-mlx-4bit": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-4bit": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-8bit": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-bf16": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-qat-mlx-4bit": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-E4B-it-qat-mlx-4bit": "google/gemma-4-E4B-it-assistant",
    "ToPo-ToPo/gemma-4-E2B-it-qat-mlx-4bit": "google/gemma-4-E2B-it-assistant",
}


# MTP ドラフター（speculative decoding 用の補助モデル）の repo-id 集合。これ自体は
# 単体のチャットモデルとして使うものではないので、発見一覧（discover_cached_models）には
# 「使えるモデル」として出さない。`org/repo:selector` 形式のドラフターは repo 部分で判定する。
_DRAFTER_REPOS = frozenset(v.split(":", 1)[0] for v in MTP_DRAFTERS.values())


def resolve_drafter(model: str, draft_model: str | None) -> str | None:
    """draft_model を解決する。

    - None / 空 … ドラフター無し（speculative decodingを使わない）。
    - "auto"   … 本体名 model から対応する MTP ドラフター（Gemma 4 / Qwen3.6）を
                 内蔵表で引く。未収載なら ValueError（HF id を明示するよう促す）。
    - それ以外 … その値（ドラフターの HF id / パス）をそのまま使う。
    """
    if not draft_model:
        return None
    if draft_model != "auto":
        return draft_model
    drafter = MTP_DRAFTERS.get(model)
    if drafter is None:
        known = ", ".join(sorted(MTP_DRAFTERS))
        raise ValueError(
            f'draft_model="auto" に対応するドラフターが見つかりません（model={model!r}）。'
            f" 自動対応している本体: {known}。"
            " 他のモデルでは draft_model にドラフターの HF id を明示してください。"
        )
    return drafter


def _hf_hub_cache() -> str:
    """HuggingFace Hub のキャッシュ（models--org--name/snapshots/...）ルートを返す。"""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(home, "hub")


def resolve_gguf(model: str) -> str:
    """llama.cpp の `model`（HF repo-id）を DL 済みキャッシュの実 GGUF パスに解決する。

    `model` は必ず **HF repo-id（`org/repo[:セレクタ]`）**で指定する（実ファイルパスは非対応）。
    HF キャッシュ（`hf download` 済み）から該当 GGUF を探して返す。`-hf` の自動DLには依存しない
    （トークン不要・401 回避）。次の場合はいずれも ValueError（取得方法を案内）:

    - repo-id 形式でない（実パス等）／キャッシュに無い／該当 GGUF が無い。
    - `org/repo` に GGUF が複数あって 1 つに定まらない（`:セレクタ` で絞る）。

    `org/repo:selector` はファイル名の一部（量子化名や `F16-MTP` 等）。セレクタ無しのときは mmproj と
    MTP ヘッドを除いた「本体」GGUF を選ぶ。
    """
    spec = model.strip()
    repo, _sep, selector = spec.partition(":")
    if repo.startswith(("/", "./", "../", "~")) or repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(
            f"model は HF repo-id（org/repo[:量子化名]）で指定してください（実パス非対応）: {model!r}"
        )
    org, name = repo.split("/", 1)
    cache_dir = os.path.join(_hf_hub_cache(), f"models--{org}--{name}", "snapshots")
    if not os.path.isdir(cache_dir):
        raise ValueError(
            f"'{repo}' がローカルキャッシュにありません。先に取得してください: "
            f"hf download {repo} <ファイル名.gguf>"
        )
    ggufs: list[str] = []
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.lower().endswith(".gguf"):
                ggufs.append(os.path.join(root, f))
    if selector:
        matched = [g for g in ggufs if selector.lower() in os.path.basename(g).lower()]
    else:
        # 本体＝mmproj でも MTP ヘッドでもないもの
        matched = [
            g for g in ggufs
            if "mmproj" not in os.path.basename(g).lower()
            and "mtp" not in os.path.basename(g).lower()
        ]
    # 複数スナップショットが同じ blob を指すことがあるので実体で重複排除する。ただし返すのは
    # スナップショット側のパス（実ファイル名が残り、隣の mmproj を検出できる）。
    by_blob: dict[str, str] = {}
    for g in sorted(matched):
        by_blob.setdefault(os.path.realpath(g), g)
    pool = sorted(by_blob.values())
    if not pool:
        hint = f"（セレクタ '{selector}' に一致なし）" if selector else ""
        raise ValueError(
            f"'{model}' に該当する GGUF がキャッシュにありません{hint}。"
            f"hf download {repo} <ファイル名.gguf> で取得してください。"
        )
    if len(pool) > 1:
        names = sorted(os.path.basename(g) for g in pool)
        raise ValueError(
            f"'{model}' に複数の GGUF が該当します {names}。"
            f"'{repo}:<量子化名など>' でファイルを 1 つに絞ってください。"
        )
    return pool[0]


def ensure_cached(repo: str, *, what: str = "モデル") -> str:
    """mlx 系（mlx / mlx-vlm）の HF repo-id がローカルキャッシュに**完全に**存在するか検証する。

    本サーバーは自動ダウンロードを行わない（事前に `hf download` 済みであることを要求する）。
    起動前にここで存在を確認し、無ければ取得方法を案内して ValueError を送出する
    （llama-cpp の resolve_gguf と同じ「事前 DL 必須」ポリシー）。返り値は確認したスナップショット
    ディレクトリ（実ファイルパス指定時はそのパス）。

    次のいずれも「未取得」とみなしてエラーにする:
      - スナップショットが存在しない。
      - ダウンロード途中の残骸（blobs/ 配下の *.incomplete）が残っている。
      - 重み（*.safetensors）の実体がキャッシュに揃っていない。
    """
    spec = repo.strip()
    # 実ファイル/ディレクトリパス指定（repo-id ではない）はそのパスの存在のみ確認する。
    if spec.startswith(("/", "./", "../", "~")):
        path = os.path.expanduser(spec)
        if not os.path.exists(path):
            raise ValueError(f"{what}のパスが見つかりません: {repo!r}")
        return path
    if spec.count("/") != 1 or not all(spec.split("/")):
        raise ValueError(
            f"{what}は HF repo-id（org/repo）で指定してください: {repo!r}"
        )
    org, name = spec.split("/", 1)
    base = os.path.join(_hf_hub_cache(), f"models--{org}--{name}")
    snap_root = os.path.join(base, "snapshots")
    # ダウンロード途中の残骸があれば「未取得」と同じ扱い（今回の不具合＝DL 停滞の主症状）。
    incomplete = glob.glob(os.path.join(base, "blobs", "*.incomplete"))
    snaps = sorted(glob.glob(os.path.join(snap_root, "*"))) if os.path.isdir(snap_root) else []
    if not snaps or incomplete:
        raise ValueError(
            f"{what} '{repo}' がローカルキャッシュにありません（自動ダウンロードは無効）。"
            f" 先に取得してください: hf download {spec}"
        )
    # 重み（safetensors / npz）の実体（シンボリックリンク先まで）が存在するか確認する。
    # whisper 系の mlx リポジトリは *.npz で重みを持つものがあるため両方を許容する。
    weights = [
        f
        for s in snaps
        for pattern in ("*.safetensors", "*.npz")
        for f in glob.glob(os.path.join(s, pattern))
        if os.path.exists(os.path.realpath(f))
    ]
    if not weights:
        raise ValueError(
            f"{what} '{repo}' の重み（*.safetensors / *.npz）がキャッシュに揃っていません。"
            f" 取得し直してください: hf download {spec}"
        )
    return os.path.dirname(weights[0])


def mtp_status(model: str) -> str | None:
    """model の MTP（Multi-Token Prediction による高速化）の利用可否を返す。

    対応表（MTP_DRAFTERS）に本体が在るかと、対応ドラフターがローカルにキャッシュ済みかで判定する:

    - "ready"     … 対応ドラフターがキャッシュ済み。そのまま MTP が効く（~2倍速）。
    - "available" … MTP には対応するがドラフターが未取得。`hf download <drafter>` で有効化できる。
    - None        … MTP 非対応（対応表に無い）。

    一覧表示（discover_cached_models / TUI）から呼ぶ。ドラフターの有無確認に ensure_cached を
    使う（自動 DL はしない方針と一貫）。
    """
    drafter = MTP_DRAFTERS.get(model)
    if not drafter:
        return None
    try:
        ensure_cached(drafter, what="ドラフター")
        return "ready"
    except ValueError:
        return "available"


_DISCOVER_CACHE: dict = {"t": -1e9, "v": []}

# チャット/生成に使わない（埋め込み・STT・分類・エンコーダ）モデルタイプ。発見一覧から除く。
_NON_CHAT_MODEL_TYPES = frozenset({
    "bert", "roberta", "xlm-roberta", "distilbert", "deberta", "deberta-v2",
    "mpnet", "camembert", "electra", "albert", "nomic_bert",
    "whisper", "wav2vec2", "clip", "siglip", "t5", "mt5",
})


def _is_generative_repo(snap_root: str) -> bool:
    """スナップショット内の config.json を見て、生成（チャット）系モデルかを判定する。

    埋め込み（e5/MiniLM 等）・STT（whisper）・分類器など非チャットのモデルを発見一覧から
    除くためのフィルタ。config.json が読めなければ True（取りこぼしを避ける＝控えめに除外）。
    """
    cfg_path = None
    for sroot, _d, files in os.walk(snap_root):
        if "config.json" in files:
            cfg_path = os.path.join(sroot, "config.json")
            break
    if not cfg_path:
        return True
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return True
    model_type = str(cfg.get("model_type", "")).lower()
    if model_type in _NON_CHAT_MODEL_TYPES:
        return False
    archs = cfg.get("architectures") or []
    if not archs:
        return True  # アーキ不明なら除外しない
    return any(
        a.endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs
    )


def discover_cached_models(ttl: float = 10.0) -> list[dict]:
    """HF キャッシュにある**実行可能なチャットモデル**を列挙する（発見用）。

    LM Studio / Ollama のように「いま手元で動かせる候補」をクライアントに見せるための一覧。
    ロード済みかどうかに関わらず、ダウンロード済みモデルを `{"id", "backend", "mtp"}` のリストで
    返す（`mtp` は "ready" / "available" / None＝mtp_status）。MTP ドラフター自体は単体で使う
    モデルではないので一覧からは除外する（_DRAFTER_REPOS）。判定はヒューリスティック:

    - GGUF を含む repo → llama-cpp。本体（mmproj / MTP ヘッドを除く）が 1 つなら `org/repo`、
      複数あれば `org/repo:<ファイル名>` を量子化ごとに列挙（そのままロードできる形）。
    - `config.json` ＋ 重み（`.safetensors` / `.npz`）を持ち、生成系アーキ（`*ForCausalLM` /
      `*ForConditionalGeneration`）の repo → mlx 系（mlx-vlm で動的ロード）。埋め込み・STT・
      分類などの非チャットモデルは除外する（`_is_generative_repo`）。

    `ttl` 秒は結果をキャッシュする（`/admin/status` の毎秒ポーリングで毎回走査しないため）。
    """
    now = time.monotonic()
    if now - _DISCOVER_CACHE["t"] < ttl:
        return list(_DISCOVER_CACHE["v"])
    root = _hf_hub_cache()
    out: list[dict] = []
    seen: set[str] = set()
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("models--") or entry.count("--") < 2:
                continue
            _, org, name = entry.split("--", 2)
            repo = f"{org}/{name}"
            # MTP ドラフターは「使えるモデル」ではないので一覧に出さない。
            if repo in _DRAFTER_REPOS:
                continue
            snap_root = os.path.join(root, entry, "snapshots")
            if not os.path.isdir(snap_root):
                continue
            files = [f for _r, _d, fs in os.walk(snap_root) for f in fs]
            ggufs = [f for f in files if f.lower().endswith(".gguf")]
            if ggufs:
                bodies = [
                    f for f in ggufs
                    if "mmproj" not in f.lower() and "mtp" not in f.lower()
                ]
                if not bodies:
                    continue  # mmproj / MTP ヘッドだけの repo は本体ではない
                if len(bodies) == 1:
                    cands = [repo]
                else:
                    cands = [f"{repo}:{os.path.splitext(f)[0]}" for f in sorted(bodies)]
                backend = "llama-cpp"
            elif (
                "config.json" in files
                and any(f.endswith((".safetensors", ".npz")) for f in files)
                # 生成系（チャット）または STT（whisper）を対象にする。埋め込み・分類器などの
                # 非チャット・非STT モデルは除外する（_is_generative_repo）。
                and (_is_generative_repo(snap_root) or infer_backend(repo) == "whisper")
            ):
                cands = [repo]
                backend = infer_backend(repo)  # whisper → STT、mlx → mlx-vlm、他は OS 既定
            else:
                continue
            # MTP（高速化）の利用可否を本体ごとに付与する（ドラフターが揃っていれば "ready"）。
            mtp = mtp_status(repo)
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    out.append({"id": c, "backend": backend, "mtp": mtp})
    _DISCOVER_CACHE["t"] = now
    _DISCOVER_CACHE["v"] = out
    return list(out)


def estimate_model_bytes(config: ServerConfig) -> int | None:
    """常駐に要するメモリの**概算**（バイト）。重みファイルのサイズを基準にする。

    - llama-cpp: 本体 GGUF（＋自動付与される mmproj、＋ドラフト GGUF）のファイルサイズ合計。
    - mlx / mlx-vlm: HF キャッシュのスナップショットに在るファイルサイズ合計（blob 実体で重複排除）。

    取得できない（未キャッシュ・未DL 等）ときは `None` を返す（メモリガードはそのモデルを
    スキップする）。KVキャッシュ・ランタイムバッファは含まない**下限寄りの見積もり**なので、
    呼び出し側で余裕係数を掛ける前提（→ docs/llama-cpp.md のメモリガード）。
    """
    try:
        if config.backend == "llama-cpp":
            path = resolve_gguf(config.model)
            total = os.path.getsize(path)
            mmproj = find_sibling_mmproj(path)
            if mmproj and not any(
                a in ("--no-mmproj",) for a in config.extra_args
            ):
                total += os.path.getsize(mmproj)
            if config.draft_model:
                try:
                    total += os.path.getsize(resolve_gguf(config.draft_model))
                except (ValueError, OSError):
                    pass  # ドラフトが解決できなくても本体分は数える
            return total
        # mlx / mlx-vlm: models--org--name/snapshots/<hash>/ の合計（blob 実体で重複排除）
        repo = config.model.strip()
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
        name for name in os.listdir(directory)
        if "mmproj" in name.lower() and name.lower().endswith(".gguf")
    )
    if not candidates:
        return None
    return os.path.abspath(os.path.join(directory, candidates[0]))


def build_command(config: ServerConfig) -> list[str]:
    """バックエンドに応じた起動コマンドを組み立てる。

    いずれも OpenAI 互換サーバーを立ち上げる:
      - mlx       : mlx_lm.server（テキスト専用。逐次処理。並列スロットの概念なし）
      - mlx-vlm   : mlx_vlm.server（vision 対応。画像入力 image_url を受けられる）
      - llama-cpp : llama-server（--parallel で並列スロットを確保）
    """
    if config.backend == "mlx":
        # 自動ダウンロードは行わない。事前に `hf download` 済みであることを起動前に確認する
        # （未取得ならここで案内付き ValueError）。
        ensure_cached(config.model)
        command = [
            "mlx_lm.server",
            "--model", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
        if config.disable_thinking:
            command += ["--chat-template-args", '{"enable_thinking": false}']
    elif config.backend == "mlx-vlm":
        # mlx_vlm の OpenAI 互換サーバー（vision 対応）。コンソールスクリプトが
        # 無い環境もあるため `python -m` で確実に起動する。逐次処理で並列スロットの
        # 概念はないため --parallel は渡さない。thinking はサーバー既定が OFF
        # （--enable-thinking を渡さなければ無効）であり、リクエスト毎の明示制御は
        # クライアントがトップレベル enable_thinking で行う（llm.py 参照）。
        # 自動ダウンロードは行わない。本体は事前に `hf download` 済みであることを確認する。
        ensure_cached(config.model)
        command = [
            sys.executable, "-m", "mlx_vlm.server",
            "--model", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
        # Gemma 4 の MTP ドラフターによる speculative decoding。draft_kind は mtp に固定する
        # （他種別＝dflash / eagle3 は今回は対象外）。draft_model="auto" は本体名から
        # 対応ドラフターを自動選択する。本体・ドラフターとも事前 DL 必須（未取得なら案内付き
        # エラーで起動を中止し、自動ダウンロードはしない）。
        drafter = resolve_drafter(config.model, config.draft_model)
        if drafter:
            ensure_cached(drafter, what="ドラフター")
            command += ["--draft-model", drafter, "--draft-kind", "mtp"]
    elif config.backend == "llama-cpp":
        # model は HF repo-id（org/repo[:selector]）。DL 済みキャッシュから実 GGUF を解決する
        # （キャッシュに無ければ ValueError。クライアントに見せる ID は repo-id のまま）。
        model_path = resolve_gguf(config.model)
        command = [
            llama_server_binary(),  # プロビジョナが導入した絶対パス、無ければ PATH の "llama-server"
            "-m", model_path,
            "--host", config.host,
            "--port", str(config.port),
        ]
        # 埋め込み MTP（Qwen3.6 等、本体 GGUF に MTP ヘッドが内蔵）。draft_model="self"/"mtp"
        # で有効化＝別ドラフトファイル不要（--spec-type draft-mtp のみ）。この方式は llama.cpp 側で
        # --mmproj（vision）と --parallel>1 が未対応なので、両者は付けない（付けると起動失敗する）。
        embedded_mtp = bool(config.draft_model) and \
            config.draft_model.strip().lower() in ("self", "mtp")
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
            if config.draft_model and "-md" not in config.extra_args:
                draft_path = resolve_gguf(config.draft_model)
                command += ["-md", draft_path]
                if "mtp" in os.path.basename(draft_path).lower():
                    command += ["--spec-type", "draft-mtp"]
        # 計算効率の自動チューニング（自動導入バイナリの accel に合わせる。extra_args 優先）。
        command += auto_llama_flags(config)
    elif config.backend == "whisper":
        # mlx-whisper を OpenAI 互換の STT サーバ（1 モデル 1 プロセス）として起動する。
        # 専用サーバは同梱の local_llm_server.stt_server（標準ライブラリのみ）。
        # 本体は事前 DL 必須（未取得なら案内付き ValueError）。音声デコードに ffmpeg CLI が要る。
        ensure_cached(config.model)
        command = [
            sys.executable, "-m", "local_llm_server.stt_server",
            "--model", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
    elif config.backend == "vllm":
        # vLLM の OpenAI 互換 API サーバを、隔離 venv の python から起動する（→ vllm_provisioner）。
        # model は HF repo-id。事前 DL 済みを前提に確認する（未取得は案内付き ValueError）。
        # クライアントに見せる id を repo-id に固定するため --served-model-name も同じ id にする。
        # 逐次でなく連続バッチングで多人数同時に強い（並列は vLLM が内部で捌く）。
        ensure_cached(config.model)
        command = [
            vllm_python(), "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--served-model-name", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
    elif config.backend == "sglang":
        # SGLang の OpenAI 互換 API サーバを、隔離 venv の python から起動する（→ sglang_provisioner）。
        # SGLang は引数が vLLM と違い、モデルは --model-path で渡す。RadixAttention で
        # 共有プレフィックス（システムプロンプト/ツール定義）の多い用途に強い。
        ensure_cached(config.model)
        command = [
            sglang_python(), "-m", "sglang.launch_server",
            "--model-path", config.model,
            "--served-model-name", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
    else:
        raise ValueError(
            f"unknown backend: {config.backend!r} (choose from {BACKENDS})"
        )
    return command + config.extra_args


def is_ready(base_url: str, timeout: float = 1.0) -> bool:
    """OpenAI互換サーバーが応答可能かを判定する。

    401/403 も「起動して応答している」と判定する（api_key を設定したゲートウェイは
    /v1/models にキーを要求するため。ここで False にすると TUI の自己ヘルスチェック
    （起動判定・常駐ポーリング）が、正常起動したゲートウェイを「応答なし」と誤判定してしまう）。
    """
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        return exc.code in (401, 403)  # 認証は要求されたが、サーバー自体は稼働中
    except (urllib.error.URLError, OSError):
        return False


def list_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """サーバーが公開する全モデル id を /v1/models から返す（取得失敗時は []）。

    単一モデルサーバーはロード済みの1件を返す。一方、複数モデルを束ねるルーター型
    サーバーはカタログとして多数を並べる（先頭が必ずしもアクティブとは限らない）ので、
    モデルの提供有無は「リストに含まれるか」で判定する（→ model_available）。
    """
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            mid = it.get("id")
            if isinstance(mid, str) and mid:
                out.append(mid)
    return out


def running_model(base_url: str, timeout: float = 5.0) -> str | None:
    """起動中サーバーの代表モデル（/v1/models の最初の id）を返す。取得不可は None。

    注意: ルーター型（多モデル）サーバーでは先頭が必ずしもアクティブとは限らない。
    設定モデルが使えるかの判定には model_available（リスト全体を見る）を使うこと。
    """
    models = list_models(base_url, timeout)
    return models[0] if models else None


def model_available(base_url: str, model: str | None, timeout: float = 5.0) -> bool | None:
    """設定モデル model がサーバーで提供されているかを返す。

    - True  … /v1/models のいずれかと一致（単一モデル一致／ルーターのカタログに存在）
    - False … モデル一覧は取れたが、その中に一致が無い
    - None  … 判定不能（model 未指定、または一覧を取得できない）→ 警告しない
    """
    if not model:
        return None
    models = list_models(base_url, timeout)
    if not models:
        return None
    return any(models_match(m, model) for m in models)


def models_match(a: str | None, b: str | None) -> bool:
    """2つのモデル名が同じものを指すかを大まかに判定する。

    パス指定とリポジトリ名のゆれ（例 /abs/path/Foo と org/Foo）を吸収するため、
    末尾要素（basename）を小文字で比較する。
    """
    if not a or not b:
        return True  # どちらか不明なら警告しない（誤検知を避ける）
    if a == b:
        return True
    base = lambda s: s.rstrip("/").split("/")[-1].lower()  # noqa: E731
    return base(a) == base(b)


def parse_host_port(base_url: str, default_port: int = 8080) -> tuple[str, int]:
    """base_url（例 http://127.0.0.1:8080/v1）から host と port を取り出す。"""
    parsed = urllib.parse.urlparse(base_url)
    return parsed.hostname or "127.0.0.1", parsed.port or default_port


class LocalServer:
    """ローカルLLMサーバーをサブプロセスとして管理する。"""

    def __init__(self, config: ServerConfig, log_path: str | None = None) -> None:
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        # サーバーの大量ログ（INFO/Stream finished 等）で対話画面が乱れないよう、
        # 標準出力・標準エラーはこのログファイルへ逃がす（端末には流さない）。
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
        if self._proc is not None:
            raise RuntimeError("server already started")
        if self.log_path is None:
            fd, self.log_path = tempfile.mkstemp(
                prefix="local-llm-server-", suffix=".log"
            )
            os.close(fd)
        else:
            # 明示ログパス（ゲートウェイの daemon_log_path 等）は親ディレクトリが
            # 無いことがあるので作る（project_cache_dir は呼び出し側が作る設計）。
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        try:
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
                env.setdefault("MLX_VLM_THINKING_START_TOKEN", "<|channel>thought")
                env.setdefault("MLX_VLM_THINKING_END_TOKEN", "<channel|>")
            self._proc = subprocess.Popen(
                build_command(self.config),
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                # 子を独立したプロセスグループにして、停止時に孫まで一括終了できるようにする
                # （POSIX のみ。Windows では無視される）。stop() の killpg と対になる。
                start_new_session=_POSIX,
            )
        except FileNotFoundError as exc:
            self._close_log()
            raise RuntimeError(
                f"バックエンド実行ファイルが見つかりません: {exc.filename}。"
                " mlx_lm / mlx_vlm / llama.cpp がインストール・PATH 上にあるか確認してください。"
            ) from exc

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
        if self._proc is None:
            self._close_log()
            return
        proc = self._proc
        if grace <= 0:
            _signal_process_tree(proc, kill=True)  # 即 SIGKILL（全体終了・graceful 不要）
        else:
            _signal_process_tree(proc, kill=False)  # SIGTERM を送って grace 秒待つ
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                # 終わらなければグループ全体へ SIGKILL で強制終了
                _signal_process_tree(proc, kill=True)
        try:
            proc.wait(timeout=5)  # SIGKILL 後の回収を見届ける（ゾンビを残さない）
        except subprocess.TimeoutExpired:
            pass
        self._proc = None
        self._close_log()

    def __enter__(self) -> "LocalServer":
        self.start()
        self.wait_until_ready()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --- マルチモデルゲートウェイ（daemon）用ヘルパ -----------------------------
def ignore_shutdown_signals() -> None:
    """SIGTERM / SIGHUP / SIGINT を一旦無視（SIG_IGN）にする。

    後始末（配下のサーバー停止など）の最中に再度シグナルが届いても中断されないよう、
    クリーンアップ開始時に呼ぶ。停止時の killpg や端末クローズで複数シグナルが連続して
    届いても、停止処理を最後までやり切って孫プロセスを残さないための保険。
    install_shutdown_handlers() の対（同じくメインスレッドからのみ有効）。
    """
    for name in ("SIGTERM", "SIGHUP", "SIGINT"):
        sig = getattr(signal, name, None)  # SIGHUP は Windows に無い
        if sig is None:
            continue
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):
            pass  # メインスレッド以外などでは登録できない


def daemon_log_path(port: int) -> str:
    """ゲートウェイが起動するモデルサーバーのログ保存先（ポート別の固定パス）。

    TUI やログ表示から参照できるよう、ランダムな tempfile ではなくポートで決まる固定パスにする。
    プロジェクト内（`./.local-llm-server/`、カレントディレクトリ相対）に置き、ホーム等の外部には
    書かない。同じポートのサーバーは同じログに追記する。
    """
    return os.path.join(project_cache_dir(), f"server-{port}.log")


class GatewayAlreadyRunning(RuntimeError):
    """このマシンで既にゲートウェイが起動している（単一起動ガードが二重起動を拒否）。

    保持者の PID（読めれば）とロックファイルのパスを添える。呼び出し側はこれを捕まえて
    「既に起動済み」を明示エラーとして返す（黙って 2 個目を立てて乱立させない）。
    """

    def __init__(self, pid: int | None, path: str) -> None:
        self.pid = pid
        self.path = path
        who = f"pid {pid}" if pid else "unknown pid"
        super().__init__(
            f"a local-llm-server gateway is already running on this machine ({who}); "
            f"stop it before starting another (single-instance lock: {path})"
        )


def gateway_lock_path() -> str:
    """マシン内で 1 つだけゲートウェイを許すロックファイルのパス（cwd 非依存の固定パス）。

    モデルサーバーのログ（`project_cache_dir()` = cwd 相対）と違い、**どのディレクトリから
    起動しても同じ 1 個**のロックを見るよう、temp ディレクトリ配下の固定名にする。これで
    「別ディレクトリから（開発ツール等が）勝手に 2 個目を起動する」ケースも 1 本に束ねられる。
    ポートに依存しないので、port を変えても二重には立たない（＝マシンにつき 1 ゲートウェイ）。
    """
    return os.path.join(tempfile.gettempdir(), "local-llm-server-gateway.lock")


def _read_lock_pid(path: str) -> int | None:
    """ロックファイルに保持者が書き込んだ PID を読む（読めなければ None）。"""
    try:
        with open(path, encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


class GatewayLock:
    """ゲートウェイの単一起動を保証する OS レベルの排他ロック（flock / msvcrt）。

    プロセス生存中だけ握る advisory ロック。プロセスが（クラッシュ・SIGKILL 含め）終われば
    OS が自動解放するので、古い PID ファイルが残っても stale ロックにはならない（＝手動の
    生存判定が要らない）。`acquire()` は取得できなければ `GatewayAlreadyRunning` を投げる。
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path or gateway_lock_path()
        self._fd: int | None = None

    def acquire(self) -> "GatewayLock":
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _flock_exclusive_nb(fd)
        except OSError as exc:  # 既に他プロセスが握っている（EWOULDBLOCK 等）
            pid = _read_lock_pid(self._path)
            os.close(fd)
            raise GatewayAlreadyRunning(pid, self._path) from exc
        # 取得できた → 自分の PID を記録（失敗した取得者がこれを読んで相手を示す）。
        # Windows のロックは番兵オフセットへ seek した状態なので、書き込み前に必ず先頭へ戻す。
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass  # PID 記録は best-effort（ロック自体は取れている）
        self._fd = fd
        return self

    def release(self) -> None:
        """ロックを解放する（プロセス終了時にも OS が自動解放するが明示的に返す）。

        ファイル自体は消さない（消すと「解放→別プロセスが再作成」の隙に取り違えが起きる）。
        残った PID は次の取得失敗時にしか読まれず、その時は必ず生きた保持者が上書き済み。
        """
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                _flock_unlock(fd)
            except OSError:
                pass
            finally:
                os.close(fd)

    def __enter__(self) -> "GatewayLock":
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()


# --- プラットフォーム別のファイルロック実装 -----------------------------------
# POSIX は fcntl.flock、Windows は msvcrt.locking を使う。どちらも「他プロセスが
# 握っていれば即エラー（非ブロッキング）」で、プロセス終了時に OS が自動解放する。
#
# Windows の msvcrt.locking は POSIX の flock（advisory）と違い**強制ロック**で、
# ロックした領域は他ハンドルからの読み書きもブロックされる。そのため保持者 PID は
# ファイル先頭に書き、ロックは PID データと重ならない**高オフセットの番兵 1 バイト**に掛ける
# （EOF を越えた領域もロック可）。こうすれば _read_lock_pid が先頭の PID を普通に読める。
_LOCK_SENTINEL_OFFSET = 1 << 30  # 1 GiB 目。PID 文字列（先頭数バイト）と絶対に重ならない
if os.name == "nt":  # pragma: no cover - Windows 専用パス
    import msvcrt

    def _flock_exclusive_nb(fd: int) -> None:
        os.lseek(fd, _LOCK_SENTINEL_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _flock_unlock(fd: int) -> None:
        os.lseek(fd, _LOCK_SENTINEL_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _flock_exclusive_nb(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _flock_unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def local_connect_host(bind_host: str) -> str:
    """bind 用ホストから、同一マシンでの自己接続に使うホストを求める。

    0.0.0.0 / :: / 空（ワイルドカード bind）は、そのアドレス宛の直接接続が不可搬なため
    （特に macOS）ループバック 127.0.0.1 で叩く。特定 IP に bind したときはその IP をそのまま
    使う。TUI/CLI の状態確認・ヘルスチェックなど「自分自身のゲートウェイ」への接続に使う。
    """
    if bind_host in ("0.0.0.0", "::", "", "*"):
        return "127.0.0.1"
    return bind_host


def primary_lan_ip() -> str | None:
    """このマシンの主要な LAN IP（外向きインターフェースのアドレス）。取得不能なら None。

    実際には通信せず、UDP ソケットの接続先選択でルーティング表からローカル側 IP を得る
    （リモートのクライアントが指す base_url を案内するために使う）。
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # 実送信はしない。ローカル側アドレスの決定だけ
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def server_status(host: str = "127.0.0.1", port: int = 8799) -> dict | None:
    """ポートで動いているローカルサーバーの状態をまとめて返す（TUI の状態表示用）。

    応答もせず LISTEN しているプロセスも無ければ None。応答可否・PID 一覧・提供モデル・
    ログパス（存在すれば）を1つの dict にまとめる。PID は POSIX で lsof が使えるときのみ
    （取得不能でも応答していれば ready=True で報告する）。
    """
    base_url = f"http://{host}:{port}/v1"
    ready = is_ready(base_url)
    pids = find_pids_on_port(port)
    if not ready and not pids:
        return None
    log = daemon_log_path(port)
    return {
        "host": host,
        "port": port,
        "base_url": base_url,
        "ready": ready,
        "pids": pids,
        "models": list_models(base_url) if ready else [],
        "log_path": log if os.path.exists(log) else None,
    }


def gateway_admin_status(
    host: str = "127.0.0.1", port: int = 8799, timeout: float = 2.0
) -> dict | None:
    """ゲートウェイの GET /admin/status を取得する（常駐モデルのライブ状態＋運用方針）。

    server_status と違い、各モデルの loaded / inflight（処理中数）や max_resident /
    idle_timeout までゲートウェイ本体から取得できる（→ TUI 監視用）。応答しない・旧版で
    エンドポイントが無い場合は None を返す（呼び出し側は server_status にフォールバックできる）。
    """
    url = f"http://{host}:{port}/admin/status"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def bench_model(
    model: str,
    base_url: str = "http://127.0.0.1:8799/v1",
    *,
    api_key: str | None = None,
    max_tokens: int = 128,
    timeout: float = 180.0,
) -> dict:
    """モデルに短文生成を投げ、生成スループット（tok/s）を測る（チューニング効果の確認用）。

    非ストリームで `max_tokens` トークンを生成させ、応答の usage.completion_tokens を
    実測秒数で割る。初回はモデルロード込みなので、TUI 側は「2 回目」を測るとよい。
    戻り値: {"model", "tokens", "seconds", "tok_per_s"}。失敗は RuntimeError。
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": "Write a short story about the sea."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers=headers)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"bench request failed: {exc}") from exc
    seconds = time.monotonic() - t0
    tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
    tps = tokens / seconds if seconds > 0 else 0.0
    return {"model": model, "tokens": tokens, "seconds": round(seconds, 2),
            "tok_per_s": round(tps, 1)}


def gateway_set_max_resident(
    value: int | None,
    host: str = "127.0.0.1",
    port: int = 8799,
    timeout: float = 5.0,
) -> dict | None:
    """稼働中のゲートウェイに POST /admin/config で max_resident を変更させる（TUI 操作用）。

    value は 1 以上の整数、または None（無制限）。稼働中（busy）のモデルは止めず、超過分は
    サーバー側でアイドルから順に非同期退避される（＝更新でリクエストが止まらない）。反映後の
    値を含む応答 dict を返す。応答しない・エラー時は None（呼び出し側が失敗として扱う）。
    """
    url = f"http://{host}:{port}/admin/config"
    body = json.dumps({"max_resident": value}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def gateway_drain(
    enable: bool = True,
    host: str = "127.0.0.1",
    port: int = 8799,
    timeout: float = 5.0,
) -> dict | None:
    """稼働中のゲートウェイに POST /admin/drain で再起動準備を要求する（TUI の自動更新用）。

    enable=True: ゲートウェイが原子的に「処理中 0・在席 0」を確認し、満たせば新規受付を
    止めて {"draining": True} を返す。busy なら {"draining": False, "inflight": n,
    "sessions": n}（何も変えない）。enable=False で解除。応答しない（未起動・旧版で
    エンドポイントが無い）ときは None。
    """
    url = f"http://{host}:{port}/admin/drain"
    body = json.dumps({"enable": enable}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def gateway_log_path(port: int) -> str:
    """バックグラウンド起動したゲートウェイ本体（公開ポート）の出力ログ保存先。

    モデルサーバーの daemon_log_path（server-<port>.log）と別に、ゲートウェイ自身の
    起動ログを gateway-<port>.log に逃がす（TUI のログ表示から参照できる固定パス）。
    """
    return os.path.join(project_cache_dir(), f"gateway-{port}.log")


def start_gateway_background(
    cwd: str,
    host: str = "127.0.0.1",
    port: int = 8799,
    *,
    start_timeout: float = 120.0,
) -> int:
    """ゲートウェイをデタッチした別プロセスで常駐起動し、応答可能になるまで待つ。

    ターミナルを占有しない常駐起動（Ollama 流）。cwd の ./gateway.toml を読む
    ヘッドレスワーカー（`python -m local_llm_server` = __main__）を、新セッション（POSIX）/
    DETACHED_PROCESS（Windows）で起動して端末・親から切り離し、出力は gateway_log_path に
    逃がす。応答可能になったら PID を返す。既に起動済みなら何もせず既存 PID（不明なら 0）を返す。
    起動失敗は RuntimeError、時間内に応答しなければ TimeoutError。
    """
    base_url = f"http://{host}:{port}/v1"
    existing = find_pids_on_port(port)
    if is_ready(base_url):
        return existing[0] if existing else 0
    if existing:
        # ポートは埋まっているのに応答しない。うちのプロセス（起動途中など）なら既存扱い、
        # 無関係なプロセスなら「起動済み」と偽らず明示エラーにする（黙って成功を返すと
        # ゲートウェイが一度も立たないまま全リクエストが失敗し続ける）。
        if any(pid_looks_like_ours(p) for p in existing):
            return existing[0]
        raise RuntimeError(
            f"port {port} is in use by an unrelated process (pid {existing}) that does not "
            f"respond as a gateway; stop it or change `port` in gateway.toml"
        )

    log_path = gateway_log_path(port)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    popen_kwargs: dict = {
        "cwd": cwd,
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        # /admin/status の launcher 表示用マーク。この関数経由（`gw start` の裏起動）で立った
        # ゲートウェイは "cli"、直接の `python -m local_llm_server` は無印で "headless" になる。
        "env": {**os.environ, "LOCAL_LLM_GW_LAUNCHER": "cli"},
    }
    if os.name == "nt":
        # 端末から切り離し、新プロセスグループにする（stop の taskkill /T と対）。
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True  # setsid: 端末/親から独立
    try:
        # ヘッドレスワーカー（__main__）を起動。裏起動は出力をログへ逃がす非 TTY なので
        # TUI を出さずゲートウェイ本体だけを回す。
        proc = subprocess.Popen(
            [sys.executable, "-m", "local_llm_server"], **popen_kwargs
        )
    finally:
        log_file.close()  # fd は子へ複製済み。親側は閉じてよい
    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"gateway exited early (code {proc.returncode}); see {log_path}"
            )
        if is_ready(base_url):
            return proc.pid
        time.sleep(0.5)
    raise TimeoutError(f"gateway not ready within {start_timeout:g}s; see {log_path}")
