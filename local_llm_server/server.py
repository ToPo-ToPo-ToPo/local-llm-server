from __future__ import annotations

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
from dataclasses import dataclass, field, replace

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
    """指定ポートを LISTEN しているプロセスの PID 一覧を返す（POSIX のみ、lsof を利用）。

    lsof が無い / Windows などでは空リストを返す（呼び出し側で案内する）。
    """
    if not _POSIX:
        return []
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


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    """PID とそのプロセスグループへ SIGTERM→（猶予後）SIGKILL を送って停止する。

    停止を試みたら True、対象が既にいなければ False（POSIX 以外も False）。
    """
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


@dataclass
class ServerConfig:
    """起動するローカルLLMサーバーの設定。"""

    backend: str  # "mlx" | "llama-cpp"
    model: str
    host: str = "127.0.0.1"
    port: int = 8080
    parallel: int | None = None  # 同時処理スロット数（llama.cpp のみ）
    disable_thinking: bool = False  # Qwen3 系の思考モードを無効化して起動
    # 投機的デコード（speculative decoding）用ドラフター。今回は公式対応する
    # Gemma 4 の MTP（Multi-Token Prediction）ドラフターに限定する。
    # draft_model にドラフターの HF id / パス（例
    # mlx-community/gemma-4-E4B-it-qat-assistant-bf16）を指定すると、本体の出力を
    # 変えずに高速化する。"auto" にすると本体名から対応ドラフターを自動選択する
    # （MTP_DRAFTERS の対応表）。指定時は本体＋ドラフターの2モデルを mlx-vlm が
    # 初回に自動ダウンロードする。MTP は vision 対応の mlx-vlm バックエンドのみ対応。
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
}


def resolve_drafter(model: str, draft_model: str | None) -> str | None:
    """draft_model を解決する。

    - None / 空 … ドラフター無し（投機的デコードを使わない）。
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


def build_command(config: ServerConfig) -> list[str]:
    """バックエンドに応じた起動コマンドを組み立てる。

    いずれも OpenAI 互換サーバーを立ち上げる:
      - mlx       : mlx_lm.server（テキスト専用。逐次処理。並列スロットの概念なし）
      - mlx-vlm   : mlx_vlm.server（vision 対応。画像入力 image_url を受けられる）
      - llama-cpp : llama-server（--parallel で並列スロットを確保）
    """
    if config.backend == "mlx":
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
        command = [
            sys.executable, "-m", "mlx_vlm.server",
            "--model", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
        # Gemma 4 の MTP ドラフターによる投機的デコード。draft_kind は mtp に固定する
        # （他種別＝dflash / eagle3 は今回は対象外）。draft_model="auto" は本体名から
        # 対応ドラフターを自動選択する。指定があれば本体とドラフターの両方を mlx-vlm が
        # 初回に自動ダウンロードする。
        drafter = resolve_drafter(config.model, config.draft_model)
        if drafter:
            command += ["--draft-model", drafter, "--draft-kind", "mtp"]
    elif config.backend == "llama-cpp":
        command = [
            "llama-server",
            "-m", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
        if config.parallel is not None:
            command += ["--parallel", str(config.parallel)]
        if config.disable_thinking:
            command += ["--chat-template-kwargs", '{"enable_thinking": false}']
    else:
        raise ValueError(
            f"unknown backend: {config.backend!r} (choose from {BACKENDS})"
        )
    return command + config.extra_args


def is_ready(base_url: str, timeout: float = 1.0) -> bool:
    """OpenAI互換サーバーが応答可能かを判定する。"""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            return resp.status == 200
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
            self._proc = subprocess.Popen(
                build_command(self.config),
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
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

    def stop(self) -> None:
        if self._proc is None:
            self._close_log()
            return
        proc = self._proc
        # graceful: プロセスグループ全体へ SIGTERM を送って 10 秒待つ
        _signal_process_tree(proc, kill=False)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # 終わらなければグループ全体へ SIGKILL で強制終了
            _signal_process_tree(proc, kill=True)
            try:
                proc.wait(timeout=5)
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


def build_pool_configs(base: ServerConfig, instances: int) -> list[ServerConfig]:
    """base を起点に、連番ポート（port, port+1, ...）の設定を instances 個作る。"""
    return [replace(base, port=base.port + i) for i in range(instances)]


class ServerPool:
    """複数のローカルLLMサーバーをまとめて起動・管理する。

    mlx のように1プロセスが逐次処理のバックエンドで並列性を得る用途を想定。
    各インスタンスは別ポートで起動し、それぞれモデルを個別にロードする
    （= 重みのメモリはインスタンス数分かかる）。
    """

    def __init__(self, configs: list[ServerConfig]) -> None:
        self._servers = [LocalServer(c) for c in configs]

    @property
    def base_urls(self) -> list[str]:
        return [server.base_url for server in self._servers]

    def start(self) -> None:
        for server in self._servers:
            server.start()

    def wait_until_ready(self, timeout: float = 120.0) -> None:
        for server in self._servers:
            server.wait_until_ready(timeout=timeout)

    def wait(self) -> None:
        for server in self._servers:
            server.wait()

    def stop(self) -> None:
        for server in self._servers:
            server.stop()

    def __enter__(self) -> "ServerPool":
        self.start()
        self.wait_until_ready()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --- マルチモデルゲートウェイ（daemon）用ヘルパ -----------------------------
def ignore_shutdown_signals() -> None:
    """SIGTERM / SIGHUP / SIGINT を一旦無視（SIG_IGN）にする。

    後始末（配下のサーバー停止など）の最中に再度シグナルが届いても中断されないよう、
    クリーンアップ開始時に呼ぶ。`--stop` の killpg や端末クローズで複数シグナルが連続して
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

    `--status` から参照できるよう、ランダムな tempfile ではなくポートで決まる固定パスにする。
    プロジェクト内（`./.local-llm-server/`、カレントディレクトリ相対）に置き、ホーム等の外部には
    書かない。同じポートのサーバーは同じログに追記する。
    """
    return os.path.join(project_cache_dir(), f"server-{port}.log")


def server_status(host: str = "127.0.0.1", port: int = 8799) -> dict | None:
    """ポートで動いているローカルサーバーの状態をまとめて返す（`--status` 表示用）。

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
