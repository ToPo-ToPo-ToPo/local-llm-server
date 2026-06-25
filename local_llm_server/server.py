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


def _hf_hub_cache() -> str:
    """HuggingFace Hub のキャッシュ（models--org--name/snapshots/...）ルートを返す。"""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(home, "hub")


def resolve_gguf(model: str) -> str:
    """llama.cpp の `model` を実ファイルパスに解決する。

    - 既存のローカルパスならそのまま返す。
    - `org/repo` 形式なら HF キャッシュ（既にDL済みの GGUF）から該当ファイルを探して返す。
      `-hf` の自動DLには依存しない（トークン不要・401 回避）。
    - `org/repo:selector` でファイル名の一部（量子化名や `F16-MTP` 等）を指定できる。
      セレクタ無しのときは mmproj と MTP ヘッドを除いた「本体」GGUF を選ぶ（1つに定まらなければ
      候補を挙げて ValueError）。キャッシュに無い／repo 形式でなければ入力をそのまま返す
      （呼び出し側で「実行ファイル/モデルが見つからない」と分かるエラーになる）。
    """
    if os.path.exists(model):
        return model
    repo, sep, selector = model.partition(":")
    if "/" not in repo:
        return model
    org, name = repo.split("/", 1)
    cache_dir = os.path.join(
        _hf_hub_cache(), f"models--{org}--{name.replace('/', '--')}", "snapshots"
    )
    if not os.path.isdir(cache_dir):
        return model
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
        return model
    if len(pool) > 1:
        names = sorted(os.path.basename(g) for g in pool)
        raise ValueError(
            f"'{model}' に複数の GGUF が該当します {names}。"
            f"'{repo}:<量子化名など>' でファイルを 1 つに絞ってください。"
        )
    return pool[0]


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
        # model は実パスでも HF repo-id（org/repo[:selector]）でも可。後者は DL 済み
        # キャッシュから実 GGUF を解決する（クライアントに見せる ID は repo-id のまま）。
        model_path = resolve_gguf(config.model)
        command = [
            "llama-server",
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
            # 別ヘッド方式の投機的デコード（gemma4 等）。draft_model にドラフト GGUF のパス
            # または HF repo-id（org/repo:F16-MTP 等）を指定すると有効化（-md）。ファイル名に
            # "mtp" を含めば MTP ヘッドとみなし --spec-type draft-mtp を付ける（それ以外は
            # llama.cpp 既定の draft-simple）。
            if config.draft_model and "-md" not in config.extra_args:
                draft_path = resolve_gguf(config.draft_model)
                command += ["-md", draft_path]
                if "mtp" in os.path.basename(draft_path).lower():
                    command += ["--spec-type", "draft-mtp"]
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


def gateway_admin_status(
    host: str = "127.0.0.1", port: int = 8799, timeout: float = 2.0
) -> dict | None:
    """ゲートウェイの GET /admin/status を取得する（常駐モデルのライブ状態＋運用方針）。

    server_status と違い、各モデルの loaded / inflight（処理中数）や max_resident /
    idle_timeout までゲートウェイ本体から取得できる（→ GUI 監視用）。応答しない・旧版で
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


def gateway_log_path(port: int) -> str:
    """バックグラウンド起動したゲートウェイ本体（公開ポート）の出力ログ保存先。

    モデルサーバーの daemon_log_path（server-<port>.log）と別に、ゲートウェイ自身の
    起動ログを gateway-<port>.log に逃がす（`--status` / GUI から参照できる固定パス）。
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
    フォアグラウンド版（`python -m local_llm_server.cli`）を、新セッション（POSIX）/
    DETACHED_PROCESS（Windows）で起動して端末・親から切り離し、出力は gateway_log_path に
    逃がす。応答可能になったら PID を返す。既に起動済みなら何もせず既存 PID（不明なら 0）を返す。
    起動失敗は RuntimeError、時間内に応答しなければ TimeoutError。
    """
    base_url = f"http://{host}:{port}/v1"
    existing = find_pids_on_port(port)
    if is_ready(base_url) or existing:
        return existing[0] if existing else 0

    log_path = gateway_log_path(port)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    popen_kwargs: dict = {
        "cwd": cwd,
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        # 端末から切り離し、新プロセスグループにする（stop の taskkill /T と対）。
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True  # setsid: 端末/親から独立
    try:
        # --headless 必須: 裏起動は出力をログへ逃がす非 TTY なので TUI を出さずゲートウェイ本体を回す。
        proc = subprocess.Popen(
            [sys.executable, "-m", "local_llm_server.cli", "--headless"], **popen_kwargs
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
