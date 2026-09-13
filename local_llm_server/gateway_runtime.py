"""Gateway runtime files, ownership ledger, locking, admin client, and launcher."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .constants import log_dir
from .process_control import (
    POSIX,
    find_pids_on_port,
    pid_is_alive,
    pid_looks_like_gateway,
    pid_looks_like_ours,
    pid_matches_record,
    process_fingerprint,
    stop_process_tree,
    stop_pid,
)
from .server_health import is_ready, list_models


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

    ログ表示から参照できるよう、ランダムな tempfile ではなくポートで決まる固定パスにする。
    場所は `log_dir()`（cwd 非依存）——起動したディレクトリに `./.local-llm-server` を
    作らない。同じポートのサーバーは同じログに追記する。
    """
    return os.path.join(log_dir(), f"server-{port}.log")


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


def runtime_dir() -> str:
    """UIDごとに分離した0700のランタイムディレクトリを返す。"""
    suffix = (
        str(os.getuid())
        if hasattr(os, "getuid")
        else hashlib.sha256(getpass.getuser().encode("utf-8", "replace")).hexdigest()[
            :12
        ]
    )
    path = os.path.join(tempfile.gettempdir(), f"local-llm-server-{suffix}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    if os.name != "nt":
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OSError(f"runtime path is not a real directory: {path}")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise PermissionError(f"runtime directory is owned by another user: {path}")
        os.chmod(path, 0o700)
    return path


def gateway_lock_path() -> str:
    """マシン内で 1 つだけゲートウェイを許すロックファイルのパス（cwd 非依存の固定パス）。

    **どのディレクトリから起動しても同じ 1 個**のロックを見るよう、temp ディレクトリ配下の
    固定名にする（ログの `log_dir()` と同じく cwd 非依存）。これで
    「別ディレクトリから（開発ツール等が）勝手に 2 個目を起動する」ケースも 1 本に束ねられる。
    ポートに依存しないので、port を変えても二重には立たない（＝マシンにつき 1 ゲートウェイ）。
    """
    return os.path.join(runtime_dir(), "gateway.lock")


def gateway_runtime_path() -> str:
    """稼働中ゲートウェイの接続先（host/port/pid/cwd）を書く固定パス（cwd 非依存）。

    単一起動（`GatewayLock`）でマシンに 1 ゲートウェイなので、この 1 ファイルを読めば
    **どのディレクトリからでも**「いま動いているゲートウェイ」の host/port を特定できる
    （`gw status` / `gw stop` を gateway.toml の無い場所から打つため）。ロックの隣に置く。
    """
    return os.path.join(runtime_dir(), "gateway.json")


def write_gateway_runtime(
    host: str, port: int, pid: int, cwd: str, started_at: str
) -> None:
    """稼働中ゲートウェイの接続先をランタイム記録に書く（デーモンが起動時に呼ぶ）。

    書き込みは best-effort（失敗しても稼働は妨げない）。単一起動なので上書きで良い。
    """
    rec = {"host": host, "port": port, "pid": pid, "cwd": cwd, "started_at": started_at}
    fingerprint = process_fingerprint(pid)
    if fingerprint is not None:
        rec.update(fingerprint)
    try:
        path = gateway_runtime_path()
        _atomic_write_json(path, rec)
    except OSError:
        pass


def read_gateway_runtime() -> dict | None:
    """ランタイム記録を読む（無い・壊れている・PID が生きていなければ None）。

    クラッシュで残った stale 記録を掴まないよう、記録の PID が生存しているときだけ返す
    （PID 生存は下限の健全性チェック。最終的な疎通は呼び出し側が /admin/status で確認する）。
    """
    try:
        with open(gateway_runtime_path(), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    host, port, pid = rec.get("host"), rec.get("port"), rec.get("pid")
    if (
        not isinstance(host, str)
        or not host
        or isinstance(port, bool)
        or not isinstance(port, int)
        or not (1 <= port <= 65535)
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
    ):
        return None
    if not pid_is_alive(pid):
        return None  # クラッシュで残った stale 記録（保持者が居ない）
    if not pid_matches_record(pid, rec):
        return None  # PID が別プロセスへ再利用された
    return rec


def clear_gateway_runtime() -> None:
    """ランタイム記録を消す（デーモンの正常終了時に呼ぶ。best-effort）。"""
    try:
        os.remove(gateway_runtime_path())
    except OSError:
        pass


# --- ワーカー台帳と孤児掃除（startup reconciliation） --------------------------
def workers_state_path() -> str:
    """起動中モデルサーバー（ワーカー）の PID 台帳のパス（cwd 非依存の固定パス）。

    LocalServer が起動時に {pid, port, model} を記録し、正常停止時に消す。デーモンが
    `kill -9` 等で死ぬと消されないまま残り、それが**次回起動時の孤児掃除
    （reap_orphan_workers）の手掛かり**になる。ロック・ランタイム記録の隣に置く。
    """
    return os.path.join(runtime_dir(), "workers.json")


_WORKERS_FILE_LOCK = threading.Lock()  # 複数スレッドの同時ロード/退避から台帳を守る


def _atomic_write_json(path: str, value: object) -> None:
    """0600の一意な一時ファイルから置換し、symlink追従と途中書きを防ぐ。"""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1
            json.dump(value, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _load_workers_unlocked() -> list[dict]:
    """台帳を読む（無い・壊れているは空扱い。_WORKERS_FILE_LOCK 保持下で呼ぶ）。"""
    try:
        with open(workers_state_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    entries = data.get("workers") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _save_workers_unlocked(entries: list[dict]) -> None:
    """台帳を書く（tmp → rename の原子的置換。best-effort・失敗しても稼働は妨げない）。"""
    path = workers_state_path()
    try:
        _atomic_write_json(path, {"workers": entries})
    except OSError:
        pass


def register_worker(pid: int, port: int, model: str) -> None:
    """起動したワーカーを台帳に記録する（LocalServer.start が呼ぶ）。"""
    with _WORKERS_FILE_LOCK:
        entries = [e for e in _load_workers_unlocked() if e.get("pid") != pid]
        record = {"pid": pid, "port": port, "model": model}
        fingerprint = process_fingerprint(pid)
        if fingerprint is not None:
            record.update(fingerprint)
        entries.append(record)
        _save_workers_unlocked(entries)


def unregister_worker(pid: int) -> None:
    """停止したワーカーを台帳から消す（LocalServer.stop が呼ぶ）。"""
    with _WORKERS_FILE_LOCK:
        _save_workers_unlocked(
            [e for e in _load_workers_unlocked() if e.get("pid") != pid]
        )


def owned_worker_pids_on_ports(ports: list[int]) -> list[int]:
    """安全な台帳指紋とLISTENポートの両方が一致するワーカーPIDだけを返す。"""
    wanted = set(ports)
    with _WORKERS_FILE_LOCK:
        entries = list(_load_workers_unlocked())
    listeners = {port: set(find_pids_on_port(port)) for port in wanted}
    return [
        e["pid"]
        for e in entries
        if (
            isinstance(e.get("pid"), int)
            and e.get("port") in wanted
            and e["pid"] in listeners.get(e["port"], set())
            and pid_matches_record(e["pid"], e)
            and pid_looks_like_ours(e["pid"])
        )
    ]


def worker_pid_is_owned(pid: int) -> bool:
    """PIDが安全なワーカー台帳の現在の指紋と一致するか。"""
    with _WORKERS_FILE_LOCK:
        record = next(
            (e for e in _load_workers_unlocked() if e.get("pid") == pid), None
        )
    return bool(
        record is not None
        and pid_matches_record(pid, record)
        and pid_looks_like_ours(pid)
    )


def reap_orphan_workers() -> list[int]:
    """前回のゲートウェイが残した孤児ワーカーを回収する（デーモン起動時に呼ぶ）。

    crash-only 設計の要: **起動処理 = 復旧処理**。前回が `kill -9` やクラッシュで死ぬと
    台帳が残るので、記録された PID のうち「まだ生きていて、かつこのパッケージ由来に見える
    （pid_looks_like_ours）」ものだけをプロセスグループごと止める。無関係なプロセス
    （PID 再利用等）には手を出さない。無関係・死亡済み記録は捨て、停止に失敗した
    所有ワーカーの記録だけは次回復旧のため残す。停止した PID の一覧を返す。
    ポート単位の後追い回収（reclaim_stale_workers）はこれの backstop として残る。
    """
    with _WORKERS_FILE_LOCK:
        entries = _load_workers_unlocked()
    victims = [
        e["pid"]
        for e in entries
        if (
            isinstance(e.get("pid"), int)
            and pid_matches_record(e["pid"], e)
            and pid_looks_like_ours(e["pid"])
        )
    ]
    # stop_pid は 1 件あたり最長 ~10s 待つため並列に止める。
    outcomes: dict[int, bool] = {}
    outcomes_lock = threading.Lock()

    def _stop_one(pid: int) -> None:
        try:
            stopped = stop_pid(pid) or not pid_is_alive(pid)
        except Exception:  # noqa: BLE001 - 失敗は台帳を残して次回に回す
            stopped = False
        with outcomes_lock:
            outcomes[pid] = stopped

    threads = [threading.Thread(target=_stop_one, args=(pid,)) for pid in victims]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    reaped = [pid for pid in victims if outcomes.get(pid, False)]
    failed = set(victims) - set(reaped)
    with _WORKERS_FILE_LOCK:
        _save_workers_unlocked([e for e in entries if e.get("pid") in failed])
    return reaped


# --- 子プロセスの親への繋留（tether） -----------------------------------------
# デーモンだけが書き込み端を握るパイプ。ワーカーは読み取り端を継承し、EOF（＝デーモンの
# 死。kill -9 でも OS が確実に閉じる）を検知したら自分のグループごと終了する（tether.py）。
_TETHER_READ_FD: int | None = None
_TETHER_WRITE_FD: int | None = None


def enable_child_tethering() -> None:
    """以後の LocalServer.start が起動するワーカーをこのプロセスへ繋留する（POSIX のみ）。

    デーモン（run_gateway）が起動時に 1 回呼ぶ。書き込み端はこのプロセスが生きている間
    ずっと握り続ける——閉じることが「死の通知」なので、明示的な close はどこにも要らない
    （プロセス終了時に OS が閉じる。os.pipe は CLOEXEC なので手動更新の execv でも閉じるが、
    その時点でワーカーは全て停止済み）。Windows は対象外（0a の起動時掃除が受け皿）。
    """
    global _TETHER_READ_FD, _TETHER_WRITE_FD
    if not POSIX or _TETHER_READ_FD is not None:
        return
    _TETHER_READ_FD, _TETHER_WRITE_FD = os.pipe()


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
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self._path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise OSError(f"gateway lock is not a regular file: {self._path}")
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
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
_LOCK_SENTINEL_OFFSET = (
    1 << 30
)  # 1 GiB 目。PID 文字列（先頭数バイト）と絶対に重ならない
if os.name == "nt":  # pragma: no cover - Windows 専用パス
    import msvcrt

    def _flock_exclusive_nb(fd: int) -> None:
        os.lseek(fd, _LOCK_SENTINEL_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]

    def _flock_unlock(fd: int) -> None:
        os.lseek(fd, _LOCK_SENTINEL_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
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


def _admin_request(
    path: str,
    host: str,
    port: int,
    timeout: float,
    body: dict | None = None,
) -> dict | None:
    """ゲートウェイの /admin/* を叩く共通口（body があれば POST、無ければ GET）。

    gateway_admin_status / gateway_set_max_resident / gateway_drain が共有する。
    応答しない・非 200・JSON でない場合はすべて None（呼び出し側は「未起動 or 旧版」として
    フォールバックする）——エンドポイントごとに例外の意味を変えないことで、呼び出し側の
    分岐を「dict か None か」だけにしている。
    """
    url = f"http://{host}:{port}{path}"
    if body is None:
        req: urllib.request.Request | str = url
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def gateway_admin_status(
    host: str = "127.0.0.1",
    port: int = 8799,
    timeout: float = 2.0,
    *,
    refresh_updates: bool = False,
) -> dict | None:
    """ゲートウェイの GET /admin/status を取得する（常駐モデルのライブ状態＋運用方針）。

    server_status と違い、各モデルの loaded / inflight（処理中数）や max_resident /
    idle_timeout までゲートウェイ本体から取得できる（→ TUI 監視用）。応答しない・旧版で
    エンドポイントが無い場合は None を返す（呼び出し側は server_status にフォールバックできる）。

    refresh_updates=True は、更新確認も済ませた状態を同じ応答で返すトレイ向けの指定。
    トレイ自身がこの呼び出しを裏スレッドで行うためメニュー表示はブロックしない。一方、通常の
    TUI/CLI 監視は従来どおり即時応答のままにし、更新確認はサーバー側の裏処理に任せる。
    """
    path = "/admin/status?refresh_updates=1" if refresh_updates else "/admin/status"
    return _admin_request(path, host, port, timeout)


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
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Write a short story about the sea."}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body, headers=headers
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"bench request failed: {exc}") from exc
    seconds = time.monotonic() - t0
    tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
    tps = tokens / seconds if seconds > 0 else 0.0
    return {
        "model": model,
        "tokens": tokens,
        "seconds": round(seconds, 2),
        "tok_per_s": round(tps, 1),
    }


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
    return _admin_request(
        "/admin/config", host, port, timeout, body={"max_resident": value}
    )


def gateway_drain(
    enable: bool = True,
    host: str = "127.0.0.1",
    port: int = 8799,
    timeout: float = 5.0,
) -> dict | None:
    """稼働中のゲートウェイに POST /admin/drain で再起動準備を要求する。

    enable=True: ゲートウェイが原子的に「処理中 0・在席 0」を確認し、満たせば新規受付を
    止めて {"draining": True} を返す。busy なら {"draining": False, "inflight": n,
    "sessions": n}（何も変えない）。enable=False で解除。応答しない（未起動・旧版で
    エンドポイントが無い）ときは None。
    """
    return _admin_request("/admin/drain", host, port, timeout, body={"enable": enable})


def gateway_log_path(port: int) -> str:
    """バックグラウンド起動したゲートウェイ本体（公開ポート）の出力ログ保存先。

    モデルサーバーの daemon_log_path（server-<port>.log）と別に、ゲートウェイ自身の
    起動ログを gateway-<port>.log に逃がす（`gw log` が参照する）。場所は `log_dir()` の
    固定パス —— cwd 非依存なので、CLI がどこから実行されてもデーモンが書く場所と一致する。
    """
    return os.path.join(log_dir(), f"gateway-{port}.log")


# ログを残す世代数（Ollama の LogRotationCount と同じ 5）。
LOG_ROTATION_COUNT = 5


def rotate_log(path: str, keep: int = LOG_ROTATION_COUNT) -> None:
    """`x.log` を `x-1.log` へ押し出し、keep 世代を超えた分を捨てる（Ollama 方式）。

    追記のみだと際限なく伸びるので、**起動のたびに**世代を繰り上げる。サイズ監視はしない
    ——書かれるのは起動・停止・モデルのロード/破棄といったイベント行だけで、リクエスト
    ごとには書かないため、世代数の上限だけで十分に頭打ちになる。
    ローテーションに失敗しても起動は止めない（そのまま追記に落ちるだけ）。
    """
    if keep < 1 or not os.path.exists(path):
        return
    root, ext = os.path.splitext(path)
    try:
        oldest = f"{root}-{keep}{ext}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for i in range(keep - 1, 0, -1):
            src = f"{root}-{i}{ext}"
            if os.path.exists(src):
                os.replace(src, f"{root}-{i + 1}{ext}")
        os.replace(path, f"{root}-1{ext}")
    except OSError:
        pass


def prune_server_logs(
    directory: str | None = None, keep: int = LOG_ROTATION_COUNT
) -> None:
    """古い `server-<port>.log` を新しい順に keep 個だけ残して削除する。

    モデルサーバーのログはポート番号がファイル名に入る（`daemon_log_path`）ので、
    ゲートウェイ本体のように世代を押し出せない——ポートが変わるたびに**別ファイルが増える**。
    そこで世代管理ではなく「新しい順に keep 個」で頭打ちにする。呼ぶのは**ゲートウェイ起動時
    だけ**：この時点ではモデルサーバーは 1 つも走っておらず、書き込み中のログを消す危険がない。
    """
    directory = directory or log_dir()
    try:
        logs = [
            os.path.join(directory, n)
            for n in os.listdir(directory)
            if n.startswith("server-") and n.endswith(".log")
        ]
        # mtime の新しい順。keep 個より後ろ（古い方）を落とす。
        for stale in sorted(logs, key=os.path.getmtime, reverse=True)[keep:]:
            os.remove(stale)
    except OSError:
        pass


def start_gateway_background(
    cwd: str,
    host: str = "127.0.0.1",
    port: int = 8799,
    *,
    start_timeout: float = 120.0,
    _find_pids=find_pids_on_port,
    _admin_status=gateway_admin_status,
    _connect_host=local_connect_host,
    _looks_like_gateway=pid_looks_like_gateway,
    _log_path=gateway_log_path,
    _rotate=rotate_log,
    _prune=prune_server_logs,
    _ready=is_ready,
    _pid_alive=pid_is_alive,
    _stop_spawned=stop_process_tree,
    _sleep=time.sleep,
    _monotonic=time.monotonic,
) -> int:
    """ゲートウェイをデタッチした別プロセスで常駐起動し、応答可能になるまで待つ。

    ターミナルを占有しない常駐起動（Ollama 流）。cwd の ./gateway.toml を読む
    ヘッドレスワーカー（`python -m local_llm_server` = __main__）を、新セッション（POSIX）/
    DETACHED_PROCESS（Windows）で起動して端末・親から切り離し、出力は gateway_log_path に
    逃がす。応答可能になったら PID を返す。既に起動済みなら何もせず既存 PID（不明なら 0）を返す。
    起動失敗は RuntimeError、時間内に応答しなければ TimeoutError。
    """
    base_url = f"http://{host}:{port}/v1"
    existing = _find_pids(port)
    admin = _admin_status(_connect_host(host), port)
    if (
        admin
        and isinstance(admin.get("pid"), int)
        and _looks_like_gateway(admin["pid"])
    ):
        return admin["pid"]
    if existing:
        # ポートは埋まっているのにまだ管理 API が応答しない。自分由来なら起動途中の
        # 可能性があるため待つが、応答を一度も確認せず成功扱いにはしない。ここで見つけた
        # プロセスはこの呼び出しが生成したものではないので、失敗しても勝手に停止しない。
        gateways = [p for p in existing if _looks_like_gateway(p)]
        if gateways:
            deadline = _monotonic() + start_timeout
            while _monotonic() < deadline:
                status = _admin_status(_connect_host(host), port)
                status_pid = status.get("pid") if status else None
                if isinstance(status_pid, int) and _looks_like_gateway(status_pid):
                    return status_pid
                if not any(_pid_alive(pid) for pid in gateways):
                    raise RuntimeError(
                        f"existing gateway exited before becoming ready on port {port}; "
                        f"see {_log_path(port)}"
                    )
                _sleep(0.5)
            raise TimeoutError(
                f"existing gateway pid {gateways} did not become ready within "
                f"{start_timeout:g}s on port {port}; see {_log_path(port)}"
            )
        raise RuntimeError(
            f"port {port} is in use by an unrelated process (pid {existing}) that does not "
            f"respond as a gateway; stop it or change `port` in gateway.toml"
        )

    # ログは log_dir() の固定パス（cwd 非依存）。デーモンの cwd（設定のある場所）が
    # どこでも同じ場所に書く——起動したディレクトリに ./.local-llm-server を作らない。
    log_path = _log_path(port)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    # 起動のたびに世代を繰り上げ、併せて古いモデルサーバーのログも刈る（ここが唯一の
    # 「モデルサーバーが 1 つも走っていない」と言える地点）。
    _rotate(log_path)
    _prune()
    log_file = open(log_path, "a", encoding="utf-8")
    popen_kwargs: dict = {
        "cwd": cwd,
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        # 正規 spawn の内部マーク。__main__ はこれが無い直接の `python -m local_llm_server`
        # を拒否する（起動の入口を `gw start` の 1 本に固定する）。
        "env": {**os.environ, "LOCAL_LLM_GW_LAUNCHER": "cli"},
    }
    if os.name == "nt":
        # 端末から切り離し、新プロセスグループにする（stop の taskkill /T と対）。
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
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
    try:
        deadline = _monotonic() + start_timeout
        while _monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"gateway exited early (code {proc.returncode}); see {log_path}"
                )
            status = _admin_status(_connect_host(host), port)
            status_pid = status.get("pid") if status else None
            if status_pid == proc.pid or _ready(base_url):
                return proc.pid
            _sleep(0.5)
        raise TimeoutError(
            f"gateway pid {proc.pid} not ready within {start_timeout:g}s; "
            f"see {log_path}"
        )
    except BaseException:
        # This call owns only the process it just spawned.  Reap it on timeout,
        # early exit, KeyboardInterrupt, and every other failed startup path.
        stopped = _stop_spawned(proc, grace=0.0, kill_timeout=5.0)
        if not stopped:
            print(
                f"warning: gateway pid {proc.pid} could not be confirmed stopped; "
                "do not discard the original startup error and inspect the process manually",
                file=sys.stderr,
            )
        raise
