"""`gw` コマンド — ゲートウェイを裏で常駐させ、CLI サブコマンドで運用する（Ollama 流）。

デーモン本体（`python -m local_llm_server` = __main__）は端末を持たず裏で常駐する。この
モジュールはそのデーモンを起動/停止/監視する **薄い CLI** で、状態は `GET /admin/status`
を叩いて取得し、停止は同パッケージ由来のプロセスだけを止める。単一起動は従来どおりデーモンが
握る `GatewayLock`（マシンに 1 ゲートウェイ）が保証するので、この CLI をいくつ起動しても
デーモンは 0 個か 1 個のまま。

サブコマンド:
  - `gw start`          … デーモンを裏で常駐起動（既に居れば何もしない）
  - `gw stop`           … このパッケージ由来のゲートウェイ/モデルサーバーを停止
  - `gw restart`        … stop → start
  - `gw serve`          … フォアグラウンドで実行（サービスの実行口。Ollama の serve 相当）
  - `gw enable`         … ログイン時自動起動＋異常終了時自動復活を OS に登録
  - `gw disable`        … 自動起動を解除して手動運用（gw start）に戻す
  - `gw status`         … 稼働/停止・PID・URL・起動経過・累計リクエストを 1 行表示
  - `gw ps`             … ロード中モデルの状態（処理中数・在席・アイドル残り）を表示
  - `gw list`           … 使えるモデル（カタログ＋HF キャッシュ）を一覧
  - `gw pull <model>`   … モデルを事前ダウンロード（進捗付き。GGUF は本体+mmproj のみ）
  - `gw rm <model>`     … モデルを HF キャッシュから削除（確認あり・ロード中は拒否）
  - `gw show <model>`   … モデルの素性（量子化・コンテキスト長・サイズ・MTP）を表示
  - `gw log [-f]`       … ゲートウェイログの末尾を表示（-f で追従）
  - `gw max <n|off>`    … max_resident を無停止で変更
  - `gw mtp [model]`    … MTP ドラフターの取得状況を確認（ダウンロードはしない）
  - `gw update`         … PyPI 新版があれば git pull で追従してデーモンを再起動
  - 引数なし `gw`       … start してから status/ps を表示（従来の `uv run gw` 相当）

設定は **カレントディレクトリの `./gateway.toml` のみ**を見る（場所は CWD 固定の 1 ルール）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib

from .daemon import load_gateway_config
from .server import (
    MTP_DRAFTERS,
    _hf_hub_cache,
    discover_cached_models,
    ensure_cached,
    find_pids_on_port,
    gateway_admin_status,
    gateway_log_path,
    gateway_set_max_resident,
    infer_backend,
    install_shutdown_handlers,
    is_ready,
    local_connect_host,
    mtp_status,
    pid_looks_like_ours,
    primary_lan_ip,
    read_gateway_runtime,
    resolve_gguf,
    start_gateway_background,
    stop_pid,
)


# --- 設定解決（1 ライブラリ 1 使い方: 設定は user_config_path の 1 箇所だけ） -----
def resolve_config() -> str | None:
    """デーモン本体（__main__）が読む gateway.toml。**起動された cwd の `./gateway.toml` のみ**。

    `gw start` は設定ディレクトリ（user_config_path の親）を cwd にしてデーモンを spawn する
    ので、デーモンから見れば常に `./gateway.toml`。存在すればそのパス、無ければ None。
    """
    path = os.path.join(os.getcwd(), "gateway.toml")
    return path if os.path.isfile(path) else None


def user_config_path() -> str:
    """gateway.toml の置き場所（`~/.config/local-llm-server/gateway.toml`）。**ここだけ**。

    Ollama 流に「設定の場所はユーザーが選ばない」——どこから `gw` を打っても常にこの 1 ファイル
    を使う（`XDG_CONFIG_HOME` があれば尊重）。無ければ初回の `gw start` が自動生成する
    （→ ensure_user_config）。ファイルの有無は問わずパスだけ返す。
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "local-llm-server", "gateway.toml")


_DEFAULT_CONFIG = """\
# local-llm-server の設定（`gw start` が初回に自動生成。編集したら保存するだけで
# ポリシー設定は稼働中でも反映される → docs/gateway.md）
host = "127.0.0.1"          # 別 PC から繋ぐなら "0.0.0.0"（api_key の設定を推奨）
port = 8799                 # クライアントの base_url はここ（http://127.0.0.1:8799/v1）
max_resident = 1            # 同時常駐モデル数の上限（超過は LRU 退避）
# モデルは事前登録不要。クライアントが指定した model をその場でロードする。
"""


def ensure_user_config() -> str:
    """設定ファイルを用意して、そのパスを返す（初回 `gw start` 用の自動生成つき）。

    無ければ自動生成する: editable インストール元のクローンに例（gateway.toml）があれば
    それを**1 回だけ複製**し、無ければ最小の既定を書く。以降の編集・参照は常に
    user_config_path の 1 ファイルだけ（クローン側の例は二度と読まない —— 読む場所を
    2 つにしない。リポジトリを汚さないので自動更新のクリーン判定も妨げない）。
    """
    path = user_config_path()
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seed = None
    try:
        from . import update
        root = update.repo_root()
        if root is not None:
            example = os.path.join(str(root), "gateway.toml")
            if os.path.isfile(example):
                with open(example, encoding="utf-8") as fh:
                    seed = fh.read()
    except Exception:  # noqa: BLE001 - 例の複製は best-effort（失敗は既定で生成）
        seed = None
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(seed if seed is not None else _DEFAULT_CONFIG)
    print(f"created {path}", file=sys.stderr)
    return path


class _RuntimeConfig:
    """ランタイム記録（稼働中デーモンの host/port）から作る最小の設定シム。

    `./gateway.toml` の無いディレクトリから `gw status` / `gw stop` 等を打ったとき用。
    事前登録カタログ（`models`）は記録に無いので空。表示に必要な値は `/admin/status` の
    ライブ状態から得る（`merge_status` は admin があればそちらを優先する）。
    """

    def __init__(self, rec: dict) -> None:
        self.host = rec.get("host", "127.0.0.1")
        self.port = int(rec.get("port", 8799))
        self.models: list = []          # 事前登録は記録に無い（動的ロード分は admin から見える）
        self.idle_timeout = 0           # アイドル残りは記録だけでは出せない（admin にも idle_for はある）
        self.max_resident = None        # 実値は admin の max_resident を使う
        # デーモンの作業ディレクトリ（＝設定 ./gateway.toml の基準）。記録の cwd から引く
        # （restart で同じ設定の場所から再起動するため。ログは log_dir() 固定で cwd 非依存）。
        self._config_dir = rec.get("cwd")


def load_config(path: str):
    """指定パスの gateway.toml を読む（壊れていれば例外を送出）。読んだ設定に
    `_config_dir`（そのファイルのあるディレクトリ）を付けて返す —— `gw start` はそこを
    デーモンの cwd にして spawn する（デーモンから見て設定が常に `./gateway.toml` になる）。"""
    gcfg = load_gateway_config(path)
    try:
        gcfg._config_dir = os.path.dirname(os.path.abspath(path))
    except Exception:  # noqa: BLE001 - 付与できなくても致命ではない（cwd フォールバック）
        pass
    return gcfg


# --- 純データ層（状態のマージ・整形。端末なしでテストできる） --------------------
def merge_status(gcfg, admin: dict | None, ready: bool | None = None) -> dict:
    """gateway.toml のカタログと `/admin/status` のライブ状態を1つのビューに統合する（純粋関数）。

    カタログの全モデルを並べ（未起動も「unloaded」で見せる）、起動中のものはライブ状態
    （loaded/idle/busy・処理中数・累計・アイドル自動解放までの残り）を重ねる。描画を含まない
    のでそのままテストできる（`gw ps` / `gw list` の描画関数がこれを使う）。
    """
    live = {m["model"]: m for m in (admin or {}).get("models", [])}
    idle_timeout = gcfg.idle_timeout

    def _row(model, backend, port, m):
        """ライブ状態 m（None=未ロード）から表示用の 1 行を作る。"""
        # MTP（高速化）の利用可否は本体名から判定する（ドラフターがキャッシュ済みなら "ready"）。
        mtp = mtp_status(model)
        if not m or not m.get("loaded"):
            return {
                "model": model, "backend": backend, "port": port,
                "state": "unloaded", "inflight": 0, "instances": 0,
                "requests": (m or {}).get("requests", 0), "idle_remaining": None,
                "sessions": (m or {}).get("sessions", 0), "mtp": mtp,
            }
        inflight = int(m.get("inflight", 0))
        idle_for = m.get("idle_for")
        if inflight > 0:
            state, remaining = "busy", None
        else:
            state = "idle"
            remaining = (
                max(0.0, idle_timeout - idle_for)
                if (idle_timeout and idle_for is not None) else None
            )
        return {
            "model": model, "backend": backend, "port": port,
            "state": state, "inflight": inflight,
            # 起動中インスタンス数（負荷ベースの複製で >1 になる。並列度の目安）。
            "instances": int(m.get("instances", 1)),
            "requests": int(m.get("requests", 0)), "idle_remaining": remaining,
            "sessions": int(m.get("sessions", 0)),  # 在席エージェント数（0 で即アンロード対象）
            "mtp": mtp,
        }

    rows = []
    listed = set()
    # 事前登録モデル（未ロードでも unloaded で見せる）。
    for c in gcfg.models:
        listed.add(c.model)
        rows.append(_row(c.model, c.backend, c.port, live.get(c.model)))
    # 動的ロードされたモデル（事前登録に無い、現在管理中のものを追加表示）。
    for model, m in live.items():
        if model not in listed:
            listed.add(model)
            rows.append(_row(model, m.get("backend", "?"), m.get("port"), m))
    # キャッシュにある DL 済みモデル（まだロードしていない候補。LM Studio 風に「選べる一覧」）。
    for d in (admin or {}).get("available", []):
        mid = d.get("id")
        if mid and mid not in listed:
            listed.add(mid)
            rows.append(_row(mid, d.get("backend", "?"), None, None))
    if ready is None:
        ready = bool(admin)
    # max_resident は実行中に変更できる（POST /admin/config）。ライブ値（admin）があれば
    # それを優先し、無ければ gateway.toml の起動時値にフォールバックする。admin では None が
    # 「無制限」を意味するので、キーが在ればその値（None 含む）をそのまま使う。
    live_max = (admin or {}).get("max_resident", gcfg.max_resident) if admin else gcfg.max_resident
    return {
        "ready": ready,
        "uptime": (admin or {}).get("uptime"),
        "requests": (admin or {}).get("requests", sum(r["requests"] for r in rows)),
        "max_resident": live_max,
        "idle_timeout": idle_timeout,
        "pid": (admin or {}).get("pid"),
        "started_at": (admin or {}).get("started_at"),
        "models": rows,
    }


def mtp_report(model: str | None) -> tuple[str, int]:
    """使う予定のモデルに必要な MTP ドラフターを、ダウンロード前に調べて文面にする。

    対応表（MTP_DRAFTERS）の辞書引きとローカルキャッシュ確認だけで、モデルのダウンロードは
    一切しない（非破壊）。gateway.toml も見ないので、どのディレクトリからでも実行できる。model を
    省略（None/空）すると対応表を全件、取得状況つきで並べる。

    戻り値: (表示テキスト, 終了コード)。指定モデルが MTP 非対応なら 1、
    それ以外（ready / available / 一覧表示）は 0。
    """

    def _describe(target: str) -> str:
        drafter = MTP_DRAFTERS[target]
        # mtp_status は "ready"（ドラフター取得済み）/ "available"（未取得）を返す。DL はしない。
        if mtp_status(target) == "ready":
            return (
                f"{target}\n"
                f"    drafter: {drafter}  [ready — 取得済み。そのまま MTP が効く]"
            )
        return (
            f"{target}\n"
            f"    drafter: {drafter}  [available — 未取得]\n"
            f"    hf download {drafter}"
        )

    if not model:
        lines = ['MTP 対応モデル（mlx-vlm・draft_model="auto" で自動解決）:']
        lines.extend(f"  {_describe(target)}" for target in sorted(MTP_DRAFTERS))
        return "\n".join(lines), 0

    if model not in MTP_DRAFTERS:
        return (
            f"{model}: MTP 非対応（対応表に無い）。使うなら gateway.toml の draft_model に "
            "ドラフターの HF id を明示してください。対応モデル一覧は引数なしの "
            "`mtp` コマンドで表示。",
            1,
        )
    return _describe(model), 0


def _fmt_hms(seconds) -> str:
    """秒を H:MM:SS / M:SS に整形する（None は「—」）。"""
    if seconds is None:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def read_log_tail(port: int, max_lines: int = 1000, max_bytes: int = 512 * 1024) -> str:
    """ゲートウェイログの末尾（最大 max_lines 行）を返す（`gw log` 表示用）。

    ログは `log_dir()` の固定パス（cwd 非依存）なので、CLI がどこから実行されても
    デーモンが実際に書いている場所を読める。末尾 max_bytes だけ読む（全読みを避ける）。
    ログがまだ無い／空のときは案内文を返す。
    """
    path = gateway_log_path(port)
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)  # 末尾へ
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return f"(ログはまだありません: {path})"
    if not data:
        return f"(ログは空です: {path})"
    lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
    if size > max_bytes and lines:
        lines = lines[1:]  # 途中から読んだ先頭の欠け行は捨てる
    return "".join(lines[-max_lines:])


# --- 表示（プレーンテキスト。textual に依存しない） ---------------------------
def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """左寄せの等幅テーブルを文字列にする（最後の列は末尾空白を付けない）。"""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: list[str]) -> str:
        parts = [cells[i].ljust(widths[i]) for i in range(len(cells) - 1)]
        parts.append(cells[-1])  # 末尾列はパディングしない
        return "  ".join(parts).rstrip()

    out = [_line(headers)]
    out.extend(_line(r) for r in rows)
    return "\n".join(out)


def _client_url(gcfg) -> str:
    """クライアントが指す base_url（公開なら LAN IP を優先案内）。"""
    if gcfg.host in ("0.0.0.0", "::", "", "*"):
        lan = primary_lan_ip()
        if lan:
            return f"http://{lan}:{gcfg.port}/v1"
    return f"http://{local_connect_host(gcfg.host)}:{gcfg.port}/v1"


def render_status(gcfg, admin: dict | None, ready: bool) -> str:
    """稼働/停止を 1〜数行で要約する（`gw status`）。"""
    url = _client_url(gcfg)
    if not ready:
        return f"gateway: stopped  ({url})\n  start it with `gw start`"
    view = merge_status(gcfg, admin, ready)
    loaded = sum(1 for m in view["models"] if m["state"] in ("idle", "busy"))
    busy = sum(1 for m in view["models"] if m["state"] == "busy")
    cap = "∞" if view["max_resident"] is None else str(view["max_resident"])
    parts = [
        "gateway: running",
        f"url {url}",
        f"pid {view['pid']}" if view.get("pid") else None,
        f"up {_fmt_hms(view['uptime'])}",
        f"requests {view['requests']}",
        f"loaded {loaded}/{cap}" + (f" ({busy} busy)" if busy else ""),
    ]
    line = "  ".join(p for p in parts if p)
    return line


def render_ps(gcfg, admin: dict | None, ready: bool) -> str:
    """ロード中モデルの状態を表にする（`gw ps`）。未ロードは出さない。"""
    if not ready:
        return "gateway: stopped  (start it with `gw start`)"
    view = merge_status(gcfg, admin, ready)
    live = [m for m in view["models"] if m["state"] in ("idle", "busy")]
    if not live:
        return "no models loaded"
    rows = []
    for m in live:
        idle = _fmt_hms(m["idle_remaining"]) if m["state"] == "idle" else "—"
        rows.append([
            m["model"], m["backend"], m["state"],
            str(m["inflight"]), str(m["instances"]),
            str(m["sessions"]), idle, str(m["requests"]),
        ])
    return _fmt_table(
        ["MODEL", "BACKEND", "STATE", "INFLIGHT", "INSTANCES", "SESSIONS", "IDLE-LEFT", "REQUESTS"],
        rows,
    )


def render_list(gcfg, admin: dict | None) -> str:
    """使えるモデル（カタログ＋動的ロード＋HF キャッシュ）を一覧する（`gw list`）。

    デーモンが稼働していれば /admin/status の available を使い、停止中はローカルの
    HF キャッシュ走査（discover_cached_models）にフォールバックして同じ一覧を出す。
    """
    if admin is None:
        # 停止中でも使える一覧を出す（カタログ＋ローカルキャッシュ）。
        admin = {"available": discover_cached_models()}
    view = merge_status(gcfg, admin, ready=bool(admin.get("models") is not None))
    if not view["models"]:
        return "(no models registered and none cached)"
    rows = [[m["model"], m["backend"], m["state"], ("mtp" if m["mtp"] == "ready" else "")]
            for m in view["models"]]
    return _fmt_table(["MODEL", "BACKEND", "STATE", "MTP"], rows)


# --- host/port の解決 -------------------------------------------------------
def _endpoint(gcfg) -> tuple[str, int, list[int]]:
    """自己接続に使う host/port と、停止対象の全ポート（公開＋事前登録モデル）。"""
    host = local_connect_host(gcfg.host)
    port = gcfg.port
    all_ports = [port] + [m.port for m in gcfg.models]
    return host, port, all_ports


def _collect_gateway_pids(host: str, port: int, all_ports: list[int]) -> list[int]:
    """停止対象（ゲートウェイ＋全モデルワーカー）の PID を、複数経路から重複なく集める。

    - `/admin/status`: 稼働中なら daemon pid（`pid`）と各モデルワーカーの pids を直接得る
      （モデルワーカーは別セッションなので、port 走査だけでは取りこぼしうる。これが主経路）。
    - 既知ポート走査: 公開ポート＋事前登録モデルポート（gateway.toml がある場合）。孤児対策。
    - ランタイム記録: `./gateway.toml` の無い場所からでも daemon pid を拾える。
    このパッケージ由来に見える PID だけに絞る（無関係プロセスを巻き添えにしない）。
    """
    pids: set[int] = set()
    admin = gateway_admin_status(host, port)
    if admin:
        if isinstance(admin.get("pid"), int):
            pids.add(admin["pid"])
        for m in admin.get("models", []):
            for p in (m.get("pids") or []):
                if isinstance(p, int):
                    pids.add(p)
    for p in all_ports:
        pids.update(find_pids_on_port(p))
    rec = read_gateway_runtime()
    if rec and isinstance(rec.get("pid"), int):
        pids.add(rec["pid"])
    return [pid for pid in pids if pid_looks_like_ours(pid)]


def _stop_pids(pids: list[int]) -> None:
    """指定 PID 群を並列に停止する（stop_pid は 1 件あたり最長 ~10s 待つため）。"""
    import threading

    threads = [threading.Thread(target=stop_pid, args=(pid,)) for pid in pids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# --- 各サブコマンドの実装 ---------------------------------------------------
def _start_cwd(gcfg) -> str:
    """デーモンを spawn する作業ディレクトリ = 設定ファイルのある場所（無ければ CWD）。

    `gw start` を gateway.toml の無い場所から打っても、発見した設定のディレクトリで
    デーモンを起動する（その cwd の `./gateway.toml` を読む。ログは log_dir() 固定）。
    """
    return getattr(gcfg, "_config_dir", None) or os.getcwd()


def cmd_start(gcfg, args) -> int:
    host, port, _ = _endpoint(gcfg)
    cwd = _start_cwd(gcfg)
    try:
        pid = start_gateway_background(cwd, host, port)
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"start failed: {exc}", file=sys.stderr)
        return 1
    admin = gateway_admin_status(host, port)
    ready = admin is not None or is_ready(f"http://{host}:{port}/v1")
    print(render_status(gcfg, admin, ready))
    return 0


def cmd_stop(gcfg, args) -> int:
    host, port, all_ports = _endpoint(gcfg)
    pids = _collect_gateway_pids(host, port, all_ports)
    _stop_pids(pids)
    if pids:
        print(f"gateway: stopped ({len(pids)} process(es))")
    else:
        print("gateway: not running")
    return 0


def cmd_restart(gcfg, args) -> int:
    host, port, all_ports = _endpoint(gcfg)
    _stop_pids(_collect_gateway_pids(host, port, all_ports))
    return cmd_start(gcfg, args)


def cmd_serve(gcfg, args) -> int:
    """`gw serve`: ゲートウェイをフォアグラウンドで実行する（Ollama の `ollama serve` 相当）。

    通常運用は `gw start`（裏で常駐）。これは自動起動サービス（gw enable）が使う実行口で、
    端末で直接回してデバッグしたいときにも使える。単一起動は従来どおり GatewayLock が保証。

    `--managed` はサービスマネージャ（launchd / systemd）から起動されたことを示す内部
    フラグ。「既に別のゲートウェイが稼働中」（終了コード 3）を成功（0）として静かに退く——
    手動 `gw start` のデーモンが居るとき、KeepAlive/Restart が失敗と誤認して再起動ループに
    入らないため。
    """
    from .daemon import run_gateway

    cwd = _start_cwd(gcfg)
    config_path = os.path.join(cwd, "gateway.toml")
    try:
        # デーモンの規約に合わせる: 設定は常に「cwd の ./gateway.toml」（__main__ と同じ視点）。
        os.chdir(cwd)
    except OSError as exc:
        print(f"serve failed: {exc}", file=sys.stderr)
        return 1
    install_shutdown_handlers()
    rc = run_gateway(gcfg, config_path=config_path)
    if rc == 3 and getattr(args, "managed", False):
        print("gateway already running; managed serve exits successfully.", file=sys.stderr)
        return 0
    return rc


def cmd_enable(gcfg, args) -> int:
    """`gw enable`: ログイン時自動起動＋異常終了時自動復活を OS に登録する。

    「ユーザー専管」の原則はそのまま——世話をユーザーの手動操作から **OS** へ委任する
    （エージェントや corp がライフサイクルに触らないルールは不変）。手動起動のデーモンが
    居れば止めてから切り替え、管理者を OS の 1 者に固定する（二重管理にしない）。
    """
    from . import service

    if service.service_kind() is None:
        print(
            "この OS では自動起動の登録に未対応です（Windows は従来どおり手動 `gw start`）。",
            file=sys.stderr,
        )
        return 1
    host, port, all_ports = _endpoint(gcfg)
    pids = _collect_gateway_pids(host, port, all_ports)
    if pids:
        print(f"稼働中のゲートウェイ（pid {pids}）を停止し、以後はサービス管理へ切り替えます…")
        _stop_pids(pids)
    try:
        service.enable()
    except RuntimeError as exc:
        print(f"enable failed: {exc}", file=sys.stderr)
        return 1
    kind = "launchd" if service.service_kind() == "launchd" else "systemd --user"
    print(f"自動起動を登録しました（{kind}）。ログイン時に自動起動し、異常終了時は自動復活します。")
    print("解除は gw disable（以後は手動 gw start）。停止だけなら従来どおり gw stop。")
    # サービスが立ち上がるのを軽く待って現況を見せる（初回はモデル導入等で遅いことも
    # あるので best-effort。間に合わなくても裏で起動は進む）。
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if is_ready(f"http://{host}:{port}/v1"):
            break
        time.sleep(0.5)
    admin = gateway_admin_status(host, port)
    ready = admin is not None or is_ready(f"http://{host}:{port}/v1")
    print(render_status(gcfg, admin, ready))
    return 0


def cmd_disable(gcfg, args) -> int:
    """`gw disable`: 自動起動を解除して手動運用（gw start / gw stop）へ完全に戻す。"""
    from . import service

    if service.service_kind() is None:
        print("この OS に自動起動の登録はありません（何もしませんでした）。")
        return 0
    try:
        removed = service.disable()
    except RuntimeError as exc:
        print(f"disable failed: {exc}", file=sys.stderr)
        return 1
    if removed:
        print("自動起動を解除しました。ゲートウェイは停止し、以後は手動 `gw start` で運用します。")
    else:
        print("自動起動は登録されていません（手動運用のままです）。")
    return 0


def cmd_status(gcfg, args) -> int:
    host, port, _ = _endpoint(gcfg)
    admin = gateway_admin_status(host, port)
    ready = admin is not None or is_ready(f"http://{host}:{port}/v1")
    print(render_status(gcfg, admin, ready))
    return 0 if ready else 1


def cmd_ps(gcfg, args) -> int:
    host, port, _ = _endpoint(gcfg)
    admin = gateway_admin_status(host, port)
    ready = admin is not None or is_ready(f"http://{host}:{port}/v1")
    print(render_ps(gcfg, admin, ready))
    return 0


def cmd_list(gcfg, args) -> int:
    host, port, _ = _endpoint(gcfg)
    admin = gateway_admin_status(host, port)
    print(render_list(gcfg, admin))
    return 0


def cmd_log(gcfg, args) -> int:
    _, port, _ = _endpoint(gcfg)
    if getattr(args, "follow", False):
        return _follow_log(port)
    print(read_log_tail(port, max_lines=args.lines))
    return 0


def _follow_log(port: int) -> int:
    """`gw log -f`: ログの新規行を追従表示する（Ctrl-C で終了）。"""
    path = gateway_log_path(port)
    print(read_log_tail(port), end="")
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            while True:
                line = fh.readline()
                if line:
                    sys.stdout.write(line.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                else:
                    time.sleep(0.3)
    except FileNotFoundError:
        print(f"(ログはまだありません: {path})", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


def cmd_max(gcfg, args) -> int:
    host, port, _ = _endpoint(gcfg)
    # 指定は「1 以上の整数」か「off（無制限）」の 2 形だけ（別名は設けない —— 1 使い方）。
    arg = args.value.strip().lower()
    if arg == "off":
        value: int | None = None
    else:
        try:
            value = int(arg)
        except ValueError:
            print(f"max_resident には 1 以上の整数か off を指定してください: '{args.value}'", file=sys.stderr)
            return 2
        if value < 1:
            print("max_resident は 1 以上の整数、または off（無制限）です", file=sys.stderr)
            return 2
    label = "∞" if value is None else str(value)
    res = gateway_set_max_resident(value, host, port)
    if res is None:
        print("max_resident の変更に失敗しました（ゲートウェイ未起動？）", file=sys.stderr)
        return 1
    print(f"max_resident → {label}")
    return 0


def cmd_mtp(gcfg, args) -> int:
    text, code = mtp_report(args.model)
    print(text)
    return code


# --- モデルファイル管理（pull / rm / show。Ollama の pull / rm / show 相当） ----
def _split_model_spec(model: str) -> tuple[str, str]:
    """`org/repo[:セレクタ]` を検証して (repo, selector) に分ける（resolve_gguf と同じ規約）。"""
    spec = model.strip()
    repo, _sep, selector = spec.partition(":")
    if (repo.startswith(("/", "./", "../", "~"))
            or repo.count("/") != 1 or not all(repo.split("/"))):
        raise ValueError(
            f"model は HF repo-id（org/repo[:セレクタ]）で指定してください: {model!r}"
        )
    return repo, selector


def _model_cache_dir(repo: str) -> str:
    """repo の HF キャッシュディレクトリ（models--org--name）。"""
    org, name = repo.split("/", 1)
    return os.path.join(_hf_hub_cache(), f"models--{org}--{name}")


def _dir_size(path: str) -> int:
    """ディレクトリの実消費バイト数。シンボリックリンクは辿らない（blob の二重計上を防ぐ）。"""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                try:
                    total += os.stat(p).st_size
                except OSError:
                    pass
    return total


def _human_size(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


_GGUF_SHARD_RE = r"-\d{5}-of-\d{5}(?=\.gguf$)"


def plan_gguf_pull(files: list[str], selector: str) -> list[str]:
    """GGUF repo の取得対象ファイルを決める（純粋関数・テスト可能）。

    GGUF repo は量子化ごとに別ファイルを持ち、全部落とすと数百 GB になり得るので
    **本体 1 種 + mmproj（画像入力用）だけ**を選ぶ。resolve_gguf と同じ規約:
    セレクタでファイル名を絞り、本体が複数種（シャード分割は 1 種と数える）なら
    候補を並べて ValueError（`repo:<量子化名>` で絞ってもらう）。
    """
    import re

    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    mmproj = [f for f in ggufs if "mmproj" in os.path.basename(f).lower()]
    bodies = [
        f for f in ggufs
        if "mmproj" not in os.path.basename(f).lower()
        and "mtp" not in os.path.basename(f).lower()
    ]
    if selector:
        bodies = [f for f in bodies if selector.lower() in os.path.basename(f).lower()]
    if not bodies:
        hint = f"（セレクタ '{selector}' に一致なし）" if selector else ""
        raise ValueError(f"取得対象の GGUF が見つかりません{hint}")
    # シャード（-00001-of-00002）は同一本体として束ねる。
    stems = sorted({re.sub(_GGUF_SHARD_RE, "", f) for f in bodies})
    if len(stems) > 1:
        names = [os.path.splitext(os.path.basename(s))[0] for s in stems]
        raise ValueError(
            "量子化が複数あります。'org/repo:<量子化名>' で 1 つに絞ってください: "
            + ", ".join(names)
        )
    return sorted(bodies) + sorted(mmproj)


def cmd_pull(gcfg, args) -> int:
    """`gw pull`: モデルを事前ダウンロードする（進捗表示付き）。

    本サーバーは「事前 DL 必須」ポリシー（ensure_cached / resolve_gguf）——これはその
    正規の取得口。GGUF repo は本体 1 種 + mmproj だけを選んで落とす（全量子化を
    巻き込まない）。mlx 系は repo 全体（重み + config + tokenizer）。進捗バーは
    huggingface_hub（tqdm）がそのまま端末に出す。
    """
    try:
        repo, selector = _split_model_spec(args.model)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    backend = infer_backend(args.model)
    try:
        from huggingface_hub import snapshot_download
        if backend == "llama-cpp":
            from huggingface_hub import HfApi
            targets = plan_gguf_pull(HfApi().list_repo_files(repo), selector)
            print(f"取得: {repo}  {len(targets)} ファイル "
                  f"({', '.join(os.path.basename(t) for t in targets[:4])}"
                  f"{' …' if len(targets) > 4 else ''})")
            snapshot_download(repo, allow_patterns=targets)
        else:
            print(f"取得: {repo}（repo 全体）")
            snapshot_download(repo)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断しました（続きから再開できます: 同じ gw pull をもう一度）",
              file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - ネットワーク・認証等はメッセージで案内
        print(f"pull failed: {exc}", file=sys.stderr)
        return 1
    # 取得結果をサーバーと同じ基準で検証してから完了を名乗る。
    try:
        if backend == "llama-cpp":
            resolve_gguf(args.model)
        else:
            ensure_cached(repo)
    except ValueError as exc:
        print(f"取得は終わりましたが検証で問題が見つかりました: {exc}", file=sys.stderr)
        return 1
    print(f"done: {args.model}（キャッシュ {_human_size(_dir_size(_model_cache_dir(repo)))}）")
    if mtp_status(repo) == "available":
        drafter = MTP_DRAFTERS.get(repo)
        print(f"ヒント: MTP（~2倍速）に対応しています。`gw pull {drafter}` で有効化できます。")
    return 0


def cmd_rm(gcfg, args) -> int:
    """`gw rm`: モデルを HF キャッシュから削除する（確認プロンプト付き）。

    削除単位は repo（models--org--name ディレクトリ全体 = その repo の全ファイル）。
    ロード中のモデルは拒否する（gw stop するかアンロードを待ってから）。自動では
    決して走らない——ユーザーが名指しで明示的に消す道具（モデルキャッシュを勝手に
    消さない方針はそのまま）。
    """
    try:
        repo, _selector = _split_model_spec(args.model)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    cache = _model_cache_dir(repo)
    if not os.path.isdir(cache):
        print(f"'{repo}' はキャッシュにありません（何もしませんでした）")
        return 1
    # ロード中なら拒否（稼働中デーモンがあるときだけ確認できる。best-effort）。
    rec = read_gateway_runtime()
    if rec is not None:
        admin = gateway_admin_status(rec.get("host", "127.0.0.1"),
                                     int(rec.get("port", 8799)))
        loaded = [
            m["model"] for m in (admin or {}).get("models", [])
            if m.get("loaded") and str(m.get("model", "")).split(":", 1)[0] == repo
        ]
        if loaded:
            print(f"'{loaded[0]}' はロード中です。アンロードを待つか gw stop してから "
                  "やり直してください。", file=sys.stderr)
            return 1
    size = _human_size(_dir_size(cache))
    if not getattr(args, "yes", False):
        print(f"削除: {repo}（{size}。この repo の全ファイル——量子化違い・mmproj も含む）")
        try:
            answer = input("よろしいですか? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("中止しました")
            return 1
    import shutil
    shutil.rmtree(cache)
    print(f"deleted: {repo}（{size} 解放）")
    return 0


def _find_config_json(cache: str) -> dict | None:
    """キャッシュのスナップショットから config.json を読む（無ければ None）。"""
    snap_root = os.path.join(cache, "snapshots")
    for root, _dirs, files in os.walk(snap_root):
        if "config.json" in files:
            try:
                import json
                with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return None
    return None


def build_show_rows(model: str) -> list[tuple[str, str]]:
    """`gw show` の表示行を組み立てる（描画なし・テスト可能）。"""
    import re

    repo, _selector = _split_model_spec(model)
    backend = infer_backend(model)
    rows: list[tuple[str, str]] = [("モデル", model), ("バックエンド", backend)]
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", repo.split("/", 1)[1])
    if m:
        rows.append(("パラメータ数", f"{m.group(1)}B（名前から推定）"))
    cache = _model_cache_dir(repo)
    if not os.path.isdir(os.path.join(cache, "snapshots")):
        rows.append(("キャッシュ", f"未取得（gw pull {model} で取得）"))
        return rows
    rows.append(("ディスクサイズ", _human_size(_dir_size(cache))))
    if backend == "llama-cpp":
        try:
            path = resolve_gguf(model)
            name = os.path.basename(path)
            rows.append(("GGUF", name))
            q = re.search(r"(?i)\b(i?q\d[a-z0-9_]*|f16|bf16|f32)\b", name)
            if q:
                rows.append(("量子化", q.group(1)))
            rows.append(("ファイルサイズ", _human_size(os.stat(os.path.realpath(path)).st_size)))
            mmproj = any(
                "mmproj" in f.lower()
                for _r, _d, fs in os.walk(os.path.join(cache, "snapshots")) for f in fs
            )
            rows.append(("画像入力", "対応（mmproj あり）" if mmproj else "-"))
        except ValueError as exc:
            rows.append(("GGUF", str(exc)))
    else:
        cfg = _find_config_json(cache) or {}
        if cfg.get("model_type"):
            rows.append(("アーキテクチャ", str(cfg["model_type"])))
        ctx = (cfg.get("max_position_embeddings")
               or (cfg.get("text_config") or {}).get("max_position_embeddings"))
        if ctx:
            rows.append(("コンテキスト長", f"{int(ctx):,}"))
        bits = (cfg.get("quantization") or {}).get("bits")
        if bits:
            rows.append(("量子化", f"{bits}bit（mlx）"))
        if "vision_config" in cfg:
            rows.append(("画像入力", "対応（vision モデル）"))
    mtp = mtp_status(repo)
    if mtp == "ready":
        rows.append(("MTP", "対応（ドラフター取得済み・そのまま効く）"))
    elif mtp == "available":
        rows.append(("MTP", f"対応（未取得。gw pull {MTP_DRAFTERS.get(repo)} で有効化）"))
    # 稼働中デーモンがあればロード状態も出す（best-effort）。
    rec = read_gateway_runtime()
    if rec is not None:
        admin = gateway_admin_status(rec.get("host", "127.0.0.1"),
                                     int(rec.get("port", 8799)))
        live = [
            m for m in (admin or {}).get("models", [])
            if str(m.get("model", "")).split(":", 1)[0] == repo
        ]
        if live and live[0].get("loaded"):
            rows.append(("状態", f"ロード中（処理中 {int(live[0].get('inflight', 0))}）"))
        else:
            rows.append(("状態", "未ロード（初回リクエストでロード）"))
    return rows


def cmd_show(gcfg, args) -> int:
    """`gw show`: モデルの素性（量子化・コンテキスト長・サイズ・MTP 等）を表示する。"""
    try:
        rows = build_show_rows(args.model)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    width = max(len(label) for label, _v in rows)
    for label, value in rows:
        print(f"  {label.ljust(width)}  {value}")
    return 0


def cmd_update(gcfg, args) -> int:
    """PyPI 新版があれば git pull で追従し、稼働中デーモンを再起動する（手動トリガ）。

    自動更新は稼働中デーモンが裏で行う（idle 時に自動適用）。このコマンドは「今すぐ確認・適用」
    したいとき用。ソース更新後、稼働中デーモンを止めて start し直す（新コードで立ち上がる）。
    """
    from . import update

    st = update.check()
    print(f"current {st.current}  latest {st.latest}")
    if not st.available:
        print("already up to date" if st.reason == "ok" else f"no update: {st.reason}")
        return 0
    if not st.can_apply:
        print(f"update available but cannot auto-apply: {st.reason}", file=sys.stderr)
        return 1
    ok, msg = update.apply_update()
    if not ok:
        print(f"update failed: {msg}", file=sys.stderr)
        return 1
    print(f"updated: {msg}")
    # 稼働中なら新コードで再起動（stop → start）。
    host, port, all_ports = _endpoint(gcfg)
    if is_ready(f"http://{host}:{port}/v1"):
        _stop_pids(_collect_gateway_pids(host, port, all_ports))
        return cmd_start(gcfg, args)
    return 0


def cmd_help(gcfg, args) -> int:
    """`gw help`: サブコマンド一覧（`gw -h` と同じ）を表示する。"""
    build_parser().print_help()
    return 0


# --- argparse ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gw",
        description="ローカル LLM ゲートウェイを裏で常駐させて運用する"
                    "（設定は ~/.config/local-llm-server/gateway.toml の 1 箇所。初回 start で自動生成）。",
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("start", help="デーモンを裏で常駐起動する")
    sub.add_parser("stop", help="ゲートウェイとモデルサーバーを停止する")
    sub.add_parser("restart", help="stop してから start する")
    serve = sub.add_parser("serve", help="フォアグラウンドで実行する（自動起動サービスの実行口）")
    serve.add_argument(
        "--managed", action="store_true",
        help=argparse.SUPPRESS,  # サービスマネージャ専用の内部フラグ（既稼働を成功として退く）
    )
    sub.add_parser("enable", help="ログイン時自動起動＋異常終了時自動復活を OS に登録する")
    sub.add_parser("disable", help="自動起動を解除して手動運用（gw start）に戻す")
    sub.add_parser("status", help="稼働状態を 1 行で表示する")
    sub.add_parser("ps", help="ロード中モデルの状態を表示する")
    sub.add_parser("list", help="使えるモデル一覧を表示する")
    log = sub.add_parser("log", help="ゲートウェイログの末尾を表示する")
    log.add_argument("-f", "--follow", action="store_true", help="新規行を追従表示する")
    log.add_argument("-n", "--lines", type=int, default=200, help="表示行数（既定 200）")
    mx = sub.add_parser("max", help="max_resident を無停止で変更する")
    mx.add_argument("value", help="常駐上限（1 以上の整数、または off で無制限）")
    mtp = sub.add_parser("mtp", help="MTP ドラフターの取得状況を確認する")
    mtp.add_argument("model", nargs="?", help="調べるモデル ID（省略で対応表を全件）")
    pull = sub.add_parser("pull", help="モデルを事前ダウンロードする（進捗表示付き）")
    pull.add_argument("model", help="HF repo-id（org/repo[:量子化名]。GGUF は本体+mmprojのみ取得）")
    rm = sub.add_parser("rm", help="モデルを HF キャッシュから削除する（確認あり）")
    rm.add_argument("model", help="HF repo-id（org/repo。repo の全ファイルが対象）")
    rm.add_argument("-y", "--yes", action="store_true", help="確認プロンプトを省略する")
    show = sub.add_parser("show", help="モデルの素性（量子化・コンテキスト長・サイズ等）を表示する")
    show.add_argument("model", help="HF repo-id（org/repo[:量子化名]）")
    sub.add_parser("update", help="PyPI 新版があれば git pull で追従して再起動する")
    sub.add_parser("help", help="このコマンド一覧を表示する")
    return p


_COMMANDS = {
    "start": cmd_start, "stop": cmd_stop, "restart": cmd_restart,
    "serve": cmd_serve, "enable": cmd_enable, "disable": cmd_disable,
    "status": cmd_status, "ps": cmd_ps, "list": cmd_list, "log": cmd_log,
    "max": cmd_max, "mtp": cmd_mtp, "update": cmd_update, "help": cmd_help,
    "pull": cmd_pull, "rm": cmd_rm, "show": cmd_show,
}


def _resolve_gcfg(cmd: str | None):
    """コマンドに応じて設定（gcfg）を解決する。戻り値: (gcfg, error_code)。

    設定の場所は user_config_path の **1 箇所だけ**（1 ライブラリ 1 使い方）。

    - **launch 系（start/restart）**: 設定を用意して（無ければ自動生成 → ensure_user_config）
      読む。設定ディレクトリを cwd にしてデーモンを起動するので、どこから打っても同じ。
    - **query/control 系（status/stop/ps/list/log/max/update）**: 稼働中デーモンの
      ランタイム記録（＝実物）を最優先し、未起動なら設定ファイルから接続先を出す。
    """
    def _load(path):
        try:
            return load_config(path), 0
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            print(f"Failed to load {path}: {exc}", file=sys.stderr)
            return None, 2

    # launch 系: 設定を用意して読む（無ければ自動生成）。serve はデーモンそのもの、
    # enable/disable はサービス登録（serve が読む設定を先に確定させる）なので同類。
    if cmd in ("start", "restart", "serve", "enable", "disable"):
        try:
            return _load(ensure_user_config())
        except OSError as exc:
            print(f"failed to create {user_config_path()}: {exc}", file=sys.stderr)
            return None, 2

    # query/control 系: 稼働中デーモン（ランタイム記録）＝実物を最優先。
    rec = read_gateway_runtime()
    if rec is not None:
        return _RuntimeConfig(rec), 0
    # 未起動なら設定ファイル（未作成でも「この URL で停止中」と案内できるよう自動生成はしない）。
    path = user_config_path()
    if os.path.isfile(path):
        return _load(path)
    print(
        f"no running gateway, and no config at {path} yet. Run `gw start` first.",
        file=sys.stderr,
    )
    return None, 2


def main(argv: list[str] | None = None) -> int:
    """`gw` コマンド本体。サブコマンドで運用する（引数なしはコマンド一覧を表示）。

    設定は user_config_path の 1 ファイルだけ・起動は `gw start` だけ、という
    「1 ライブラリ 1 使い方」。query/control 系はどのディレクトリからでも、稼働中デーモンの
    ランタイム記録を辿って同じ 1 つのデーモンに届く。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # 設定にまったく依存しないコマンド。引数なしは Ollama 流にコマンド一覧を出す
    # （起動の入口は `gw start` の 1 つだけにする）。
    if args.cmd in (None, "help"):
        return cmd_help(None, args)
    # モデルファイル管理は設定に依存しない（HF キャッシュ直接 + 稼働中デーモンは
    # ランタイム記録から best-effort で参照）。
    if args.cmd in ("mtp", "pull", "rm", "show"):
        return _COMMANDS[args.cmd](None, args)

    gcfg, code = _resolve_gcfg(args.cmd)
    if gcfg is None:
        return code
    return _COMMANDS[args.cmd](gcfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
