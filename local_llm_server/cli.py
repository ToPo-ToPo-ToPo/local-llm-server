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
  - `gw status`         … 稼働/停止・PID・URL・起動経過・累計リクエストを 1 行表示
  - `gw ps`             … ロード中モデルの状態（処理中数・在席・アイドル残り）を表示
  - `gw list`           … 使えるモデル（カタログ＋HF キャッシュ）を一覧
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
    discover_cached_models,
    find_pids_on_port,
    gateway_admin_status,
    gateway_log_path,
    gateway_set_max_resident,
    is_ready,
    local_connect_host,
    mtp_status,
    pid_looks_like_ours,
    primary_lan_ip,
    read_gateway_runtime,
    start_gateway_background,
    stop_pid,
)


# --- 設定解決 ---------------------------------------------------------------
def resolve_config() -> str | None:
    """使う gateway.toml を決める。**カレントディレクトリの `./gateway.toml` のみ**。

    存在すればそのパス、無ければ None（呼び出し側がエラーにする）。場所は CWD 固定で、
    位置引数やホーム等の外部は見ない（「gateway.toml は CWD に置く」という 1 ルール）。
    """
    path = os.path.join(os.getcwd(), "gateway.toml")
    return path if os.path.isfile(path) else None


def user_config_path() -> str:
    """ユーザー既定の gateway.toml パス（`~/.config/local-llm-server/gateway.toml`）。

    `gw` を PATH に入れて**どこからでも起動**するとき用の常設置き場（Ollama 流）。
    `XDG_CONFIG_HOME` があれば尊重する。ファイルの有無は問わずパスだけ返す。
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "local-llm-server", "gateway.toml")


def _fallback_config_path() -> str | None:
    """CWD 以外の既定の gateway.toml を探す（ユーザー既定 → editable クローン）。

    - **ユーザー既定**（`~/.config/local-llm-server/gateway.toml`）—— `gw` を PATH に入れた
      常設運用の置き場。
    - **editable インストール元のクローンの `gateway.toml`** —— `uv tool install --editable`
      で入れた場合、リポジトリの gateway.toml を自動発見する（何も移動せず動く）。
    """
    user = user_config_path()
    if os.path.isfile(user):
        return user
    try:
        from . import update
        root = update.repo_root()
        if root is not None:
            clone = os.path.join(str(root), "gateway.toml")
            if os.path.isfile(clone):
                return clone
    except Exception:  # noqa: BLE001 - 発見は best-effort（失敗は「無し」扱い）
        pass
    return None


def find_config_path() -> str | None:
    """`gw start` が使う gateway.toml を優先順位つきで探す（どこからでも起動できるように）。

    CWD の `./gateway.toml`（最優先）→ ユーザー既定 → editable クローンの gateway.toml。
    見つからなければ None。デーモン本体（__main__）は spawn 時の cwd で `./gateway.toml` を
    読むので、CLI はここで決めた**設定のあるディレクトリ**を cwd として渡す。
    """
    return resolve_config() or _fallback_config_path()


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


def load_config(path: str):
    """指定パスの gateway.toml を読む（壊れていれば例外を送出）。読んだ設定に
    `_config_dir`（そのファイルのあるディレクトリ）を付けて返す —— `gw start` はそこを
    デーモンの cwd にして spawn する（設定・ログの位置が一貫する）。"""
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
        "launcher": (admin or {}).get("launcher"),
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

    ログはローテーションされず肥大化しうるので、末尾 max_bytes だけ読む（全読みを避ける）。
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
    if view.get("launcher"):
        line += f"\n  launcher {view['launcher']}"
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
    デーモンを起動する（その cwd の `./gateway.toml` を読み、ログもそこに置く）。
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
    if cwd != os.getcwd():
        print(f"(using {os.path.join(cwd, 'gateway.toml')})")
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
    arg = args.value.strip().lower()
    if arg in ("off", "none", "unlimited", "inf", "∞", "0"):
        value: int | None = None
    else:
        try:
            value = int(arg)
        except ValueError:
            print(f"max_resident には数値か off を指定してください: '{args.value}'", file=sys.stderr)
            return 2
        if value < 1:
            print("max_resident は 1 以上、または off（無制限）です", file=sys.stderr)
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


def cmd_default(gcfg, args) -> int:
    """引数なし `gw`: start してから status/ps を表示する（従来の `uv run gw` 相当）。"""
    host, port, _ = _endpoint(gcfg)
    try:
        start_gateway_background(_start_cwd(gcfg), host, port)
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"start failed: {exc}", file=sys.stderr)
        return 1
    admin = gateway_admin_status(host, port)
    ready = admin is not None or is_ready(f"http://{host}:{port}/v1")
    print(render_status(gcfg, admin, ready))
    ps = render_ps(gcfg, admin, ready)
    if ps != "no models loaded":
        print("\n" + ps)
    return 0


# --- argparse ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gw",
        description="ローカル LLM ゲートウェイ（./gateway.toml）を裏で常駐させて運用する。",
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("start", help="デーモンを裏で常駐起動する")
    sub.add_parser("stop", help="ゲートウェイとモデルサーバーを停止する")
    sub.add_parser("restart", help="stop してから start する")
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
    sub.add_parser("update", help="PyPI 新版があれば git pull で追従して再起動する")
    sub.add_parser("help", help="このコマンド一覧を表示する")
    return p


_COMMANDS = {
    "start": cmd_start, "stop": cmd_stop, "restart": cmd_restart,
    "status": cmd_status, "ps": cmd_ps, "list": cmd_list, "log": cmd_log,
    "max": cmd_max, "mtp": cmd_mtp, "update": cmd_update, "help": cmd_help,
}

# `./gateway.toml` を **必須** とするコマンド（何を配信するか＝models を知る必要がある）。
# それ以外（status/stop/ps/list/log/max/update）は、CWD 設定が無ければ稼働中デーモンの
# ランタイム記録から接続先を得て**どこからでも**動く（引数なし＝start なので必須側）。
_NEEDS_CONFIG = {"start", "restart", None}
_NO_CONFIG = {"help", "mtp"}  # 設定にまったく依存しない


def _resolve_gcfg(cmd: str | None):
    """コマンドに応じて設定（gcfg）を解決する。戻り値: (gcfg, error_code)。

    **launch 系（start/restart/引数なし）**: `find_config_path`（CWD → `~/.config` →
    editable クローン）で gateway.toml を探し、その設定のあるディレクトリでデーモンを
    起動する（どこからでも `gw start`）。見つからなければエラー。

    **query/control 系（status/stop/ps/list/log/max/update）**: マシンに 1 つのデーモンが
    真実なので、**CWD の明示設定 → 稼働中デーモンのランタイム記録 → ユーザー/クローン設定**
    の順で接続先を決める。これで別ディレクトリの設定（別ポート）を掴んで実際の稼働デーモンを
    見失う事故を防ぐ（記録＝実際に動いているデーモンを優先）。
    """
    def _load(path):
        try:
            return load_config(path), 0
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            print(f"Failed to load {path}: {exc}", file=sys.stderr)
            return None, 2

    if cmd in _NEEDS_CONFIG:  # launch 系: 設定必須
        path = find_config_path()
        if path is None:
            print(
                "no gateway.toml found. `gw start` looks in: the current directory, "
                f"{user_config_path()}, and (for editable installs) the clone. "
                "Create one in a directory and run `gw start` there, or place it at the path above.",
                file=sys.stderr,
            )
            return None, 2
        return _load(path)

    # query/control 系: CWD の明示設定が最優先。
    cwd_path = resolve_config()
    if cwd_path is not None:
        return _load(cwd_path)
    # 次に稼働中デーモン（ランタイム記録）—— 実際に動いているものを優先する。
    rec = read_gateway_runtime()
    if rec is not None:
        return _RuntimeConfig(rec), 0
    # 最後にユーザー/クローン設定（未起動時に「そのURLで停止中」や list を出すため）。
    path = _fallback_config_path()
    if path is not None:
        return _load(path)
    print(
        "no running gateway found, and no gateway.toml in the current directory / "
        f"{user_config_path()}. Start one with `gw start`.",
        file=sys.stderr,
    )
    return None, 2


def main(argv: list[str] | None = None) -> int:
    """`gw` コマンド本体。サブコマンドで運用する（引数なしは start + status）。

    `status`/`stop`/`ps`/`list`/`log`/`max`/`update` は、`./gateway.toml` の無い
    ディレクトリからでも、稼働中デーモンのランタイム記録を辿って実行できる。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # 設定にまったく依存しないコマンド（help / mtp）は先に処理する。
    if args.cmd == "help":
        return cmd_help(None, args)
    if args.cmd == "mtp":
        return cmd_mtp(None, args)

    gcfg, code = _resolve_gcfg(args.cmd)
    if gcfg is None:
        return code

    handler = _COMMANDS.get(args.cmd, cmd_default)
    return handler(gcfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
