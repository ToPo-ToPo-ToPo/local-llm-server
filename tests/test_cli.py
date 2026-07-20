"""CLI（`gw` サブコマンド）の純ロジック（catalog × ライブ状態のマージ・時間整形・ログ末尾・
描画）と、argparse ディスパッチ（start/stop/status/ps を mock で）を検証する。TUI は廃止済み。
"""
import pytest

from local_llm_server import cli
from local_llm_server.daemon import load_gateway_config


def _write_cfg(tmp_path, body):
    p = tmp_path / "gateway.toml"
    # 純ロジックのテストは自動更新を切って hermetic に保つ（起動時の PyPI HTTP / git を避ける）。
    if "auto_update" not in body:
        body = "auto_update = false\n" + body
    p.write_text(body, encoding="utf-8")
    return p


# --- 純データ層 -------------------------------------------------------------
def test_read_log_tail(tmp_path, monkeypatch):
    log = tmp_path / "gw.log"
    log.write_text("\n".join(f"line {i}" for i in range(1, 2001)) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "gateway_log_path", lambda port: str(log))
    out = cli.read_log_tail(123, max_lines=10)
    lines = out.strip().splitlines()
    assert lines[-1] == "line 2000" and len(lines) == 10
    # ファイルが無いときは案内文
    monkeypatch.setattr(cli, "gateway_log_path", lambda port: str(tmp_path / "nope.log"))
    assert cli.read_log_tail(123).startswith("(ログはまだ")


def test_fmt_hms():
    assert cli._fmt_hms(None) == "—"
    assert cli._fmt_hms(0) == "0:00"
    assert cli._fmt_hms(65) == "1:05"
    assert cli._fmt_hms(3661) == "1:01:01"


def test_merge_status_lists_catalog_live_and_cached(tmp_path, monkeypatch):
    # カタログ（未ロード）・動的ロード分・HF キャッシュの候補が重複なく 1 ビューに並ぶこと。
    monkeypatch.setattr(cli, "mtp_status", lambda m: None)
    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    admin = {
        "uptime": 12.0, "requests": 3, "max_resident": 2,
        "models": [{"model": "org/Live-4bit", "backend": "mlx-vlm", "loaded": True,
                    "inflight": 1, "requests": 3, "idle_for": None, "sessions": 0}],
        "available": [{"id": "org/Cached-GGUF", "backend": "llama-cpp"}],
    }
    view = cli.merge_status(gcfg, admin, ready=True)
    by = {m["model"]: m for m in view["models"]}
    assert by["org/Live-4bit"]["state"] == "busy"          # inflight>0 → busy
    assert by["org/Cached-GGUF"]["state"] == "unloaded"    # キャッシュ候補は未ロード
    assert view["max_resident"] == 2 and view["requests"] == 3


# --- 描画 -------------------------------------------------------------------
def test_render_status_running_and_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "mtp_status", lambda m: None)
    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    stopped = cli.render_status(gcfg, None, ready=False)
    assert "stopped" in stopped and "gw start" in stopped
    admin = {"uptime": 5.0, "requests": 7, "max_resident": 1, "pid": 4242,
             "models": [], "available": []}
    running = cli.render_status(gcfg, admin, ready=True)
    assert "running" in running and "pid 4242" in running and "requests 7" in running


def test_render_ps_only_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "mtp_status", lambda m: None)
    gcfg = load_gateway_config(str(_write_cfg(tmp_path, "port = 8799\n")))
    # ロード無し → 案内文
    assert cli.render_ps(gcfg, {"models": [], "available": []}, ready=True) == "no models loaded"
    admin = {
        "models": [
            {"model": "org/Busy", "backend": "mlx-vlm", "loaded": True,
             "inflight": 2, "instances": 1, "requests": 9, "idle_for": None, "sessions": 1},
            {"model": "org/Cold", "backend": "mlx-vlm", "loaded": False},
        ],
        "available": [],
    }
    out = cli.render_ps(gcfg, admin, ready=True)
    assert "org/Busy" in out and "busy" in out
    assert "org/Cold" not in out          # 未ロードは ps に出さない


# --- argparse ディスパッチ --------------------------------------------------
# 設定は user_config_path の 1 箇所だけ。テストでは tmp 配下に差し替えて hermetic に保つ。
def _use_cfg(tmp_path, monkeypatch, body="port = 8799\n"):
    """user_config_path を tmp の gateway.toml に差し替える（実 ~/.config を触らない）。"""
    p = _write_cfg(tmp_path, body)
    monkeypatch.setattr(cli, "user_config_path", lambda: str(p))
    return p


def test_status_dispatch_running(tmp_path, monkeypatch):
    _use_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "read_gateway_runtime", lambda: None)  # 記録なし → 設定で解決
    monkeypatch.setattr(cli, "mtp_status", lambda m: None)
    monkeypatch.setattr(cli, "gateway_admin_status",
                        lambda h, p: {"uptime": 1.0, "requests": 0, "max_resident": 1,
                                      "pid": 1, "models": [], "available": []})
    monkeypatch.setattr(cli, "is_ready", lambda url, **k: True)
    assert cli.main(["status"]) == 0      # 稼働中 → 0


def test_status_dispatch_stopped(tmp_path, monkeypatch):
    _use_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "read_gateway_runtime", lambda: None)
    monkeypatch.setattr(cli, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(cli, "is_ready", lambda url, **k: False)
    assert cli.main(["status"]) == 1      # 停止中 → 1


def test_start_dispatch_calls_background(tmp_path, monkeypatch):
    _use_cfg(tmp_path, monkeypatch)
    called = {}
    monkeypatch.setattr(cli, "start_gateway_background",
                        lambda cwd, host, port: called.setdefault("cwd", cwd) or 111)
    monkeypatch.setattr(cli, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(cli, "is_ready", lambda url, **k: True)
    monkeypatch.setattr(cli, "mtp_status", lambda m: None)
    assert cli.main(["start"]) == 0
    # どこから打っても、設定ファイルのあるディレクトリを cwd にして起動する。
    assert called["cwd"] == str(tmp_path)


def test_start_autocreates_config(tmp_path, monkeypatch, capsys):
    # 設定が未作成でも `gw start` が自動生成して起動する（初回のゼロ設定）。
    from local_llm_server import update

    cfg = tmp_path / "conf" / "gateway.toml"
    monkeypatch.setattr(cli, "user_config_path", lambda: str(cfg))
    monkeypatch.setattr(update, "repo_root", lambda: None)  # クローン例の複製ではなく既定を生成
    monkeypatch.setattr(cli, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(cli, "is_ready", lambda url, **k: True)
    monkeypatch.setattr(cli, "mtp_status", lambda m: None)
    seen = {}
    monkeypatch.setattr(cli, "start_gateway_background",
                        lambda cwd, host, port: seen.update(cwd=cwd, port=port) or 1)
    assert cli.main(["start"]) == 0
    assert cfg.is_file()                      # 自動生成された
    assert seen["cwd"] == str(cfg.parent)     # 設定ディレクトリで起動
    assert "created" in capsys.readouterr().err


def test_stop_dispatch_only_kills_our_pids(tmp_path, monkeypatch):
    _use_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "read_gateway_runtime", lambda: None)
    monkeypatch.setattr(cli, "gateway_admin_status", lambda h, p: None)
    monkeypatch.setattr(cli, "find_pids_on_port", lambda p: [900, 901] if p == 8799 else [])
    monkeypatch.setattr(cli, "pid_looks_like_ours", lambda pid: pid == 900)  # 901 は無関係
    killed = []
    monkeypatch.setattr(cli, "stop_pid", lambda pid, **k: killed.append(pid))
    assert cli.main(["stop"]) == 0
    assert killed == [900]                 # 巻き添えにしない


def test_missing_config_returns_2(tmp_path, monkeypatch):
    # 稼働中デーモンも設定ファイルも無ければ、query 系は「まず gw start」を案内して終わる。
    monkeypatch.setattr(cli, "user_config_path", lambda: str(tmp_path / "gateway.toml"))
    monkeypatch.setattr(cli, "read_gateway_runtime", lambda: None)
    assert cli.main(["status"]) == 2


def test_status_from_anywhere_uses_runtime_record(tmp_path, monkeypatch):
    # 設定ファイルが無くても、稼働中デーモンのランタイム記録（＝実物）から接続先を引いて動く。
    monkeypatch.setattr(cli, "user_config_path", lambda: str(tmp_path / "gateway.toml"))
    monkeypatch.setattr(cli, "mtp_status", lambda m: None)
    monkeypatch.setattr(cli, "read_gateway_runtime",
                        lambda: {"host": "127.0.0.1", "port": 8795, "pid": 4242})
    seen = {}
    monkeypatch.setattr(cli, "gateway_admin_status",
                        lambda h, p: seen.update(port=p) or
                        {"uptime": 3.0, "requests": 5, "max_resident": 1,
                         "pid": 4242, "models": [], "available": []})
    monkeypatch.setattr(cli, "is_ready", lambda url, **k: True)
    assert cli.main(["status"]) == 0
    assert seen["port"] == 8795     # 記録のポート（＝実際に動いているデーモン）を見ている


def test_remote_cmd_without_record_or_config_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "user_config_path", lambda: str(tmp_path / "gateway.toml"))
    monkeypatch.setattr(cli, "read_gateway_runtime", lambda: None)
    assert cli.main(["status"]) == 2
    assert "gw start" in capsys.readouterr().err


def test_stop_collects_pids_from_admin_and_record(tmp_path, monkeypatch):
    # stop は /admin/status の daemon pid＋各ワーカー pids と、ポート走査、記録の pid を
    # まとめて（このパッケージ由来だけ）停止する。
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "read_gateway_runtime",
                        lambda: {"host": "127.0.0.1", "port": 8799, "pid": 100})
    monkeypatch.setattr(cli, "gateway_admin_status",
                        lambda h, p: {"pid": 100,
                                      "models": [{"model": "m", "pids": [201, 202]}]})
    monkeypatch.setattr(cli, "find_pids_on_port", lambda p: [])
    monkeypatch.setattr(cli, "pid_looks_like_ours", lambda pid: True)
    killed = []
    monkeypatch.setattr(cli, "stop_pid", lambda pid, **k: killed.append(pid))
    assert cli.main(["stop"]) == 0
    assert sorted(killed) == [100, 201, 202]


def test_help_lists_commands_without_config(tmp_path, monkeypatch, capsys):
    # `gw help` は gateway.toml が無くてもコマンド一覧を表示する（設定に依らない）。
    monkeypatch.chdir(tmp_path)
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    assert all(c in out for c in ("start", "stop", "status", "ps", "list", "max", "update"))


def test_max_dispatch_rejects_bad_value(tmp_path, monkeypatch):
    _use_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "read_gateway_runtime", lambda: None)
    assert cli.main(["max", "-3"]) == 2       # 1 未満はエラー
    assert cli.main(["max", "none"]) == 2     # 無制限の指定は off の 1 形だけ（別名なし）


# --- デーモンの自動更新ウォッチャー ---------------------------------------
def test_update_watcher_fetches_then_restarts_on_drain(monkeypatch):
    # 新版あり・適用可能なら稼働中に取得（apply_update）し、drain 成功で restart_requested。
    import threading
    from local_llm_server import daemon, update

    st = update.UpdateStatus(current="0.1.0", latest="0.2.0", available=True,
                             can_apply=True, reason="ok")
    monkeypatch.setattr(update, "check", lambda timeout=3.0: st)
    applied = {}
    monkeypatch.setattr(update, "apply_update",
                        lambda *a, **k: (applied.setdefault("ok", True), "done"))

    class _Mgr:
        def begin_drain(self, ttl=120.0):
            return {"ok": True, "inflight": 0, "sessions": 0}  # idle → drain 成功

    stop = threading.Event()
    restart = threading.Event()
    calls = {"n": 0}
    monkeypatch.setattr(stop, "wait", lambda timeout=None: (calls.__setitem__("n", calls["n"] + 1) or calls["n"] > 1))

    daemon._update_watcher(_Mgr(), stop, restart)
    assert applied.get("ok") is True       # 取得は先に済ませる
    assert restart.is_set()                # drain が通ったので再起動


def test_update_watcher_holds_when_busy(monkeypatch):
    # 取得は済ませても、drain が通らない（busy）あいだは再起動を保留する（生成を殺さない）。
    import threading
    from local_llm_server import daemon, update

    st = update.UpdateStatus(current="0.1.0", latest="0.2.0", available=True,
                             can_apply=True, reason="ok")
    monkeypatch.setattr(update, "check", lambda timeout=3.0: st)
    applied = {}
    monkeypatch.setattr(update, "apply_update",
                        lambda *a, **k: (applied.setdefault("ok", True), "done"))

    class _Mgr:
        def begin_drain(self, ttl=120.0):
            return {"ok": False, "inflight": 1, "sessions": 0}  # busy → drain 拒否

    stop = threading.Event()
    restart = threading.Event()
    calls = {"n": 0}
    monkeypatch.setattr(stop, "wait", lambda timeout=None: (calls.__setitem__("n", calls["n"] + 1) or calls["n"] > 3))

    daemon._update_watcher(_Mgr(), stop, restart)
    assert applied.get("ok") is True       # 取得は済ませる（稼働中に先に pull）
    assert not restart.is_set()            # だが busy のあいだは再起動しない


def test_run_gateway_reexecs_on_restart_code(monkeypatch):
    # _run_gateway_locked が _RESTART_CODE を返したら、ロック解放後に reexec_daemon で
    # 自分を新コードに置き換える（execv）。ロック取得と execv はスタブして seam だけ検証。
    from local_llm_server import daemon, update

    events = []

    class _Lock:
        def acquire(self):
            events.append("acquire")
            return self

        def release(self):
            events.append("release")

    monkeypatch.setattr(daemon, "GatewayLock", _Lock)
    monkeypatch.setattr(daemon, "_run_gateway_locked",
                        lambda cfg, config_path=None: daemon._RESTART_CODE)
    monkeypatch.setattr(update, "reexec_daemon", lambda: events.append("reexec"))

    daemon.run_gateway(cfg=object(), config_path=None)
    # ロックを解放し切った **後** に reexec する（execv 前にロック fd を手放す契約）。
    assert events == ["acquire", "release", "reexec"]
