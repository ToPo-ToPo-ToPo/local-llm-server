"""OS サービス登録（`gw enable` / `gw disable`）——常駐の世話を OS に委任する。

Ollama が「サーバーを意識せず使える」のは常駐を OS が管理しているから。同じく、
ログイン時に自動起動し・異常終了時に自動復活するユーザーサービスとして gw を登録する
（docs/ollama-clone-plan.md Phase 0c）。登録先:

- macOS: launchd の LaunchAgent（`~/Library/LaunchAgents/<label>.plist`）
- Linux: systemd の user unit（`~/.config/systemd/user/<unit>`）
- Windows: 未対応（サービス管理が別系統。従来どおり手動 `gw start`）

サービスが実行するのは `gw serve --managed`（フォアグラウンド実行口）。復活の対象は
**異常終了だけ**——`gw stop`（SIGTERM → 終了コード 0）は正常終了なので復活しない
（launchd は KeepAlive.SuccessfulExit=false、systemd は Restart=on-failure）。
クラッシュループはスロットリングで頭打ちにする。「エージェントや corp がライフサイクルに
触らない」ルールは不変——委任先はあくまで OS であり、真実は常に `gw status` の 1 箇所。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from .constants import log_dir

# launchd のラベル（= plist ファイル名）と systemd のユニット名。
LAUNCHD_LABEL = "com.local-llm-server.gw"
SYSTEMD_UNIT = "local-llm-server.service"


def service_kind() -> str | None:
    """この OS で使うサービスマネージャ（"launchd" / "systemd" / None=未対応）。"""
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux") and shutil.which("systemctl"):
        return "systemd"
    return None


def gw_executable() -> str:
    """サービス定義に書く gw 実行ファイルの絶対パスを解決する。

    実行中の自分自身（sys.argv[0]）を優先する——`uv tool install` の
    `~/.local/bin/gw` は再インストールでも変わらない安定パスで、venv の実体
    （realpath）より寿命が長い。見つからなければ PATH から引く。
    """
    argv0 = sys.argv[0] or ""
    if os.path.basename(argv0) == "gw":
        path = os.path.abspath(argv0)
        if os.access(path, os.X_OK):
            return path
    path = shutil.which("gw")
    if path:
        return os.path.abspath(path)
    raise RuntimeError(
        "gw 実行ファイルが見つかりません。`make install` で導入してから gw enable してください"
    )


def service_log_path() -> str:
    """サービス（launchd 経由の gw serve）の標準出力/エラーの逃がし先。"""
    return os.path.join(log_dir(), "service.log")


def launchd_plist_path() -> str:
    return os.path.join(
        os.path.expanduser("~"), "Library", "LaunchAgents", f"{LAUNCHD_LABEL}.plist"
    )


def systemd_unit_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(base, "systemd", "user", SYSTEMD_UNIT)


def render_launchd_plist(gw: str) -> str:
    """LaunchAgent の plist を組み立てる（純粋関数・テスト可能）。

    - RunAtLoad: ログイン（bootstrap）時に起動する
    - KeepAlive.SuccessfulExit=false: **異常終了のときだけ**復活する。`gw stop`
      （SIGTERM → 正常終了 0）では復活しない——手動停止の意思を上書きしない
    - ThrottleInterval: クラッシュループでも 15 秒に 1 回までに頭打ち
    """
    log = service_log_path()
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{gw}</string>
        <string>serve</string>
        <string>--managed</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key><false/>
    </dict>
    <key>ThrottleInterval</key><integer>15</integer>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def render_systemd_unit(gw: str) -> str:
    """systemd user unit を組み立てる（純粋関数・テスト可能）。

    Restart=on-failure が launchd の SuccessfulExit=false と同じ「異常終了のみ復活」。
    StartLimit* はクラッシュループの頭打ち（60 秒に 5 回まで）。
    """
    return f"""[Unit]
Description=local-llm-server gateway (gw)
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
ExecStart={gw} serve --managed
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def is_enabled() -> bool:
    """自動起動が登録済みか（サービス定義ファイルの有無で判定）。"""
    kind = service_kind()
    if kind == "launchd":
        return os.path.isfile(launchd_plist_path())
    if kind == "systemd":
        return os.path.isfile(systemd_unit_path())
    return False


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def enable() -> None:
    """自動起動を登録して今すぐ起動する（冪等。失敗は RuntimeError）。"""
    kind = service_kind()
    if kind is None:
        raise RuntimeError("この OS では自動起動の登録に未対応です（手動 `gw start` を使ってください）")
    gw = gw_executable()
    if kind == "launchd":
        _enable_launchd(gw)
    else:
        _enable_systemd(gw)


def disable() -> bool:
    """自動起動を解除する（稼働中のサービスも止める）。登録が無ければ False。"""
    kind = service_kind()
    if kind is None:
        return False
    if kind == "launchd":
        return _disable_launchd()
    return _disable_systemd()


def _enable_launchd(gw: str) -> None:
    path = launchd_plist_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(log_dir(), exist_ok=True)  # StandardOutPath の親（無いと launchd が書けない）
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_launchd_plist(gw))
    uid = os.getuid()
    # 旧登録が残っていても載せ直せるよう、先に下ろす（未登録エラーは無視 = 冪等）。
    _run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"])
    res = _run(["launchctl", "bootstrap", f"gui/{uid}", path])
    if res.returncode != 0:
        # 古い macOS 向けフォールバック（bootstrap が無い/失敗する環境では load -w）。
        res = _run(["launchctl", "load", "-w", path])
        if res.returncode != 0:
            raise RuntimeError(
                f"launchctl への登録に失敗しました: {res.stderr.strip() or res.stdout.strip()}"
            )


def _disable_launchd() -> bool:
    path = launchd_plist_path()
    existed = os.path.isfile(path)
    uid = os.getuid()
    _run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"])  # 稼働中なら止まる
    _run(["launchctl", "unload", "-w", path])  # 旧方式の残骸も掃除（無ければ無害）
    try:
        os.remove(path)
    except OSError:
        pass
    return existed


def _enable_systemd(gw: str) -> None:
    path = systemd_unit_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_systemd_unit(gw))
    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT],
    ):
        res = _run(cmd)
        if res.returncode != 0:
            raise RuntimeError(
                f"{' '.join(cmd)} に失敗しました: {res.stderr.strip() or res.stdout.strip()}"
            )


def _disable_systemd() -> bool:
    path = systemd_unit_path()
    existed = os.path.isfile(path)
    _run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT])
    try:
        os.remove(path)
    except OSError:
        pass
    _run(["systemctl", "--user", "daemon-reload"])
    return existed
