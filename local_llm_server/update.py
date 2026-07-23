"""PyPI 公開版の検知と、ソースの自動更新（git クローン運用向け）。

このリポジトリは PyPI に公開しつつ、実運用は **GitHub から clone して `uv run gw`** で
動かす。そのためバージョンアップが手作業になりがち。ここでは「PyPI に新版が出たら検知し、
作業ツリーがクリーンなら `git pull --ff-only` で追従する」ための小さな道具を提供する。
常駐デーモン（daemon._run_gateway_locked の更新ウォッチャー）が idle 時にこれを使い、適用後は
run_gateway が reexec_daemon で自分自身を新コードに置き換える（手動なら `gw update`）。

方針（安全側）:
  - **git クローン & upstream 追跡ブランチ & 作業ツリーがクリーンな時だけ**適用する
    （開発中の PC＝未コミット変更がある場合は適用せず「保留」を表示。WIP を壊さない）。
  - ネットワーク I/O は短いタイムアウトで、失敗しても常に None/False を返す（オフラインでも
    TUI の起動を妨げない）。
  - PyPI 公開版を「トリガー」に使う（公開時に main へ push 済みなので pull で同じコードが得られる）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_PKG = "local-llm-server"
_PYPI_JSON = f"https://pypi.org/pypi/{_PKG}/json"


def installed_version() -> str | None:
    """今動いているパッケージのバージョン（取得不可なら None）。"""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # 念のため（3.8+ では標準）
        return None
    try:
        return version(_PKG)
    except PackageNotFoundError:
        return None


def latest_pypi_version(timeout: float = 3.0) -> str | None:
    """PyPI の最新公開版（取得失敗・オフラインは None）。"""
    try:
        with urllib.request.urlopen(_PYPI_JSON, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    info = data.get("info") if isinstance(data, dict) else None
    ver = info.get("version") if isinstance(info, dict) else None
    return ver if isinstance(ver, str) and ver else None


def _version_key(v: str):
    """比較用キー。packaging があればそれを使い、無ければ数値タプルに落とす。"""
    try:
        from packaging.version import Version

        return Version(v)
    except Exception:  # noqa: BLE001 - packaging 不在や不正版はタプル比較にフォールバック
        parts = []
        for token in v.split("."):
            num = "".join(ch for ch in token if ch.isdigit())
            parts.append(int(num) if num else 0)
        return tuple(parts)


def is_newer(candidate: str | None, current: str | None) -> bool:
    """candidate が current より新しいバージョンか（どちらか不明なら False）。"""
    if not candidate or not current:
        return False
    try:
        return _version_key(candidate) > _version_key(current)
    except TypeError:
        # packaging 版とタプルが混ざる等の異常時は保守的に「更新なし」。
        return False


def repo_root() -> Path | None:
    """git クローン運用なら、この起動が読んでいるソースの repo ルートを返す。

    パッケージソース（local_llm_server/__init__.py）の 2 つ上に `.git` と `pyproject.toml`
    があれば「編集可能な git クローン」とみなす。PyPI から `uv tool install` した場合等は
    `.git` が無いので None（＝自動更新の対象外）。
    """
    try:
        import local_llm_server

        root = Path(local_llm_server.__file__).resolve().parent.parent
    except Exception:  # noqa: BLE001
        return None
    if (root / ".git").is_dir() and (root / "pyproject.toml").is_file():
        return root
    return None


def _find_uv() -> str | None:
    """uv 実行ファイルを探す（PATH → 標準の導入先の順。無ければ None）。

    launchd / systemd 配下のデーモンは PATH が最小（/usr/bin:/bin 等）で、`uv` が
    PATH に居ないことがある。その環境でも自動更新の依存同期が黙って失敗しないよう、
    標準の導入先を直接当たる。
    """
    import os
    import shutil

    found = shutil.which("uv")
    if found:
        return found
    for cand in (
        os.path.expanduser("~/.local/bin/uv"),   # スタンドアロンインストーラ既定
        "/opt/homebrew/bin/uv",                   # Homebrew（Apple Silicon）
        "/usr/local/bin/uv",                      # Homebrew（Intel）等
    ):
        if os.access(cand, os.X_OK):
            return cand
    return None


def tool_env_root() -> Path | None:
    """uv tool の隔離 venv（`uv tool install` 導入）から動いていれば、その venv ルート。

    `make install` の標準導入では gw は `~/.local/share/uv/tools/local-llm-server/` の
    venv で動く。この venv の依存は `uv sync`（プロジェクト venv 用）では更新されず、
    `uv tool install --reinstall` でしか入れ直せない——自動更新が依存の追加を
    取りこぼさないための判定に使う。`uv run gw`（プロジェクト venv）なら None。
    """
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.name == _PKG and parent.parent.name == "tools" \
                and parent.parent.parent.name == "uv":
            return parent
    return None


def refresh_tool_env(root: Path | None, timeout: float = 600.0) -> tuple[bool, str]:
    """gw が動いている uv tool venv の依存を、更新後の pyproject に合わせて入れ直す。

    editable 導入はコード変更こそ即反映されるが、**依存の追加は再インストールしないと
    入らない**（例: メニューバーアイコンの pyobjc 追加が旧環境では欠けて、トレイが
    静かに機能を落とした）。tool venv で動いていないとき（`uv run gw`）は何もしない
    ——そちらは `uv sync` / 次回 `uv run` が同期する。

    呼ぶタイミングは**再起動の直前**（モデルサーバー停止後）。venv を入れ直した後に
    古いプロセスが新規 import をする時間を最小にするため。
    """
    if tool_env_root() is None:
        return True, "tool venv ではない（uv sync が受け持つ）"
    if root is None:
        return False, "git クローン運用ではない"
    uv = _find_uv()
    if uv is None:
        return False, "uv が見つからない（PATH と ~/.local/bin を確認）"
    try:
        res = subprocess.run(
            [uv, "tool", "install", "--editable", str(root), "--reinstall", "--quiet"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"uv tool install を実行できませんでした: {exc}"
    if res.returncode != 0:
        return False, (res.stderr or res.stdout).strip()[:200] or "uv tool install 失敗"
    return True, "tool venv の依存を入れ直しました"


def _git(root: Path, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _tracks_upstream(root: Path) -> bool:
    """現在のブランチが upstream（origin/... 等）を追跡しているか（ff pull の前提）。"""
    try:
        r = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def _source_version(root: Path) -> str | None:
    """クローンの pyproject.toml が宣言するバージョン（＝いまチェックアウト中のソースの版）。

    editable インストールでは `importlib.metadata` の版はインストール時に固定される一方、
    `git pull` は pyproject.toml を書き換える。更新判定は「次に動くコードの版」で行うべきなので
    ソース（pyproject）の版を真とする —— これを使わないと、pull でコードが最新になっても
    固定メタデータが古いまま `available` が真に居座り、**再起動ループ**になる。
    """
    try:
        with open(root / "pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return None
    v = data.get("project", {}).get("version") if isinstance(data, dict) else None
    return v if isinstance(v, str) and v else None


def _current_branch(root: Path) -> str | None:
    try:
        r = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _default_branch(root: Path) -> str | None:
    """origin の既定ブランチ名（例 "main"）。取得できなければ None。"""
    try:
        r = _git(root, "rev-parse", "--abbrev-ref", "origin/HEAD")
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    name = r.stdout.strip()          # 例 "origin/main"
    return name.split("/", 1)[1] if "/" in name else (name or None)


def _on_default_branch(root: Path) -> bool:
    """現在のブランチが origin の既定ブランチ（main）か。

    自動更新は**既定ブランチにいるときだけ**行う。開発用の機能ブランチでは `git pull --ff-only`
    してもリリース（main）は取り込まれず版が上がらないので、放置すると再起動ループになる。
    さらに「開発中の別ブランチを勝手に触らない」という直感にも合う（機能ブランチでは自動更新
    しない）。既定を特定できないときは main/master 名で保守的に判定する。
    """
    cur = _current_branch(root)
    if cur is None:
        return False
    default = _default_branch(root)
    if default is None:
        return cur in ("main", "master")
    return cur == default


def _working_tree_clean(root: Path) -> bool:
    """未コミットの変更（追跡ファイル）が無いか。開発中 PC の WIP を守るためのガード。"""
    try:
        r = _git(root, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and not r.stdout.strip()


@dataclass
class UpdateStatus:
    """更新チェックの結果（TUI のバナー表示と適用判断に使う）。"""

    current: str | None       # 稼働中（＝チェックアウト中のソース）バージョン
    latest: str | None        # PyPI 最新
    available: bool           # PyPI が現行より新しい
    can_apply: bool           # 既定ブランチ & git クローン & クリーン & upstream 追跡（＝自動適用してよい）
    # "ok" / "not-a-git-clone" / "not-on-default-branch" / "no-upstream" / "dirty" / "offline"
    reason: str


def check(timeout: float = 3.0) -> UpdateStatus:
    """現行と PyPI 最新を比べ、自動適用できるかまで含めて判定する（副作用なし）。

    現行版はクローンの pyproject.toml（ソース）から取る。editable インストールで固定される
    メタデータ版ではなく、pull で上がる版を見る（→ 再起動ループを防ぐ）。自動適用は**既定
    ブランチ（main）でクリーン & upstream 追跡**のときだけ。機能ブランチや WIP は触らない。
    """
    latest = latest_pypi_version(timeout)
    root = repo_root()
    # 現行版はソース（pyproject）優先。取れないときだけインストールメタデータへフォールバック。
    cur = (_source_version(root) if root else None) or installed_version()
    available = is_newer(latest, cur)
    if latest is None:
        return UpdateStatus(cur, latest, False, False, "offline")
    if root is None:
        return UpdateStatus(cur, latest, available, False, "not-a-git-clone")
    if not _on_default_branch(root):
        return UpdateStatus(cur, latest, available, False, "not-on-default-branch")
    if not _tracks_upstream(root):
        return UpdateStatus(cur, latest, available, False, "no-upstream")
    if not _working_tree_clean(root):
        return UpdateStatus(cur, latest, available, False, "dirty")
    return UpdateStatus(cur, latest, available, True, "ok")


def apply_update(root: Path | None = None, timeout: float = 120.0) -> tuple[bool, str]:
    """`git pull --ff-only`（＋可能なら `uv sync`）でソースを最新へ更新する。

    成功したら (True, メッセージ)。呼び出し側は**プロセスを再起動**して新コードを読み込むこと
    （実行中の Python は古いコードを保持したままなので、reexec_daemon で入れ替える）。
    直前に作業ツリーを再確認し、汚れていれば適用しない（チェック〜適用間の変更に対する保険）。
    """
    root = root or repo_root()
    if root is None:
        return False, "git クローン運用ではありません（自動更新の対象外）"
    if not _working_tree_clean(root):
        return False, "作業ツリーに未コミットの変更があります"
    try:
        pull = _git(root, "pull", "--ff-only", timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"git pull を実行できませんでした: {exc}"
    if pull.returncode != 0:
        return False, f"git pull 失敗: {(pull.stderr or pull.stdout).strip()[:200]}"
    # 依存が変わっている可能性があるので uv sync を試みる（プロジェクト venv 用。
    # tool venv の依存入れ直しは refresh_tool_env が再起動直前に行う）。uv は launchd
    # 配下だと PATH に居ないことがあるので _find_uv で標準の導入先まで探す。
    uv = _find_uv()
    if uv is not None:
        try:
            subprocess.run(
                [uv, "sync", "--quiet"], cwd=str(root),
                capture_output=True, text=True, timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            pass  # 致命ではない（再起動側の uv run が拾う）
    return True, (pull.stdout or "").strip()[:200] or "更新しました"


def reexec_daemon() -> None:
    """現在の Python でゲートウェイ本体を再 exec する（更新後、新コードを読み込むため）。

    デーモン（`python -m local_llm_server`）が idle 時に自動更新を適用したあと、自分自身を
    新コードで置き換えるために呼ぶ。呼ぶ前に **単一起動ロックの解放とポートの解放（server_close）
    を済ませておくこと**（execv は開いた fd を引き継ぐため、握ったままだと再取得で自分自身と
    衝突する）。CWD を保つ（./gateway.toml の解決が変わらない）。同一 venv の python を使うので、
    git pull 済みの新ソースと uv sync 済みの依存で立ち上がる。呼ぶと戻らない。
    """
    import os

    os.execv(sys.executable, [sys.executable, "-m", "local_llm_server"])
