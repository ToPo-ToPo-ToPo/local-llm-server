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

    current: str | None       # 稼働中バージョン
    latest: str | None        # PyPI 最新
    available: bool           # PyPI が現行より新しい
    can_apply: bool           # git クローン & クリーン & upstream 追跡（＝自動適用してよい）
    reason: str               # "ok" / "not-a-git-clone" / "no-upstream" / "dirty" / "offline"


def check(timeout: float = 3.0) -> UpdateStatus:
    """現行と PyPI 最新を比べ、自動適用できるかまで含めて判定する（副作用なし）。"""
    cur = installed_version()
    latest = latest_pypi_version(timeout)
    available = is_newer(latest, cur)
    if latest is None:
        return UpdateStatus(cur, latest, False, False, "offline")
    root = repo_root()
    if root is None:
        return UpdateStatus(cur, latest, available, False, "not-a-git-clone")
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
    # 依存が変わっている可能性があるので uv sync を試みる（無ければ次回 `uv run gw` が同期する）。
    try:
        subprocess.run(
            ["uv", "sync", "--quiet"], cwd=str(root),
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        pass  # uv 不在等。致命ではない（再起動側の uv run が拾う）
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
