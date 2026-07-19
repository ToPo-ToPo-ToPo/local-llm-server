"""update.py（PyPI 新版検知・git 追従）のテスト。ネットワーク/git は monkeypatch で隔離。"""
from __future__ import annotations

from pathlib import Path

from local_llm_server import update


# --- バージョン比較 --------------------------------------------------------
def test_is_newer():
    assert update.is_newer("0.22.0", "0.21.0") is True
    assert update.is_newer("0.21.1", "0.21.0") is True
    assert update.is_newer("1.0.0", "0.21.0") is True
    assert update.is_newer("0.21.0", "0.21.0") is False
    assert update.is_newer("0.9.0", "0.21.0") is False   # 文字列比較なら 9>2 で誤判定する所
    assert update.is_newer("0.20.1", "0.21.0") is False


def test_is_newer_handles_missing():
    assert update.is_newer(None, "0.21.0") is False
    assert update.is_newer("0.22.0", None) is False
    assert update.is_newer(None, None) is False


# --- check（判定の分岐）----------------------------------------------------
def test_check_offline_returns_offline(monkeypatch):
    monkeypatch.setattr(update, "installed_version", lambda: "0.21.0")
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: None)
    st = update.check()
    assert st.available is False and st.can_apply is False and st.reason == "offline"


def test_check_not_a_git_clone(monkeypatch):
    monkeypatch.setattr(update, "installed_version", lambda: "0.21.0")
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: "0.22.0")
    monkeypatch.setattr(update, "repo_root", lambda: None)
    st = update.check()
    assert st.available is True and st.can_apply is False
    assert st.reason == "not-a-git-clone"
    assert st.latest == "0.22.0"


def test_check_dirty_tree_holds(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "installed_version", lambda: "0.21.0")
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: "0.22.0")
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "_on_default_branch", lambda root: True)
    monkeypatch.setattr(update, "_tracks_upstream", lambda root: True)
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: False)
    st = update.check()
    assert st.available is True and st.can_apply is False and st.reason == "dirty"


def test_check_no_upstream_holds(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "installed_version", lambda: "0.21.0")
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: "0.22.0")
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "_on_default_branch", lambda root: True)
    monkeypatch.setattr(update, "_tracks_upstream", lambda root: False)
    st = update.check()
    assert st.can_apply is False and st.reason == "no-upstream"


def test_check_non_default_branch_holds(monkeypatch, tmp_path):
    # 機能ブランチ（既定ブランチでない）では、新版があっても自動適用しない（開発を邪魔しない）。
    monkeypatch.setattr(update, "installed_version", lambda: "0.21.0")
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: "0.22.0")
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "_on_default_branch", lambda root: False)
    st = update.check()
    assert st.available is True and st.can_apply is False
    assert st.reason == "not-on-default-branch"


def test_check_ok_can_apply(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "installed_version", lambda: "0.21.0")
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: "0.22.0")
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "_on_default_branch", lambda root: True)
    monkeypatch.setattr(update, "_tracks_upstream", lambda root: True)
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: True)
    st = update.check()
    assert st.available is True and st.can_apply is True and st.reason == "ok"


def test_check_uses_source_version_over_metadata(monkeypatch, tmp_path):
    # 現行版はクローンの pyproject（ソース）優先 —— pull で版が上がれば available が False に
    # なりループしないことの担保。固定メタデータ(0.21.0)ではなくソース(0.22.0)を見る。
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "local-llm-server"\nversion = "0.22.0"\n', encoding="utf-8")
    monkeypatch.setattr(update, "installed_version", lambda: "0.21.0")  # 固定メタデータは古い
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: "0.22.0")
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "_on_default_branch", lambda root: True)
    monkeypatch.setattr(update, "_tracks_upstream", lambda root: True)
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: True)
    st = update.check()
    assert st.current == "0.22.0"          # ソース版を採用
    assert st.available is False           # ソース==PyPI なので更新なし（＝ループしない）


def test_check_same_version_not_available(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "installed_version", lambda: "0.22.0")
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: "0.22.0")
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "_on_default_branch", lambda root: True)
    monkeypatch.setattr(update, "_tracks_upstream", lambda root: True)
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: True)
    st = update.check()
    assert st.available is False and st.can_apply is True  # 追従可能だが更新は無い


# --- apply_update（git 呼び出しは monkeypatch）-----------------------------
def test_apply_update_refuses_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: False)
    ok, msg = update.apply_update(root=tmp_path)
    assert ok is False and "変更" in msg


def test_apply_update_runs_pull_and_sync(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: True)
    calls = []

    class _R:
        def __init__(self, rc=0, out="Updating abc..def", err=""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def fake_git(root, *args, timeout=30.0):
        calls.append(("git", args))
        return _R()

    def fake_run(cmd, **kw):
        calls.append(("run", tuple(cmd)))
        return _R()

    monkeypatch.setattr(update, "_git", fake_git)
    monkeypatch.setattr(update.subprocess, "run", fake_run)
    ok, msg = update.apply_update(root=tmp_path)
    assert ok is True
    # git pull --ff-only が呼ばれ、続いて uv sync が試行される。
    assert ("git", ("pull", "--ff-only")) in calls
    assert any(c[0] == "run" and c[1][:2] == ("uv", "sync") for c in calls)


def test_apply_update_reports_pull_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: True)

    class _R:
        returncode = 1
        stdout = ""
        stderr = "Not possible to fast-forward"

    monkeypatch.setattr(update, "_git", lambda root, *a, **k: _R())
    ok, msg = update.apply_update(root=tmp_path)
    assert ok is False and "fast-forward" in msg


def test_apply_update_no_repo(monkeypatch):
    monkeypatch.setattr(update, "repo_root", lambda: None)
    ok, msg = update.apply_update()
    assert ok is False and "git" in msg
