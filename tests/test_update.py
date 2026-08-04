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
    # git pull --ff-only が呼ばれ、続いて uv sync が試行される（uv は絶対パス解決あり）。
    # Windows では which("uv") が uv.exe を返すため、拡張子を除いた名前で判定する。
    assert ("git", ("pull", "--ff-only")) in calls
    assert any(
        c[0] == "run" and Path(c[1][0]).stem == "uv" and c[1][1] == "sync"
        for c in calls
    )


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


# --- tool venv の依存入れ直し（refresh_tool_env） -----------------------------
# make install（uv tool install --editable）導入では、コードは git pull で即反映される
# 一方、依存の追加は tool venv に入らない。自動更新がこれを取りこぼすと
# 「コードだけ新しく依存が古い」静かな機能欠けになる（実例: pyobjc 不在でトレイ不表示）。
def test_tool_env_root_detects_uv_tools_python(monkeypatch):
    monkeypatch.setattr(update.sys, "prefix",
                        "/Users/x/.local/share/uv/tools/local-llm-server")
    monkeypatch.setattr(
        update.sys, "executable",
        "/Users/x/.local/share/uv/tools/local-llm-server/bin/python3",
    )
    root = update.tool_env_root()
    assert root is not None and root.name == "local-llm-server"


def test_tool_env_root_uses_prefix_not_resolved_executable(monkeypatch, tmp_path):
    """venv の python がシンボリックリンクでも tool venv を検出できる（実バグの回帰テスト）。

    venv の bin/python は uv 管理の素の CPython への symlink であり、旧実装の
    `Path(sys.executable).resolve()` は venv の**外**へ解決されて None を返していた。
    その結果、自動更新の依存入れ直し（refresh_tool_env）が本番で一度も走らず、
    「コードだけ新しく依存が古い」静かな機能欠けを防ぐ仕組み自体が死んでいた。
    sys.prefix（稼働中 venv のルート）で判定することを、実 symlink で検証する。
    """
    base = tmp_path / "python" / "cpython-3.13" / "bin"; base.mkdir(parents=True)
    real = base / "python3.13"; real.write_bytes(b"")
    env = tmp_path / "uv" / "tools" / "local-llm-server"
    (env / "bin").mkdir(parents=True)
    link = env / "bin" / "python"
    link.symlink_to(real)
    monkeypatch.setattr(update.sys, "prefix", str(env))
    monkeypatch.setattr(update.sys, "executable", str(link))
    root = update.tool_env_root()
    assert root is not None and root == env


def test_tool_env_root_none_for_project_venv(monkeypatch):
    monkeypatch.setattr(update.sys, "prefix",
                        "/Users/x/my_program/local-llm-server/.venv")
    monkeypatch.setattr(
        update.sys, "executable",
        "/Users/x/my_program/local-llm-server/.venv/bin/python3",
    )
    assert update.tool_env_root() is None


def test_refresh_tool_env_runs_reinstall(monkeypatch, tmp_path):
    """tool venv 稼働時は uv tool install --editable <root> --reinstall を実行する。"""
    calls = []

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        update, "tool_env_root", lambda: update.Path("/x/uv/tools/local-llm-server"))
    monkeypatch.setattr(update, "_find_uv", lambda: "/opt/homebrew/bin/uv")
    monkeypatch.setattr(update.subprocess, "run",
                        lambda cmd, **kw: calls.append(tuple(cmd)) or _R())
    ok, _msg = update.refresh_tool_env(tmp_path)
    assert ok is True
    assert calls and calls[0][:3] == ("/opt/homebrew/bin/uv", "tool", "install")
    assert "--reinstall" in calls[0] and "--editable" in calls[0]


def _repo_with_deps(tmp_path, lock=b"lock-v1", pyproject=b"[project]"):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "uv.lock").write_bytes(lock)
    (root / "pyproject.toml").write_bytes(pyproject)
    return root


def test_refresh_tool_env_skips_when_deps_unchanged(monkeypatch, tmp_path):
    """依存（uv.lock + pyproject）が前回と同一なら再インストールしない。

    無条件の --reinstall は ~5 秒かかり、zero-drop restart で accept キューに並んだ
    接続の待ち時間の支配項だった。自動更新の大半はコードだけの変更なので、この
    スキップで再起動の窓が ~1〜2 秒になる。
    """
    env_root = tmp_path / "toolenv"; env_root.mkdir()
    root = _repo_with_deps(tmp_path)
    monkeypatch.setattr(update, "tool_env_root", lambda: env_root)
    monkeypatch.setattr(update, "_find_uv", lambda: "/usr/bin/uv")
    calls = []

    class _R:
        returncode = 0; stdout = ""; stderr = ""
    monkeypatch.setattr(update.subprocess, "run",
                        lambda cmd, **kw: calls.append(tuple(cmd)) or _R())

    # 1 回目: マーカーが無い → 再インストールし、成功したのでマーカーを書く
    ok, msg = update.refresh_tool_env(root)
    assert ok and len(calls) == 1
    assert (env_root / update._DEPS_FINGERPRINT_NAME).exists()

    # 2 回目: 依存が同一 → スキップ（uv を呼ばない）
    ok, msg = update.refresh_tool_env(root)
    assert ok and len(calls) == 1 and "スキップ" in msg

    # 依存が変わったら再インストールし、マーカーも更新される
    (root / "uv.lock").write_bytes(b"lock-v2")
    ok, _ = update.refresh_tool_env(root)
    assert ok and len(calls) == 2
    ok, msg = update.refresh_tool_env(root)   # 変更後の 2 回目はまたスキップ
    assert ok and len(calls) == 2 and "スキップ" in msg


def test_refresh_tool_env_no_marker_after_failure(monkeypatch, tmp_path):
    """再インストールが失敗したらマーカーを書かない（次回も安全側＝再試行する）。"""
    env_root = tmp_path / "toolenv"; env_root.mkdir()
    root = _repo_with_deps(tmp_path)
    monkeypatch.setattr(update, "tool_env_root", lambda: env_root)
    monkeypatch.setattr(update, "_find_uv", lambda: "/usr/bin/uv")

    class _Fail:
        returncode = 1; stdout = ""; stderr = "boom"
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, **kw: _Fail())
    ok, _ = update.refresh_tool_env(root)
    assert ok is False
    assert not (env_root / update._DEPS_FINGERPRINT_NAME).exists()


def test_refresh_tool_env_noop_outside_tool_venv(monkeypatch, tmp_path):
    """uv run（プロジェクト venv）稼働時は何もしない（uv sync が受け持つ）。"""
    monkeypatch.setattr(update, "tool_env_root", lambda: None)
    monkeypatch.setattr(update.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    ok, msg = update.refresh_tool_env(tmp_path)
    assert ok is True and "uv sync" in msg


def test_refresh_tool_env_reports_missing_uv(monkeypatch, tmp_path):
    monkeypatch.setattr(
        update, "tool_env_root", lambda: update.Path("/x/uv/tools/local-llm-server"))
    monkeypatch.setattr(update, "_find_uv", lambda: None)
    ok, msg = update.refresh_tool_env(tmp_path)
    assert ok is False and "uv" in msg


# --- 走っているコードが古いことの検知（editable 運用の穴） --------------------
# `git pull` するとディスク上のソース版だけが上がり、稼働中プロセスは古いコードを保持したまま
# になる。ところが更新判定はソース版を見るので「もう最新」と結論し、誰も再起動を促さなかった。
# mark_running_source() が「このプロセスが読み込んだ版」を基準点として記録し、その差を見る。

def test_running_source_version_is_none_before_marking(monkeypatch):
    monkeypatch.setattr(update, "_RUNNING_SOURCE_VERSION", None)
    assert update.running_source_version() is None


def test_mark_running_source_records_the_source_version(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8")
    monkeypatch.setattr(update, "_RUNNING_SOURCE_VERSION", None)
    assert update.mark_running_source(tmp_path) == "1.2.3"
    assert update.running_source_version() == "1.2.3"


def _stub_check(monkeypatch, tmp_path, source_version, latest):
    """pyproject が source_version、PyPI が latest の状況を作る。"""
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{source_version}"\n', encoding="utf-8")
    monkeypatch.setattr(update, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(update, "latest_pypi_version", lambda timeout=3.0: latest)
    monkeypatch.setattr(update, "_on_default_branch", lambda root: True)
    monkeypatch.setattr(update, "_tracks_upstream", lambda root: True)
    monkeypatch.setattr(update, "_working_tree_clean", lambda root: True)


def test_check_flags_restart_required_when_process_is_stale(monkeypatch, tmp_path):
    # pull 済み（ソース 0.37.1）だが、プロセスは 0.37.0 を読み込んだまま。
    _stub_check(monkeypatch, tmp_path, "0.37.1", "0.37.1")
    monkeypatch.setattr(update, "_RUNNING_SOURCE_VERSION", "0.37.0")
    st = update.check()
    assert st.available is False          # 取ってくるものは無い
    assert st.restart_required is True    # でも再起動は要る
    assert st.current == "0.37.1"


def test_check_no_restart_required_when_process_is_current(monkeypatch, tmp_path):
    _stub_check(monkeypatch, tmp_path, "0.37.1", "0.37.1")
    monkeypatch.setattr(update, "_RUNNING_SOURCE_VERSION", "0.37.1")
    assert update.check().restart_required is False


def test_check_no_restart_required_when_unmarked(monkeypatch, tmp_path):
    # CLI のような短命プロセスは記録していない → 誤検知しない。
    _stub_check(monkeypatch, tmp_path, "0.37.1", "0.37.1")
    monkeypatch.setattr(update, "_RUNNING_SOURCE_VERSION", None)
    assert update.check().restart_required is False


def test_check_reports_restart_required_even_when_offline(monkeypatch, tmp_path):
    # PyPI に届かなくても、プロセスが古いことはローカルだけで分かる。
    _stub_check(monkeypatch, tmp_path, "0.37.1", None)
    monkeypatch.setattr(update, "_RUNNING_SOURCE_VERSION", "0.37.0")
    st = update.check()
    assert st.reason == "offline" and st.restart_required is True
