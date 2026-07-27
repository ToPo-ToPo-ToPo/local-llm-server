"""`gateway.toml` のスキーマ移行（廃止キーの削除・改名）を検証する。

要点は「ユーザーの設定を壊さない」こと —— コメントと並びを保つ、トップレベル以外や複数行の値には
触らない、二度走らせても同じ、書き込みは原子的。
"""
import os

import pytest

from local_llm_server import migrate


@pytest.fixture
def obsolete(monkeypatch):
    """テスト用の廃止キー表に差し替える（本物の表の中身が変わってもテストが揺れない）。"""
    monkeypatch.setattr(migrate, "OBSOLETE_KEYS", {"gone_key": ("9.9.9", "もう使わない")})
    monkeypatch.setattr(migrate, "RENAMED_KEYS", {"old_key": ("new_key", "9.9.9")})


# --- テキスト移行（純関数） ---------------------------------------------------
def test_removes_obsolete_key_and_its_hanging_comments(obsolete):
    text = (
        'port = 8799\n'
        'gone_key = "org/model"  # 行末コメント\n'
        '            # ぶら下がった説明の続き\n'
        '            # もう1行\n'
        'max_resident = 2\n'
    )
    new, notes = migrate.migrate_text(text)
    assert new == 'port = 8799\nmax_resident = 2\n'
    assert len(notes) == 1 and "gone_key" in notes[0] and "9.9.9" in notes[0]


def test_keeps_unindented_comment_after_removed_key(obsolete):
    # 行頭から始まるコメントは独立した見出しなので残す（道連れにしない）。
    text = 'gone_key = "x"\n# 次のセクションの説明\nport = 8799\n'
    new, _ = migrate.migrate_text(text)
    assert new == '# 次のセクションの説明\nport = 8799\n'


def test_renames_key_keeping_value_and_comment(obsolete):
    text = 'old_key = 42   # 大事なメモ\n'
    new, notes = migrate.migrate_text(text)
    assert new == 'new_key = 42   # 大事なメモ\n'
    assert "old_key -> new_key" in notes[0]


def test_leaves_same_name_key_inside_a_table(obsolete):
    # [[models]] の中の同名キーは別物なので触らない。
    text = 'port = 8799\n[[models]]\nmodel = "org/m"\ngone_key = "x"\n'
    new, notes = migrate.migrate_text(text)
    assert new == text and notes == []


def test_does_not_touch_multiline_value_but_reports_it(obsolete):
    # 複数行にまたがる値は自動で消さず、手動削除を促すだけ（設定を壊さない）。
    text = 'gone_key = [\n  "a",\n]\nport = 8799\n'
    new, notes = migrate.migrate_text(text)
    assert new == text
    assert len(notes) == 1 and "要手動削除" in notes[0]


def test_no_change_when_already_migrated(obsolete):
    text = 'port = 8799\nmax_resident = 2\n'
    assert migrate.migrate_text(text) == (text, [])


def test_is_idempotent(obsolete):
    text = 'gone_key = "x"\nport = 8799\n'
    once, notes1 = migrate.migrate_text(text)
    twice, notes2 = migrate.migrate_text(once)
    assert twice == once and notes1 and notes2 == []


# --- ファイル移行 -------------------------------------------------------------
def _write(tmp_path, body):
    p = tmp_path / "gateway.toml"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_migrate_file_rewrites_and_backs_up(tmp_path, obsolete):
    path = _write(tmp_path, 'gone_key = "x"\nport = 8799\n')
    notes = migrate.migrate_file(path)
    assert notes and open(path, encoding="utf-8").read() == 'port = 8799\n'
    # 元の内容は .bak に残る（取り返しがつく）。
    assert open(path + ".bak", encoding="utf-8").read() == 'gone_key = "x"\nport = 8799\n'
    # 一時ファイルは残さない（os.replace で片付く）。
    assert not os.path.exists(path + ".tmp")


def test_migrate_file_dry_run_does_not_write(tmp_path, obsolete):
    body = 'gone_key = "x"\nport = 8799\n'
    path = _write(tmp_path, body)
    notes = migrate.migrate_file(path, dry_run=True)
    assert notes                                     # 何が変わるかは分かる
    assert open(path, encoding="utf-8").read() == body   # でも書き換えない
    assert not os.path.exists(path + ".bak")


def test_migrate_file_no_change_does_not_back_up(tmp_path, obsolete):
    path = _write(tmp_path, "port = 8799\n")
    assert migrate.migrate_file(path) == []
    assert not os.path.exists(path + ".bak")


def test_migrate_file_missing_is_noop(tmp_path):
    assert migrate.migrate_file(str(tmp_path / "nope.toml")) == []


def test_migrate_quietly_survives_write_failure(tmp_path, obsolete, monkeypatch):
    # 書けなくても起動を止めない（旧キーが無視されるだけの元の状態に戻る）。
    path = _write(tmp_path, 'gone_key = "x"\n')
    monkeypatch.setattr(migrate, "migrate_file",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    logged = []
    migrate.migrate_quietly(path, log=logged.append)
    assert any("失敗" in m for m in logged)


def test_migrate_quietly_ignores_missing_path():
    migrate.migrate_quietly(None)     # 例外を出さない


# --- 実際の廃止キー表 ---------------------------------------------------------
def test_real_table_drops_vision_model(tmp_path):
    # 0.36.3 で廃止した vision_model が、実際の表で消えること（回帰防止）。
    path = _write(
        tmp_path,
        'max_resident = 2\n'
        'vision_model = "ToPo-ToPo/gemma-4-31b-it-mlx-4bit"  # 画像入りの振り分け先\n'
        '                          # 続きの説明\n'
        'dynamic = true\n',
    )
    notes = migrate.migrate_file(path)
    text = open(path, encoding="utf-8").read()
    assert "vision_model" not in text
    assert text == 'max_resident = 2\ndynamic = true\n'
    assert any("vision_model" in n for n in notes)
