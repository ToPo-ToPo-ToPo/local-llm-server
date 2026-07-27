"""`gateway.toml` のスキーマ移行 —— 更新で廃止・改名されたキーに設定を自動追従させる。

設定ファイルはユーザーの持ち物（`~/.config/local-llm-server/gateway.toml`）で、更新しても
勝手には変わらない。一方コード側でキーが廃止されると、設定に残った旧キーは**黙って無視される**
だけになり、「書いてあるのに効かない」という一番わかりにくい状態になる。ここはその差を埋める:
更新後の起動時（と `gw update` / `gw migrate`）に旧キーを実際に書き換え・削除する。

方針:

- **TOML を読み書きし直さない**。行単位のテキスト編集で済ませ、ユーザーのコメント・並び・
  書式をそのまま残す（`tomllib` は読み取り専用で、書き戻すとコメントが全部消える）。
- **冪等**。既に移行済みなら何もしない（毎起動で走らせて安全）。
- **対象はトップレベルのキーだけ**。`[[models]]` 等のテーブル内の同名キーには触らない。
- **判断がつかない行は触らない**。複数行にまたがる値（配列・インラインテーブル）は自動で
  消さず、「手で消してください」と伝えるだけにする（設定を壊さない方を優先）。
"""
from __future__ import annotations

import os
import re
import shutil

# 廃止されたトップレベルキー: キー名 -> (廃止したバージョン, 理由)。
# ここに足すだけで、次回起動時に既存の設定から消える。
OBSOLETE_KEYS: dict[str, tuple[str, str]] = {
    "vision_model": (
        "0.36.3",
        "画像入りリクエストの自動振り分けを廃止（mlx-vlm 0.6.7 で Qwen3.6 の画像入力が直り、"
        "gemma-4 系へ逃がす回避策が不要になった）",
    ),
}

# 改名されたトップレベルキー: 旧キー名 -> (新キー名, 改名したバージョン)。
# 値と行末コメントは保ったままキー名だけ差し替える。
RENAMED_KEYS: dict[str, tuple[str, str]] = {}

# `key = value` 行（行頭の字下げも許す）。TOML のベアキーだけを見る（クォートキーは対象外）。
_KEY_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*=\s*)(.*)$", re.DOTALL)


def _is_table_header(line: str) -> bool:
    """`[table]` / `[[array of tables]]` の見出し行か（以降はトップレベルではない）。"""
    return line.lstrip().startswith("[")


def _is_hanging_comment(line: str) -> bool:
    """直前のキーにぶら下がる継続コメント行か（**字下げされた** `#` だけの行）。

    このリポジトリの `gateway.toml` は、キーの説明を次行以降に字下げして続ける書き方をする。
    キーを消すときはその説明も道連れにしないと、宙に浮いたコメントだけが残る。行頭から始まる
    コメント（字下げ無し）は独立した見出しコメントなので残す。
    """
    if not line[:1].isspace():
        return False
    return line.lstrip().startswith("#")


def _value_is_self_contained(value: str) -> bool:
    """`=` の右側がその行だけで完結しているか（複数行の配列・インラインテーブルを弾く）。

    行末コメントは値ではないので、`#` 以降は見ない（ただし文字列中の `#` は区別できないので、
    クォートを含む行は素直に「クォートが閉じていれば完結」とみなす）。
    """
    v = value.strip()
    if not v:
        return False           # 値が次行から始まる書き方（`key =` で改行）は触らない
    if v.count('"') % 2 or v.count("'") % 2:
        return False           # クォートが閉じていない＝複数行文字列
    head = v.split("#", 1)[0]
    return head.count("[") <= head.count("]") and head.count("{") <= head.count("}")


def migrate_text(text: str) -> tuple[str, list[str]]:
    """設定テキストを移行して `(新しいテキスト, 変更点の説明)` を返す。

    変更が無ければ説明は空リスト（テキストは元のまま）。純粋関数なのでファイルを触らない。
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    notes: list[str] = []
    in_root = True
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_table_header(line):
            in_root = False
        m = _KEY_RE.match(line) if in_root else None
        key = m.group(2) if m else None

        if key in OBSOLETE_KEYS:
            version, why = OBSOLETE_KEYS[key]
            if not _value_is_self_contained(m.group(4)):
                notes.append(
                    f"要手動削除: {key} は {version} で廃止（{why}）。値が複数行にまたがるため"
                    "自動では消しませんでした"
                )
                out.append(line)
                i += 1
                continue
            i += 1
            while i < len(lines) and _is_hanging_comment(lines[i]):
                i += 1               # ぶら下がっている説明コメントも一緒に消す
            notes.append(f"削除: {key}（{version} で廃止 — {why}）")
            continue

        if key in RENAMED_KEYS:
            new_key, version = RENAMED_KEYS[key]
            out.append(f"{m.group(1)}{new_key}{m.group(3)}{m.group(4)}")
            notes.append(f"改名: {key} -> {new_key}（{version}）")
            i += 1
            continue

        out.append(line)
        i += 1
    return "".join(out), notes


def migrate_file(path: str, *, dry_run: bool = False, backup: bool = True) -> list[str]:
    """`path` の設定を移行し、変更点の説明を返す（変更なし・ファイル無しなら空リスト）。

    書き込みは一時ファイル経由の `os.replace` で原子的に行う —— 稼働中デーモンの
    ホットリロード監視が書きかけの中身を読んで「壊れた TOML」と判定しないため。
    `backup=True` なら上書き前に `<path>.bak` を残す。`dry_run=True` なら書き換えずに
    説明だけ返す（`gw migrate --dry-run` 用）。
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        return []
    new_text, notes = migrate_text(text)
    if dry_run or new_text == text:
        return notes
    if backup:
        shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    os.replace(tmp, path)
    return notes


def migrate_quietly(path: str | None, *, log=None) -> None:
    """起動経路から呼ぶ移行（**失敗しても起動を止めない**）。

    設定の移行はあくまで親切機能なので、権限エラー等で書けなくても旧キーが無視されるだけの
    元の状態に戻るだけ。起動そのものは続ける。`log` は 1 行受け取る callable（既定は無出力）。
    """
    if not path:
        return
    try:
        notes = migrate_file(path)
    except OSError as exc:  # noqa: BLE001 - 移行の失敗で起動を止めない
        if log:
            log(f"gateway.toml の自動移行に失敗しました（設定はそのまま使います）: {exc}")
        return
    if notes and log:
        log(f"gateway.toml を更新に合わせて書き換えました（元の内容は {path}.bak）:")
        for note in notes:
            log(f"  - {note}")
