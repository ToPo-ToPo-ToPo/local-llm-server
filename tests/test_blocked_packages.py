"""導入禁止パッケージ（litellm 等）が uv 側・隔離 venv の pip 側の両方で塞がれていることの検証。"""
from __future__ import annotations

import tomllib
from pathlib import Path

from local_llm_server import _venv_backend as vb

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_constraints_match_venv_backend():
    # uv（本体 .venv / uv.lock）と隔離 venv の pip で禁止集合がずれないこと。
    tool_uv = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["tool"]["uv"]
    blocked = {value for value in tool_uv["constraint-dependencies"] if value.endswith("<0")}
    assert blocked == {f"{n}<0" for n in vb.BLOCKED_PACKAGES}


def test_lock_has_no_blocked_packages():
    lock = tomllib.loads((ROOT / "uv.lock").read_text("utf-8"))
    names = {p["name"] for p in lock["package"]}
    assert names.isdisjoint(vb.BLOCKED_PACKAGES)



def test_multidict_security_floor_excludes_affected_release():
    from packaging.requirements import Requirement
    tool_uv = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["tool"]["uv"]
    requirements = {r.name: r for r in map(Requirement, tool_uv["constraint-dependencies"])}
    assert "6.9.0" not in requirements["multidict"].specifier
    assert "6.9.1" in requirements["multidict"].specifier
