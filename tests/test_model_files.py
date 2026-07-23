"""モデルファイル管理 CLI（Phase 2: gw pull / rm / show）のテスト。

ネットワーク（HF Hub）には触れない——ファイル選定・削除・表示の組み立てを、
偽のキャッシュディレクトリと純粋関数で検証する。
"""
from __future__ import annotations

import json
import os
import types

import pytest

from local_llm_server import cli as cli_mod
from local_llm_server import server as srv_mod
from local_llm_server.cli import (
    _split_model_spec,
    build_show_rows,
    cmd_rm,
    plan_gguf_pull,
)


# --- spec 解釈 ---------------------------------------------------------------
def test_split_model_spec():
    assert _split_model_spec("org/repo:Q4_K_M") == ("org/repo", "Q4_K_M")
    assert _split_model_spec("org/repo") == ("org/repo", "")
    for bad in ("/abs/path", "./rel", "~/home", "noslash", "a/b/c"):
        with pytest.raises(ValueError):
            _split_model_spec(bad)


# --- pull のファイル選定（GGUF） ---------------------------------------------
def test_plan_gguf_pull_single_body_includes_mmproj():
    files = ["model-Q4_K_M.gguf", "mmproj-F16.gguf", "README.md"]
    assert plan_gguf_pull(files, "") == ["model-Q4_K_M.gguf", "mmproj-F16.gguf"]


def test_plan_gguf_pull_requires_selector_for_multiple_quants():
    files = ["m-Q4_K_M.gguf", "m-Q8_0.gguf"]
    with pytest.raises(ValueError) as exc:
        plan_gguf_pull(files, "")
    assert "Q4_K_M" in str(exc.value) and "Q8_0" in str(exc.value)  # 候補を案内する
    assert plan_gguf_pull(files, "q8_0") == ["m-Q8_0.gguf"]  # セレクタは大小無視


def test_plan_gguf_pull_shards_count_as_one_body():
    files = ["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf"]
    assert plan_gguf_pull(files, "") == files  # シャードは 1 本体 → 両方取得


def test_plan_gguf_pull_excludes_mtp_head():
    files = ["m-Q4.gguf", "m-F16-MTP.gguf"]
    assert plan_gguf_pull(files, "") == ["m-Q4.gguf"]


def test_plan_gguf_pull_no_match():
    with pytest.raises(ValueError):
        plan_gguf_pull(["m-Q4.gguf"], "Q8")
    with pytest.raises(ValueError):
        plan_gguf_pull(["README.md"], "")


# --- 偽キャッシュのヘルパー ---------------------------------------------------
def _make_cache(tmp_path, monkeypatch, repo: str, files: dict[str, bytes]):
    """models--org--name/snapshots/main/ 配下にファイルを置いた偽キャッシュを作る。"""
    org, name = repo.split("/", 1)
    snap = tmp_path / f"models--{org}--{name}" / "snapshots" / "main"
    snap.mkdir(parents=True)
    for fname, content in files.items():
        (snap / fname).write_bytes(content)
    monkeypatch.setattr(srv_mod, "_hf_hub_cache", lambda: str(tmp_path))
    monkeypatch.setattr(cli_mod, "_hf_hub_cache", lambda: str(tmp_path))
    return snap


# --- rm ----------------------------------------------------------------------
def test_rm_deletes_repo_with_yes(tmp_path, monkeypatch):
    _make_cache(tmp_path, monkeypatch, "org/gone", {"model.safetensors": b"x" * 100})
    monkeypatch.setattr(cli_mod, "read_gateway_runtime", lambda: None)  # デーモン無し
    rc = cmd_rm(None, types.SimpleNamespace(model="org/gone", yes=True))
    assert rc == 0
    assert not os.path.isdir(str(tmp_path / "models--org--gone"))


def test_rm_refuses_when_loaded(tmp_path, monkeypatch):
    _make_cache(tmp_path, monkeypatch, "org/busy", {"model.safetensors": b"x"})
    monkeypatch.setattr(cli_mod, "read_gateway_runtime",
                        lambda: {"host": "127.0.0.1", "port": 1})
    monkeypatch.setattr(cli_mod, "gateway_admin_status",
                        lambda h, p: {"models": [{"model": "org/busy", "loaded": True}]})
    rc = cmd_rm(None, types.SimpleNamespace(model="org/busy", yes=True))
    assert rc == 1
    assert os.path.isdir(str(tmp_path / "models--org--busy"))  # 消えていない


def test_rm_missing_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_hf_hub_cache", lambda: str(tmp_path))
    rc = cmd_rm(None, types.SimpleNamespace(model="org/nothing", yes=True))
    assert rc == 1


def test_rm_aborts_without_confirmation(tmp_path, monkeypatch):
    _make_cache(tmp_path, monkeypatch, "org/keep", {"model.safetensors": b"x"})
    monkeypatch.setattr(cli_mod, "read_gateway_runtime", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_a: "n")
    rc = cmd_rm(None, types.SimpleNamespace(model="org/keep", yes=False))
    assert rc == 1
    assert os.path.isdir(str(tmp_path / "models--org--keep"))


# --- show --------------------------------------------------------------------
def test_show_mlx_model(tmp_path, monkeypatch):
    cfg = {"model_type": "qwen3", "max_position_embeddings": 32768,
           "quantization": {"bits": 4}, "vision_config": {}}
    _make_cache(tmp_path, monkeypatch, "org/Foo-27B-mlx-4bit",
                {"config.json": json.dumps(cfg).encode(),
                 "model.safetensors": b"w" * 2048})
    monkeypatch.setattr(cli_mod, "read_gateway_runtime", lambda: None)
    rows = dict(build_show_rows("org/Foo-27B-mlx-4bit"))
    assert rows["バックエンド"] == "mlx-vlm"
    assert rows["パラメータ数"].startswith("27B")
    assert rows["コンテキスト長"] == "32,768"
    assert rows["量子化"] == "4bit（mlx）"
    assert "対応" in rows["画像入力"]


def test_show_gguf_model(tmp_path, monkeypatch):
    _make_cache(tmp_path, monkeypatch, "org/Bar-9B-GGUF",
                {"bar-Q4_K_M.gguf": b"g" * 4096, "mmproj-F16.gguf": b"m"})
    monkeypatch.setattr(cli_mod, "read_gateway_runtime", lambda: None)
    rows = dict(build_show_rows("org/Bar-9B-GGUF"))
    assert rows["バックエンド"] == "llama-cpp"
    assert rows["GGUF"] == "bar-Q4_K_M.gguf"
    assert rows["量子化"] == "Q4_K_M"
    assert "対応" in rows["画像入力"]


def test_show_uncached_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_hf_hub_cache", lambda: str(tmp_path))
    monkeypatch.setattr(srv_mod, "_hf_hub_cache", lambda: str(tmp_path))
    rows = dict(build_show_rows("org/NotYet-8B-mlx"))
    assert "未取得" in rows["キャッシュ"]
    assert "gw pull" in rows["キャッシュ"]


def test_show_reports_loaded_state(tmp_path, monkeypatch):
    _make_cache(tmp_path, monkeypatch, "org/live-mlx",
                {"config.json": b"{}", "model.safetensors": b"w"})
    monkeypatch.setattr(cli_mod, "read_gateway_runtime",
                        lambda: {"host": "127.0.0.1", "port": 1})
    monkeypatch.setattr(cli_mod, "gateway_admin_status",
                        lambda h, p: {"models": [{"model": "org/live-mlx",
                                                  "loaded": True, "inflight": 2}]})
    rows = dict(build_show_rows("org/live-mlx"))
    assert rows["状態"].startswith("ロード中")
