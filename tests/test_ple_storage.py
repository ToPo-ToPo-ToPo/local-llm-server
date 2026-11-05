"""n-gram 表の外出し(_ple_storage): どの量子化でも、外に出して行ごとに読んだ値が、表を丸ごと戻した値と一致する。"""
import json
import sys

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
ps = pytest.importorskip("local_llm_server._ple_storage")

PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards"
ROWS, WIDTH = (37, 23), 128      # 断片 2 つ(行数は違ってよい)、1 行 128 個


def _checkpoint(root, quantization, *, per_module=None, unquantized=False):
    """小さな表を 2 断片で量子化して保存し、元の値(表を丸ごと戻したもの)を返す。"""
    root.mkdir()
    tensors = {"language_model.model.embed_tokens.weight": mx.zeros((4, 8), dtype=mx.bfloat16)}
    full = []
    for i, rows in enumerate(ROWS):
        table = mx.random.normal((rows, WIDTH), key=mx.random.key(i)).astype(mx.bfloat16)
        if unquantized:
            tensors[f"{PREFIX}.{i}.weight"] = table
            full.append(table)
            continue
        q = per_module or quantization
        parts = mx.quantize(table, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"))
        for name, value in zip(("weight", "scales", "biases"), parts):
            tensors[f"{PREFIX}.{i}.{name}"] = value
        full.append(mx.dequantize(*parts, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine")))
    half = len(tensors) // 2
    names = sorted(tensors)
    files = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for file_name, keys in files.items():
        mx.save_safetensors(str(root / file_name), {k: tensors[k] for k in keys}, metadata={"format": "mlx"})
        weight_map.update({k: file_name for k in keys})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    config = {"model_type": "qwen4_exp", "text_config": {}}
    if quantization is not None:
        config["quantization"] = dict(quantization)
        if per_module is not None:
            for i in range(len(ROWS)):
                config["quantization"][f"{PREFIX}.{i}"] = per_module
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    return mx.concatenate(full, axis=0)


def _check_lookup(table, reference):
    ids = np.array([[0, 5, 36, 37, 59], [59, 0, 12, 40, 5]])          # 断片をまたぎ、重複も含む
    got = table(ids)
    assert got.shape == (*ids.shape, WIDTH)
    want = reference[mx.array(ids.reshape(-1))].reshape(*ids.shape, WIDTH)
    assert mx.array_equal(got.astype(mx.float32), want.astype(mx.float32)).item()


CASES = [
    {"bits": 2, "group_size": 32, "mode": "affine"},
    {"bits": 3, "group_size": 32, "mode": "affine"},
    {"bits": 4, "group_size": 32, "mode": "affine"},
    {"bits": 4, "group_size": 64, "mode": "affine"},
    {"bits": 5, "group_size": 32, "mode": "affine"},
    {"bits": 6, "group_size": 64, "mode": "affine"},
    {"bits": 8, "group_size": 32, "mode": "affine"},
    {"bits": 4, "group_size": 32, "mode": "mxfp4"},
    {"bits": 4, "group_size": 16, "mode": "nvfp4"},
    {"bits": 8, "group_size": 32, "mode": "mxfp8"},
]


@pytest.mark.parametrize("q", CASES, ids=lambda q: f"{q['mode']}-{q['bits']}bit-g{q['group_size']}")
def test_external_view_reads_rows_like_the_full_table(tmp_path, q):
    reference = _checkpoint(tmp_path / "src", q)
    ps.prepare_external_ple_model(tmp_path / "src", tmp_path / "view")
    config = json.loads((tmp_path / "view" / "config.json").read_text())
    assert config["text_config"]["ple_storage"] == {"manifest": "ple-store.json", "cache_rows": 0}
    index = json.loads((tmp_path / "view" / "model.safetensors.index.json").read_text())
    assert not any(ps.PLE_MARKER in key for key in index["weight_map"])
    table = ps.QuantizedMMapNGramEmbedding(tmp_path / "view" / "ple-store.json")
    assert (table.row_count, table.row_width) == (sum(ROWS), WIDTH)
    _check_lookup(table, reference)


def test_unquantized_table(tmp_path):
    reference = _checkpoint(tmp_path / "src", None, unquantized=True)
    ps.prepare_external_ple_model(tmp_path / "src", tmp_path / "view")
    _check_lookup(ps.QuantizedMMapNGramEmbedding(tmp_path / "view" / "ple-store.json"), reference)


def test_per_module_entry_without_mode_inherits_the_top_level(tmp_path):
    """項目ごとの量子化に mode が無い config(上位の mode を継ぐ前提)でも読める。"""
    top = {"bits": 8, "group_size": 32, "mode": "affine"}
    reference = _checkpoint(tmp_path / "src", top, per_module={"bits": 4, "group_size": 32})
    manifest = ps.build_quantized_ple_manifest(tmp_path / "src", tmp_path / "view" / "ple-store.json")
    assert manifest["quantization"] == {"bits": 4, "group_size": 32, "mode": "affine"}
    _check_lookup(ps.QuantizedMMapNGramEmbedding(tmp_path / "view" / "ple-store.json"), reference)


@pytest.mark.parametrize("q", [CASES[6], CASES[8]], ids=["affine-8bit", "nvfp4"])
def test_interleaved_rows_match(tmp_path, q):
    reference = _checkpoint(tmp_path / "src", q)
    ps.materialize_interleaved_ple_store(tmp_path / "src", tmp_path / "store" / "ple-store.json", chunk_rows=10)
    table = ps.QuantizedMMapNGramEmbedding(tmp_path / "store" / "ple-store.json", cache_rows=3)
    _check_lookup(table, reference)
    _check_lookup(table, reference)          # 2 回目は行の残し(cache_rows)からも読む
    assert table.stats.cache_hits > 0


def test_quantization_that_does_not_match_the_tensors_is_refused(tmp_path):
    _checkpoint(tmp_path / "src", {"bits": 4, "group_size": 32, "mode": "affine"})
    config = json.loads((tmp_path / "src" / "config.json").read_text())
    config["quantization"]["bits"] = 8                     # 実際の詰め方と違う
    (tmp_path / "src" / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        ps.build_quantized_ple_manifest(tmp_path / "src", tmp_path / "view" / "ple-store.json")


def test_shim_installs_the_generalized_module(monkeypatch):
    from local_llm_server import _mlx_vlm_shims
    monkeypatch.delitem(sys.modules, _mlx_vlm_shims._PLE_MODULE, raising=False)
    assert _mlx_vlm_shims._install_ple_storage() == "applied"
    assert sys.modules[_mlx_vlm_shims._PLE_MODULE] is ps


def _snapshot(tmp_path, q):
    """HF キャッシュと同じ形(<hub>/models--org--name/snapshots/<rev>)に小さな表を置く。"""
    snap = tmp_path / "hub" / "models--org--name" / "snapshots" / "rev1"
    snap.parent.mkdir(parents=True)
    return snap, _checkpoint(snap, q)


def test_external_view_is_made_once_next_to_the_hub(tmp_path):
    """表を持つモデルは、hub の隣に見え方を自動で作る。2 回目は作り直さない。"""
    snap, reference = _snapshot(tmp_path, CASES[6])
    view = ps.external_view(snap)
    assert view == tmp_path / "ple-views" / "models--org--name" / "rev1"
    stamp = (view / "EXTERNAL_PLE.json").stat().st_mtime_ns
    assert ps.external_view(snap) == view
    assert (view / "EXTERNAL_PLE.json").stat().st_mtime_ns == stamp
    assert not list(view.parent.glob(".rev1.*"))              # 作り途中の場所は残さない
    _check_lookup(ps.QuantizedMMapNGramEmbedding(view / "ple-store.json"), reference)


def test_models_without_tables_and_views_are_used_as_they_are(tmp_path):
    snap, _ = _snapshot(tmp_path, CASES[2])
    view = ps.external_view(snap)
    assert ps.external_view(view) is None                     # 見え方そのもの
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text("{}")
    (plain / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "a.safetensors"}}))
    assert ps.external_view(plain) is None


def test_memory_estimate_leaves_out_the_table(tmp_path, monkeypatch):
    """ゲートウェイの常駐の見積もりは、表(外に出して読む)の分を数えない。"""
    from local_llm_server import model_catalog, server
    snap, _ = _snapshot(tmp_path, CASES[6])
    files = sum(p.stat().st_size for p in snap.glob("*.safetensors"))
    table = model_catalog.ple_table_bytes(str(snap))
    assert 0 < table < files
    monkeypatch.setattr(server, "_hf_hub_cache", lambda: str(tmp_path / "hub"))
    config = server.ServerConfig(model="org/name", backend="mlx-vlm")
    whole = sum(p.stat().st_size for p in snap.iterdir())
    assert server.estimate_model_bytes(config) == whole - table


def test_shim_reads_the_view_under_the_hf_name(tmp_path, monkeypatch):
    """mlx-vlm がモデルの置き場を決める所で、表を持つモデルは見え方に替わる。"""
    utils = pytest.importorskip("mlx_vlm.utils")
    from local_llm_server import _mlx_vlm_shims
    snap, _ = _snapshot(tmp_path, CASES[6])
    monkeypatch.setattr(utils, "get_model_path", lambda repo, *a, **k: snap)
    assert _mlx_vlm_shims._install_external_view() == "applied"
    assert utils.get_model_path("org/name") == tmp_path / "ple-views" / "models--org--name" / "rev1"
