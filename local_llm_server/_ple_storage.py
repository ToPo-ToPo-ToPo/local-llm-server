"""Qwen4-Exp(Qwen3.8-Flash-Next)の n-gram 埋め込み表を、常駐させずに行単位で読む置き場。

mlx-vlm の ``mlx_vlm.models.qwen4_exp.ple_storage``(MIT License, Copyright © 2025 Prince Canuma)を元に、
量子化の決め打ちを外したもの。上流(0.7.11 時点)は 4bit の affine(g32)と nvfp4(g16)しか読めず、
行の幅・詰め込みの幅・復元の bits と group_size が 4bit 前提で書かれている。ここでは量子化の値(bits・
group_size・mode)をチェックポイントの config と safetensors のヘッダから読み、MLX が扱える全ての組み合わせ
(affine の 2〜8bit・任意の group_size、mxfp4・nvfp4・mxfp8 など)と、量子化していない表を同じ道で扱う。
項目ごとの量子化に ``mode`` が無いとき(上位の値を継ぐ前提の config)は、上位の値を使う。

使い方は上流と同じ(``prepare_external_ple_model`` で表を除いた見え方を作り、実行時は
``QuantizedMMapNGramEmbedding`` が必要な行だけ読む)。モデルサーバーでは ``_mlx_vlm_shims`` が
上流のモジュールの代わりにこれを入れる。上流が直ったら、このファイルとシムの差し替えを消す。

    python -m local_llm_server._ple_storage <元のモデル> <見え方を作る場所> [--cache-rows N]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model_catalog import PLE_MARKER  # noqa: E402 — 印は 1 か所(ゲートウェイの見積もりと同じ)
MANIFEST_VERSION = 2
# safetensors の dtype 名 → 読み出しに使う numpy の型(BF16 は numpy に無いので 16bit の整数で持ち、mx へ渡すときに直す)
_DTYPES = {
    "U8": np.dtype("u1"),
    "U16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
}
_PACK_BITS = 32  # MLX の量子化は値を uint32 に詰める


# --- 量子化の読み取り -----------------------------------------------------------------


def module_quantization(config: dict, prefix: str) -> dict | None:
    """1 つの断片(``prefix``)の量子化。量子化していなければ None。

    項目ごとの値に無いもの(よくあるのは ``mode``)は上位の値を継ぎ、どこにも無い ``mode`` は affine。
    """
    table = config.get("quantization_config") or config.get("quantization")
    if not table:
        return None
    params = table.get(prefix, table)
    if params is False:
        return None
    if not isinstance(params, dict):
        raise ValueError(f"unsupported quantization entry for {prefix!r}: {params!r}")
    merged = {key: params.get(key, table.get(key)) for key in ("bits", "group_size", "mode")}
    merged["mode"] = merged["mode"] or "affine"
    if merged["bits"] is None or merged["group_size"] is None:
        raise ValueError(f"quantization for {prefix!r} lacks bits or group_size: {params!r}")
    merged["bits"], merged["group_size"] = int(merged["bits"]), int(merged["group_size"])
    return merged


def tensor_names(quantization: dict | None) -> tuple[str, ...]:
    """断片を作るテンソルの名前。affine は scales と biases、ほかの mode(mxfp4 など)は scales だけ。"""
    if quantization is None:
        return ("weight",)
    if quantization["mode"] == "affine":
        return ("weight", "scales", "biases")
    return ("weight", "scales")


def row_width_from_packed(quantization: dict | None, packed_columns: int) -> int:
    """weight の列数から 1 行の値の数を出す(量子化していなければ同じ)。"""
    if quantization is None:
        return packed_columns
    bits = quantization["bits"]
    if (packed_columns * _PACK_BITS) % bits:
        raise ValueError(f"packed width {packed_columns} is not a whole number of {bits}-bit values")
    return packed_columns * _PACK_BITS // bits


def column_counts(quantization: dict | None, row_width: int) -> dict[str, int]:
    """各テンソルの 1 行あたりの列数。"""
    if quantization is None:
        return {"weight": row_width}
    bits, group = quantization["bits"], quantization["group_size"]
    if (row_width * bits) % _PACK_BITS or row_width % group:
        raise ValueError(f"row width {row_width} does not fit bits={bits}, group_size={group}")
    counts = {"weight": row_width * bits // _PACK_BITS}
    for name in tensor_names(quantization)[1:]:
        counts[name] = row_width // group
    return counts


def _to_mx(values: np.ndarray, dtype_name: str) -> mx.array:
    if dtype_name == "BF16":
        as_float = (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        return mx.array(as_float).astype(mx.bfloat16)
    return mx.array(values)


# --- 表の読み出し -------------------------------------------------------------------


@dataclass(frozen=True)
class LookupStats:
    lookups: int
    rows: int
    cache_hits: int
    cache_misses: int
    bytes_read: int
    elapsed_seconds: float


class QuantizedMMapNGramEmbedding:
    """読み出し専用の n-gram 表。safetensors のバイト範囲(または行ごとに並べ直した 1 ファイル)を mmap し、
    使う行だけを読んで値に戻す。量子化していない表もそのまま読む。"""

    MANIFEST_VERSION = MANIFEST_VERSION

    def __init__(self, manifest_path, *, cache_rows=None):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        missing = {"version", "source_root", "row_width", "quantization"} - manifest.keys()
        if missing:
            raise ValueError(f"manifest missing fields: {sorted(missing)}")
        if manifest["version"] != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {manifest['version']}")
        self.quantization = manifest["quantization"]
        self._names = tensor_names(self.quantization)
        self.row_width = int(manifest["row_width"])
        if self.row_width <= 0:
            raise ValueError("row_width must be positive")
        self._columns = column_counts(self.quantization, self.row_width)

        source_root = Path(manifest["source_root"])
        if not source_root.is_absolute():
            source_root = (manifest_path.parent / source_root).resolve()

        self._interleaved = None
        layout = manifest.get("layout", "safetensors_ranges")
        if layout in ("interleaved_rows", "interleaved_q4"):
            self._dtypes = manifest.get("dtypes") or _legacy_dtypes(self.quantization)
            data_name = Path(manifest["data_file"])
            if data_name.is_absolute() or data_name.name != str(data_name):
                raise ValueError("data_file must be beside the manifest")
            data_path = manifest_path.parent / data_name
            self.row_count = int(manifest["row_count"])
            row_stride = int(manifest["row_stride"])
            if row_stride != self._row_bytes():
                raise ValueError("interleaved row stride does not match the row layout")
            expected_size = self.row_count * row_stride
            if data_path.stat().st_size != expected_size:
                raise ValueError(
                    f"interleaved PLE size mismatch: expected {expected_size}, got {data_path.stat().st_size}"
                )
            self._interleaved = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(self.row_count, row_stride))
            self._shards = []
        elif layout == "safetensors_ranges":
            if "shards" not in manifest:
                raise ValueError("manifest missing fields: ['shards']")
            self._init_safetensors_ranges(manifest, source_root)
        else:
            raise ValueError(f"unsupported PLE storage layout: {layout}")

        configured = manifest.get("cache_rows", 0) if cache_rows is None else cache_rows
        self.cache_rows = int(configured)
        if self.cache_rows < 0:
            raise ValueError("cache_rows must be non-negative")
        self._cache: OrderedDict = OrderedDict()
        self._lookups = self._rows = self._hits = self._misses = self._bytes = 0
        self._elapsed = 0.0

    def _row_bytes(self) -> int:
        return sum(self._columns[name] * _DTYPES[self._dtypes[name]].itemsize for name in self._names)

    def _init_safetensors_ranges(self, manifest, source_root):
        self._shards = []
        self._dtypes = {}
        expected_start = 0
        for entry in manifest["shards"]:
            start, count = int(entry["row_start"]), int(entry["row_count"])
            if start != expected_start or count <= 0:
                raise ValueError("PLE shards must be positive, contiguous, and ordered")
            arrays = {}
            for name in self._names:
                tensor = entry[name]
                dtype_name = tensor["dtype"]
                if dtype_name not in _DTYPES:
                    raise ValueError(f"unsupported {name} dtype: {dtype_name}")
                if self._dtypes.setdefault(name, dtype_name) != dtype_name:
                    raise ValueError(f"{name} dtype differs between PLE shards")
                file_name = Path(tensor["file"])
                if file_name.is_absolute() or file_name.name != str(file_name):
                    raise ValueError("tensor files must be filenames under source_root")
                path = source_root / file_name
                offset = int(tensor["offset"])
                shape = tuple(int(dim) for dim in tensor["shape"])
                if shape != (count, self._columns[name]):
                    raise ValueError(f"{name} shape {shape} does not match the row layout")
                dtype = _DTYPES[dtype_name]
                byte_count = int(np.prod(shape)) * dtype.itemsize
                if offset < 0 or path.stat().st_size < offset + byte_count:
                    raise ValueError(f"{name} byte range exceeds {path}")
                arrays[name] = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
            self._shards.append((start, start + count, arrays))
            expected_start += count
        self.row_count = expected_start

    def _remember(self, row_id, row):
        if self.cache_rows:
            self._cache[row_id] = row
            self._cache.move_to_end(row_id)
            while len(self._cache) > self.cache_rows:
                self._cache.popitem(last=False)

    def _split_interleaved(self, raw: np.ndarray) -> list[np.ndarray]:
        columns, offset = [], 0
        for name in self._names:
            dtype = _DTYPES[self._dtypes[name]]
            byte_count = self._columns[name] * dtype.itemsize
            columns.append(raw[:, offset : offset + byte_count].reshape(len(raw), -1).view(dtype))
            offset += byte_count
        return columns

    def _read_rows(self, row_ids):
        """違う行を 1 回ずつ読む(置き場の並びに沿ってまとめて読む)。"""
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if row_ids.ndim != 1:
            raise ValueError("row_ids must be one-dimensional")
        if not row_ids.size:
            return tuple(
                np.empty((0, self._columns[name]), dtype=_DTYPES[self._dtypes[name]]) for name in self._names
            )
        rows: list = [None] * len(row_ids)
        missing_positions, missing_ids = [], []
        for position, row_id in enumerate(row_ids.tolist()):
            cached = self._cache.get(row_id)
            if cached is None:
                missing_positions.append(position)
                missing_ids.append(row_id)
                continue
            self._hits += 1
            self._cache.move_to_end(row_id)
            rows[position] = cached

        loaded: list
        if missing_ids and self._interleaved is not None:
            raw = np.array(self._interleaved[np.asarray(missing_ids)], copy=True)
            loaded = list(zip(*self._split_interleaved(raw)))
            self._bytes += raw.nbytes
        elif missing_ids:
            loaded = [None] * len(missing_ids)
            missing_array = np.asarray(missing_ids)
            for start, end, arrays in self._shards:
                selected = np.flatnonzero((missing_array >= start) & (missing_array < end))
                if not selected.size:
                    continue
                local = missing_array[selected] - start
                columns = [np.array(arrays[name][local], copy=True) for name in self._names]
                self._bytes += sum(column.nbytes for column in columns)
                for output_index, row in zip(selected.tolist(), zip(*columns)):
                    loaded[output_index] = row
            if any(row is None for row in loaded):
                raise IndexError("n-gram row outside mmap table")
        else:
            loaded = []

        self._misses += len(missing_ids)
        for position, row_id, row in zip(missing_positions, missing_ids, loaded):
            rows[position] = row
            self._remember(row_id, row)
        return tuple(np.stack([row[index] for row in rows]) for index in range(len(self._names)))

    def _values(self, tensors) -> mx.array:
        """読んだ行(weight と、量子化していれば scales・biases)を値に戻す。"""
        if self.quantization is None:
            return _to_mx(tensors[0], self._dtypes["weight"])
        q = self.quantization
        parts = dict(zip(self._names, tensors))
        packed = mx.array(parts["weight"]).astype(mx.uint32)
        scales = _to_mx(parts["scales"], self._dtypes["scales"])
        biases = _to_mx(parts["biases"], self._dtypes["biases"]) if "biases" in parts else None
        return mx.dequantize(packed, scales, biases, group_size=q["group_size"], bits=q["bits"], mode=q["mode"])

    def __call__(self, row_ids):
        started = time.perf_counter()
        ids = np.asarray(row_ids).astype(np.int64, copy=False)
        if ids.size and (ids.min() < 0 or ids.max() >= self.row_count):
            raise IndexError("n-gram row outside mmap table")
        flat_ids = ids.reshape(-1)
        if not flat_ids.size:
            return mx.zeros((*ids.shape, self.row_width), dtype=mx.bfloat16)
        unique_ids, inverse = np.unique(flat_ids, return_inverse=True)
        values = self._values(self._read_rows(unique_ids))
        self._hits += flat_ids.size - unique_ids.size
        # np.unique は並べ替えるので、呼び出し側の順へ inverse で戻す(重複が無くても要る)
        values = values[mx.array(inverse, dtype=mx.int64)]
        self._lookups += 1
        self._rows += ids.size
        self._elapsed += time.perf_counter() - started
        return values.reshape(*ids.shape, self.row_width)

    lookup = __call__

    @property
    def stats(self):
        return LookupStats(self._lookups, self._rows, self._hits, self._misses, self._bytes, self._elapsed)


def _legacy_dtypes(quantization: dict | None) -> dict[str, str]:
    """dtypes を持たない上流の interleaved_q4 の目録向け(affine は BF16、ほかの mode の scales は U8)。"""
    if quantization is None:
        return {"weight": "BF16"}
    aux = "BF16" if quantization["mode"] == "affine" else "U8"
    return {name: ("U32" if name == "weight" else aux) for name in tensor_names(quantization)}


# --- 目録づくり ---------------------------------------------------------------------


def build_quantized_ple_manifest(model_path, output_path, *, cache_rows=0):
    """n-gram 表の断片の置き場所(ファイル・位置・形)を目録にする。中身はコピーしない。"""
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    config = json.loads((model_path / "config.json").read_text())
    weight_map = index["weight_map"]
    prefixes = {key.rsplit(".", 1)[0] for key in weight_map if PLE_MARKER in key}
    if not prefixes:
        raise ValueError("checkpoint contains no Qwen4-Exp PLE shards")

    headers: dict = {}

    def descriptor(key):
        file_name = weight_map[key]
        if file_name not in headers:
            with (model_path / file_name).open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                headers[file_name] = (8 + header_size, json.loads(stream.read(header_size)))
        payload_start, header = headers[file_name]
        info = header[key]
        return {
            "file": file_name,
            "offset": payload_start + int(info["data_offsets"][0]),
            "dtype": info["dtype"],
            "shape": info["shape"],
        }

    shards, row_start = [], 0
    manifest_quantization = row_width = None
    dtypes: dict[str, str] = {}
    for prefix in sorted(prefixes, key=lambda value: int(value.rsplit(".", 1)[1])):
        quantization = module_quantization(config, prefix)
        names = tensor_names(quantization)
        present = {key.rsplit(".", 1)[1] for key in weight_map if key.startswith(prefix + ".")}
        if set(names) != present:
            raise ValueError(
                f"PLE tensor {prefix!r} has {sorted(present)}, but its quantization {quantization} needs {list(names)}"
            )
        if row_start == 0:
            manifest_quantization = quantization
        elif quantization != manifest_quantization:
            raise ValueError("all PLE shards must use the same quantization")
        tensors = {name: descriptor(f"{prefix}.{name}") for name in names}
        weight = tensors["weight"]
        if len(weight["shape"]) != 2:
            raise ValueError(f"PLE tensor {prefix!r} weight must be two-dimensional")
        count = int(weight["shape"][0])
        width = row_width_from_packed(quantization, int(weight["shape"][1]))
        if row_width is None:
            row_width = width
        elif width != row_width:
            raise ValueError("all PLE shards must have the same row width")
        columns = column_counts(quantization, width)
        for name, tensor in tensors.items():
            if tensor["shape"] != [count, columns[name]]:
                raise ValueError(f"PLE tensor {prefix}.{name} shape {tensor['shape']} != {[count, columns[name]]}")
            if dtypes.setdefault(name, tensor["dtype"]) != tensor["dtype"]:
                raise ValueError(f"PLE tensor {prefix}.{name} dtype differs between shards")
        shards.append({"row_start": row_start, "row_count": count, **tensors})
        row_start += count
    manifest = {
        "version": MANIFEST_VERSION,
        "source_root": os.path.relpath(model_path, output_path.parent),
        "layout": "safetensors_ranges",
        "row_width": row_width,
        "row_count": row_start,
        "quantization": manifest_quantization,
        "dtypes": dtypes,
        "cache_rows": int(cache_rows),
        "shards": shards,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def materialize_interleaved_ple_store(model_path, manifest_path, *, data_file="ple-rows.bin", cache_rows=0,
                                      chunk_rows=131_072):
    """散らばった断片を、1 行 1 レコードの 1 ファイルに並べ直す(mmap で行をまとめて読むため)。"""
    manifest_path = Path(manifest_path).resolve()
    data_name = Path(data_file)
    if data_name.is_absolute() or data_name.name != str(data_name):
        raise ValueError("data_file must be a filename beside the manifest")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    data_path = manifest_path.parent / data_name
    if data_path.exists():
        raise FileExistsError(f"PLE row store already exists: {data_path}")
    ranges = build_quantized_ple_manifest(model_path, manifest_path, cache_rows=cache_rows)
    quantization, row_width, dtypes = ranges["quantization"], int(ranges["row_width"]), ranges["dtypes"]
    names = tensor_names(quantization)
    columns = column_counts(quantization, row_width)
    widths = {name: columns[name] * _DTYPES[dtypes[name]].itemsize for name in names}
    row_stride = sum(widths.values())
    total_size = int(ranges["row_count"]) * row_stride
    descriptor, partial_name = tempfile.mkstemp(prefix=f".{data_name.name}.", suffix=".partial",
                                                dir=manifest_path.parent)
    os.close(descriptor)
    partial_path = Path(partial_name)
    try:
        with partial_path.open("wb") as stream:
            stream.truncate(total_size)
        source_root = (manifest_path.parent / ranges["source_root"]).resolve()
        for shard in ranges["shards"]:
            arrays = {
                name: np.memmap(source_root / shard[name]["file"], dtype=_DTYPES[shard[name]["dtype"]], mode="r",
                                offset=int(shard[name]["offset"]), shape=tuple(shard[name]["shape"]))
                for name in names
            }
            row_start, row_count = int(shard["row_start"]), int(shard["row_count"])
            for local_start in range(0, row_count, chunk_rows):
                local_end = min(local_start + chunk_rows, row_count)
                output = np.memmap(partial_path, dtype=np.uint8, mode="r+",
                                   offset=(row_start + local_start) * row_stride,
                                   shape=(local_end - local_start, row_stride))
                offset = 0
                for name in names:
                    output[:, offset : offset + widths[name]] = (
                        np.asarray(arrays[name][local_start:local_end]).view(np.uint8)
                        .reshape(local_end - local_start, widths[name])
                    )
                    offset += widths[name]
                output.flush()
                del output
            del arrays
        os.link(partial_path, data_path)
    finally:
        partial_path.unlink(missing_ok=True)
    manifest = {
        "version": MANIFEST_VERSION,
        "source_root": os.path.relpath(Path(model_path).resolve(), manifest_path.parent),
        "layout": "interleaved_rows",
        "data_file": data_name.name,
        "row_width": row_width,
        "row_count": int(ranges["row_count"]),
        "row_stride": row_stride,
        "quantization": quantization,
        "dtypes": dtypes,
        "cache_rows": int(cache_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_external_ple_model(source_path, target_path, *, cache_rows=0):
    """n-gram 表を除いた「見え方」を作る。重みは元のファイルへのハードリンクで、ディスクは増えない。"""
    source_path = Path(source_path).resolve()
    target_path = Path(target_path).resolve()
    if target_path.exists() and any(target_path.iterdir()):
        raise FileExistsError(f"target model directory is not empty: {target_path}")
    target_path.mkdir(parents=True, exist_ok=True)
    if source_path.stat().st_dev != target_path.stat().st_dev:
        raise ValueError("source and target must share a filesystem for hard links")

    manifest_path = target_path / "ple-store.json"
    ple_manifest = build_quantized_ple_manifest(source_path, manifest_path, cache_rows=cache_rows)
    source_index = json.loads((source_path / "model.safetensors.index.json").read_text())
    weight_map = {key: name for key, name in source_index["weight_map"].items() if PLE_MARKER not in key}
    used_files = sorted(set(weight_map.values()))
    for file_name in used_files:
        os.link(source_path / file_name, target_path / file_name)
    target_index = dict(source_index)
    target_index["weight_map"] = weight_map
    names = tensor_names(ple_manifest["quantization"])
    target_index.setdefault("metadata", {})["external_ple_bytes"] = str(sum(
        int(np.prod(shard[name]["shape"])) * _DTYPES[shard[name]["dtype"]].itemsize
        for shard in ple_manifest["shards"] for name in names
    ))
    (target_path / "model.safetensors.index.json").write_text(json.dumps(target_index, indent=2) + "\n")

    for source_file in source_path.iterdir():
        if not source_file.is_file() or source_file.name == "config.json" or source_file.suffix == ".safetensors":
            continue
        if source_file.name in {"model.safetensors.index.json", "MANIFEST.sha256", "MEMORY_CONTEXT_SWEEP.jsonl"}:
            continue
        shutil.copy2(source_file, target_path / source_file.name)

    config = json.loads((source_path / "config.json").read_text())
    config["text_config"]["ple_storage"] = {"manifest": manifest_path.name, "cache_rows": int(cache_rows)}
    for key in ("quantization", "quantization_config"):
        if isinstance(config.get(key), dict):
            config[key] = {name: value for name, value in config[key].items() if PLE_MARKER not in name}
    (target_path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "format": "qwen4_exp_external_ple_v1",
        "source_model": str(source_path),
        "source_index": "model.safetensors.index.json",
        "external_ple_manifest": manifest_path.name,
        "linked_weight_files": used_files,
    }
    (target_path / "EXTERNAL_PLE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def view_dir(model_dir) -> Path:
    """見え方を置く場所。元と同じファイルシステムに置く(ハードリンクのため)。

    HF キャッシュのスナップショット(``<hub>/models--org--name/snapshots/<rev>``)なら ``<hub>`` の隣の
    ``ple-views/models--org--name/<rev>``、それ以外のフォルダなら隣の ``.ple-views/<名前>``。
    """
    model_dir = Path(model_dir).resolve()
    if model_dir.parent.name == "snapshots" and model_dir.parents[1].name.startswith("models--"):
        hub = model_dir.parents[2]
        return hub.parent / "ple-views" / model_dir.parents[1].name / model_dir.name
    return model_dir.parent / ".ple-views" / model_dir.name


def external_view(model_dir) -> Path | None:
    """n-gram 表を持つモデルなら、表を外に出した見え方の場所を返す(無ければ作る)。表が無ければ None。

    見え方は重みのハードリンクと小さな設定だけなので、作るのは数秒でディスクも増えない。作り途中を
    読まれないよう、別の場所に作ってから置き換える。
    """
    model_dir = Path(model_dir)
    try:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        config = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not any(PLE_MARKER in key for key in index.get("weight_map", {})):
        return None
    if (config.get("text_config") or {}).get("ple_storage"):
        return None                      # もう見え方(表は外にある)
    target = view_dir(model_dir)
    if (target / "EXTERNAL_PLE.json").is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        prepare_external_ple_model(model_dir, work)
        try:
            work.rename(target)
        except OSError:                   # 同時に作った別の起動が先に置いた
            if not (target / "EXTERNAL_PLE.json").is_file():
                raise
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m local_llm_server._ple_storage",
        description="Qwen4-Exp の n-gram 表を常駐させない「見え方」を作る(重みはハードリンク)。",
    )
    parser.add_argument("source", help="元のモデルのフォルダ(HF キャッシュの snapshot など)")
    parser.add_argument("target", help="見え方を作るフォルダ(空か、無いこと。元と同じファイルシステム)")
    parser.add_argument("--cache-rows", type=int, default=0, help="読んだ行をメモリに残す数(0 で残さない)")
    args = parser.parse_args(argv)
    provenance = prepare_external_ple_model(args.source, args.target, cache_rows=args.cache_rows)
    manifest = json.loads((Path(args.target) / provenance["external_ple_manifest"]).read_text())
    print(f"{args.target}: rows={manifest['row_count']} width={manifest['row_width']} "
          f"quantization={manifest['quantization']} dtypes={manifest['dtypes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
