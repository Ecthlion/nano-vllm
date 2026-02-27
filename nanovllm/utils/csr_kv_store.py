from __future__ import annotations

import json
import os
import struct
import zlib
from typing import Any

import torch


_MAGIC = b"CSRKVC02"
_VERSION = 2
_HEADER_FMT = "<8sIIIIIIIIIQQQQ"
_HEADER_SIZE = 128


def _align_up(value: int, align: int) -> int:
    if align <= 0:
        return value
    rem = value % align
    return value if rem == 0 else value + (align - rem)


def _dtype_to_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.float32:
        return 3
    raise ValueError(f"unsupported dtype for CSR-KV store: {dtype}")


def _code_to_dtype(code: int) -> torch.dtype:
    if code == 1:
        return torch.float16
    if code == 2:
        return torch.bfloat16
    if code == 3:
        return torch.float32
    raise ValueError(f"unsupported dtype code for CSR-KV store: {code}")


def _record_key(text_id: Any) -> str:
    return json.dumps(text_id, ensure_ascii=False, separators=(",", ":"))


def _crc32(blob: bytes) -> int:
    return int(zlib.crc32(blob) & 0xFFFFFFFF)


def _record_crc32(
    indices_blob: bytes,
    indptr_blob: bytes,
    key_blob: bytes,
    value_blob: bytes,
) -> int:
    return int(
        zlib.crc32(
            value_blob,
            zlib.crc32(
                key_blob,
                zlib.crc32(indptr_blob, zlib.crc32(indices_blob)),
            ),
        )
        & 0xFFFFFFFF
    )


def _to_cpu_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.device.type != "cpu":
        t = t.detach().cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _tensor_from_blob(blob: bytes, dtype: torch.dtype) -> torch.Tensor:
    view = memoryview(blob)
    if dtype == torch.bfloat16:
        return torch.frombuffer(view, dtype=torch.uint16).clone().view(torch.bfloat16)
    return torch.frombuffer(view, dtype=dtype).clone()


class CSRKVStore:
    """
    Append-only CSR-KV storage with alignment-aware layout.

    Layout (per record):
      [header][indices][indptr][pad_to_data_align][key][pad_to_data_align][value][pad_to_record_align]

    Design notes:
    - Keep key/value 4KB-aligned for GPUDirect-friendly reads.
    - Avoid aligning indices/indptr individually to reduce write amplification.
    - Use group commit to amortize fsync overhead.
    """

    def __init__(
        self,
        root_dir: str,
        manifest_name: str = "index_manifest.jsonl",
        data_name: str = "kv_data.bin",
        data_align_bytes: int = 4096,
        record_align_bytes: int = 512,
        commit_interval: int = 32,
    ) -> None:
        self.root_dir = root_dir
        self.data_align_bytes = max(256, int(data_align_bytes))
        self.record_align_bytes = max(256, int(record_align_bytes))
        self.commit_interval = max(1, int(commit_interval))

        self.manifest_path = os.path.join(root_dir, manifest_name)
        self.data_path = os.path.join(root_dir, data_name)

        os.makedirs(self.root_dir, exist_ok=True)
        if not os.path.exists(self.data_path):
            with open(self.data_path, "ab"):
                pass

        self._records: dict[str, dict[str, Any]] = {}
        self._pending_manifest: list[str] = []
        self._pending_count = 0
        self._load_manifest()

    def _load_manifest(self) -> None:
        self._records.clear()
        if not os.path.exists(self.manifest_path):
            return

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    meta = json.loads(line)
                except Exception:
                    continue
                key = _record_key(meta.get("text_id"))
                self._records[key] = meta

    def has_records(self) -> bool:
        return bool(self._records)

    def _flush_manifest(self, do_fsync: bool = False) -> None:
        if not self._pending_manifest:
            return

        with open(self.manifest_path, "a", encoding="utf-8") as mf:
            mf.write("\n".join(self._pending_manifest) + "\n")
            mf.flush()
            if do_fsync:
                os.fsync(mf.fileno())

        self._pending_manifest.clear()

    def _sync_data_file(self) -> None:
        with open(self.data_path, "rb") as f:
            os.fsync(f.fileno())

    def finalize(self) -> None:
        self._flush_manifest(do_fsync=True)
        self._sync_data_file()
        self._pending_count = 0

    def build_index_entries(self) -> dict[Any, dict[str, Any]]:
        out: dict[Any, dict[str, Any]] = {}
        for meta in self._records.values():
            text_id = meta.get("text_id")
            num_layers = int(meta.get("num_layers", 0))
            num_tokens = int(meta.get("num_tokens", 0))
            num_heads = int(meta.get("num_heads", 0))
            head_dim = int(meta.get("head_dim", 0))
            entry = {
                "kv": None,
                "kv_shape": (2, num_layers, num_tokens, num_heads, head_dim),
                "kv_dtype": str(meta.get("dtype", "torch.float16")),
                "pruning_len": int(meta.get("pruning_len", 0)),
                "text_tokens_pruned": meta.get("text_tokens_pruned", []),
                "ilh_bounds": meta.get("ilh_bounds"),
                "task_type": meta.get("task_type", "generic"),
                "sparsity_ratio": float(meta.get("sparsity_ratio", 0.0)),
                "csr_offset": int(meta.get("offset", 0)),
                "csr_length": int(meta.get("length", 0)),
                "csr_record_key": meta.get("record_key", _record_key(text_id)),
            }
            out[text_id] = entry
        return out

    def get_meta(self, text_id: Any) -> dict[str, Any] | None:
        return self._records.get(_record_key(text_id))

    def write_amplification_summary(self) -> dict[str, float]:
        if not self._records:
            return {
                "logical_bytes": 0.0,
                "physical_bytes": 0.0,
                "write_amplification": 0.0,
            }

        logical = 0
        physical = 0
        for meta in self._records.values():
            logical += int(meta.get("logical_bytes", 0))
            physical += int(meta.get("physical_bytes", 0))

        amp = float(physical / logical) if logical > 0 else 0.0
        return {
            "logical_bytes": float(logical),
            "physical_bytes": float(physical),
            "write_amplification": amp,
        }

    def append_record(
        self,
        text_id: Any,
        kv_tensor: torch.Tensor,
        kept_indices: list[int],
        pruning_len: int,
        text_tokens_pruned: list[int],
        ilh_bounds: dict[str, int],
        task_type: str,
        force_sync: bool = False,
    ) -> dict[str, Any]:
        if kv_tensor.ndim != 5 or kv_tensor.shape[0] != 2:
            raise ValueError(
                "kv_tensor must have shape [2, num_layers, num_tokens, num_heads, head_dim]"
            )

        kv_cpu = _to_cpu_contiguous(kv_tensor)
        _, num_layers, num_tokens, num_heads, head_dim = kv_cpu.shape
        if num_tokens <= 0:
            raise ValueError("empty sparse token dimension is not supported")
        if len(kept_indices) != num_tokens:
            raise ValueError(
                f"kept_indices length mismatch: {len(kept_indices)} vs {num_tokens}"
            )

        indices = torch.tensor(kept_indices, dtype=torch.int32, device="cpu").contiguous()
        indptr = torch.arange(
            0,
            (num_layers + 1) * num_tokens,
            num_tokens,
            dtype=torch.int32,
            device="cpu",
        ).contiguous()
        key = kv_cpu[0].contiguous()
        value = kv_cpu[1].contiguous()

        dtype_code = _dtype_to_code(kv_cpu.dtype)
        indices_blob = indices.numpy().tobytes()
        indptr_blob = indptr.numpy().tobytes()
        key_blob = key.numpy().tobytes()
        value_blob = value.numpy().tobytes()

        indices_offset_rel = _HEADER_SIZE
        indptr_offset_rel = indices_offset_rel + len(indices_blob)
        key_offset_rel = _align_up(
            indptr_offset_rel + len(indptr_blob),
            self.data_align_bytes,
        )
        value_offset_rel = _align_up(
            key_offset_rel + len(key_blob),
            self.data_align_bytes,
        )

        header = struct.pack(
            _HEADER_FMT,
            _MAGIC,
            _VERSION,
            self.data_align_bytes,
            int(num_layers),
            int(num_tokens),
            int(num_heads),
            int(head_dim),
            int(dtype_code),
            int(indices.numel()),
            int(indptr.numel()),
            int(len(indices_blob)),
            int(len(indptr_blob)),
            int(len(key_blob)),
            int(len(value_blob)),
        )
        if len(header) > _HEADER_SIZE:
            raise RuntimeError("CSR-KV header overflow")
        header = header + (b"\0" * (_HEADER_SIZE - len(header)))

        with open(self.data_path, "r+b") as f:
            f.seek(0, os.SEEK_END)
            cur = f.tell()
            rec_offset = _align_up(cur, self.record_align_bytes)
            if rec_offset > cur:
                f.write(b"\0" * (rec_offset - cur))

            f.write(header)
            f.write(indices_blob)
            f.write(indptr_blob)

            cur_pos = rec_offset + indptr_offset_rel + len(indptr_blob)
            if key_offset_rel > (cur_pos - rec_offset):
                f.write(b"\0" * (key_offset_rel - (cur_pos - rec_offset)))

            f.write(key_blob)

            cur_pos = rec_offset + key_offset_rel + len(key_blob)
            if value_offset_rel > (cur_pos - rec_offset):
                f.write(b"\0" * (value_offset_rel - (cur_pos - rec_offset)))

            f.write(value_blob)
            rec_end_raw = rec_offset + value_offset_rel + len(value_blob)
            rec_end = _align_up(rec_end_raw, self.record_align_bytes)
            if rec_end > rec_end_raw:
                f.write(b"\0" * (rec_end - rec_end_raw))

            f.flush()
            if force_sync:
                os.fsync(f.fileno())

        logical_bytes = (
            _HEADER_SIZE + len(indices_blob) + len(indptr_blob) + len(key_blob) + len(value_blob)
        )
        physical_bytes = int(rec_end - rec_offset)

        meta: dict[str, Any] = {
            "text_id": text_id,
            "record_key": _record_key(text_id),
            "offset": int(rec_offset),
            "length": int(rec_end - rec_offset),
            "data_align": int(self.data_align_bytes),
            "record_align": int(self.record_align_bytes),
            "shape": [2, int(num_layers), int(num_tokens), int(num_heads), int(head_dim)],
            "num_layers": int(num_layers),
            "num_tokens": int(num_tokens),
            "num_heads": int(num_heads),
            "head_dim": int(head_dim),
            "dtype": str(kv_cpu.dtype),
            "dtype_code": int(dtype_code),
            "indices_count": int(indices.numel()),
            "indptr_count": int(indptr.numel()),
            "indices_nbytes": int(len(indices_blob)),
            "indptr_nbytes": int(len(indptr_blob)),
            "key_nbytes": int(len(key_blob)),
            "value_nbytes": int(len(value_blob)),
            "offsets": {
                "record": int(rec_offset),
                "header": int(rec_offset),
                "indices": int(rec_offset + indices_offset_rel),
                "indptr": int(rec_offset + indptr_offset_rel),
                "key": int(rec_offset + key_offset_rel),
                "value": int(rec_offset + value_offset_rel),
            },
            "checksums": {
                "indices_crc32": _crc32(indices_blob),
                "indptr_crc32": _crc32(indptr_blob),
                "key_crc32": _crc32(key_blob),
                "value_crc32": _crc32(value_blob),
                "record_crc32": _record_crc32(
                    indices_blob,
                    indptr_blob,
                    key_blob,
                    value_blob,
                ),
            },
            "pruning_len": int(pruning_len),
            "text_tokens_pruned": list(text_tokens_pruned),
            "ilh_bounds": {
                "core": int(ilh_bounds.get("core", 0)),
                "important": int(ilh_bounds.get("important", 0)),
                "optional": int(ilh_bounds.get("optional", 0)),
            },
            "task_type": str(task_type),
            "sparsity_ratio": float(pruning_len / max(1, pruning_len + num_tokens)),
            "logical_bytes": int(logical_bytes),
            "physical_bytes": int(physical_bytes),
            "write_amplification": float(physical_bytes / max(1, logical_bytes)),
        }

        self._records[meta["record_key"]] = meta
        self._pending_manifest.append(json.dumps(meta, ensure_ascii=False))
        self._pending_count += 1

        should_commit = force_sync or self._pending_count >= self.commit_interval
        if should_commit:
            self._flush_manifest(do_fsync=True)
            self._sync_data_file()
            self._pending_count = 0

        return meta

    def load_record(self, text_id: Any) -> dict[str, Any] | None:
        meta = self.get_meta(text_id)
        if meta is None:
            return None

        offset = int(meta.get("offset", 0))
        with open(self.data_path, "rb") as f:
            f.seek(offset)
            header_blob = f.read(_HEADER_SIZE)
            if len(header_blob) < _HEADER_SIZE:
                return None

            parsed = struct.unpack(_HEADER_FMT, header_blob[: struct.calcsize(_HEADER_FMT)])
            (
                magic,
                version,
                align,
                num_layers,
                num_tokens,
                num_heads,
                head_dim,
                dtype_code,
                _indices_count,
                _indptr_count,
                indices_nbytes,
                indptr_nbytes,
                key_nbytes,
                value_nbytes,
            ) = parsed

            if magic != _MAGIC or version != _VERSION:
                return None

            offsets = meta.get("offsets", {})
            indices_offset = int(offsets.get("indices", offset + _HEADER_SIZE))
            indptr_offset = int(offsets.get("indptr", indices_offset + indices_nbytes))
            key_offset = int(
                offsets.get("key", _align_up(indptr_offset + indptr_nbytes, align))
            )
            value_offset = int(
                offsets.get("value", _align_up(key_offset + key_nbytes, align))
            )

            f.seek(indices_offset)
            indices_blob = f.read(indices_nbytes)
            f.seek(indptr_offset)
            indptr_blob = f.read(indptr_nbytes)
            f.seek(key_offset)
            key_blob = f.read(key_nbytes)
            f.seek(value_offset)
            value_blob = f.read(value_nbytes)

        if (
            len(indices_blob) != indices_nbytes
            or len(indptr_blob) != indptr_nbytes
            or len(key_blob) != key_nbytes
            or len(value_blob) != value_nbytes
        ):
            return None

        checksums = meta.get("checksums", {})
        if isinstance(checksums, dict):
            if "indices_crc32" in checksums and int(checksums["indices_crc32"]) != _crc32(indices_blob):
                return None
            if "indptr_crc32" in checksums and int(checksums["indptr_crc32"]) != _crc32(indptr_blob):
                return None
            if "key_crc32" in checksums and int(checksums["key_crc32"]) != _crc32(key_blob):
                return None
            if "value_crc32" in checksums and int(checksums["value_crc32"]) != _crc32(value_blob):
                return None
            if "record_crc32" in checksums:
                record_crc = _record_crc32(indices_blob, indptr_blob, key_blob, value_blob)
                if int(checksums["record_crc32"]) != record_crc:
                    return None

        dtype = _code_to_dtype(dtype_code)
        indices = _tensor_from_blob(indices_blob, torch.int32)
        indptr = _tensor_from_blob(indptr_blob, torch.int32)
        key = _tensor_from_blob(key_blob, dtype).view(
            int(num_layers), int(num_tokens), int(num_heads), int(head_dim)
        )
        value = _tensor_from_blob(value_blob, dtype).view(
            int(num_layers), int(num_tokens), int(num_heads), int(head_dim)
        )
        kv = torch.stack([key, value], dim=0)

        return {
            "kv": kv,
            "indices": indices,
            "indptr": indptr,
            "meta": meta,
        }
