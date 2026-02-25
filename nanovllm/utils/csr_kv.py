from __future__ import annotations

from dataclasses import dataclass
import os
import struct
from typing import Any

import torch


@dataclass
class CSRKV:
    """
    CSR-KV compact sparse format.

    - indices: retained token positions in original text order, shape [K]
    - indptr: per-layer row pointer, shape [num_layers + 1], stride K
    - key: sparse key tensor, shape [num_layers, K, num_heads, head_dim]
    - value: sparse value tensor, shape [num_layers, K, num_heads, head_dim]
    """

    indices: torch.Tensor
    indptr: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


def build_csr_kv(kv_tensor: torch.Tensor, kept_indices: list[int]) -> CSRKV:
    """Convert [2, L, K, H, D] sparse KV tensor to CSR-KV record."""
    if kv_tensor.ndim != 5 or kv_tensor.size(0) != 2:
        raise ValueError("kv_tensor must have shape [2, num_layers, num_tokens, num_heads, head_dim]")

    _, num_layers, kept_tokens, _, _ = kv_tensor.shape
    if kept_tokens != len(kept_indices):
        raise ValueError(
            f"kept index length mismatch: tensor has {kept_tokens}, indices have {len(kept_indices)}"
        )

    indices = torch.tensor(kept_indices, dtype=torch.int32, device="cpu")
    indptr = torch.arange(
        0,
        (num_layers + 1) * kept_tokens,
        kept_tokens,
        dtype=torch.int32,
        device="cpu",
    )

    key = kv_tensor[0].contiguous()
    value = kv_tensor[1].contiguous()

    return CSRKV(indices=indices, indptr=indptr, key=key, value=value)


def csr_kv_to_record(csr: CSRKV) -> dict[str, Any]:
    return {
        "indices": csr.indices,
        "indptr": csr.indptr,
        "key": csr.key,
        "value": csr.value,
    }


def csr_kv_from_record(record: dict[str, Any]) -> CSRKV | None:
    indices = record.get("indices")
    indptr = record.get("indptr")
    key = record.get("key")
    value = record.get("value")
    if not all(isinstance(x, torch.Tensor) for x in (indices, indptr, key, value)):
        return None
    return CSRKV(indices=indices, indptr=indptr, key=key, value=value)


def csr_kv_to_dense(csr: CSRKV) -> torch.Tensor:
    """Reconstruct dense sparse-KV tensor [2, L, K, H, D] from CSR-KV components."""
    return torch.stack([csr.key, csr.value], dim=0)


def pin_csr_kv(csr: CSRKV) -> CSRKV:
    def _pin(x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu" and not x.is_pinned():
            return x.pin_memory()
        return x

    return CSRKV(
        indices=_pin(csr.indices),
        indptr=_pin(csr.indptr),
        key=_pin(csr.key),
        value=_pin(csr.value),
    )


def estimate_csr_kv_nbytes(csr: CSRKV) -> int:
    return (
        csr.indices.numel() * csr.indices.element_size()
        + csr.indptr.numel() * csr.indptr.element_size()
        + csr.key.numel() * csr.key.element_size()
        + csr.value.numel() * csr.value.element_size()
    )


def serialize_csr_kv_aligned(path: str, csr: CSRKV, align_bytes: int = 4096) -> str:
    """
    Serialize CSR-KV with 4KB-style alignment for DMA-friendly layout.

    Layout:
    [header(64B)][indices][pad][indptr][pad][key][pad][value]
    """
    align = max(256, int(align_bytes))

    def _pad_size(n: int) -> int:
        rem = n % align
        return 0 if rem == 0 else (align - rem)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    indices = csr.indices.contiguous().numpy().tobytes()
    indptr = csr.indptr.contiguous().numpy().tobytes()
    key = csr.key.contiguous().numpy().tobytes()
    value = csr.value.contiguous().numpy().tobytes()

    with open(path, "wb") as f:
        # Magic(8) + version(4) + align(4) + tensor dims(6*4) + reserved(24) = 64B
        header = struct.pack(
            "<8sii6i24s",
            b"CSRKVC01",
            1,
            align,
            int(csr.key.shape[0]),
            int(csr.key.shape[1]),
            int(csr.key.shape[2]),
            int(csr.key.shape[3]),
            int(csr.indices.numel()),
            int(csr.key.element_size()),
            b"",
        )
        f.write(header)

        f.write(indices)
        f.write(b"\0" * _pad_size(len(indices)))

        f.write(indptr)
        f.write(b"\0" * _pad_size(len(indptr)))

        f.write(key)
        f.write(b"\0" * _pad_size(len(key)))

        f.write(value)

    return path
