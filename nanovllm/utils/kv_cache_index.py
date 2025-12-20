import os
from threading import Event
from typing import Optional

import torch

from nanovllm.engine.sequence import Sequence

try:
    import kvikio  # type: ignore
    from kvikio import CuFile  # type: ignore

    _HAS_GDS = True
except Exception:
    _HAS_GDS = False
    CuFile = None  # type: ignore


class KVCacheIndex:
    # TODO: cpu cache overflow ssd
    def __init__(self, gpu_kv_cache, index_name="imdb_kvcache.pt") -> None:
        # Tensor[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.gpu_kv_cache = gpu_kv_cache

        self.save_dir = os.environ.get("NANOVLLM_KV_DIR", "/data/zwt/")
        self.index_name = index_name
        self.use_gds = (
            os.environ.get("NANOVLLM_USE_GPUDIRECT", "0") == "1" and _HAS_GDS
        )
        self.force_gds = os.environ.get("NANOVLLM_FORCE_GPUDIRECT", "0") == "1"

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.gds_dir = os.path.join(self.save_dir, "kv_cache_gds")
        if self.use_gds:
            os.makedirs(self.gds_dir, exist_ok=True)

        path = f"{self.save_dir}{self.index_name}"

        self.dirty = False
        self.indexed = False
        # text_id -> {
        #   "kv": Tensor[2, num_layers, seq_len_post_prune, num_kv_heads, head_dim],
        #   "pruning_len": int,                       # number of pruned text tokens (layer 0 reference)
        #   "text_tokens_pruned": list[int] | None    # text tokens after pruning (layer 0 reference)
        # }
        self.kv_cache_index: dict = {}
        if os.path.isfile(path):
            self.kv_cache_index = torch.load(path)
            self.indexed = True

            # pin memory when init
            for _, item in list(self.kv_cache_index.items()):
                kv = item.get("kv")
                if isinstance(kv, torch.Tensor):
                    if not kv.is_pinned():
                        item["kv"] = kv.pin_memory()
                # legacy files may not include kv shape/dtype metadata
                if "kv" in item:
                    item.setdefault("kv_shape", tuple(item["kv"].shape))  # type: ignore
                    item.setdefault("kv_dtype", str(item["kv"].dtype))  # type: ignore

    def store_kv_cache(
        self,
        seqs: list[Sequence],
        stream: torch.cuda.Stream,
        return_timing: bool = False,
    ):
        _, num_layers, _, block_size, num_kv_heads, head_dim = self.gpu_kv_cache.shape
        total_token = 0.0
        total_prune = 0.0
        with torch.cuda.stream(stream):
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)
            for seq in seqs:
                if seq.text_id is None or seq.text_id in self.kv_cache_index:
                    continue

                # Determine per-layer pruned indices (local prompt positions)
                base_pruned = seq.pruning_indices or []
                layer_pruned: list[list[int]] = []
                for layer_idx in range(num_layers):
                    if layer_idx < len(base_pruned):
                        layer_pruned.append(sorted(base_pruned[layer_idx]))
                    else:
                        layer_pruned.append([])

                first_layer_pruned = [
                    idx for idx in layer_pruned[0] if idx < seq.text_token_len
                ]
                post_prune_len = seq.text_token_len - len(first_layer_pruned)
                cpu_kv_cache = torch.empty(
                    2,
                    num_layers,
                    post_prune_len,
                    num_kv_heads,
                    head_dim,
                    device="cpu",
                    dtype=self.gpu_kv_cache.dtype,
                    pin_memory=True,
                )

                # Build flattened slot list for the first text_token_len tokens
                token_slots: list[int] = []
                taken = 0
                for block_id in seq.block_table:
                    if taken >= seq.text_token_len:
                        break
                    # tokens available to take from this block
                    cnt = min(block_size, seq.text_token_len - taken)
                    base = block_id * block_size
                    token_slots.extend(range(base, base + cnt))
                    taken += cnt

                if len(token_slots) != seq.text_token_len:
                    # Fallback guard: lengths must match; otherwise skip indexing
                    token_slots = token_slots[: seq.text_token_len]

                assert post_prune_len > 0

                flat_blocks = self.gpu_kv_cache.shape[2] * block_size
                device = self.gpu_kv_cache.device

                for layer_idx in range(num_layers):
                    pruned = [
                        idx for idx in layer_pruned[layer_idx] if idx < seq.text_token_len
                    ]
                    if pruned:
                        pruned_set = set(pruned)
                        kept_slots = [
                            slot
                            for i, slot in enumerate(token_slots)
                            if i not in pruned_set
                        ]
                    else:
                        kept_slots = token_slots

                    keep_len = len(kept_slots)
                    if keep_len != post_prune_len:
                        raise RuntimeError(
                            "Pruned length mismatch across layers; ensure pruning_len is consistent."
                        )

                    kept_slots_tensor = torch.tensor(
                        kept_slots, dtype=torch.int64, device=device
                    )

                    for kv_idx in range(2):
                        src_flat = self.gpu_kv_cache[kv_idx, layer_idx].reshape(
                            flat_blocks, num_kv_heads, head_dim
                        )
                        selected = src_flat.index_select(0, kept_slots_tensor)
                        dst = cpu_kv_cache[kv_idx, layer_idx, :keep_len]
                        dst.copy_(selected, non_blocking=True)

                first_layer_pruned_set = set(first_layer_pruned)
                kept_local_indices = [
                    i for i in range(seq.text_token_len)
                    if i not in first_layer_pruned_set
                ]
                text_tokens_pruned = [seq.token_ids[i] for i in kept_local_indices]

                self.dirty = True
                kv_path = None
                if self.use_gds:
                    kv_path = self._dump_kv_to_gds(seq.text_id, cpu_kv_cache)
                self.kv_cache_index[seq.text_id] = {
                    "kv": cpu_kv_cache,
                    "kv_path": kv_path,
                    "kv_shape": tuple(cpu_kv_cache.shape),
                    "kv_dtype": str(cpu_kv_cache.dtype),
                    "pruning_len": len(first_layer_pruned),
                    "text_tokens_pruned": text_tokens_pruned,
                }
                total_token += len(seq.token_ids) - 1
                total_prune += len(first_layer_pruned)

        # if total_token != 0:
        #     print(f"[real sparsity]: {(total_prune / total_token):.2f}")
        # No global synchronize here; let transfers overlap with subsequent work
        if self.dirty:
            event = torch.cuda.Event(blocking=False, enable_timing=return_timing)
            event.record(stream)
            if return_timing and start_event is not None:
                return event, start_event
            else:
                return event
        else:
            return None

    def get_kv_cache(
        self,
        seqs: list[Sequence],
        cancel_event: Event,
        stream: torch.cuda.Stream,
        return_timing: bool = False,
    ):
        """
        Schedule CPU->GPU H2D copies for KV cache on the provided CUDA stream.
        Returns a CUDA event recorded on that stream to signal completion, or None if no copies were enqueued.
        Does not call global synchronize.
        """
        _, num_layers, _, block_size, _, _ = self.gpu_kv_cache.shape
        any_copied = False
        # Use provided stream (preferred) or current stream
        with torch.cuda.stream(stream):
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)
            for seq in seqs:
                if seq.text_id is None or seq.num_cached_tokens >= seq.text_token_len:
                    continue
                item = self.kv_cache_index.get(seq.text_id)
                if item is None:
                    continue
                # Support legacy tensor or new dict format
                cpu_kv_cache = item.get("kv") if not self.force_gds else None
                if cpu_kv_cache is None and "gds_cache" in item:
                    cpu_kv_cache = item.get("gds_cache")
                kv_path = item.get("kv_path")
                # Prefer GPUDirect when enabled and metadata exists
                if self.use_gds and kv_path is not None:
                    gpu_cache = self._load_kv_with_gds(
                        kv_path,
                        item.get("kv_shape"),
                        item.get("kv_dtype"),
                        stream,
                    )
                    if gpu_cache is not None:
                        item["gds_cache"] = gpu_cache
                        cpu_kv_cache = gpu_cache

                if cpu_kv_cache is None:
                    continue

                # Only copy tokens that aren't already cached (full blocks only)
                start_token = seq.num_cached_tokens
                start_block_idx = start_token // block_size
                token_offset = start_token
                # gpu_kv_cache[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                # cpu_kv_cache[2, num_layers, seq_len, num_kv_heads, head_dim]
                # Skip fully cached leading blocks
                for block_id in seq.block_table[start_block_idx:]:
                    if cancel_event.is_set():
                        break

                    remaining = seq.text_token_len - token_offset
                    if remaining <= 0:
                        break

                    block_tokens = remaining if remaining < block_size else block_size

                    for kv_idx in range(2):
                        for layer_idx in range(num_layers):
                            dst = self.gpu_kv_cache[
                                kv_idx, layer_idx, block_id, :block_tokens
                            ]
                            src = cpu_kv_cache[
                                kv_idx,
                                layer_idx,
                                token_offset : token_offset + block_tokens,
                            ]
                            # One copy: CPU -> GPU (non_blocking if src pinned)
                            dst.copy_(src, non_blocking=True)
                            any_copied = True

                    token_offset += block_tokens
                    seq.num_cached_tokens = token_offset

        if any_copied:
            event = torch.cuda.Event(blocking=False, enable_timing=return_timing)
            event.record(stream)
            if return_timing and start_event is not None:
                return event, start_event
            else:
                return event
        else:
            return None

    def is_indexed(self, seq: Sequence):
        return seq.text_id and seq.text_id in self.kv_cache_index

    def persistence(self):
        print("[persistence]")
        if self.dirty:
            path = f"{self.save_dir}{self.index_name}"
            serializable_index: dict = {}
            for text_id, item in self.kv_cache_index.items():
                to_save = dict(item)
                # Drop GPU-resident cache before persisting
                to_save.pop("gds_cache", None)
                if self.use_gds and "kv_path" in to_save:
                    # avoid serializing full tensor when GPUDirect file exists
                    to_save.pop("kv", None)
                serializable_index[text_id] = to_save

            torch.save(serializable_index, path)
        self.dirty = False

    def _dump_kv_to_gds(self, text_id: str | int, cpu_kv_cache: torch.Tensor):
        """
        Persist KV tensor as a raw binary so it can be loaded via GPUDirect Storage.
        """
        os.makedirs(self.gds_dir, exist_ok=True)
        kv_path = os.path.join(self.gds_dir, f"{text_id}.bin")
        contiguous = cpu_kv_cache.contiguous()
        with open(kv_path, "wb") as f:
            f.write(contiguous.numpy().tobytes())
        return kv_path

    def _load_kv_with_gds(
        self,
        kv_path: Optional[str],
        kv_shape: Optional[tuple],
        kv_dtype: Optional[str],
        stream: torch.cuda.Stream,
    ):
        """
        Load KV tensor directly into GPU memory via GPUDirect Storage.
        Falls back to None if GDS is unavailable or metadata is missing.
        """
        if not (_HAS_GDS and kv_path and kv_shape and kv_dtype):
            return None

        if isinstance(kv_shape, list):
            kv_shape = tuple(kv_shape)

        dtype = getattr(torch, str(kv_dtype).split(".")[-1], None)
        if dtype is None:
            return None

        with torch.cuda.stream(stream):
            gpu_tensor = torch.empty(
                tuple(kv_shape), dtype=dtype, device=self.gpu_kv_cache.device
            )
            try:
                with CuFile(kv_path, "r") as f:  # type: ignore
                    expected = gpu_tensor.numel() * gpu_tensor.element_size()
                    read_n = f.readinto(gpu_tensor)  # type: ignore
                    if read_n != expected:
                        raise IOError(
                            f"GPUDirect read truncated: expected {expected}, got {read_n}"
                        )
            except Exception as exc:  # pragma: no cover - GPUDirect may be unavailable in CI
                print(f"[kv_cache_index] GPUDirect read failed for {kv_path}: {exc}")
                return None

        # Cache the GPU tensor for reuse
        return gpu_tensor


# Triton can only load GPU memory, so useless for now

# @triton.jit
# def get_kv_cache_kernel(
#     cpu_key_ptr,
#     cpu_value_ptr,
#     gpu_key_ptr,
#     gpu_value_ptr,
#     slot_mapping_ptr,
#     cpu_layer_stride,
#     gpu_layer_stride,
#     num_layers: tl.constexpr,
#     D: tl.constexpr,
# ):
#     idx = tl.program_id(0)
#     slot = tl.load(slot_mapping_ptr + idx)
#     if slot == -1:
#         return
#     for i in range(num_layers):
#         # offsets for source (key/value) rows
#         cpu_offsets = i * cpu_layer_stride + idx * D + tl.arange(0, D)
#         key = tl.load(cpu_key_ptr + cpu_offsets)
#         value = tl.load(cpu_value_ptr + cpu_offsets)
#
#         # offsets for destination cache (flattened)
#         gpu_offsets = i * gpu_layer_stride + slot * D + tl.arange(0, D)
#         tl.store(gpu_key_ptr + gpu_offsets, key)
#         tl.store(gpu_value_ptr + gpu_offsets, value)
#
#
# def get_kv_cache(
#     gpu_key_cache: torch.Tensor,
#     gpu_value_cache: torch.Tensor,
#     cpu_key_cache: torch.Tensor,
#     cpu_value_cache: torch.Tensor,
#     slot_mapping: torch.Tensor,
# ):
#     # get cpu to gpu
#     # gpu_key, gpu_value: [num_layers, num_blocks * block_size, num_heads, head_dim]
#     # cpu_key, cpu_value: [num_layers, seq_len, num_heads, head_dim]
#
#     num_layers, _, num_heads, head_dim = gpu_key_cache.shape
#     _, seq_len, num_heads, head_dim = cpu_key_cache.shape
#
#     D = num_heads * head_dim
#     assert slot_mapping.numel() == seq_len
#     # Launch one program per token for every layers to scatter into cache
#     get_kv_cache_kernel[(seq_len,)](
#         cpu_key_cache,
#         cpu_value_cache,
#         gpu_key_cache,
#         gpu_value_cache,
#         slot_mapping,
#         cpu_key_cache.stride(0),
#         gpu_key_cache.stride(0),
#         num_layers,  # type: ignore
#         D,  # type: ignore
#     )
