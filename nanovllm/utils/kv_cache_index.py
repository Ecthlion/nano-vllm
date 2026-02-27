import os
from threading import Event
from typing import Optional

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.csr_kv_store import CSRKVStore
from nanovllm.utils.two_level_cache import GPUHotCache

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

        self.use_csr_kv = os.environ.get("NANOVLLM_USE_CSR_KV", "1") == "1"
        self.csr_data_align_bytes = int(
            os.environ.get("NANOVLLM_CSR_DATA_ALIGN_BYTES", "4096")
        )
        self.csr_record_align_bytes = int(
            os.environ.get("NANOVLLM_CSR_RECORD_ALIGN_BYTES", "512")
        )
        self.csr_commit_interval = int(
            os.environ.get("NANOVLLM_CSR_COMMIT_INTERVAL", "32")
        )

        self.enable_gpu_hot_cache = (
            os.environ.get("NANOVLLM_GPU_HOT_CACHE_ENABLE", "1") == "1"
        )
        self.gpu_hot_cache_gb = float(
            os.environ.get("NANOVLLM_GPU_HOT_CACHE_GB", "1.5")
        )

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.gds_dir = os.path.join(self.save_dir, "kv_cache_gds")
        if self.use_gds:
            os.makedirs(self.gds_dir, exist_ok=True)

        path = os.path.join(self.save_dir, self.index_name)

        self.dirty = False
        self.indexed = False
        # text_id -> {
        #   "kv": Tensor[2, num_layers, seq_len_post_prune, num_kv_heads, head_dim],
        #   "pruning_len": int,
        #   "text_tokens_pruned": list[int] | None,
        #   "ilh_bounds": {"core": int, "important": int, "optional": int},
        #   "task_type": str,
        # }
        self.kv_cache_index: dict = {}

        self.csr_store: CSRKVStore | None = None
        if self.use_csr_kv:
            self.csr_store = CSRKVStore(
                root_dir=self.save_dir,
                data_align_bytes=self.csr_data_align_bytes,
                record_align_bytes=self.csr_record_align_bytes,
                commit_interval=self.csr_commit_interval,
            )
            persisted_entries = self.csr_store.build_index_entries()
            if persisted_entries:
                self.kv_cache_index.update(persisted_entries)
                self.indexed = True

        if not self.kv_cache_index and os.path.isfile(path):
            self.kv_cache_index = torch.load(path)
            self.indexed = True

            # pin memory when init
            for _, item in list(self.kv_cache_index.items()):
                kv = item.get("kv")
                if isinstance(kv, torch.Tensor) and kv.device.type == "cpu":
                    if not kv.is_pinned():
                        item["kv"] = kv.pin_memory()
                # legacy files may not include kv shape/dtype metadata
                if "kv" in item:
                    item.setdefault("kv_shape", tuple(item["kv"].shape))  # type: ignore
                    item.setdefault("kv_dtype", str(item["kv"].dtype))  # type: ignore

        self.gpu_hot_cache: GPUHotCache | None = None
        if self.enable_gpu_hot_cache:
            self.gpu_hot_cache = GPUHotCache(self.gpu_hot_cache_gb)

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
                if post_prune_len <= 0:
                    continue

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
                    cnt = min(block_size, seq.text_token_len - taken)
                    base = block_id * block_size
                    token_slots.extend(range(base, base + cnt))
                    taken += cnt

                if len(token_slots) != seq.text_token_len:
                    token_slots = token_slots[: seq.text_token_len]

                flat_blocks = self.gpu_kv_cache.shape[2] * block_size
                device = self.gpu_kv_cache.device

                for layer_idx in range(num_layers):
                    pruned = [
                        idx
                        for idx in layer_pruned[layer_idx]
                        if idx < seq.text_token_len
                    ]
                    if pruned:
                        pruned_set = set(pruned)
                        kept_slots = [
                            slot for i, slot in enumerate(token_slots) if i not in pruned_set
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
                    i for i in range(seq.text_token_len) if i not in first_layer_pruned_set
                ]
                text_tokens_pruned = [seq.token_ids[i] for i in kept_local_indices]

                core_tokens = max(1, int(post_prune_len * 0.70))
                important_tokens = max(core_tokens, int(post_prune_len * 0.85))
                important_tokens = min(important_tokens, post_prune_len)
                ilh_bounds = {
                    "core": core_tokens,
                    "important": important_tokens,
                    "optional": post_prune_len,
                }

                self.dirty = True
                kv_path = None
                if self.use_gds:
                    kv_path = self._dump_kv_to_gds(seq.text_id, cpu_kv_cache)

                item = {
                    "kv": cpu_kv_cache,
                    "kv_path": kv_path,
                    "kv_shape": tuple(cpu_kv_cache.shape),
                    "kv_dtype": str(cpu_kv_cache.dtype),
                    "pruning_len": len(first_layer_pruned),
                    "text_tokens_pruned": text_tokens_pruned,
                    "ilh_bounds": ilh_bounds,
                    "task_type": getattr(seq, "task_type", "generic"),
                    "sparsity_ratio": float(len(first_layer_pruned) / max(1, seq.text_token_len)),
                }

                if self.csr_store is not None:
                    csr_meta = self.csr_store.append_record(
                        text_id=seq.text_id,
                        kv_tensor=cpu_kv_cache,
                        kept_indices=kept_local_indices,
                        pruning_len=len(first_layer_pruned),
                        text_tokens_pruned=text_tokens_pruned,
                        ilh_bounds=ilh_bounds,
                        task_type=getattr(seq, "task_type", "generic"),
                        force_sync=False,
                    )
                    item["csr_offset"] = int(csr_meta["offset"])
                    item["csr_length"] = int(csr_meta["length"])
                    item["csr_record_key"] = str(csr_meta["record_key"])
                    item["write_amplification"] = float(csr_meta.get("write_amplification", 0.0))

                self.kv_cache_index[seq.text_id] = item
                total_token += len(seq.token_ids) - 1
                total_prune += len(first_layer_pruned)

        if self.dirty:
            event = torch.cuda.Event(blocking=False, enable_timing=return_timing)
            event.record(stream)
            if return_timing and start_event is not None:
                return event, start_event
            return event
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
        Returns a CUDA event recorded on that stream to signal completion.
        """
        _, num_layers, _, block_size, _, _ = self.gpu_kv_cache.shape
        any_copied = False

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

                precision_tier = getattr(seq, "precision_tier", "balanced")
                if precision_tier == "fast":
                    level_name = "core"
                elif precision_tier == "high":
                    level_name = "optional"
                else:
                    level_name = "important"

                load_token_limit = seq.text_token_len
                bounds = item.get("ilh_bounds") if isinstance(item, dict) else None
                if isinstance(bounds, dict):
                    level_limit = bounds.get(level_name)
                    if isinstance(level_limit, int):
                        load_token_limit = min(load_token_limit, level_limit)

                cache_key = str(seq.text_id)
                cpu_kv_cache = None if self.force_gds else item.get("kv")

                if cpu_kv_cache is None and self.gpu_hot_cache is not None:
                    hot = self.gpu_hot_cache.get(cache_key)
                    if hot is not None:
                        cpu_kv_cache = hot

                if cpu_kv_cache is None and "gds_cache" in item:
                    cached_tokens = item.get("gds_cache_tokens")
                    if not isinstance(cached_tokens, int) or cached_tokens >= load_token_limit:
                        cpu_kv_cache = item.get("gds_cache")

                kv_path = item.get("kv_path")
                if self.use_gds and kv_path is not None:
                    gpu_cache = self._load_kv_with_gds(
                        kv_path,
                        item.get("kv_shape"),
                        item.get("kv_dtype"),
                        stream,
                        load_tokens=load_token_limit,
                    )
                    if gpu_cache is not None:
                        item["gds_cache"] = gpu_cache
                        item["gds_cache_tokens"] = int(gpu_cache.shape[2])
                        cpu_kv_cache = gpu_cache

                if (
                    cpu_kv_cache is None
                    and self.use_gds
                    and self.csr_store is not None
                ):
                    gpu_cache = self._load_csr_kv_with_gds(
                        text_id=seq.text_id,
                        item=item,
                        stream=stream,
                        load_tokens=load_token_limit,
                    )
                    if gpu_cache is not None:
                        item["gds_cache"] = gpu_cache
                        item["gds_cache_tokens"] = int(gpu_cache.shape[2])
                        cpu_kv_cache = gpu_cache

                if cpu_kv_cache is None and self.csr_store is not None:
                    loaded = self.csr_store.load_record(seq.text_id)
                    if loaded is not None:
                        cpu_kv_cache = loaded.get("kv")
                        if (
                            isinstance(cpu_kv_cache, torch.Tensor)
                            and cpu_kv_cache.device.type == "cpu"
                            and not cpu_kv_cache.is_pinned()
                        ):
                            cpu_kv_cache = cpu_kv_cache.pin_memory()
                        item["kv"] = cpu_kv_cache
                        if isinstance(cpu_kv_cache, torch.Tensor):
                            item.setdefault("kv_shape", tuple(cpu_kv_cache.shape))
                            item.setdefault("kv_dtype", str(cpu_kv_cache.dtype))
                        item.setdefault("pruning_len", loaded["meta"].get("pruning_len", 0))
                        item.setdefault(
                            "text_tokens_pruned",
                            loaded["meta"].get("text_tokens_pruned", []),
                        )
                        item.setdefault("ilh_bounds", loaded["meta"].get("ilh_bounds"))
                        item.setdefault("task_type", loaded["meta"].get("task_type", "generic"))

                if cpu_kv_cache is None:
                    continue

                if self.gpu_hot_cache is not None and isinstance(cpu_kv_cache, torch.Tensor):
                    if cpu_kv_cache.device.type == "cuda":
                        self.gpu_hot_cache.put(cache_key, cpu_kv_cache)

                if (
                    isinstance(cpu_kv_cache, torch.Tensor)
                    and cpu_kv_cache.ndim >= 3
                    and cpu_kv_cache.shape[2] > load_token_limit
                ):
                    cpu_kv_cache = cpu_kv_cache[:, :, :load_token_limit]

                # Only copy tokens that are not already cached
                start_token = seq.num_cached_tokens
                start_block_idx = start_token // block_size
                token_offset = start_token
                target_token_len = min(
                    seq.text_token_len,
                    load_token_limit,
                    int(cpu_kv_cache.shape[2]),  # type: ignore[index]
                )

                for block_id in seq.block_table[start_block_idx:]:
                    if cancel_event.is_set():
                        break

                    remaining = target_token_len - token_offset
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
                            dst.copy_(src, non_blocking=True)
                            any_copied = True

                    token_offset += block_tokens
                    seq.num_cached_tokens = token_offset

        if any_copied:
            event = torch.cuda.Event(blocking=False, enable_timing=return_timing)
            event.record(stream)
            if return_timing and start_event is not None:
                return event, start_event
            return event
        return None

    def is_indexed(self, seq: Sequence):
        return seq.text_id and seq.text_id in self.kv_cache_index

    def persistence(self):
        print("[persistence]")
        if self.csr_store is not None:
            self.csr_store.finalize()

        if self.dirty:
            path = os.path.join(self.save_dir, self.index_name)
            serializable_index: dict = {}
            for text_id, item in self.kv_cache_index.items():
                to_save = dict(item)
                # Drop GPU-resident cache before persisting
                to_save.pop("gds_cache", None)
                to_save.pop("gds_cache_tokens", None)
                if self.use_csr_kv:
                    # CSR-KV records are persisted in aligned binary + manifest.
                    to_save.pop("kv", None)
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
        load_tokens: Optional[int] = None,
    ):
        """
        Load KV tensor directly into GPU memory via GPUDirect Storage.
        Falls back to None if GDS is unavailable or metadata is missing.
        """
        if not (_HAS_GDS and kv_path and kv_shape and kv_dtype):
            return None

        if isinstance(kv_shape, list):
            kv_shape = tuple(kv_shape)
        kv_shape = tuple(int(x) for x in kv_shape)
        if len(kv_shape) < 3:
            return None

        target_tokens = int(kv_shape[2])
        if load_tokens is not None:
            load_tokens = max(0, int(load_tokens))
            if load_tokens == 0:
                return None
            target_tokens = min(load_tokens, int(kv_shape[2]))

        dtype = getattr(torch, str(kv_dtype).split(".")[-1], None)
        if dtype is None:
            return None

        with torch.cuda.stream(stream):
            gpu_tensor = torch.empty(
                kv_shape, dtype=dtype, device=self.gpu_kv_cache.device
            )
            try:
                with CuFile(kv_path, "r") as f:  # type: ignore
                    expected = gpu_tensor.numel() * gpu_tensor.element_size()
                    read_n = f.readinto(gpu_tensor)  # type: ignore
                    if read_n != expected:
                        raise IOError(
                            f"GPUDirect read truncated: expected {expected}, got {read_n}"
                        )
            except Exception as exc:  # pragma: no cover
                print(f"[kv_cache_index] GPUDirect read failed for {kv_path}: {exc}")
                return None

        if target_tokens < int(kv_shape[2]):
            gpu_tensor = gpu_tensor[:, :, :target_tokens]

        return gpu_tensor

    def _cufile_readinto_with_offset(
        self,
        cu_file,
        dst_tensor: torch.Tensor,
        file_offset: int,
    ) -> Optional[int]:
        """
        Best-effort wrapper over different kvikio API variants.
        """
        try:
            return cu_file.pread(dst_tensor, file_offset=file_offset)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return cu_file.pread(dst_tensor, file_offset)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return cu_file.readinto(dst_tensor, file_offset=file_offset)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return cu_file.readinto(dst_tensor, file_offset)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            cu_file.seek(file_offset)  # type: ignore[attr-defined]
            return cu_file.readinto(dst_tensor)  # type: ignore[attr-defined]
        except Exception:
            return None

    def _load_csr_kv_with_gds(
        self,
        text_id: str | int,
        item: dict,
        stream: torch.cuda.Stream,
        load_tokens: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not (_HAS_GDS and self.csr_store is not None):
            return None

        meta = self.csr_store.get_meta(text_id)
        if meta is None:
            return None

        offsets = meta.get("offsets")
        if not isinstance(offsets, dict):
            return None

        try:
            key_offset = int(offsets["key"])
            value_offset = int(offsets["value"])
            key_nbytes = int(meta["key_nbytes"])
            value_nbytes = int(meta["value_nbytes"])
            num_layers = int(meta["num_layers"])
            num_tokens = int(meta["num_tokens"])
            num_heads = int(meta["num_heads"])
            head_dim = int(meta["head_dim"])
        except Exception:
            return None

        if load_tokens is not None:
            num_tokens = min(num_tokens, max(0, int(load_tokens)))
            if num_tokens <= 0:
                return None
            elem_size = torch.tensor([], dtype=getattr(torch, str(meta.get("dtype", "torch.float16")).split(".")[-1], torch.float16)).element_size()
            key_nbytes = int(num_layers * num_tokens * num_heads * head_dim * elem_size)
            value_nbytes = int(num_layers * num_tokens * num_heads * head_dim * elem_size)

        dtype_name = str(meta.get("dtype", item.get("kv_dtype", "torch.float16")))
        dtype = getattr(torch, dtype_name.split(".")[-1], None)
        if dtype is None:
            return None

        with torch.cuda.stream(stream):
            gpu_kv = torch.empty(
                (2, num_layers, num_tokens, num_heads, head_dim),
                dtype=dtype,
                device=self.gpu_kv_cache.device,
            )
            try:
                with CuFile(self.csr_store.data_path, "r") as f:  # type: ignore
                    key_read_n = self._cufile_readinto_with_offset(
                        f, gpu_kv[0], key_offset
                    )
                    value_read_n = self._cufile_readinto_with_offset(
                        f, gpu_kv[1], value_offset
                    )
                    if key_read_n != key_nbytes or value_read_n != value_nbytes:
                        raise IOError(
                            f"GPUDirect CSR read truncated: key {key_read_n}/{key_nbytes}, "
                            f"value {value_read_n}/{value_nbytes}"
                        )
            except Exception as exc:  # pragma: no cover
                print(f"[kv_cache_index] GPUDirect CSR read failed for {text_id}: {exc}")
                return None

        return gpu_kv

    def cache_stats(self) -> dict[str, object]:
        stats = {"indexed_items": len(self.kv_cache_index)}
        if self.gpu_hot_cache is not None:
            stats["gpu_hot_cache"] = self.gpu_hot_cache.stats()
        if self.csr_store is not None:
            stats["csr_write_amplification"] = self.csr_store.write_amplification_summary()
        return stats
