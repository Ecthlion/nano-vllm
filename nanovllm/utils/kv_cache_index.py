import os
from threading import Event
from typing import Optional

import numpy as np
import torch

from nanovllm.engine.sequence import Sequence


class KVCacheIndex:
    # TODO: cpu cache overflow ssd
    def __init__(self, gpu_kv_cache, index_name="imdb_kvcache.pt") -> None:
        # Tensor[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.gpu_kv_cache = gpu_kv_cache
        self.path = f"/data/zhangyuyun/{index_name}"
        self.dirty = False
        self.indexed = False
        # text_id -> kv_cache[2, num_layers, seq_len, num_kv_heads, head_dim]
        self.kv_cache_index: dict = {}
        if os.path.isfile(self.path):
            self.kv_cache_index = torch.load(self.path)
            self.indexed = True

            # pin memory when init
            for id, kv_cache in list(self.kv_cache_index.items()):
                if isinstance(kv_cache, torch.Tensor) and not kv_cache.is_pinned():
                    self.kv_cache_index[id] = kv_cache.pin_memory()

    def store_kv_cache(
        self,
        seqs: list[Sequence],
        stream: torch.cuda.Stream,
        return_timing: bool = False,
    ):
        _, num_layers, _, block_size, num_kv_heads, head_dim = self.gpu_kv_cache.shape
        with torch.cuda.stream(stream):
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)
            for seq in seqs:
                if seq.text_id is None or seq.text_id in self.kv_cache_index:
                    # If already indexed, skip
                    continue

                # 1. allocate cpu kv cache
                cpu_kv_cache = torch.empty(
                    2,
                    num_layers,
                    seq.text_token_len,
                    num_kv_heads,
                    head_dim,
                    device="cpu",
                    dtype=self.gpu_kv_cache.dtype,
                    pin_memory=True,
                )

                # 2. gpu_kv_cache -> cpu_kv_cache
                for kv_idx in range(2):
                    for layer_idx in range(num_layers):
                        token_offset = 0  # reset for each layer
                        for block_id in seq.block_table:
                            remaining = seq.text_token_len - token_offset
                            if remaining <= 0:
                                break
                            block_tokens = (
                                remaining if remaining < block_size else block_size
                            )

                            dst = cpu_kv_cache[
                                kv_idx,
                                layer_idx,
                                token_offset : token_offset + block_tokens,
                            ]
                            src = self.gpu_kv_cache[
                                kv_idx, layer_idx, block_id, :block_tokens
                            ]
                            # Async D2H copy. Pinned dst enables non_blocking behavior.
                            dst.copy_(src, non_blocking=True)

                            token_offset += block_tokens

                self.dirty = True
                self.kv_cache_index[seq.text_id] = cpu_kv_cache
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
                cpu_kv_cache = self.kv_cache_index.get(seq.text_id)
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

                    block_tokens = (
                        remaining if remaining < block_size else block_size
                    )

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
        # TODO: pruning kv cache
        if self.dirty:
            torch.save(self.kv_cache_index, self.path)
        self.dirty = False


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
