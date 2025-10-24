import os

import numpy as np
import torch

from nanovllm.engine.sequence import Sequence


class KVCacheIndex:
    # TODO: cpu cache -> ssd
    def __init__(self, gpu_kv_cache, database_name="imdb") -> None:
        # Tensor[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.gpu_kv_cache = gpu_kv_cache
        self.path = f"/data/zwt/{database_name}_kvcache.pt"
        self.dirty = False

        # text_id -> kv_cache[2, num_layers, seq_len, num_kv_heads, head_dim]
        self.kv_cache_index: dict = {}
        if os.path.isfile(self.path):
            self.kv_cache_index = torch.load(self.path)

    def store_kv_cache(self, seqs: list[Sequence]):
        for seq in seqs:
            if seq.text_id is None or seq.text_id in self.kv_cache_index:
                continue

            # 1. Allocate cpu mem for kv cache transfer
            _, num_layers, _, block_size, num_kv_heads, head_dim = (
                self.gpu_kv_cache.shape
            )

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
                        if token_offset + block_size > seq.text_token_len:
                            block_tokens = seq.text_token_len - token_offset
                        else:
                            block_tokens = block_size

                        if block_tokens <= 0:
                            break

                        dst = cpu_kv_cache[
                            kv_idx,
                            layer_idx,
                            token_offset : token_offset + block_tokens,
                        ]
                        src = self.gpu_kv_cache[
                            kv_idx, layer_idx, block_id, :block_tokens
                        ]
                        dst.copy_(src, non_blocking=True)

                        token_offset += block_tokens
                        if token_offset >= seq.text_token_len:
                            break

            self.dirty = True
            self.kv_cache_index[seq.text_id] = cpu_kv_cache

        torch.cuda.synchronize()

    def get_kv_cache(self, seqs: list[Sequence]):
        _, num_layers, _, block_size, _, _ = self.gpu_kv_cache.shape
        for seq in seqs:
            if seq.text_id is None or seq.num_cached_tokens >= seq.text_token_len:
                continue
            cpu_kv_cache = self.kv_cache_index.get(seq.text_id)
            if cpu_kv_cache is None:
                continue

            for kv_idx in range(2):
                for layer_idx in range(num_layers):
                    token_offset = 0
                    for block_id in seq.block_table:
                        remaining = seq.text_token_len - token_offset
                        if remaining <= 0:
                            break
                        block_tokens = (
                            remaining if remaining < block_size else block_size
                        )

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

                        token_offset += block_tokens
                        if token_offset >= seq.text_token_len:
                            break

            # the prefix is already considered
            seq.num_cached_tokens = seq.text_token_len

        torch.cuda.synchronize()

    def is_indexed(self, seq: Sequence):
        return seq.text_id and seq.text_id in self.kv_cache_index

    def persistence(self):
        if self.dirty:
            torch.save(self.kv_cache_index, self.path)
        self.dirty = False
