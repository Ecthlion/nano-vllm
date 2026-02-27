from __future__ import annotations

from collections import OrderedDict

import torch


class GPUHotCache:
    """
    L1 cache for GPU-resident hot KV tensors with LRU eviction.
    """

    def __init__(self, budget_gb: float = 2.0) -> None:
        self.capacity_bytes = max(0, int(float(budget_gb) * (1024**3)))
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def get(self, key: str) -> torch.Tensor | None:
        value = self.cache.get(key)
        if value is None:
            self.misses += 1
            return None
        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.capacity_bytes <= 0:
            return

        value_bytes = self._nbytes(value)
        if value_bytes > self.capacity_bytes:
            return

        old = self.cache.pop(key, None)
        if old is not None:
            self.current_bytes -= self._nbytes(old)

        self.cache[key] = value
        self.current_bytes += value_bytes

        while self.current_bytes > self.capacity_bytes and self.cache:
            _, evicted = self.cache.popitem(last=False)
            self.current_bytes -= self._nbytes(evicted)
            self.evictions += 1

    def stats(self) -> dict[str, int]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "current_bytes": self.current_bytes,
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
