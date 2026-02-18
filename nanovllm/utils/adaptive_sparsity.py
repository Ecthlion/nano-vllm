from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    base_sparsity: float


class AdaptiveSparsityManager:
    """
    Task-layer-head adaptive sparsity policy used by Chapter 4.

    The policy is intentionally lightweight at runtime:
    - infer task type once per request
    - build a per-layer/per-head sparsity table offline in O(L*H)
    - attention kernels only consume the table
    """

    def __init__(self) -> None:
        self._profiles = {
            "sentiment_classification": TaskProfile(
                "sentiment_classification", 0.90
            ),
            "summarization": TaskProfile("summarization", 0.90),
            "scientific_qa": TaskProfile("scientific_qa", 0.85),
            "multi_hop_reasoning": TaskProfile("multi_hop_reasoning", 0.80),
            "code_generation": TaskProfile("code_generation", 0.70),
            "generic": TaskProfile("generic", 0.83),
        }

    def infer_task_type(self, prompt: str, user_task_type: str | None = None) -> str:
        if user_task_type:
            return user_task_type
        text = prompt.lower()
        if any(x in text for x in ("sentiment", "classify", "classification")):
            return "sentiment_classification"
        if any(x in text for x in ("summary", "summarize", "abstract")):
            return "summarization"
        if any(x in text for x in ("hotpot", "multi-hop", "reason", "strategyqa")):
            return "multi_hop_reasoning"
        if any(x in text for x in ("code", "program", "function", "humaneval", "mbpp")):
            return "code_generation"
        if any(x in text for x in ("qasper", "s2orc", "paper", "scientific", "citation")):
            return "scientific_qa"
        return "generic"

    def build_layer_head_sparsity(
        self,
        task_type: str,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        seq_len_p95: int = 2048,
    ) -> list[list[float]]:
        profile = self._profiles.get(task_type, self._profiles["generic"])
        base = profile.base_sparsity
        length_guard = self._length_guard(seq_len, seq_len_p95)
        out: list[list[float]] = []
        for layer_idx in range(num_layers):
            lf = self._layer_factor(layer_idx, num_layers, task_type)
            row = []
            for head_idx in range(num_heads):
                hf = self._head_factor(head_idx, num_heads)
                s = base * lf * hf * length_guard
                row.append(self._clip_sparsity(s))
            out.append(row)
        return out

    @staticmethod
    def _clip_sparsity(v: float) -> float:
        return max(0.50, min(0.97, float(v)))

    @staticmethod
    def _head_factor(head_idx: int, num_heads: int) -> float:
        if num_heads <= 1:
            return 1.0
        phase = (head_idx + 1) / num_heads
        return 1.0 + 0.08 * math.sin(2.0 * math.pi * phase)

    @staticmethod
    def _layer_factor(layer_idx: int, num_layers: int, task_type: str) -> float:
        if num_layers <= 0:
            return 1.0
        ratio = (layer_idx + 0.5) / num_layers
        if ratio < 0.25:
            factor = 0.95
        elif ratio < 0.75:
            factor = 1.00
        else:
            factor = 1.05

        # Code and reasoning tasks are usually more sensitive in upper layers.
        if task_type == "code_generation" and ratio >= 0.70:
            factor *= 0.88
        elif task_type == "multi_hop_reasoning" and ratio >= 0.65:
            factor *= 0.92
        return factor

    @staticmethod
    def _length_guard(seq_len: int, seq_len_p95: int) -> float:
        if seq_len_p95 <= 0 or seq_len <= seq_len_p95:
            return 1.0
        overflow_ratio = min(1.0, (seq_len - seq_len_p95) / seq_len_p95)
        # Long-tail sequences apply a conservative multiplier.
        return 1.0 - 0.05 * overflow_ratio

    def offline_evolution_search(
        self,
        objective: Callable[[list[float]], tuple[float, float]],
        dim: int,
        max_loss: float = 0.05,
        population: int = 50,
        parents: int = 25,
        generations: int = 80,
        seed: int = 42,
    ) -> dict[str, object]:
        """
        Lightweight CMA-ES-style search used for offline profile generation.

        objective(vector) -> (speedup, loss)
        """
        rng = random.Random(seed)
        parents = max(1, min(parents, population))
        dim = max(1, dim)

        mean = [0.83] * dim
        diag_var = [0.08] * dim
        sigma = 0.12
        best_vec = mean[:]
        best_fit = -1e9

        for _ in range(generations):
            candidates: list[tuple[float, list[float]]] = []
            for _ in range(population):
                vec = []
                for i in range(dim):
                    std = max(1e-5, sigma * math.sqrt(diag_var[i]))
                    v = rng.gauss(mean[i], std)
                    vec.append(self._clip_sparsity(v))
                speedup, loss = objective(vec)
                penalty = max(0.0, loss - max_loss) * 100.0
                fit = speedup - penalty
                candidates.append((fit, vec))
                if fit > best_fit:
                    best_fit = fit
                    best_vec = vec

            candidates.sort(key=lambda x: x[0], reverse=True)
            elites = [x[1] for x in candidates[:parents]]

            new_mean = []
            for i in range(dim):
                new_mean.append(sum(v[i] for v in elites) / len(elites))
            mean = [self._clip_sparsity(v) for v in new_mean]

            new_var = []
            for i in range(dim):
                mu = mean[i]
                var = sum((v[i] - mu) ** 2 for v in elites) / len(elites)
                new_var.append(max(1e-5, var))
            diag_var = new_var
            sigma = min(0.30, max(0.01, 0.9 * sigma + 0.1 * math.sqrt(sum(diag_var) / dim)))

        return {
            "best_vector": best_vec,
            "best_fitness": best_fit,
            "population": population,
            "parents": parents,
            "generations": generations,
        }
