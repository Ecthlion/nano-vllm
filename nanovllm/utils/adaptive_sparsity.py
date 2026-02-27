from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
from typing import Callable


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    base_sparsity: float


class AdaptiveSparsityManager:
    """
    Task-aware sparsity manager with lightweight sampling adaptation.

    Runtime path:
    - infer task type
    - load/calibrate task profile
    - build layer-head sparsity table
    """

    def __init__(self) -> None:
        self._profiles = {
            "sentiment_classification": TaskProfile("sentiment_classification", 0.90),
            "summarization": TaskProfile("summarization", 0.88),
            "scientific_qa": TaskProfile("scientific_qa", 0.84),
            "multi_hop_reasoning": TaskProfile("multi_hop_reasoning", 0.80),
            "code_generation": TaskProfile("code_generation", 0.74),
            "generic": TaskProfile("generic", 0.83),
        }
        self._store_path = os.environ.get(
            "NANOVLLM_TASK_PROFILE_STORE",
            "experiments/chapter4/task_profile_store.json",
        )
        self._runtime_stats: dict[str, dict[str, float]] = {}
        self._load_profile_store()

    def _load_profile_store(self) -> None:
        if not os.path.exists(self._store_path):
            return
        try:
            raw = json.loads(open(self._store_path, "r", encoding="utf-8").read())
        except Exception:
            return

        if not isinstance(raw, dict):
            return

        for task_type, payload in raw.items():
            try:
                base = float(payload["base_sparsity"])
            except Exception:
                continue
            self._profiles[str(task_type)] = TaskProfile(
                str(task_type), self._clip_sparsity(base)
            )

    def _save_profile_store(self) -> None:
        directory = os.path.dirname(self._store_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        serializable = {
            task: {"base_sparsity": float(profile.base_sparsity)}
            for task, profile in self._profiles.items()
        }
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def has_profile(self, task_type: str) -> bool:
        return task_type in self._profiles

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
        return 1.0 - 0.05 * overflow_ratio

    def sample_prompts_for_task(
        self,
        task_type: str,
        source_prompts: list[str],
        k: int = 16,
        seed: int = 42,
    ) -> list[str]:
        if not source_prompts:
            return []
        rng = random.Random(seed)
        if len(source_prompts) <= k:
            return list(source_prompts)
        indices = list(range(len(source_prompts)))
        rng.shuffle(indices)
        return [source_prompts[i] for i in indices[:k]]

    def calibrate_task_profile(
        self,
        task_type: str,
        sample_prompts: list[str],
        evaluator: Callable[[str, list[str], float], tuple[float, float]],
        candidates: list[float] | None = None,
        max_acc_drop: float = 0.02,
    ) -> dict[str, float]:
        """
        Lightweight sampling adaptation:
        evaluator(task_type, prompts, sparsity) -> (accuracy, speedup)
        """
        if not sample_prompts:
            return {
                "task_type": task_type,
                "selected_sparsity": self._profiles.get(task_type, self._profiles["generic"]).base_sparsity,
                "baseline_accuracy": 1.0,
                "selected_accuracy": 1.0,
                "selected_speedup": 1.0,
            }

        if candidates is None:
            candidates = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92]

        baseline_s = 0.70
        baseline_acc, _ = evaluator(task_type, sample_prompts, baseline_s)

        best: dict[str, float] | None = None
        for s in candidates:
            acc, speedup = evaluator(task_type, sample_prompts, s)
            if baseline_acc - acc <= max_acc_drop:
                if best is None or speedup > best["selected_speedup"]:
                    best = {
                        "task_type": task_type,
                        "selected_sparsity": float(s),
                        "baseline_accuracy": float(baseline_acc),
                        "selected_accuracy": float(acc),
                        "selected_speedup": float(speedup),
                    }

        if best is None:
            fallback = self._profiles.get(task_type, self._profiles["generic"]).base_sparsity
            best = {
                "task_type": task_type,
                "selected_sparsity": float(fallback),
                "baseline_accuracy": float(baseline_acc),
                "selected_accuracy": float(baseline_acc),
                "selected_speedup": 1.0,
            }

        self._profiles[task_type] = TaskProfile(
            task_type,
            self._clip_sparsity(best["selected_sparsity"]),
        )
        self._save_profile_store()
        return best

    def update_profile_ema(
        self,
        task_type: str,
        observed_latency_ms: float,
        observed_quality: float,
        alpha: float = 0.2,
    ) -> TaskProfile:
        stat = self._runtime_stats.setdefault(
            task_type,
            {
                "latency_ms": float(observed_latency_ms),
                "quality": float(observed_quality),
            },
        )

        stat["latency_ms"] = (1.0 - alpha) * stat["latency_ms"] + alpha * float(observed_latency_ms)
        stat["quality"] = (1.0 - alpha) * stat["quality"] + alpha * float(observed_quality)

        profile = self._profiles.get(task_type, self._profiles["generic"])
        new_sparsity = profile.base_sparsity

        if stat["quality"] < 0.92:
            new_sparsity -= 0.02
        elif stat["latency_ms"] > 120.0:
            new_sparsity += 0.01

        updated = TaskProfile(task_type, self._clip_sparsity(new_sparsity))
        self._profiles[task_type] = updated
        self._save_profile_store()
        return updated

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
