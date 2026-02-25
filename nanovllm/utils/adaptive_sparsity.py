from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    base_sparsity: float


class AdaptiveSparsityManager:
    """
    Task-Layer-Head adaptive sparsity manager.

    Runtime path:
    - infer task type
    - fetch cached (or heuristic) per-layer/per-head sparsity table

    Offline path:
    - run CMA-ES to optimize a task-specific table
    - cache optimized table for O(1) online lookup
    """

    def __init__(self) -> None:
        self._profiles = {
            "sentiment_classification": TaskProfile("sentiment_classification", 0.90),
            "summarization": TaskProfile("summarization", 0.90),
            "scientific_qa": TaskProfile("scientific_qa", 0.85),
            "multi_hop_reasoning": TaskProfile("multi_hop_reasoning", 0.80),
            "code_generation": TaskProfile("code_generation", 0.70),
            "generic": TaskProfile("generic", 0.83),
        }
        # (task_type, num_layers, num_heads) -> table[num_layers][num_heads]
        self._optimized_tables: dict[tuple[str, int, int], list[list[float]]] = {}

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
        use_cmaes: bool = False,
    ) -> list[list[float]]:
        """
        Build a per-layer/per-head sparsity table.

        If `use_cmaes=True`, this uses (or lazily creates) a CMA-ES optimized table.
        """
        num_layers = max(1, int(num_layers))
        num_heads = max(1, int(num_heads))

        key = (task_type, num_layers, num_heads)
        if use_cmaes:
            if key not in self._optimized_tables:
                self.optimize_task_table_with_cmaes(
                    task_type=task_type,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    seed=42,
                )
            table = self._optimized_tables.get(key)
            if table is not None:
                # Apply only sequence-length guard online.
                length_guard = self._length_guard(seq_len, seq_len_p95)
                return [
                    [self._clip_sparsity(v * length_guard) for v in row]
                    for row in table
                ]

        return self._build_heuristic_table(
            task_type=task_type,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            seq_len_p95=seq_len_p95,
        )

    def optimize_task_table_with_cmaes(
        self,
        task_type: str,
        num_layers: int,
        num_heads: int,
        max_loss: float = 0.05,
        population: int = 50,
        parents: int = 25,
        generations: int = 80,
        seed: int = 42,
        force: bool = False,
    ) -> list[list[float]]:
        """
        Offline CMA-ES optimizer for a task-specific layer-head sparsity table.

        The objective is a surrogate of the thesis objective:
        maximize speedup under an accuracy-loss constraint.
        """
        num_layers = max(1, int(num_layers))
        num_heads = max(1, int(num_heads))
        key = (task_type, num_layers, num_heads)
        if not force and key in self._optimized_tables:
            return self._optimized_tables[key]

        baseline = self._build_heuristic_table(
            task_type=task_type,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=2048,
            seq_len_p95=2048,
        )
        baseline_vec = [v for row in baseline for v in row]
        dim = len(baseline_vec)

        def objective(vec: list[float]) -> tuple[float, float]:
            # Speed proxy: higher sparsity => lower I/O and attention compute.
            avg_sparsity = sum(vec) / max(1, len(vec))
            speedup = 1.0 / max(1e-3, 1.0 - avg_sparsity)

            # Loss proxy: deviation from task/layer/head priors + roughness penalty.
            l1 = sum(abs(a - b) for a, b in zip(vec, baseline_vec)) / max(1, dim)

            smooth_pen = 0.0
            # Layer smoothness (same head across adjacent layers)
            for l in range(num_layers - 1):
                row_a = vec[l * num_heads : (l + 1) * num_heads]
                row_b = vec[(l + 1) * num_heads : (l + 2) * num_heads]
                smooth_pen += sum(abs(a - b) for a, b in zip(row_a, row_b)) / num_heads
            smooth_pen /= max(1, num_layers - 1)

            # Head smoothness (adjacent heads in same layer)
            head_pen = 0.0
            for l in range(num_layers):
                row = vec[l * num_heads : (l + 1) * num_heads]
                if len(row) > 1:
                    head_pen += sum(abs(row[i] - row[i - 1]) for i in range(1, len(row))) / (
                        len(row) - 1
                    )
            head_pen /= max(1, num_layers)

            loss = 0.70 * l1 + 0.20 * smooth_pen + 0.10 * head_pen
            return speedup, loss

        result = self.offline_evolution_search(
            objective=objective,
            dim=dim,
            max_loss=max_loss,
            population=population,
            parents=parents,
            generations=generations,
            seed=seed,
        )

        best_vec = result["best_vector"]
        table: list[list[float]] = []
        cursor = 0
        for _ in range(num_layers):
            row = [self._clip_sparsity(v) for v in best_vec[cursor : cursor + num_heads]]
            table.append(row)
            cursor += num_heads

        self._optimized_tables[key] = table
        return table

    def _build_heuristic_table(
        self,
        task_type: str,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        seq_len_p95: int,
    ) -> list[list[float]]:
        profile = self._profiles.get(task_type, self._profiles["generic"])
        base = profile.base_sparsity
        length_guard = self._length_guard(seq_len, seq_len_p95)
        out: list[list[float]] = []
        for layer_idx in range(num_layers):
            lf = self._layer_factor(layer_idx, num_layers, task_type)
            row: list[float] = []
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

        # Code/reasoning tasks are more sensitive in upper layers.
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

    def offline_evolution_search(
        self,
        objective: Callable[[list[float]], tuple[float, float]],
        dim: int,
        max_loss: float = 0.05,
        population: int = 50,
        parents: int = 25,
        generations: int = 80,
        seed: int = 42,
        stall_generations: int = 10,
        trace_tol: float = 1e-4,
    ) -> dict[str, object]:
        """
        CMA-ES optimizer for black-box sparsity-search objective.

        objective(vector) -> (speedup, loss)
        fitness = speedup - penalty(loss > max_loss)
        """
        parents = max(1, min(int(parents), int(population)))
        population = max(parents, int(population))
        dim = max(1, int(dim))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        mean = torch.full((dim,), 0.83, dtype=torch.float64)
        cov = torch.eye(dim, dtype=torch.float64)
        sigma = torch.tensor(0.12, dtype=torch.float64)

        w = torch.log(torch.tensor(parents + 0.5, dtype=torch.float64)) - torch.log(
            torch.arange(1, parents + 1, dtype=torch.float64)
        )
        weights = w / torch.sum(w)
        mu_eff = 1.0 / torch.sum(weights * weights)

        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        c1 = 2.0 / (((dim + 1.3) ** 2) + mu_eff)
        c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (((dim + 2.0) ** 2) + mu_eff))

        p_sigma = torch.zeros(dim, dtype=torch.float64)
        p_c = torch.zeros(dim, dtype=torch.float64)
        chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        best_fit = -1e12
        best_vec = mean.tolist()
        best_speedup = 0.0
        best_loss = 1e12
        stall = 0

        for g in range(max(1, generations)):
            jitter = 1e-10 * torch.eye(dim, dtype=torch.float64)
            chol = torch.linalg.cholesky(cov + jitter)

            z = torch.randn((population, dim), generator=gen, dtype=torch.float64)
            y = z @ chol.T
            x = mean.unsqueeze(0) + sigma * y
            x = torch.clamp(x, 0.50, 0.97)

            scored: list[tuple[float, float, float, int]] = []
            for i in range(population):
                vec = x[i].tolist()
                speedup, loss = objective(vec)
                penalty = max(0.0, loss - max_loss) * 100.0
                fit = float(speedup - penalty)
                scored.append((fit, float(speedup), float(loss), i))
                if fit > best_fit:
                    best_fit = fit
                    best_vec = vec
                    best_speedup = float(speedup)
                    best_loss = float(loss)
                    stall = 0

            scored.sort(key=lambda t: t[0], reverse=True)
            elite_idx = [idx for _, _, _, idx in scored[:parents]]
            x_sel = x[elite_idx]
            y_sel = y[elite_idx]

            prev_best = best_fit

            mean = torch.sum(weights.unsqueeze(1) * x_sel, dim=0)
            y_w = torch.sum(weights.unsqueeze(1) * y_sel, dim=0)

            # C^{-1/2} y_w from Cholesky factor.
            c_inv_half_y = torch.linalg.solve_triangular(
                chol, y_w.unsqueeze(1), upper=False
            ).squeeze(1)

            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(
                c_sigma * (2.0 - c_sigma) * mu_eff
            ) * c_inv_half_y

            norm_p_sigma = float(torch.linalg.vector_norm(p_sigma).item())
            sigma_cond_denom = math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (g + 1)))
            h_sigma = 1.0 if norm_p_sigma / sigma_cond_denom < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0

            p_c = (1.0 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w

            rank_mu = torch.zeros((dim, dim), dtype=torch.float64)
            for wi, yi in zip(weights, y_sel):
                rank_mu += wi * torch.outer(yi, yi)

            hsig_term = (1.0 - h_sigma) * c_c * (2.0 - c_c)
            cov = (
                (1.0 - c1 - c_mu + hsig_term * c1) * cov
                + c1 * torch.outer(p_c, p_c)
                + c_mu * rank_mu
            )
            cov = (cov + cov.T) * 0.5

            sigma = sigma * math.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1.0))
            sigma = torch.clamp(sigma, min=1e-3, max=0.5)

            trace_cov = float(torch.trace(cov).item())
            if abs(best_fit - prev_best) < 1e-7:
                stall += 1

            if stall >= max(1, stall_generations) and trace_cov < trace_tol and float(sigma.item()) < 0.01:
                break

        return {
            "best_vector": [self._clip_sparsity(v) for v in best_vec],
            "best_fitness": float(best_fit),
            "best_speedup": float(best_speedup),
            "best_loss": float(best_loss),
            "population": population,
            "parents": parents,
            "generations": generations,
            "stopped_early": stall >= max(1, stall_generations),
        }
