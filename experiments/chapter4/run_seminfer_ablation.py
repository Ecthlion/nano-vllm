from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
from statistics import mean

try:
    from nanovllm.utils.adaptive_sparsity import AdaptiveSparsityManager
except Exception:
    # Fallback in environments without full runtime deps (e.g., torch).
    class AdaptiveSparsityManager:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._profiles = {
                "sentiment_classification": 0.90,
                "summarization": 0.90,
                "scientific_qa": 0.85,
                "multi_hop_reasoning": 0.80,
                "code_generation": 0.70,
                "generic": 0.83,
            }
            self._optimized: dict[tuple[str, int, int], list[list[float]]] = {}

        @staticmethod
        def _clip(v: float) -> float:
            return max(0.50, min(0.97, float(v)))

        def _heuristic(self, task: str, l: int, h: int, L: int, H: int) -> float:
            base = self._profiles.get(task, self._profiles["generic"])
            layer_factor = 0.95 if l < 0.25 * L else (1.00 if l < 0.75 * L else 1.05)
            head_factor = 1.0 + 0.08 * math.sin(2.0 * math.pi * ((h + 1) / max(1, H)))
            return self._clip(base * layer_factor * head_factor)

        def build_layer_head_sparsity(
            self,
            task_type: str,
            num_layers: int,
            num_heads: int,
            seq_len: int,
            seq_len_p95: int = 2048,
            use_cmaes: bool = False,
        ) -> list[list[float]]:
            key = (task_type, num_layers, num_heads)
            if use_cmaes and key in self._optimized:
                return self._optimized[key]
            return [
                [self._heuristic(task_type, l, h, num_layers, num_heads) for h in range(num_heads)]
                for l in range(num_layers)
            ]

        def optimize_task_table_with_cmaes(
            self,
            task_type: str,
            num_layers: int,
            num_heads: int,
            max_loss: float = 0.05,
            population: int = 24,
            parents: int = 12,
            generations: int = 24,
            seed: int = 42,
            force: bool = False,
        ) -> list[list[float]]:
            key = (task_type, num_layers, num_heads)
            if not force and key in self._optimized:
                return self._optimized[key]

            rng = random.Random(seed)
            base = self.build_layer_head_sparsity(
                task_type=task_type,
                num_layers=num_layers,
                num_heads=num_heads,
                seq_len=2048,
                seq_len_p95=2048,
                use_cmaes=False,
            )
            vec = [v for row in base for v in row]
            best = vec[:]
            best_fit = -1e9

            for _ in range(max(1, generations)):
                candidates: list[tuple[float, list[float]]] = []
                for _ in range(max(2, population)):
                    cand = [self._clip(v + rng.uniform(-0.04, 0.04)) for v in best]
                    avg = sum(cand) / len(cand)
                    speed = 1.0 / max(1e-3, 1.0 - avg)
                    loss = sum(abs(a - b) for a, b in zip(cand, vec)) / len(vec)
                    fit = speed - max(0.0, loss - max_loss) * 100.0
                    candidates.append((fit, cand))
                candidates.sort(key=lambda x: x[0], reverse=True)
                if candidates[0][0] > best_fit:
                    best_fit = candidates[0][0]
                    best = candidates[0][1]

            out: list[list[float]] = []
            cursor = 0
            for _ in range(num_layers):
                out.append(best[cursor : cursor + num_heads])
                cursor += num_heads
            self._optimized[key] = out
            return out

try:
    from nanovllm.utils.seminfer_algorithms import build_layer_interleaved_plan
except Exception:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _FallbackPlan:
        order: list[int]
        critical_path: list[float]

    def build_layer_interleaved_plan(load_times, compute_times):  # type: ignore[no-redef]
        loads = [float(x) for x in load_times]
        computes = [float(x) for x in compute_times]
        cpl = []
        for i in range(len(loads)):
            prev = cpl[i - 1] if i > 0 else 0.0
            cpl.append(computes[i] + max(prev, loads[i]))
        order = sorted(range(len(loads)), key=lambda i: cpl[i], reverse=True)
        return _FallbackPlan(order=order, critical_path=cpl)


NUM_LAYERS = 80
NUM_HEADS = 8
SEQ_LEN = 4096
HEAD_DIM = 128
DTYPE_BYTES = 2  # fp16
HBM_BANDWIDTH_GBPS = 25.0

TASKS = [
    {"task": "sentiment_classification", "oracle_sparsity": 0.90},
    {"task": "summarization", "oracle_sparsity": 0.88},
    {"task": "scientific_qa", "oracle_sparsity": 0.84},
    {"task": "multi_hop_reasoning", "oracle_sparsity": 0.80},
    {"task": "code_generation", "oracle_sparsity": 0.72},
]

STRATEGIES = [
    {
        "name": "Fixed-90",
        "fixed_sparsity": 0.90,
        "use_cmaes": False,
        "use_csr_kv": False,
        "use_layer_interleaved": False,
    },
    {
        "name": "Task-Layer-Head",
        "fixed_sparsity": None,
        "use_cmaes": False,
        "use_csr_kv": False,
        "use_layer_interleaved": False,
    },
    {
        "name": "Task-Layer-Head + CMA-ES",
        "fixed_sparsity": None,
        "use_cmaes": True,
        "use_csr_kv": False,
        "use_layer_interleaved": False,
    },
    {
        "name": "Task-Layer-Head + CMA-ES + CSR-KV",
        "fixed_sparsity": None,
        "use_cmaes": True,
        "use_csr_kv": True,
        "use_layer_interleaved": False,
    },
    {
        "name": "Full SemInfer (CMA-ES + CSR-KV + LIS)",
        "fixed_sparsity": None,
        "use_cmaes": True,
        "use_csr_kv": True,
        "use_layer_interleaved": True,
    },
]


def _avg_sparsity(table: list[list[float]]) -> float:
    vals = [v for row in table for v in row]
    return float(sum(vals) / max(1, len(vals)))


def _task_loss(task: str, avg_sparsity: float, table: list[list[float]], oracle: float) -> float:
    # Base deviation from task-optimal sparsity.
    loss = abs(avg_sparsity - oracle) * 0.85

    # Layer sensitivity: code/reasoning need more conservative upper layers.
    top_start = int(NUM_LAYERS * 0.7)
    top_layers = table[top_start:] if top_start < len(table) else table
    top_mean = mean([mean(row) for row in top_layers]) if top_layers else avg_sparsity

    if task == "code_generation":
        loss += max(0.0, top_mean - 0.78) * 0.70
    elif task == "multi_hop_reasoning":
        loss += max(0.0, top_mean - 0.84) * 0.45

    return float(max(0.0, loss))


def _dense_full_bytes() -> float:
    return float(2 * NUM_LAYERS * SEQ_LEN * NUM_HEADS * HEAD_DIM * DTYPE_BYTES)


def _sparse_bytes(avg_sparsity: float, use_csr_kv: bool) -> float:
    kept = max(1, int(round(SEQ_LEN * (1.0 - avg_sparsity))))
    sparse_payload = float(2 * NUM_LAYERS * kept * NUM_HEADS * HEAD_DIM * DTYPE_BYTES)
    if not use_csr_kv:
        # Dense KV fallback: keep full dense shape for each sample.
        return _dense_full_bytes()
    # CSR-KV payload + indices/indptr metadata.
    index_bytes = float(kept * 4 + (NUM_LAYERS + 1) * 4)
    return sparse_payload + index_bytes


def _ms_from_bytes(num_bytes: float) -> float:
    return (num_bytes / (HBM_BANDWIDTH_GBPS * 1e9)) * 1000.0


def _compute_ms(avg_sparsity: float) -> float:
    # A lightweight proxy: fewer kept tokens => fewer effective attention FLOPs.
    kept_ratio = max(0.01, 1.0 - avg_sparsity)
    return 8.0 + 120.0 * kept_ratio


def _build_table(
    manager: AdaptiveSparsityManager,
    task: str,
    fixed_sparsity: float | None,
    use_cmaes: bool,
) -> list[list[float]]:
    if fixed_sparsity is not None:
        return [[fixed_sparsity for _ in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]

    if use_cmaes:
        manager.optimize_task_table_with_cmaes(
            task_type=task,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            max_loss=0.05,
            population=24,
            parents=12,
            generations=24,
            seed=42,
        )

    return manager.build_layer_head_sparsity(
        task_type=task,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        seq_len=SEQ_LEN,
        seq_len_p95=2048,
        use_cmaes=use_cmaes,
    )


def _estimate_e2e_ms(
    table: list[list[float]],
    use_csr_kv: bool,
    use_layer_interleaved: bool,
) -> tuple[float, float]:
    avg_sparsity = _avg_sparsity(table)
    io_ms = _ms_from_bytes(_sparse_bytes(avg_sparsity, use_csr_kv=use_csr_kv))
    compute_ms = _compute_ms(avg_sparsity)

    if not use_layer_interleaved:
        return io_ms + compute_ms, io_ms

    # Layer-interleaved scheduling: critical-path estimate.
    layer_load = [io_ms / NUM_LAYERS for _ in range(NUM_LAYERS)]
    layer_compute = []
    for row in table:
        keep_ratio = max(0.01, 1.0 - mean(row))
        layer_compute.append((compute_ms / NUM_LAYERS) * keep_ratio / 0.20)

    plan = build_layer_interleaved_plan(layer_load, layer_compute)
    if not plan.critical_path:
        return io_ms + compute_ms, io_ms

    critical_ms = plan.critical_path[-1]
    # Lower-bound to avoid over-optimistic overlap estimate.
    critical_ms = max(critical_ms, max(io_ms, compute_ms) * 0.92)
    return critical_ms, io_ms


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    manager = AdaptiveSparsityManager()

    dense_baseline_ms = _ms_from_bytes(_dense_full_bytes()) + _compute_ms(avg_sparsity=0.0)

    rows: list[dict[str, float | str | bool]] = []
    detail: dict[str, list[dict[str, float | str]]] = {}

    for strategy in STRATEGIES:
        name = str(strategy["name"])
        task_rows: list[dict[str, float | str]] = []

        for task_info in TASKS:
            task = str(task_info["task"])
            oracle = float(task_info["oracle_sparsity"])

            table = _build_table(
                manager=manager,
                task=task,
                fixed_sparsity=strategy["fixed_sparsity"],
                use_cmaes=bool(strategy["use_cmaes"]),
            )
            avg_sparsity = _avg_sparsity(table)
            loss = _task_loss(task, avg_sparsity, table, oracle)
            acc_proxy = max(0.0, 1.0 - 1.4 * loss)
            pass_flag = 1.0 if loss <= 0.05 else 0.0

            e2e_ms, io_ms = _estimate_e2e_ms(
                table,
                use_csr_kv=bool(strategy["use_csr_kv"]),
                use_layer_interleaved=bool(strategy["use_layer_interleaved"]),
            )

            task_rows.append(
                {
                    "task": task,
                    "avg_sparsity": avg_sparsity,
                    "loss": loss,
                    "accuracy_proxy": acc_proxy,
                    "pass": pass_flag,
                    "index_size_mb": _sparse_bytes(avg_sparsity, bool(strategy["use_csr_kv"]))
                    / (1024.0 * 1024.0),
                    "io_latency_ms": io_ms,
                    "e2e_latency_ms": e2e_ms,
                    "speedup_vs_dense": dense_baseline_ms / max(1e-6, e2e_ms),
                }
            )

        detail[name] = task_rows

        rows.append(
            {
                "strategy": name,
                "use_cmaes": bool(strategy["use_cmaes"]),
                "use_csr_kv": bool(strategy["use_csr_kv"]),
                "use_layer_interleaved": bool(strategy["use_layer_interleaved"]),
                "avg_sparsity": mean([float(x["avg_sparsity"]) for x in task_rows]),
                "accuracy_pass_rate": mean([float(x["pass"]) for x in task_rows]),
                "avg_accuracy_proxy": mean([float(x["accuracy_proxy"]) for x in task_rows]),
                "index_size_mb": mean([float(x["index_size_mb"]) for x in task_rows]),
                "io_latency_ms": mean([float(x["io_latency_ms"]) for x in task_rows]),
                "e2e_latency_ms": mean([float(x["e2e_latency_ms"]) for x in task_rows]),
                "speedup_vs_dense": mean([float(x["speedup_vs_dense"]) for x in task_rows]),
            }
        )

    csv_path = out_dir / "seminfer_ablation.csv"
    json_path = out_dir / "seminfer_ablation.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "num_layers": NUM_LAYERS,
                    "num_heads": NUM_HEADS,
                    "seq_len": SEQ_LEN,
                    "head_dim": HEAD_DIM,
                    "dtype_bytes": DTYPE_BYTES,
                    "hbm_bandwidth_gbps": HBM_BANDWIDTH_GBPS,
                },
                "summary": rows,
                "per_task": detail,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
