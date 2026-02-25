from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_qwen3_8b_simulated_results(out_dir: Path) -> dict[str, list[dict]]:
    """
    Simulate thesis-aligned SemInfer results for Qwen3-8B.

    Scope:
    - algorithm layer (ablation, sparsity sensitivity, CMA-ES optimizer comparison)
    - system layer (I/O path, scheduler, latency breakdown)
    - application/system benchmark layer (SOTA, batch scaling, stress tests)
    """

    algorithm_ablation = [
        {
            "strategy": "Fixed-90",
            "avg_sparsity": 0.90,
            "accuracy_pass_rate": 0.78,
            "avg_accuracy": 0.903,
            "index_size_gb": 218.0,
            "e2e_latency_ms": 68.4,
            "speedup_vs_dense": 8.2,
        },
        {
            "strategy": "Task-level",
            "avg_sparsity": 0.84,
            "accuracy_pass_rate": 0.90,
            "avg_accuracy": 0.931,
            "index_size_gb": 272.4,
            "e2e_latency_ms": 73.1,
            "speedup_vs_dense": 7.7,
        },
        {
            "strategy": "Task-Layer",
            "avg_sparsity": 0.85,
            "accuracy_pass_rate": 0.94,
            "avg_accuracy": 0.944,
            "index_size_gb": 256.1,
            "e2e_latency_ms": 61.8,
            "speedup_vs_dense": 9.1,
        },
        {
            "strategy": "Task-Layer-Head",
            "avg_sparsity": 0.83,
            "accuracy_pass_rate": 0.97,
            "avg_accuracy": 0.952,
            "index_size_gb": 240.5,
            "e2e_latency_ms": 55.2,
            "speedup_vs_dense": 10.2,
        },
        {
            "strategy": "Task-Layer-Head + CMA-ES",
            "avg_sparsity": 0.84,
            "accuracy_pass_rate": 0.985,
            "avg_accuracy": 0.958,
            "index_size_gb": 236.8,
            "e2e_latency_ms": 51.7,
            "speedup_vs_dense": 10.9,
        },
        {
            "strategy": "Full SemInfer (CMA-ES + CSR-KV + GDS + LIS)",
            "avg_sparsity": 0.84,
            "accuracy_pass_rate": 0.985,
            "avg_accuracy": 0.958,
            "index_size_gb": 71.9,
            "e2e_latency_ms": 41.6,
            "speedup_vs_dense": 13.5,
        },
    ]

    task_optimal_sparsity = [
        {
            "task": "sentiment_classification",
            "optimal_sparsity": 0.91,
            "accuracy_drop_pct": 0.9,
            "best_strategy": "Task-Layer-Head + CMA-ES",
        },
        {
            "task": "long_summarization",
            "optimal_sparsity": 0.88,
            "accuracy_drop_pct": 2.0,
            "best_strategy": "Task-Layer-Head + CMA-ES",
        },
        {
            "task": "scientific_qa",
            "optimal_sparsity": 0.85,
            "accuracy_drop_pct": 2.8,
            "best_strategy": "Task-Layer-Head + CMA-ES",
        },
        {
            "task": "multi_hop_reasoning",
            "optimal_sparsity": 0.81,
            "accuracy_drop_pct": 3.9,
            "best_strategy": "Task-Layer-Head + CMA-ES",
        },
        {
            "task": "code_generation",
            "optimal_sparsity": 0.74,
            "accuracy_drop_pct": 4.4,
            "best_strategy": "Task-Layer-Head + CMA-ES",
        },
    ]

    sparsity_sensitivity = [
        {"sparsity": 0.70, "accuracy": 0.992, "latency_ms": 103.8, "speedup_vs_dense": 5.4},
        {"sparsity": 0.75, "accuracy": 0.988, "latency_ms": 90.5, "speedup_vs_dense": 6.2},
        {"sparsity": 0.80, "accuracy": 0.979, "latency_ms": 78.1, "speedup_vs_dense": 7.2},
        {"sparsity": 0.85, "accuracy": 0.963, "latency_ms": 64.0, "speedup_vs_dense": 8.8},
        {"sparsity": 0.90, "accuracy": 0.934, "latency_ms": 52.3, "speedup_vs_dense": 10.8},
        {"sparsity": 0.92, "accuracy": 0.914, "latency_ms": 48.4, "speedup_vs_dense": 11.7},
        {"sparsity": 0.95, "accuracy": 0.865, "latency_ms": 43.4, "speedup_vs_dense": 13.1},
    ]

    optimizer_comparison = [
        {
            "method": "GridSearch",
            "num_evaluations": 200000,
            "best_speedup": 10.6,
            "accuracy_pass_rate": 0.964,
            "wall_clock_hours": 72.0,
            "convergence_generation": 999,
        },
        {
            "method": "RandomSearch",
            "num_evaluations": 5000,
            "best_speedup": 10.1,
            "accuracy_pass_rate": 0.921,
            "wall_clock_hours": 6.4,
            "convergence_generation": 212,
        },
        {
            "method": "BayesianOpt",
            "num_evaluations": 5000,
            "best_speedup": 10.4,
            "accuracy_pass_rate": 0.949,
            "wall_clock_hours": 6.9,
            "convergence_generation": 141,
        },
        {
            "method": "CMA-ES",
            "num_evaluations": 3000,
            "best_speedup": 10.9,
            "accuracy_pass_rate": 0.985,
            "wall_clock_hours": 4.2,
            "convergence_generation": 58,
        },
    ]

    io_path_comparison = [
        {
            "engine": "POSIX+Memcpy",
            "bandwidth_gbps": 3.1,
            "p50_io_latency_ms": 28.2,
            "p99_io_latency_ms": 46.7,
            "cpu_util_pct": 56.0,
        },
        {
            "engine": "PinnedMemory+AsyncH2D",
            "bandwidth_gbps": 11.8,
            "p50_io_latency_ms": 9.4,
            "p99_io_latency_ms": 15.9,
            "cpu_util_pct": 24.0,
        },
        {
            "engine": "GDS",
            "bandwidth_gbps": 24.7,
            "p50_io_latency_ms": 4.1,
            "p99_io_latency_ms": 6.8,
            "cpu_util_pct": 4.8,
        },
    ]

    scheduler_comparison = [
        {
            "scheduler": "Sequential",
            "compute_io_overlap_pct": 0.0,
            "e2e_latency_ms": 71.4,
            "relative_speedup": 1.0,
        },
        {
            "scheduler": "DualStreamPipeline",
            "compute_io_overlap_pct": 71.2,
            "e2e_latency_ms": 49.7,
            "relative_speedup": 1.44,
        },
        {
            "scheduler": "LayerInterleavedScheduler",
            "compute_io_overlap_pct": 92.0,
            "e2e_latency_ms": 41.6,
            "relative_speedup": 1.72,
        },
    ]

    sota_comparison = [
        {
            "system": "vLLM-Dense",
            "e2e_latency_ms": 562.0,
            "throughput_req_s": 42.0,
            "accuracy": 1.000,
            "index_size_gb": 0.0,
        },
        {
            "system": "H2O",
            "e2e_latency_ms": 154.0,
            "throughput_req_s": 139.0,
            "accuracy": 0.910,
            "index_size_gb": 245.0,
        },
        {
            "system": "SnapKV",
            "e2e_latency_ms": 128.0,
            "throughput_req_s": 165.0,
            "accuracy": 0.892,
            "index_size_gb": 230.0,
        },
        {
            "system": "PyramidKV",
            "e2e_latency_ms": 101.0,
            "throughput_req_s": 198.0,
            "accuracy": 0.931,
            "index_size_gb": 176.0,
        },
        {
            "system": "SemInfer-Qwen3-8B",
            "e2e_latency_ms": 41.6,
            "throughput_req_s": 415.0,
            "accuracy": 0.958,
            "index_size_gb": 71.9,
        },
    ]

    latency_breakdown = [
        {"stage": "query_parse", "latency_ms": 2.6, "ratio_pct": 6.3},
        {"stage": "task_lookup_and_dispatch", "latency_ms": 1.8, "ratio_pct": 4.3},
        {"stage": "gds_io_transfer", "latency_ms": 14.9, "ratio_pct": 35.8},
        {"stage": "attention_compute", "latency_ms": 18.2, "ratio_pct": 43.8},
        {"stage": "result_assembly", "latency_ms": 4.1, "ratio_pct": 9.8},
    ]

    batch_scaling = [
        {"batch_size": 1, "throughput_req_s": 14.0, "p50_latency_ms": 41.6, "scaling_efficiency": 1.00},
        {"batch_size": 2, "throughput_req_s": 28.0, "p50_latency_ms": 42.1, "scaling_efficiency": 1.00},
        {"batch_size": 4, "throughput_req_s": 55.0, "p50_latency_ms": 43.2, "scaling_efficiency": 0.98},
        {"batch_size": 8, "throughput_req_s": 108.0, "p50_latency_ms": 44.5, "scaling_efficiency": 0.96},
        {"batch_size": 16, "throughput_req_s": 210.0, "p50_latency_ms": 47.0, "scaling_efficiency": 0.94},
        {"batch_size": 32, "throughput_req_s": 398.0, "p50_latency_ms": 50.9, "scaling_efficiency": 0.89},
        {"batch_size": 64, "throughput_req_s": 742.0, "p50_latency_ms": 57.7, "scaling_efficiency": 0.83},
    ]

    concurrency_stress = [
        {"concurrency": 1, "avg_latency_ms": 41.6, "p99_latency_ms": 55.2, "throughput_req_s": 14.0, "error_rate_pct": 0.00, "gpu_util_pct": 32.0},
        {"concurrency": 2, "avg_latency_ms": 42.8, "p99_latency_ms": 57.4, "throughput_req_s": 27.0, "error_rate_pct": 0.00, "gpu_util_pct": 43.0},
        {"concurrency": 4, "avg_latency_ms": 45.6, "p99_latency_ms": 62.9, "throughput_req_s": 52.0, "error_rate_pct": 0.00, "gpu_util_pct": 58.0},
        {"concurrency": 8, "avg_latency_ms": 49.7, "p99_latency_ms": 74.6, "throughput_req_s": 96.0, "error_rate_pct": 0.10, "gpu_util_pct": 72.0},
        {"concurrency": 16, "avg_latency_ms": 63.8, "p99_latency_ms": 96.8, "throughput_req_s": 178.0, "error_rate_pct": 0.20, "gpu_util_pct": 86.0},
        {"concurrency": 24, "avg_latency_ms": 78.9, "p99_latency_ms": 131.7, "throughput_req_s": 236.0, "error_rate_pct": 0.40, "gpu_util_pct": 93.0},
        {"concurrency": 32, "avg_latency_ms": 97.4, "p99_latency_ms": 180.8, "throughput_req_s": 271.0, "error_rate_pct": 0.70, "gpu_util_pct": 96.0},
    ]

    dataset_scale_stress = [
        {
            "records": 10000,
            "index_build_time_min": 6.2,
            "sparse_index_size_gb": 14.4,
            "p50_load_latency_ms": 18.1,
            "gpu_mem_peak_gb": 11.2,
        },
        {
            "records": 50000,
            "index_build_time_min": 30.8,
            "sparse_index_size_gb": 71.9,
            "p50_load_latency_ms": 21.9,
            "gpu_mem_peak_gb": 11.3,
        },
        {
            "records": 100000,
            "index_build_time_min": 61.7,
            "sparse_index_size_gb": 143.7,
            "p50_load_latency_ms": 24.8,
            "gpu_mem_peak_gb": 11.4,
        },
        {
            "records": 500000,
            "index_build_time_min": 311.2,
            "sparse_index_size_gb": 718.6,
            "p50_load_latency_ms": 31.6,
            "gpu_mem_peak_gb": 11.5,
        },
        {
            "records": 1000000,
            "index_build_time_min": 629.5,
            "sparse_index_size_gb": 1437.3,
            "p50_load_latency_ms": 37.8,
            "gpu_mem_peak_gb": 11.6,
        },
    ]

    semantic_dataframe_queries = [
        {
            "query_type": "nl_filter_only",
            "example": "find negative reviews mentioning pacing",
            "precision": 0.93,
            "recall": 0.91,
            "f1": 0.92,
            "latency_ms": 39.4,
        },
        {
            "query_type": "structured_filter_only",
            "example": "rating >= 8 AND year >= 2020",
            "precision": 0.99,
            "recall": 0.99,
            "f1": 0.99,
            "latency_ms": 8.3,
        },
        {
            "query_type": "hybrid_filter",
            "example": "genre='Sci-Fi' AND contains optimism about future",
            "precision": 0.94,
            "recall": 0.90,
            "f1": 0.92,
            "latency_ms": 42.7,
        },
    ]

    return {
        "algorithm_ablation": algorithm_ablation,
        "task_optimal_sparsity": task_optimal_sparsity,
        "sparsity_sensitivity": sparsity_sensitivity,
        "optimizer_comparison": optimizer_comparison,
        "io_path_comparison": io_path_comparison,
        "scheduler_comparison": scheduler_comparison,
        "sota_comparison": sota_comparison,
        "latency_breakdown": latency_breakdown,
        "batch_scaling": batch_scaling,
        "concurrency_stress": concurrency_stress,
        "dataset_scale_stress": dataset_scale_stress,
        "semantic_dataframe_queries": semantic_dataframe_queries,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "qwen3_8b_simulated"
    bundle = generate_qwen3_8b_simulated_results(out_dir)

    filename_map = {
        "algorithm_ablation": "qwen3_8b_algorithm_ablation.csv",
        "task_optimal_sparsity": "qwen3_8b_task_optimal_sparsity.csv",
        "sparsity_sensitivity": "qwen3_8b_sparsity_sensitivity.csv",
        "optimizer_comparison": "qwen3_8b_optimizer_comparison.csv",
        "io_path_comparison": "qwen3_8b_io_path_comparison.csv",
        "scheduler_comparison": "qwen3_8b_scheduler_comparison.csv",
        "sota_comparison": "qwen3_8b_sota_comparison.csv",
        "latency_breakdown": "qwen3_8b_latency_breakdown.csv",
        "batch_scaling": "qwen3_8b_batch_scaling.csv",
        "concurrency_stress": "qwen3_8b_concurrency_stress.csv",
        "dataset_scale_stress": "qwen3_8b_dataset_scale_stress.csv",
        "semantic_dataframe_queries": "qwen3_8b_semantic_dataframe_queries.csv",
    }

    generated = []
    for key, rows in bundle.items():
        path = out_dir / filename_map[key]
        _write_csv(path, rows)
        generated.append(str(path))

    meta_path = out_dir / "qwen3_8b_simulated_bundle.json"
    meta_path.write_text(
        json.dumps({"model": "Qwen3-8B", "generated_files": generated}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Generated simulated Qwen3-8B results:")
    for p in generated:
        print(p)
    print(meta_path)


if __name__ == "__main__":
    main()
