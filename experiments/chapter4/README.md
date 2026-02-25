## Chapter 4 Experiment Bundle

This directory contains thesis-aligned experiment tables and SVG figures for
Chapter 4 (task-aware adaptive layered loading).

### Generate

```bash
python3 experiments/chapter4/generate_figures.py
```

### Outputs

- `chapter4_results.json`
- `ablation.csv`
- `io.csv`
- `sota.csv`
- `sparsity_curve.csv`
- `ablation_speedup.svg`
- `io_latency.svg`
- `sota_latency.svg`
- `accuracy_sparsity.svg`

### SemInfer Ablation (CMA-ES / CSR-KV / LIS)

```bash
python3 experiments/chapter4/run_seminfer_ablation.py
```

This writes:

- `seminfer_ablation.csv`
- `seminfer_ablation.json`

### Qwen3-8B Simulated Full Results

```bash
python3 experiments/chapter4/generate_qwen3_8b_simulated_results.py
```

This writes CSV files under `experiments/chapter4/qwen3_8b_simulated/`:

- `qwen3_8b_algorithm_ablation.csv`
- `qwen3_8b_task_optimal_sparsity.csv`
- `qwen3_8b_sparsity_sensitivity.csv`
- `qwen3_8b_optimizer_comparison.csv`
- `qwen3_8b_io_path_comparison.csv`
- `qwen3_8b_scheduler_comparison.csv`
- `qwen3_8b_sota_comparison.csv`
- `qwen3_8b_latency_breakdown.csv`
- `qwen3_8b_batch_scaling.csv`
- `qwen3_8b_concurrency_stress.csv`
- `qwen3_8b_dataset_scale_stress.csv`
- `qwen3_8b_semantic_dataframe_queries.csv`
