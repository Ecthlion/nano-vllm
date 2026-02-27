# New Ablation Runner

## Run on server (H800)

Recommended entry script:

```bash
bash scripts/server/zyy_h800_run.sh
```

This will:
1. setup a user-space `uv` environment (no root required)
2. download `Qwen/Qwen3-8B` via ModelScope to `/data/zhangyuyun/models/models`
3. run all ablations and store results in `experiments/new/data`
4. plot both `experiments/final_delivery/data` and `experiments/new/data`

## Direct runner

```bash
python experiments/new/run_all_ablations.py \
  --data-path data/imdb.csv \
  --model-path /data/zhangyuyun/models/models/Qwen/Qwen3-8B \
  --kv-dir /data/zhangyuyun/kvcache_index \
  --out-dir experiments/new \
  --limit 64 --task-limit 48 --repeats 2
```

## Output files

- `experiments/new/data/point1_pipeline_vs_vllm.csv`
- `experiments/new/data/point1_transfer_breakdown.csv`
- `experiments/new/data/point2_task_adaptive_sampling.csv`
- `experiments/new/data/point2_csr_kv_write_amplification.csv`
- `experiments/new/data/overall_stack_summary.csv`
- `experiments/new/data/ablation_bundle.json`
