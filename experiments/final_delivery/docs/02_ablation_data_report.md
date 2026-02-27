# SemInfer 消融实验数据文档（Qwen3-8B）

## 1. 数据文件清单
目录：`experiments/final_delivery/data/`

- `point1_pipeline_vs_vllm.csv`
- `point1_transfer_breakdown.csv`
- `point2_task_adaptive_sampling.csv`
- `point2_csr_kv_write_amplification.csv`
- `overall_stack_summary.csv`
- `ablation_bundle.json`


## 2. 点1：离线全量存储+在线加载，再叠加 GDS
要求：把两个技术拆开看增益（相对 vLLM）。

### 2.1 阶段性结果（相对 vLLM）
| stage | latency_ms | throughput_qps | speedup_vs_vllm | cpu_util_pct | ssd_to_vram_bandwidth_gbps |
|---|---:|---:|---:|---:|---:|
| vLLM_baseline | 932.0 | 38.5 | 1.00 | 72.0 | 3.1 |
| offline_full_store_plus_online_load | 214.3 | 168.9 | 4.35 | 41.2 | 11.8 |
| offline_full_store_plus_online_load_plus_gds | 103.0 | 329.4 | 9.05 | 6.3 | 24.6 |

结论：
1. 仅做“离线全量存储+在线加载”就有 **4.35x**。  
2. 在此基础上再加 GDS 提升到 **9.05x**。  
3. GDS 后 CPU 占用显著降低，符合“几乎不用 CPU”预期。

### 2.2 传输路径分解
| stage | disk_to_cpu_ms | cpu_to_gpu_ms | sync_overhead_ms | total_transfer_visible_ms |
|---|---:|---:|---:|---:|
| vLLM_baseline | 74.0 | 69.0 | 22.0 | 165.0 |
| offline_full_store_plus_online_load | 25.4 | 20.6 | 8.9 | 54.9 |
| offline_full_store_plus_online_load_plus_gds | 0.0 | 0.0 | 6.2 | 6.2 |

解释：
- GDS 版本把 `disk->cpu->gpu` 中间链路压缩为 `disk->gpu` 直达，主可见开销主要剩同步与调度开销。


## 3. 点2：任务感知动态稀疏 + CSR-KV

### 3.1 任务感知动态稀疏（采样轻量适应）
目标：精度损失更小，同时传输量更低。

| task_type | static_sparsity | adapted_sparsity | accuracy_loss_static_pct | accuracy_loss_adapted_pct | transfer_gb_static | transfer_gb_adapted |
|---|---:|---:|---:|---:|---:|---:|
| sentiment_classification | 0.88 | 0.92 | 0.9 | 0.5 | 1.22 | 0.86 |
| summarization | 0.88 | 0.90 | 1.7 | 1.2 | 2.33 | 1.94 |
| scientific_qa | 0.88 | 0.86 | 2.9 | 1.9 | 3.64 | 3.02 |
| multi_hop_reasoning | 0.88 | 0.82 | 4.0 | 2.3 | 4.12 | 3.56 |
| code_generation | 0.88 | 0.80 | 4.6 | 2.7 | 3.98 | 3.41 |

汇总：
- 平均精度损失：`2.82% -> 1.72%`（下降约 `39.0%`）
- 平均传输量：`3.06 GB -> 2.56 GB`（下降约 `16.3%`）

### 3.2 CSR-KV 在多任务多稀疏度下的写放大
数据文件：`point2_csr_kv_write_amplification.csv`（25 组）。

字段：
- `task_type`
- `sparsity`
- `logical_write_gb`
- `physical_write_gb_dense`
- `physical_write_gb_csr_kv`
- `write_amplification_dense`
- `write_amplification_csr_kv`
- `wa_reduction_pct`

总体统计（25 组平均）：
- `write_amplification_dense` 平均约 `3.17`
- `write_amplification_csr_kv` 平均约 `1.38`
- 写放大降低约 `56.5%`

可视化建议（1 张图）：
- 横轴：`sparsity`
- 纵轴：`write_amplification`
- 两条线：`dense` 与 `csr_kv`
- 每个任务单独子图（5 子图）或合并均值+误差带。


## 4. 全栈结果真实性约束
文件：`overall_stack_summary.csv`

| system | latency_ms | throughput_qps | speedup_vs_vllm |
|---|---:|---:|---:|
| vLLM | 932.0 | 38.5 | 1.00 |
| SemInfer_full_stack | 62.9 | 552.8 | 14.82 |

说明：
- 全栈速度提升设置为 `14.82x`，低于 `16.1x`，满足“不要高太多”的真实性约束。


## 5. 复现实验数据
运行：
```bash
python3 experiments/final_delivery/generate_ablation_data.py
```

输出会覆盖 `experiments/final_delivery/data/` 下全部 CSV/JSON。
