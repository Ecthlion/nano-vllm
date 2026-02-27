# SemInfer 技术实现文档（Chapter3 + Chapter4）

## 1. 设计目标与工程原则
本次实现以 `Qwen3-8B` 可加载为前提，围绕三条工程原则设计：

1. 可演进：把索引存储、缓存策略、稀疏策略解耦为独立模块，便于后续替换。
2. 可观测：每个关键路径保留统计字段（命中、写放大、吞吐、延迟）。
3. 可回退：保留旧路径开关，任何新机制都可通过环境变量关闭。

关键开关：
- `NANOVLLM_USE_CSR_KV`：启用 CSR-KV 持久化。
- `NANOVLLM_USE_GPUDIRECT`：启用 GDS。
- `NANOVLLM_GPU_HOT_CACHE_ENABLE`：启用 L1 GPU 热缓存。
- `NANOVLLM_ENABLE_TASK_CALIBRATION`：启用首次任务采样校准。


## 2. Chapter3：离线预计算 + 在线检索

### 2.1 范式与瓶颈转换
实现链路：
- 离线预计算：`store_kv_cache()` 将 prefill KV 序列化到索引。
- 在线检索：`get_kv_cache()` 根据 `text_id` 恢复 KV，并按需加载到显存。

瓶颈变化：
- 从“每次全量注意力计算”转换为“索引检索 + 传输”。
- 随后通过异步重叠和 GDS，把 I/O 代价进一步隐藏。


### 2.2 索引持久化：CSR-KV 对齐格式
实现文件：
- `nanovllm/utils/csr_kv_store.py`

数据布局（单条记录）：
- `[header][indices][indptr][pad_to_data_align][key][pad_to_data_align][value][pad_to_record_align]`

核心字段：
- `offsets`: `record/header/indices/indptr/key/value`
- `checksums`: `indices_crc32/indptr_crc32/key_crc32/value_crc32/record_crc32`
- `shape/dtype/task_type/sparsity_ratio`

关键点：
1. key/value 4KB 对齐，适配 GDS 偏移读取。
2. indices/indptr 不单独强对齐，减少小块 padding。
3. group commit（`commit_interval`）降低 fsync 频率，减少写放大。

关键代码（摘录）：
```python
key_offset_rel = _align_up(
    indptr_offset_rel + len(indptr_blob),
    self.data_align_bytes,
)
value_offset_rel = _align_up(
    key_offset_rel + len(key_blob),
    self.data_align_bytes,
)
...
should_commit = force_sync or self._pending_count >= self.commit_interval
if should_commit:
    self._flush_manifest(do_fsync=True)
    self._sync_data_file()
    self._pending_count = 0
```


### 2.3 两级缓存：L1 GPU 热数据 + L2 NVMe/SSD 冷数据
实现文件：
- `nanovllm/utils/two_level_cache.py`
- `nanovllm/utils/kv_cache_index.py`

策略：
- L1: `GPUHotCache`（LRU，按显存预算淘汰）
- L2: CSR-KV（NVMe/SSD）

读取优先级：
1. L1 热缓存（GPU tensor）
2. 已缓存 `gds_cache`
3. GDS 读 raw kv 文件
4. GDS 按 offset 读 CSR-KV
5. CPU 回退读 CSR-KV 并异步 H2D

关键代码（摘录）：
```python
if cpu_kv_cache is None and self.gpu_hot_cache is not None:
    hot = self.gpu_hot_cache.get(cache_key)
    if hot is not None:
        cpu_kv_cache = hot
...
if cpu_kv_cache is None and self.use_gds and self.csr_store is not None:
    gpu_cache = self._load_csr_kv_with_gds(...)
```


### 2.4 GDS 高速传输
实现文件：
- `nanovllm/utils/kv_cache_index.py`

两类 GDS 路径：
1. raw kv 文件：`_load_kv_with_gds`
2. CSR-KV 偏移读取：`_load_csr_kv_with_gds`

重点：
- 对 `kvikio` 不同 API 形态做兼容封装（`pread/readinto/seek+readinto`）。
- 按 `offsets.key/value` 直接读到 GPU 张量，绕过 CPU 中转。


### 2.5 异步预取与计算重叠
实现文件：
- `nanovllm/engine/llm_engine.py`

实现方式：
- 预取线程只发起传输并返回 `transfer_event`。
- 主计算流在 `step()` 中执行 `wait_event()`，不在预取线程里硬同步。

关键代码（摘录）：
```python
# prefetch thread
self._prefetch_queue.put((seqs, is_prefill, transfer_event, start_event))

# compute thread
seqs, is_prefill, transfer_event, start_event = item
if transfer_event is not None:
    self._compute_stream.wait_event(transfer_event)
```


### 2.6 三类实验验证实现
实现文件：
- `experiments/chapter3/run_bench.py`

输出：
- `paradigm_conversion.csv`
- `async_overlap.csv`
- `two_level_cache.csv`

对应章节论点：
1. 范式转换收益（离线预计算 + 在线检索）
2. 异步重叠收益（sync vs async）
3. 两级缓存收益（无热缓存 vs L1+L2）


## 3. Chapter4：任务感知动态稀疏

### 3.1 实现链路
实现文件：
- `nanovllm/utils/adaptive_sparsity.py`
- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/attention.py`

执行路径：
1. `LLMEngine.add_request()` 推断任务类型。
2. `AdaptiveSparsityManager` 生成 task-layer-head 稀疏表。
3. 稀疏表写入 `Sequence.adaptive_sparsity`。
4. `ModelRunner.prepare_prefill()` 将表搬运到 GPU context。
5. `Attention` 在每层按当前 layer/head 稀疏权重决定裁剪比例。


### 3.2 采样轻量适应（首次任务校准）
实现文件：
- `nanovllm/utils/adaptive_sparsity.py`
- `nanovllm/engine/llm_engine.py`

机制：
- 首次任务出现时，抽样少量 prompt。
- 在候选稀疏率上快速评估（代理 evaluator）。
- 在精度损失阈值内选速度最优稀疏率。
- 结果持久化到 `experiments/chapter4/task_profile_store.json`。

关键代码（摘录）：
```python
self.adaptive_sparsity.calibrate_task_profile(
    task_type=task_type,
    sample_prompts=sample_prompts,
    evaluator=_evaluator,
    max_acc_drop=0.02,
)
```


### 3.3 按需加载（precision tier + ILH）
实现文件：
- `nanovllm/utils/kv_cache_index.py`

按需策略：
- `fast -> core`
- `balanced -> important`
- `high -> optional`

关键代码（摘录）：
```python
if precision_tier == "fast":
    level_name = "core"
elif precision_tier == "high":
    level_name = "optional"
else:
    level_name = "important"
```


## 4. Qwen3-8B 兼容性说明
- 模型加载入口未变：`BackendAPI` 仍通过
  `LLM(path="/data/zwt/model/models/Qwen/Qwen3-8B/")` 初始化。
- 本次改动主要位于索引/缓存/调度/稀疏策略层，不修改权重格式和 tokenizer 接口。
- 因此与 Qwen3-8B 的模型加载路径兼容。


## 5. 本次新增/修改文件（核心）
- 新增：
  - `nanovllm/utils/two_level_cache.py`
  - `nanovllm/utils/adaptive_sparsity.py`
  - `experiments/chapter3/run_bench.py`
  - `experiments/chapter4/run_task_adaptive_ablation.py`
  - `experiments/final_delivery/generate_ablation_data.py`
- 主要修改：
  - `nanovllm/utils/csr_kv_store.py`
  - `nanovllm/utils/kv_cache_index.py`
  - `nanovllm/engine/llm_engine.py`
  - `nanovllm/engine/model_runner.py`
  - `nanovllm/layers/attention.py`
  - `nanovllm/utils/context.py`
  - `nanovllm/engine/sequence.py`
  - `nanovllm/sampling_params.py`
  - `nanovllm/utils/backend.py`
