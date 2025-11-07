import os
from time import time

import pandas as pd
from termcolor import colored
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


class ImdbDataset:
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        self.data = pd.read_csv("./data/imdb.csv")

    def sample(
        self,
        tokenizer,
        base_prompt,
        num_input_lines=1000,
    ) -> tuple[list[tuple[int, str]], int]:
        chose_lines = self.data["review"][:num_input_lines]
        lines_per_prompt = 1
        assert lines_per_prompt == 1
        duplicate = 1
        samples = []
        num_requests = int(num_input_lines / lines_per_prompt)
        base_token_len = len(tokenizer(base_prompt).input_ids)

        for i in range(num_requests):
            texts = "\n".join(
                chose_lines[i * lines_per_prompt : (i + 1) * lines_per_prompt]
            )
            for _ in range(duplicate):
                # base_prompt = f'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
                # base_prompt = f'Given the above film review, answer whether the film is suitable for kids. Respond ONLY with "yes" or "no", in all lower case.\n'
                # base_prompt = f"Given the above film review, answer whether it contains names. Respond ONLY with \"yes\" or \"no\", in all lower case.\n"
                # base_prompt = f"Given the above film review, answer whether it contains violent elements. Respond ONLY with \"yes\" or \"no\", in all lower case.\n"
                prompt = f"{texts}\n{base_prompt}"
                samples.append((i, prompt))
        return samples, base_token_len


class MultiTaskRunner:
    def __init__(self, llm, dataset, tokenizer):
        self.llm = llm
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tasks = []

    def add_task(self, name: str, base_prompt: str, num_input_lines: int, max_output_len: int, temperature: float = 1.0):
        self.tasks.append({
            "name": name,
            "base_prompt": base_prompt,
            "num_input_lines": num_input_lines,
            "max_output_len": max_output_len,
            "temperature": temperature,
        })

    def build_index(self, num_input_lines: int):
        # 用空白提示构建索引，仅缓存文本前缀
        base_prompt = " "
        samples, base_token_len = self.dataset.sample(self.tokenizer, base_prompt, num_input_lines)
        sp = SamplingParams(temperature=1.0, max_tokens=1)
        sp.base_token_len = base_token_len
        self.llm.generate(samples, sp, use_index=True, use_tqdm=False)
        # 清空前缀缓存的 block_table，以便任务阶段独立调度
        self.llm.scheduler.block_manager.reset()

    def run(self):
        all_samples = []
        all_sps = []
        partitions = []

        for t in self.tasks:
            samples, base_token_len = self.dataset.sample(self.tokenizer, t["base_prompt"], t["num_input_lines"])
            partitions.append(len(samples))
            for _ in range(len(samples)):
                sp = SamplingParams(temperature=t["temperature"], max_tokens=t["max_output_len"])
                sp.base_token_len = base_token_len
                sp.task_name = t["name"]
                all_sps.append(sp)
            all_samples.extend(samples)

        outputs = self.llm.generate(all_samples, all_sps, use_index=True, use_tqdm=False)

        results = {}
        offset = 0
        for t, count in zip(self.tasks, partitions):
            results[t["name"]] = outputs[offset:offset + count]
            offset += count
        return results

def run_serial(llm, dataset, tokenizer, tasks):
    import time as _time
    all_outputs = {}
    start = _time.time()
    for t in tasks:
        samples, base_token_len = dataset.sample(tokenizer, t["base_prompt"], t["num_input_lines"])
        sps = []
        for _ in range(len(samples)):
            sp = SamplingParams(temperature=t["temperature"], max_tokens=t["max_output_len"])
            sp.base_token_len = base_token_len
            sp.task_name = t["name"]
            sps.append(sp)
        outputs = llm.generate(samples, sps, use_index=True, use_tqdm=False)  # 串行：逐任务调用
        all_outputs[t["name"]] = outputs
    end = _time.time()
    return all_outputs, end - start

def monitor_resources(stop_event, interval=0.5, log_path="graph/resource_logs.json"):
    import json, os
    os.makedirs("graph", exist_ok=True)
    logs = []
    try:
        import psutil
    except:
        psutil = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except:
        pynvml = None
    import torch
    from time import time as _time
    while not stop_event.is_set():
        ts = _time()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        gpu_util = None
        cpu_util = None
        if pynvml:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except:
                gpu_util = None
        if psutil:
            try:
                cpu_util = psutil.cpu_percent(interval=None)
            except:
                cpu_util = None
        logs.append({"ts": ts, "gpu_mem_used": int(used), "gpu_mem_total": int(total), "gpu_util": gpu_util, "cpu_util": cpu_util})
        import time as _t
        _t.sleep(interval)
    try:
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"write resource logs failed: {e}")

def plot_timeline(seq_logs_path="graph/seq_logs.json", out_path="graph/task_timeline.png"):
    import json, os
    import matplotlib.pyplot as plt
    if not os.path.isfile(seq_logs_path):
        return
    with open(seq_logs_path) as f:
        data = json.load(f)
    # data: dict of seq_id -> {task_name, t_start, t_end}
    items = [v for v in data.values() if v.get("t_end", 0) and v.get("t_start", 0)]
    if not items:
        return
    # 按任务分组
    tasks = {}
    for v in items:
        tasks.setdefault(v["task_name"], []).append((v["t_start"], v["t_end"]))
    # 归一化到第一个开始为0
    t0 = min(v["t_start"] for v in items)
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks = []
    ylabels = []
    for i, (name, spans) in enumerate(tasks.items()):
        for (s, e) in spans:
            ax.broken_barh([(s - t0, e - s)], (i - 0.4, 0.8))
        yticks.append(i)
        ylabels.append(name)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Task Timeline (per seq spans)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def analyze_and_report(parallel_time, serial_time, batch_logs_path="graph/batch_logs.json", resource_logs_path="graph/resource_logs.json"):
    import json, os
    out_md = "graph/report.md"
    try:
        with open(batch_logs_path) as f:
            batches = json.load(f)
    except:
        batches = []
    # 吞吐量估计：总完成序列数 / 总时间
    num_batches = len(batches)
    total_seqs = sum(b["bs"] for b in batches)
    total_compute_ms = sum(b.get("compute_ms", 0.0) for b in batches)
    total_xfer_ms = sum(b.get("xfer_ms", 0.0) for b in batches)
    # 重叠程度估计：prefetch_done_at <= batch_start 表示预取完成在计算开始前，可部分算作重叠
    overlap_cnt = sum(1 for b in batches if b.get("prefetch_done_at") and b["prefetch_done_at"] <= b["start_time"])
    overlap_ratio = overlap_cnt / num_batches if num_batches else 0.0
    speedup = (serial_time / parallel_time) if parallel_time > 0 else 0.0
    # 写报告
    lines = []
    lines.append("# 并行 vs 串行 对比报告\n")
    lines.append(f"- 串行总时间: {serial_time:.3f}s")
    lines.append(f"- 并行总时间: {parallel_time:.3f}s")
    lines.append(f"- 并行加速比: {speedup:.2f}x")
    lines.append(f"- 完成序列数: {total_seqs}")
    lines.append(f"- 批次数: {num_batches}")
    lines.append(f"- 平均每批Compute时间: { (total_compute_ms/num_batches) if num_batches else 0.0:.2f} ms")
    lines.append(f"- 平均每批H2D时间: { (total_xfer_ms/num_batches) if num_batches else 0.0:.2f} ms")
    lines.append(f"- H2D-Compute重叠比例(批级近似): {overlap_ratio:.2f}")
    try:
        with open(out_md, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"write report failed: {e}")

def main():
    # Init
    path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
    max_output_len = 1
    num_input_lines = 1000
    sampling_params = SamplingParams(temperature=1, max_tokens=max_output_len)
    dataset = ImdbDataset()

    ###################################################################
    num_warmup = 10
    if not llm.kv_cache_index.indexed:
        num_warmup = num_input_lines

    # 并行多任务：先构建索引，再并发执行多任务
    runner = MultiTaskRunner(llm, dataset, tokenizer)

    print(colored("\nBuild index / Warm up", "yellow"))
    num_warmup = num_input_lines if not llm.kv_cache_index.indexed else 10
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time()
        runner.build_index(num_warmup)
        end = time()
    prof.export_chrome_trace("trace_task1.json")
    print(colored(f"Build index time: {(end - start):.4f} s", "blue"))

    # 定义任务（共享文本前缀，指令不同）
    runner.add_task(
        name="sentiment",
        base_prompt='Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n',
        num_input_lines=num_input_lines,
        max_output_len=1,
        temperature=1.0,
    )
    # 可继续添加其他任务
    # runner.add_task(...)

    # 开启资源监控
    import threading
    stop_evt = threading.Event()
    mon = threading.Thread(target=monitor_resources, args=(stop_evt, 0.5, "graph/resource_logs.json"), daemon=True)
    mon.start()

    # 并行跑
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=False) as prof:
        print(colored("\nRun Multi-Tasks (Parallel)", "yellow"))
        start = time()
        task_outputs = runner.run()
        parallel_end = time()
    prof.export_chrome_trace("trace_task2.json")
    parallel_time = parallel_end - start
    print(colored(f"Parallel total time: {(parallel_time):.4f} s", "blue"))

    # 串行跑（逐任务调用）
    serial_outputs, serial_time = run_serial(llm, dataset, tokenizer, runner.tasks)
    print(colored(f"Serial total time: {(serial_time):.4f} s", "blue"))

    # 关闭资源监控并生成图表与报告
    stop_evt.set()
    mon.join()
    plot_timeline("graph/seq_logs.json", "graph/task_timeline.png")
    analyze_and_report(parallel_time, serial_time, "graph/batch_logs.json", "graph/resource_logs.json")

    # 示例：对 sentiment 任务评估准确率
    if "sentiment" in task_outputs:
        generated = [o["text"] for o in task_outputs["sentiment"]]
        data = pd.read_csv("./data/imdb.csv").head(len(generated))
        correct_predictions = (data["sentiment"] == generated).sum()
        accuracy = correct_predictions / len(generated)
        print(f"Task sentiment Accuracy:{accuracy}\n")

    # print(data[data["suitable"] != generated]["review"])


if __name__ == "__main__":
    main()
