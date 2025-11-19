import os

import pandas as pd

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


class BackendAPI:
    def __init__(self) -> None:
        path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
        self.llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
        self.sp = SamplingParams(temperature=1, max_tokens=1)

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data["__id"] = range(0, len(self.data))

    def build_index(self, sparsity: float, field: str, limit: int | None = 1000):
        """
        Build index for csv data
        Args:
            sparsity: sparsity to prune tokens
            field: text field
        """
        # clear kv cache index
        self.llm.kv_cache_index.kv_cache_index = {}

        if limit is not None:
            data = self.data[["__id", field]].head(limit)
        else:
            data = self.data[["__id", field]]

        samples = []
        for _, row in data.iterrows():
            id = row["__id"]
            text = row[field]
            prompt = f"{text}\n "
            samples.append((id, prompt))

        self.sp.task_str_len = 1

        _ = self.llm.generate(
            samples,
            self.sp,
            use_index=True,
            use_tqdm=False,
            pruning=True,
            sparsity=sparsity,
        )

    def query(self, query, use_index):
        # 1. parse query and make tasks
        # query example: LLM("User defined prompt1") == "positive"
        #                and LLM("User defined prompt2") == "yes"

        # 2. run tasks and filter data

        # 3. return data results and profiling information

        pass

    def analyse(self, data):
        # draw some analyse picture:
        # 1. 带索引/不带索引的运行时间柱状图比较
        # 2. 不同sparsity下，与 (不带索引的 结果不同的 比率) 的折线图&索引大小折线图
        # 3. kv cache transfer 和 compute 的运行时间图
        pass
