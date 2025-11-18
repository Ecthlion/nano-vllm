import os

from nanovllm import LLM, SamplingParams


class BackendAPI:
    def __init__(self) -> None:
        path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
        self.llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
        self.sp = SamplingParams(temperature=1, max_tokens=1)

    def build_index(self, path):
        pass
