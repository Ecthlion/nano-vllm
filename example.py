import os
from time import time

import pandas as pd
from termcolor import colored
from torch.profiler import ProfilerActivity, profile

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
        base_prompt,
        num_input_lines=1000,
    ) -> tuple[list[tuple[int, str]], int]:
        chose_lines = self.data["review"][:num_input_lines]
        lines_per_prompt = 1
        assert lines_per_prompt == 1
        duplicate = 1
        samples = []
        num_requests = int(num_input_lines / lines_per_prompt)
        task_str_len = len(base_prompt)

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
        return samples, task_str_len


def build_index(llm: LLM, dataset: ImdbDataset, num_input_lines: int, max_output_len: int):
    """
    Build KV cache index for later reuse.
    """
    sampling_params = SamplingParams(temperature=1, max_tokens=max_output_len)
    num_warmup = num_input_lines

    print(colored("\nBuild index / Warm up", "yellow"))
    base_prompt = " "
    samples, task_str_len = dataset.sample(base_prompt, num_warmup)
    sampling_params.task_str_len = task_str_len

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ):
        start = time()
        llm.generate(samples, sampling_params, use_index=True, use_tqdm=False, pruning=True)
        end = time()

    llm.kv_cache_index.persistence()
    print(colored(f"Build index time: {(end - start):.4f} s", "blue"))


def benchmark_cached_reads(
    model_path: str,
    dataset: ImdbDataset,
    num_input_lines: int,
    max_output_len: int,
    enable_gds: bool,
):
    os.environ["NANOVLLM_USE_GPUDIRECT"] = "1" if enable_gds else "0"
    mode = "GPUDirect" if enable_gds else "CPU pinned memory"
    print(colored(f"\n===== Cached read benchmark: {mode} =====", "cyan"))

    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=1, max_tokens=max_output_len)

    if not llm.kv_cache_index.indexed:
        build_index(llm, dataset, num_input_lines, max_output_len)

    base_prompt = (
        'Given the above film review, answer whether the film is suitable for kids. '
        'Respond ONLY with "yes" or "no", in all lower case.\n'
    )
    samples, task_str_len = dataset.sample(base_prompt, num_input_lines)
    sampling_params.task_str_len = task_str_len

    start = time()
    outputs = llm.generate(
        samples, sampling_params, use_index=True, use_tqdm=False, optimize=True
    )
    end = time()
    print(colored(f"Total cached generate time: {(end - start):.4f} s", "blue"))
    return outputs


def main():
    path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
    max_output_len = 1
    num_input_lines = 1000
    dataset = ImdbDataset()

    # Build index once with GPUDirect enabled so kv binaries are dumped.
    os.environ["NANOVLLM_USE_GPUDIRECT"] = "1"
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
    build_index(llm, dataset, num_input_lines, max_output_len)

    # Baseline CPU pinned memory read
    cpu_outputs = benchmark_cached_reads(
        path, dataset, num_input_lines, max_output_len, enable_gds=False
    )

    # GPUDirect read
    gds_outputs = benchmark_cached_reads(
        path, dataset, num_input_lines, max_output_len, enable_gds=True
    )

    # Simple consistency check between modes
    matches = sum(
        1 for cpu_out, gds_out in zip(cpu_outputs, gds_outputs) if cpu_out["text"] == gds_out["text"]
    )
    recall = matches / len(cpu_outputs) if cpu_outputs else 0
    print(colored(f"Recall (CPU vs GPUDirect cached outputs): {recall:.4f}", "green"))


if __name__ == "__main__":
    main()
