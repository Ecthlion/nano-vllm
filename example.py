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

    print(colored("\nBuild index / Warm up", "yellow"))
    base_prompt = " "
    samples, base_token_len = dataset.sample(tokenizer, base_prompt, num_warmup)
    sampling_params.base_token_len = base_token_len

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time()
        outputs = llm.generate(samples, sampling_params, use_index=True, use_tqdm=False)
        end = time()
    # prof.export_chrome_trace("trace_task1.json")

    print(colored(f"Build index time: {(end - start):.4f} s", "blue"))

    # remove prefix cache
    llm.scheduler.block_manager.reset()

    ###################################################################
    # print(colored("Task1: suitable for kids", "yellow"))
    #
    # base_prompt = f'Given the above film review, answer whether the film is suitable for kids. Respond ONLY with "yes" or "no", in all lower case.\n'
    # samples, base_token_len = dataset.sample(tokenizer, base_prompt, 20)
    # sampling_params.base_token_len = base_token_len
    #
    # # The first task will generate kv cache index for texts
    # # with profile(
    # #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # #     profile_memory=True,
    # #     with_stack=True,
    # # ) as prof:
    # start = time()
    # outputs = llm.generate(samples, sampling_params, use_index=True, use_tqdm=False)
    # end = time()
    # # prof.export_chrome_trace("trace_task1.json")
    #
    # print(colored(f"Total generate time: {( end - start ):.4f} s", "blue"))
    #
    # generated = [output["text"] for output in outputs]
    # # print(f"{generated[:10]}")

    ###################################################################
    print(colored("\nTask2: sentiment", "yellow"))
    base_prompt = f'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
    samples, base_token_len = dataset.sample(tokenizer, base_prompt, num_input_lines)
    sampling_params.base_token_len = base_token_len

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time()
        outputs = llm.generate(samples, sampling_params, use_index=True, use_tqdm=False)
        end = time()
    # prof.export_chrome_trace("trace_task2.json")

    print(colored(f"Total generate time: {( end - start ):.4f} s", "blue"))
    print(f"output: {len(outputs)}")

    generated = [output["text"] for output in outputs]
    # print(f"{generated[:10]}")

    data = pd.read_csv("./data/imdb.csv").head(len(outputs))
    correct_predictions = (data["sentiment"] == generated).sum()
    accuracy = correct_predictions / len(outputs)
    print(f"Task 2 Accuracy:{accuracy}\n")

    # print(data[data["suitable"] != generated]["review"])


if __name__ == "__main__":
    main()
