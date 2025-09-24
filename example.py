import os

import pandas as pd
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
        self.data = pd.read_csv("./imdb.csv")

    def sample(
        self,
        tokenizer,
        **kwargs,
    ) -> list:
        num_input_lines = 1000
        chose_lines = self.data["review"][:num_input_lines]
        lines_per_prompt = 1
        duplicate = 1
        samples = []
        num_requests = int(num_input_lines / lines_per_prompt)

        for i in range(num_requests):
            texts = "\n".join(
                chose_lines[i * lines_per_prompt : (i + 1) * lines_per_prompt]
            )
            for d in range(duplicate):
                base_prompt = f'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
                # base_prompt = f'Given the above film review, answer whether the film is suitable for kids. Respond ONLY with "yes" or "no", in all lower case.\n'
                # base_prompt = f"Given the above film review, answer whether it contains names. Respond ONLY with \"yes\" or \"no\", in all lower case.\n"
                # base_prompt = f"Given the above film review, answer whether it contains violent elements. Respond ONLY with \"yes\" or \"no\", in all lower case.\n"
                prompt = f"{texts}\n{base_prompt}"
                prompt_len = len(tokenizer(prompt).input_ids)
                samples.append(prompt)
        return samples


def main():
    path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1)

    dataset = ImdbDataset()
    outputs = llm.generate(dataset.sample(tokenizer), sampling_params)

    data = pd.read_csv("./imdb.csv").head(len(outputs))
    generated = [output['text'] for output in outputs]
    print(f"\n{generated[:10]}")
    correct_predictions = (data["sentiment"] == generated).sum()
    accuracy = correct_predictions / len(outputs)
    print(f"Accuracy:{accuracy}\n")
    # print(data[data["suitable"] != generated]["review"])

if __name__ == "__main__":
    main()
