import argparse
import csv
import json
import os
import re
import sys
import time

import tiktoken
import torch.multiprocessing as mp
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm import LLM, SamplingParams

model_map = json.loads(
    open("LongBench/config/model2path.json", encoding="utf-8").read()
)
maxlen_map = json.loads(
    open("LongBench/config/model2maxlen.json", encoding="utf-8").read()
)

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
template_rag = open("LongBench/prompts/0shot_rag.txt", encoding="utf-8").read()
template_no_context = open(
    "LongBench/prompts/0shot_no_context.txt", encoding="utf-8"
).read()
template_0shot = open("LongBench/prompts/0shot.txt", encoding="utf-8").read()
template_0shot_cot = open("LongBench/prompts/0shot_cot.txt", encoding="utf-8").read()
template_0shot_cot_ans = open(
    "LongBench/prompts/0shot_cot_ans.txt", encoding="utf-8"
).read()


def query_llm(
    prompt,
    model,
    tokenizer,
    llm,
    task_id,
    task_str_len,
    pruning=True,
    temperature=1,
    max_new_tokens=1,
):
    # truncate
    max_len = maxlen_map[model]
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    if model in model_map:
        model = model_map[model]

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    sampling_params.task_str_len = task_str_len
    outputs = llm.generate(
        [(task_id, prompt)],
        sampling_params,
        use_index=True,
        use_tqdm=False,
        pruning=pruning,
    )
    return outputs[0]["text"]


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_map[model], trust_remote_code=True
        )

    # Init
    path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
    count = 0

    prompts = []
    sps = []
    for item in tqdm(data):
        count += 1
        context = item["context"]
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][: args.rag]
            retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
            context = "\n\n".join(
                [
                    f"Retrieved chunk {idx+1}: {x['content']}"
                    for idx, x in enumerate(retrieved)
                ]
            )
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = (
            template.replace("$DOC$", context.strip())
            .replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip())
            .replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip())
            .replace("$C_D$", item["choice_D"].strip())
        )
        task_str_len = len(prompt) - (len(context) + 17)

        max_len = maxlen_map[model]
        if model in model_map:
            input_ids = tokenizer.encode(prompt)
            if len(input_ids) > max_len:
                input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)  # type: ignore

        prompt = prompt[:len(context) + 17]
        prompts.append((count, prompt))
        sps.append(
            SamplingParams(temperature=1, max_tokens=1, task_str_len=task_str_len)
        )

    # pruning first
    outputs = llm.generate(prompts, sps, use_index=True, use_tqdm=False, pruning=True)
    for output, item in zip(outputs, data):
        output = output["text"]
        if output == "":
            continue
        response = output.strip()

        item["response"] = response
        item["pred"] = response
        item["judge"] = item["pred"] == item["answer"]
        context = item["context"]
        item["context"] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        fout.flush()


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(
            args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl"
        )
    elif args.no_context:
        out_file = os.path.join(
            args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl"
        )
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset(
        "THUDM/LongBench-v2", split="train"
    )  # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [
        {
            "_id": item["_id"],  # type: ignore
            "domain": item["domain"],  # type: ignore
            "sub_domain": item["sub_domain"],  # type: ignore
            "difficulty": item["difficulty"],  # type: ignore
            "length": item["length"],  # type: ignore
            "question": item["question"],  # type: ignore
            "choice_A": item["choice_A"],  # type: ignore
            "choice_B": item["choice_B"],  # type: ignore
            "choice_C": item["choice_C"],  # type: ignore
            "choice_D": item["choice_D"],  # type: ignore
            "answer": item["answer"],  # type: ignore
            "context": item["context"],  # type: ignore
        }
        for item in dataset
    ]

    # cache
    # has_data = {}
    # if os.path.exists(out_file):
    #     with open(out_file, encoding="utf-8") as f:
    #         has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, "w", encoding="utf-8")
    # data = []
    # for item in data_all:
    #     if item["_id"] not in has_data:
    #         data.append(item)

    get_pred(data_all[:200], args, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument(
        "--cot", "-cot", action="store_true", default=False
    )  # set to True if using COT
    parser.add_argument(
        "--no_context", "-nc", action="store_true", default=False
    )  # set to True if using no context (directly measuring memorization)
    parser.add_argument(
        "--rag", "-rag", type=int, default=0
    )  # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    args = parser.parse_args()
    main()
