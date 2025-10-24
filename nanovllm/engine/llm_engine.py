import atexit
from dataclasses import fields
from time import perf_counter, time
from termcolor import colored
from torch.profiler import record_function
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.kv_cache_index import KVCacheIndex


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        assert(config.tensor_parallel_size == 1) # not supported for now
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.kv_cache_index = KVCacheIndex(self.model_runner.kv_cache)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int] | tuple[int, str], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        elif isinstance(prompt, tuple):
            prompt = (prompt[0], self.tokenizer.encode(prompt[1]))

        seq = Sequence(prompt, sampling_params) # type: ignore
        self.scheduler.add(seq)

    def step(self, use_index):
        seqs, is_prefill = self.scheduler.schedule()
        print(colored(f"schedule {len(seqs)} seq", "magenta"))
        if use_index:
            # with record_function("get kv index"):
            start = time()
            self.kv_cache_index.get_kv_cache(seqs)
            end = time()
            print(colored(f"get kv cache: {( end - start ):.4f} s", "white"))
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        if use_index:
            # with record_function("store kv index"):
            # start = time()
            self.kv_cache_index.store_kv_cache(seqs)
            # end = time()
            # print(colored(f"store kv cache: {( end - start ):.4f} s", "white"))
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[tuple[int, str]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        use_index: bool = False,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step(use_index)
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({ # type: ignore
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1) # type: ignore
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close() # type: ignore
        if use_index:
            self.kv_cache_index.persistence()
        return outputs
