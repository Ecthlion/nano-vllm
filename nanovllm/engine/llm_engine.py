import atexit
import threading
from dataclasses import fields
from queue import Empty, Queue
from time import perf_counter, sleep, time

import torch
import torch.multiprocessing as mp
from termcolor import colored
from torch.profiler import record_function
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.adaptive_sparsity import AdaptiveSparsityManager
from nanovllm.utils.kv_cache_index import KVCacheIndex


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        assert config.tensor_parallel_size == 1  # not supported for now
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.kv_cache_index = KVCacheIndex(self.model_runner.kv_cache)
        self.adaptive_sparsity = AdaptiveSparsityManager()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        # Prefetch machinery
        self._prefetch_thread: threading.Thread
        self._prefetch_queue: Queue
        self._cancel_prefetch = threading.Event()

        # for not optimize
        self._prefetch_stream = torch.cuda.Stream()

        # Store machinery
        self._store_thread: threading.Thread
        self._store_queue: Queue

        # Stats
        self._stats = {
            "total_xfer_ms": 0.0,
            "total_compute_ms": 0.0,
            "num_batches": 0,
        }

        # [{
        #   "batch": 0,
        #   "start_time": 0.0,
        #   "end_time": 0.0,
        #   "is_transfer": true,
        # }]

        self._analyse = []
        atexit.register(self.exit)
        self.use_index = False

    def exit(self):
        if self.use_index:
            self.kv_cache_index.persistence()

        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int] | tuple[int, str],
        sampling_params: SamplingParams,
        use_index,
    ):
        text_token_ids = []
        pruning_len = 0
        prompt_text = prompt[1] if isinstance(prompt, tuple) else (prompt if isinstance(prompt, str) else "")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        elif isinstance(prompt, tuple):
            if use_index:
                text_id = prompt[0]
                item = self.kv_cache_index.kv_cache_index.get(text_id)
                if isinstance(item, dict):
                    # remove text decode time
                    text_token_ids = item.get("text_tokens_pruned")
                    if isinstance(text_token_ids, torch.Tensor):
                        text_token_ids = text_token_ids.tolist()
                    # print(f"[kept text]: { self.tokenizer.decode(text_token_ids) }")
                    pruning_len = item.get("pruning_len")  # type: ignore
                    if text_token_ids is None:
                        text_token_ids = self.tokenizer.encode(
                            prompt[1][: len(prompt[1]) - sampling_params.task_str_len]
                        )
                else:
                    text_token_ids = self.tokenizer.encode(
                        prompt[1][: len(prompt[1]) - sampling_params.task_str_len]
                    )
                task_token_ids = self.tokenizer.encode(
                    prompt[1][-sampling_params.task_str_len :]
                )
                prompt = (text_id, text_token_ids + task_token_ids)
            else:
                prompt = (prompt[0], self.tokenizer.encode(prompt[1]))

        if isinstance(prompt, tuple):
            prompt_len = len(prompt[1])
        else:
            prompt_len = len(prompt)

        task_type = self.adaptive_sparsity.infer_task_type(
            prompt_text, sampling_params.task_type
        )
        precision_tier = sampling_params.precision_tier.lower()
        if precision_tier not in {"fast", "balanced", "high"}:
            precision_tier = "balanced"
        _, num_layers, _, _, num_kv_heads, _ = self.model_runner.kv_cache.shape
        adaptive_sparsity = self.adaptive_sparsity.build_layer_head_sparsity(
            task_type=task_type,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            seq_len=prompt_len,
            seq_len_p95=sampling_params.seq_len_p95,
        )

        seq = Sequence(
            prompt,  # type: ignore[arg-type]
            len(text_token_ids),
            pruning_len,
            sampling_params,
            task_type=task_type,
            precision_tier=precision_tier,
            adaptive_sparsity=adaptive_sparsity,
        )
        self.scheduler.add(seq)

    def _start_prefetcher(self):
        self._prefetch_queue: Queue = Queue(maxsize=4)

        def _prefetch_loop():
            prefetch_stream = torch.cuda.Stream()
            batch = 0
            while True:
                # Try to fill GPU blocks as much as possible by scheduling
                try:
                    seqs, is_prefill = self.scheduler.schedule()
                except AssertionError:
                    # No schedulable seqs at the moment
                    if self.scheduler.is_finished():
                        break
                    sleep(0.001)
                    continue

                # Kick off H2D KV transfer if indexed
                transfer_event = None
                xfer_ms = 0.0
                with record_function("get kv index"):
                    ret = self.kv_cache_index.get_kv_cache(
                        seqs,
                        stream=prefetch_stream,  # type: ignore
                        return_timing=True,
                        cancel_event=self._cancel_prefetch,
                    )

                if self._cancel_prefetch.is_set():
                    self._cancel_prefetch.clear()

                if ret is not None:
                    if isinstance(ret, tuple):
                        transfer_event, start_event = ret
                    else:
                        transfer_event, start_event = ret, None
                    transfer_event.synchronize()
                    if start_event is not None:
                        xfer_ms = start_event.elapsed_time(transfer_event)
                
                end_time = time()
                # Enqueue ready batch for compute
                self._analyse.append(
                    {
                        "batch": batch,
                        "start_time": end_time - (xfer_ms / 1000),
                        "end_time": end_time,
                        "is_transfer": True,
                    }
                )
                batch += 1
                self._prefetch_queue.put((seqs, is_prefill, xfer_ms))

            print(colored("prefetch thread quit!", "red"))

        self._prefetch_thread = threading.Thread(
            target=_prefetch_loop, name="kv-prefetch", daemon=True
        )
        self._prefetch_thread.start()

    def _start_storer(self):
        # Initialize ready queue and transfer stream
        self._store_queue = Queue(maxsize=-1)

        def _store_loop():
            store_stream = torch.cuda.Stream()
            while True:
                # Kick off H2D KV transfer if indexed
                transfer_event = None
                seqs = self._store_queue.get()
                if len(seqs) == 0:
                    break

                with record_function("store kv index"):
                    ret = self.kv_cache_index.store_kv_cache(
                        seqs, stream=store_stream, return_timing=False  # type: ignore
                    )

                if ret is not None:
                    if isinstance(ret, tuple):
                        transfer_event, start_event = ret
                    else:
                        transfer_event, start_event = ret, None
                    transfer_event.synchronize()

                    if start_event is not None:
                        print(
                            f"store kv cache: {start_event.elapsed_time(transfer_event)}"
                        )

                # deallocate blocks here
                for seq in seqs:
                    seq.lock_block = False
                    self.scheduler.block_manager.deallocate(seq)

            print(colored("store thread quit!", "red"))

        self._store_thread = threading.Thread(
            target=_store_loop, name="kv-store", daemon=True
        )
        self._store_thread.start()

    def step(self, use_index, optimize):
        xfer_ms = 0.0
        if use_index:
            if optimize:
                # Pop a ready batch (blocks until available or sentinel)
                try:
                    # item = self._prefetch_queue.get_nowait()
                    item = self._prefetch_queue.get()
                except Empty:
                    self._cancel_prefetch.set()
                    item = self._prefetch_queue.get()

                # backward-compat if queue carries 2-tuple
                seqs, is_prefill, xfer_ms = item
                if not seqs and self.scheduler.is_finished():
                    return [], 0
                print(colored(f"schedule {len(seqs)} seq", "magenta"))
            else:
                seqs, is_prefill = self.scheduler.schedule()
                transfer_event = None
                xfer_ms = 0.0
                with record_function("get kv index"):
                    ret = self.kv_cache_index.get_kv_cache(
                        seqs,
                        stream=self._prefetch_stream,  # type: ignore
                        return_timing=True,
                        cancel_event=self._cancel_prefetch,
                    )

                if self._cancel_prefetch.is_set():
                    self._cancel_prefetch.clear()

                if ret is not None:
                    if isinstance(ret, tuple):
                        transfer_event, start_event = ret
                    else:
                        transfer_event, start_event = ret, None
                    transfer_event.synchronize()
                    if start_event is not None:
                        xfer_ms = start_event.elapsed_time(transfer_event)
                print(colored(f"schedule {len(seqs)} seq", "magenta"))
        else:
            seqs, is_prefill = self.scheduler.schedule()
            print(colored(f"schedule {len(seqs)} seq", "magenta"))

        start = time()
        with record_function("run model"):
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        end = time()
        compute_ms = (end - start) * 1000

        self._analyse.append(
            {
                "batch": self._stats["num_batches"],
                "start_time": start,
                "end_time": end,
                "is_transfer": False,
            }
        )

        if use_index:
            for seq in seqs:
                seq.lock_block = True
            assert len(seqs) > 0
            self._store_queue.put_nowait(seqs)  # type: ignore

        self.scheduler.postprocess(seqs, token_ids)
        # Stats and prints
        if use_index:
            self._stats["total_xfer_ms"] += xfer_ms
            self._stats["total_compute_ms"] += compute_ms
            self._stats["num_batches"] += 1
            print(
                colored(f"H2D: {xfer_ms:.2f} ms | Compute: {compute_ms:.2f} ms", "cyan")
            )
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
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
        pruning: bool = False,
        sparsity: float = 0.9,
        optimize: bool = True,
    ) -> list[dict]:
        self._analyse = []
        self.model_runner.sparsity = sparsity
        init_start = time()
        self.use_index = self.use_index or use_index
        # Toggle pruning feature for this generation session
        # Propagate to model runner so it can prepare the runtime context.
        self.model_runner.pruning_enabled = pruning  # type: ignore[attr-defined]
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp, use_index)
        if use_index:
            # Start prefetch worker once requests are queued
            if optimize:
                self._start_prefetcher()
            else:
                self._prefetch_queue: Queue = Queue(maxsize=4)
            self._start_storer()
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        end = time()
        print(f"init: {end - init_start}")
        while not self.is_finished() or (
            use_index and not self._prefetch_queue.empty()
        ):
            t = perf_counter()
            output, num_tokens = self.step(use_index, optimize)
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(  # type: ignore
                    {  # type: ignore
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # type: ignore
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()  # type: ignore
        torch.cuda.synchronize()
        if use_index:
            total_xfer = self._stats["total_xfer_ms"]
            total_comp = self._stats["total_compute_ms"]
            nb = max(self._stats["num_batches"], 1)
            avg_xfer = total_xfer / nb if nb else 0.0
            avg_comp = total_comp / nb if nb else 0.0
            print(
                colored(
                    f"\nTiming summary (per batch): H2D avg {avg_xfer:.2f} ms | Compute avg {avg_comp:.2f} ms | batches {nb}",
                    "green",
                )
            )
            self._stats = {
                "total_xfer_ms": 0.0,
                "total_compute_ms": 0.0,
                "num_batches": 0,
            }
            # join background threads
            if self._prefetch_thread is not None:
                self._prefetch_thread.join()

            if self._store_thread is not None:
                self._store_queue.put_nowait([])  # type: ignore
                self._store_thread.join()
        return outputs
