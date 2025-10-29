import atexit
import threading
from queue import Queue, Empty
from dataclasses import fields
from time import perf_counter, time
from termcolor import colored
from torch.profiler import record_function
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
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
        self.kv_cache_index = KVCacheIndex(self.model_runner.kv_cache, index_name="imdb_kvcache.pt")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # Prefetch machinery (initialized lazily when use_index=True)
        self._prefetch_thread: threading.Thread | None = None
        self._ready_batches: Queue | None = None
        self._stop_prefetch = threading.Event()
        self._transfer_stream: torch.cuda.Stream | None = None
        # Stats
        self._stats = {
            "total_xfer_ms": 0.0,
            "total_compute_ms": 0.0,
            "num_batches": 0,
        }
        atexit.register(self.exit)

    def exit(self):
        if self.use_index:
            self.kv_cache_index.persistence()
        # Stop prefetch thread if running
        if self._prefetch_thread is not None:
            self._stop_prefetch.set()
            self._prefetch_thread.join(timeout=1.0)
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
        # Pre-reserve pinned CPU buffer for KV cache of indexed sequences to avoid
        # cudaHostAlloc in the hot path (observed to block compute)
        if seq.text_id is not None and not self.kv_cache_index.is_indexed(seq):
            self.kv_cache_index.reserve(seq)
        self.scheduler.add(seq)

    def _ensure_prefetcher(self):
        if self._prefetch_thread is not None:
            return
        # Initialize ready queue and transfer stream
        self._ready_batches = Queue(maxsize=16)
        self._stop_prefetch.clear()
        self._transfer_stream = torch.cuda.Stream()

        def _prefetch_loop():
            assert self._ready_batches is not None
            while not self._stop_prefetch.is_set():
                # If queue is full, wait a bit
                if self._ready_batches.full():
                    # Avoid busy wait
                    self._stop_prefetch.wait(0.001)
                    continue
                # Try to fill GPU blocks as much as possible by scheduling
                try:
                    seqs, is_prefill = self.scheduler.schedule()
                except AssertionError:
                    # No schedulable seqs at the moment
                    if self.scheduler.is_finished():
                        break
                    self._stop_prefetch.wait(0.001)
                    continue

                # Kick off H2D KV transfer if indexed
                transfer_event = None
                xfer_ms = 0.0
                if self._transfer_stream is not None:
                    with record_function("get kv index"):
                        ret = self.kv_cache_index.get_kv_cache(seqs, stream=self._transfer_stream, return_timing=True)
                    # Wait for H2D completion before marking ready, to guarantee compute sees ready KV
                    if ret is not None:
                        if isinstance(ret, tuple):
                            transfer_event, start_event = ret
                        else:
                            transfer_event, start_event = ret, None
                        transfer_event.synchronize()
                        if start_event is not None:
                            try:
                                xfer_ms = start_event.elapsed_time(transfer_event)
                            except Exception:
                                xfer_ms = 0.0
                # Enqueue ready batch for compute
                self._ready_batches.put((seqs, is_prefill, xfer_ms))
            # Signal termination with sentinel
            try:
                self._ready_batches.put_nowait(([], False, 0.0))
            except Exception:
                pass

            self._prefetch_thread = None

        self._prefetch_thread = threading.Thread(target=_prefetch_loop, name="kv-prefetch", daemon=True)
        self._prefetch_thread.start()

    def step(self, use_index):
        if use_index:
            assert self._ready_batches is not None, "Prefetcher not initialized"
            # Pop a ready batch (blocks until available or sentinel)
            item = self._ready_batches.get()
            # backward-compat if queue carries 2-tuple
            if isinstance(item, tuple) and len(item) == 3:
                seqs, is_prefill, xfer_ms = item
            else:
                seqs, is_prefill = item  # type: ignore
                xfer_ms = 0.0
            if not seqs and self.scheduler.is_finished():
                return [], 0
            print(colored(f"schedule {len(seqs)} seq (ready)", "magenta"))
        else:
            seqs, is_prefill = self.scheduler.schedule()
            print(colored(f"schedule {len(seqs)} seq", "magenta"))

        if use_index and seqs:
            # Transfer already completed in prefetch thread
            pass
        # Measure compute GPU time via CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with record_function("run model"):
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        end_event.record()
        end_event.synchronize()
        compute_ms = start_event.elapsed_time(end_event)
        if use_index:
            # with record_function("store kv index"):
            # start = time()
            self.kv_cache_index.store_kv_cache(seqs)
            # end = time()
            # print(colored(f"store kv cache: {( end - start ):.4f} s", "white"))
        self.scheduler.postprocess(seqs, token_ids)
        # Stats and prints
        if use_index:
            self._stats["total_xfer_ms"] += xfer_ms
            self._stats["total_compute_ms"] += compute_ms
            self._stats["num_batches"] += 1
            print(colored(f"H2D: {xfer_ms:.2f} ms | Compute: {compute_ms:.2f} ms", "cyan"))
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
        self.use_index = use_index
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        if use_index:
            # Start prefetch worker once requests are queued
            self._ensure_prefetcher()
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
        if use_index and self._stats["num_batches"] > 0:
            total_xfer = self._stats["total_xfer_ms"]
            total_comp = self._stats["total_compute_ms"]
            nb = self._stats["num_batches"]
            avg_xfer = total_xfer / nb
            avg_comp = total_comp / nb
            print(colored(f"\nTiming summary (per batch): H2D avg {avg_xfer:.2f} ms | Compute avg {avg_comp:.2f} ms | batches {nb}", "green"))
            self._stats = {
                "total_xfer_ms": 0.0,
                "total_compute_ms": 0.0,
                "num_batches": 0,
            }
        return outputs
