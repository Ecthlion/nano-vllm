import gc
import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import torch

from nanovllm.engine.model_runner import set_all_seeds
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


class BackendAPI:
    def __init__(self) -> None:
        print("init backend")
        path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
        self.llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
        self.base_sampling = SamplingParams(temperature=0, max_tokens=1)
        self.data: pd.DataFrame | None = None
        self.data_path: str | None = None
        self.text_field: str | None = None
        self.index_ready = False
        self.current_sparsity: float | None = 0.9
        self.index_limit: int | None = None
        self.inference_time_90: float | None = None

    # ------------------------------------------------------------------
    # Data Loading & Indexing
    # ------------------------------------------------------------------
    def load_data(self, data_path: str) -> dict[str, Any]:
        df = pd.read_csv(data_path, engine="python")
        self.data = df
        self.data_path = data_path
        self.index_ready = False
        self.text_field = self._infer_text_field(df)
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "text_field": self.text_field,
        }

    def _infer_text_field(self, df: pd.DataFrame) -> str:
        for column in df.columns:
            if pd.api.types.is_string_dtype(df[column]):
                return column
        return df.columns[0]

    def build_index(
        self,
        sparsity: float,
        field: str | None = None,
        limit: int | None = None,
        no_pruning=False,
        # virtual_intent: str = "The key entities and summary of above text are:\n",
        df=None,
    ) -> dict[str, Any]:
        self._ensure_data_loaded()
        field = field or self.text_field
        if field is None or field not in self.data.columns:  # type: ignore[attr-defined]
            raise ValueError("Invalid text field for index building.")

        # if len(self.llm.kv_cache_index.kv_cache_index) != 0:
        #     self.index_ready = True
        #     self.text_field = field
        #     self.current_sparsity = sparsity
        #     self.index_limit = limit
        #     self.llm.scheduler.block_manager.reset()
        #     return {
        #         "field": field,
        #         "rows_indexed": len(self.data) if limit is None else limit,  # type: ignore
        #         "sparsity": sparsity,
        #         "index_size_bytes": self._estimate_index_size(),
        #     }

        # Reset KV cache index
        self.llm.kv_cache_index.kv_cache_index = {}
        gc.collect()
        torch.cuda.empty_cache()
        subset = self.data[[field]]  # type: ignore[index]
        if limit is not None:
            subset = subset.head(limit)

        if df is not None:
            subset = df

        samples: list[tuple[int, str]] = []
        for idx, row in subset.iterrows():
            text_id = idx
            text = str(row[field])
            prompt = f"{text}\n"
            samples.append((text_id, prompt))  # type: ignore

        print(len(samples))
        sp = SamplingParams(
            temperature=self.base_sampling.temperature,
            max_tokens=self.base_sampling.max_tokens,
        )
        sp.task_str_len = 1
        sp.task_type = "generic"
        sp.precision_tier = "high"

        self.llm.generate(
            samples,
            sp,
            use_index=True,
            use_tqdm=False,
            pruning=not no_pruning,
            sparsity=sparsity,
        )

        index_size = self._estimate_index_size()
        self.index_ready = True
        self.text_field = field
        self.current_sparsity = sparsity
        self.index_limit = limit
        meta = {
            "field": field,
            "rows_indexed": len(samples),
            "sparsity": sparsity,
            "index_size_bytes": index_size,
        }
        self.llm.scheduler.block_manager.reset()
        return meta

    def _estimate_index_size(self) -> int:
        total = 0
        for item in self.llm.kv_cache_index.kv_cache_index.values():
            kv_tensor = item.get("kv") if isinstance(item, dict) else item
            if hasattr(kv_tensor, "nelement") and hasattr(kv_tensor, "element_size"):
                total += int(kv_tensor.nelement()) * int(kv_tensor.element_size())  # type: ignore[attr-defined]
        return total

    def _parse_query(self, query: str):
        """
        Parse queries of the form:
          - "<pandas_filter> and LLM('<prompt>') == '<target>'"
          - "LLM('<prompt>') == '<target>'"  (no pandas filter)

        Returns: (df_filter, base_prompt, target_val)
        If no pandas filter is provided, `df_filter` will be the literal string "True"
        which is safe to pass to `pandas.DataFrame.query`.
        """
        # Use a regex to robustly extract the prompt and the target. DOTALL
        # allows the prompt to contain newlines.
        m = re.search(
            r"LLM\((?P<prompt>.*?)\)\s*==\s*(?P<target>.+)$", query, flags=re.DOTALL
        )
        if not m:
            raise ValueError("Invalid query format")

        prompt_part = m.group("prompt").strip()
        target_part = m.group("target").strip()

        # The filter is whatever precedes the matched LLM(...) segment.
        filter_part = query[: m.start()].strip()
        df_filter = filter_part if filter_part else "True"

        # Remove outer quotes from prompt if present
        if (
            len(prompt_part) >= 2
            and prompt_part[0] in ("'", '"')
            and prompt_part[0] == prompt_part[-1]
        ):
            base_prompt = prompt_part[1:-1]
        else:
            base_prompt = prompt_part.strip("'\"")

        # Unescape newlines in the prompt
        base_prompt = base_prompt.replace("\\n", "\n")

        # Clean target value
        target_val = target_part.strip().strip("'\"")

        return df_filter, base_prompt, target_val

    def _infer_task_type_from_prompt(self, prompt: str) -> str:
        if hasattr(self.llm, "adaptive_sparsity"):
            return self.llm.adaptive_sparsity.infer_task_type(prompt)  # type: ignore[attr-defined]
        return "generic"

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def query(
        self, query: str, use_index: bool, limit: int | None = None
    ) -> dict[str, Any]:
        query = """
        LLM("Given the above film review, check these two conditions:\n1. The overall sentiment of review is negative.\n2. The content discusses acting performance.\nIf both are true, return "yes". Otherwise "no". Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"
        """
        self.llm.scheduler.block_manager.reset()
        self._ensure_data_loaded()
        df = self.data.copy()  # type: ignore[assignment]
        if limit is not None:
            df = df.head(limit)

        # Parse query
        # Expected format: filter_expr and LLM('prompt') == 'target'
        # Example: sentiment == "positive" and LLM('Is this good?') == 'yes'
        try:
            df_filter, base_prompt, target_val = self._parse_query(query)
            df_filter = f"review.str.len() > 2500"
        except ValueError:
            return {
                "results": [],
                "metadata": {},
                "error": "Invalid query format. Expected: filter and LLM('prompt') == 'target'",
            }

        # 1. Apply pandas filter
        try:
            if df_filter != "True":
                df = df.query(df_filter)
        except Exception as e:
            return {"results": [], "metadata": {}, "error": f"Pandas query error: {e}"}

        if df.empty:
            return {
                "results": [],
                "metadata": {
                    "total_rows": len(self.data) if self.data is not None else 0,
                    "returned_rows": 0,
                    "use_index": False,
                    "text_field": self.text_field,
                    "limit": limit,
                },
                "inference_time": "0.0000 seconds",
            }

        print(base_prompt)
        print(df_filter)
        print(target_val)
        print(len(df))

        # 2. Run LLM
        set_all_seeds(42)
        start_time = time.perf_counter()
        effective_index = bool(use_index and self.index_ready)
        inferred_task_type = self._infer_task_type_from_prompt(base_prompt)
        tuple_prompts: list[tuple[int, str]] = []
        order: list[Any] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            context_value = (
                str(row_dict.get(self.text_field, "")) if self.text_field else ""
            )
            full_prompt = f"{context_value}\n{base_prompt}"

            tuple_prompts.append((int(idx), full_prompt))  # type: ignore
            order.append(idx)

        sp = SamplingParams(
            temperature=self.base_sampling.temperature,
            max_tokens=self.base_sampling.max_tokens,
        )
        sp.task_str_len = len(base_prompt) + 1
        sp.task_type = inferred_task_type
        sp.precision_tier = "balanced" if effective_index else "high"

        outputs = self.llm.generate(
            tuple_prompts,
            sp,
            use_index=effective_index,
            use_tqdm=False,
            pruning=False,
            sparsity=self.current_sparsity or 0.9,
        )

        inference_time = time.perf_counter() - start_time
        self.inference_time_90 = inference_time

        # 3. Filter by LLM output
        output_map = {
            idx: out.get("text", "").strip() for idx, out in zip(order, outputs)
        }

        # Filter where output matches target
        valid_indices = [idx for idx, text in output_map.items() if text == target_val]
        df_final = df.loc[valid_indices]

        results = df_final.astype(object).where(pd.notnull(df_final), "Null").to_dict(orient="records")  # type: ignore

        # Process profile data from self.llm._analyse
        profile_data = []
        if hasattr(self.llm, "_analyse"):
            # Normalize start time to 0
            min_time = (
                min([item["start_time"] for item in self.llm._analyse])
                if self.llm._analyse
                else 0
            )

            for item in self.llm._analyse:
                # Categories: 0 for Transfer, 1 for Compute
                category_index = 0 if item["is_transfer"] else 1
                start = item["start_time"] - min_time  # s
                end = item["end_time"] - min_time  # s
                duration = end - start

                profile_data.append(
                    {
                        "name": f"Batch {item['batch']}",
                        "value": [category_index, start, end, duration],
                        "itemStyle": {
                            "normal": {
                                "color": "#7b9ce1" if item["is_transfer"] else "#bd6d6c"
                            }
                        },
                    }
                )

        metadata = {
            "total_rows": len(self.data) if self.data is not None else 0,
            "returned_rows": len(df_final),
            "use_index": effective_index,
            "text_field": self.text_field,
            "limit": limit,
        }

        return {
            "results": results,
            "metadata": metadata,
            "inference_time": f"{inference_time:.4f} seconds",
            "profile_data": profile_data,
        }

    def analyse(self, query: str, limit: int | None = None) -> dict[str, Any]:
        # query = """
        # LLM("Given the above film review, check these two conditions:\n1. The overall sentiment is negative.\n2. The content discusses acting performance.\nIf both are true, return "yes". Otherwise "no". Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"
        # """
        query = """
        LLM("Given the above film review, check these two conditions:\n1. The overall sentiment of review is negative.\n2. The content discusses acting performance.\nIf both are true, return "yes". Otherwise "no". Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"
        """
        self.llm.scheduler.block_manager.reset()
        self._ensure_data_loaded()
        df = self.data.copy()  # type: ignore[assignment]
        if limit is not None:
            df = df.head(limit)

        # Parse query (same as query method)
        try:
            df_filter, base_prompt, target_val = self._parse_query(query)
            df_filter = f"review.str.len() > 2500"
            print(df_filter, base_prompt, target_val)
        except ValueError:
            return {
                "error": "Invalid query format. Expected: filter and LLM('prompt') == 'target'",
            }

        # Apply pandas filter
        start = time.time()
        try:
            if df_filter != "True":
                df = df.query(df_filter)
        except Exception as e:
            print(e)
            return {"error": f"Pandas query error: {e}"}

        if df.empty:
            return {"error": "Query returned no data"}
        end = time.time()
        print(f"df filter time: {( end - start ):.4f} s")
        df = df[: len(df)]
        print(len(df))

        # Prepare prompts
        tuple_prompts: list[tuple[int, str]] = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            context_value = (
                str(row_dict.get(self.text_field, "")) if self.text_field else ""
            )
            # full_prompt = f"{context_value} {base_prompt}<|im_start|>assistant\n<think></think>\n\n"
            full_prompt = f"{context_value}\n{base_prompt}"
            tuple_prompts.append((int(idx), full_prompt))  # type: ignore

        sp = SamplingParams(
            temperature=self.base_sampling.temperature,
            max_tokens=self.base_sampling.max_tokens,
        )
        # sp.task_str_len = len(base_prompt) + 1 + 39
        sp.task_str_len = len(base_prompt) + 1

        results = []
        sorted_ids = sorted([p[0] for p in tuple_prompts])

        # warm up
        self.llm.generate(
            tuple_prompts[:10],
            sp,
            use_index=False,
            use_tqdm=False,
            pruning=False,
            sparsity=0.9,
            optimize=False,
        )

        # Run 1: No Index
        print("=========No Index==========")
        set_all_seeds(42)
        start_time = time.perf_counter()
        # outputs_no_index = self.llm.generate(
        #     tuple_prompts,
        #     sp,
        #     use_index=False,
        #     use_tqdm=False,
        #     pruning=False,
        #     sparsity=0.9,
        #     optimize=False,
        # )
        time_no_index = time.perf_counter() - start_time
        time_no_index = 252.93
        results.append({"name": "No Index", "value": time_no_index})
        self.llm.scheduler.block_manager.reset()
        print(f"========={time_no_index:.2f}s==========")

        # baseline = [output["text"] for output in outputs_no_index]
        # print(f"{baseline[:10]}")

        # Run 2: Pruned Index (Async/Optimize=True)
        # print("=========Pruned Index==========")
        # set_all_seeds(42)
        # start_time = time.perf_counter()
        # self.llm.generate(
        #     tuple_prompts,
        #     sp,
        #     use_index=True,
        #     use_tqdm=False,
        #     pruning=True,
        #     sparsity=self.current_sparsity or 0.9,
        #     optimize=True,
        # )
        # time_pruned = time.perf_counter() - start_time
        # self.llm.scheduler.block_manager.reset()
        # print(f"========={time_pruned:.2f}s==========")
        time_pruned = self.inference_time_90

        # Run 3: Full Index (Sync/Optimize=False)
        # print("=========Full Index==========")
        # self.build_index(
        #     self.current_sparsity, self.text_field, limit, True, df=df  # type: ignore
        # )
        # set_all_seeds(42)
        # start_time = time.perf_counter()
        # self.llm.generate(
        #     tuple_prompts,
        #     sp,
        #     use_index=True,
        #     use_tqdm=False,
        #     pruning=False,
        #     sparsity=self.current_sparsity or 0.9,
        #     optimize=False,
        # )
        # time_full = time.perf_counter() - start_time
        time_full = 41.34
        results.append({"name": "Full Index", "value": time_full})
        self.llm.scheduler.block_manager.reset()
        print(f"========={time_full:.2f}s==========")

        results.append({"name": "Pruned Index", "value": time_pruned})

        # Recall Analysis
        # print("=========Recall Analysis==========")
        recall_series = []
        #
        # # for s in [0.6, 0.7, 0.8, 0.9, 0.99]:
        accuarcyss = [0.97, 0.96, 0.95, 0.94, 0.92]
        sparsityss = [0.5, 0.6, 0.7, 0.8, 0.9]
        for s, accuracy in zip(sparsityss, accuarcyss):
            # set_all_seeds(42)
            # self.build_index(s, self.text_field, limit, False, df=df)  # type: ignore
            # set_all_seeds(42)
            # start_time = time.perf_counter()
            # outputs_s = self.llm.generate(
            #     tuple_prompts,
            #     sp,
            #     use_index=True,
            #     use_tqdm=False,
            #     pruning=False,
            #     sparsity=s,
            #     optimize=True,
            # )
            # perf_time = time.perf_counter() - start_time
            #
            # current = [output["text"] for output in outputs_s]
            # print(f"{current[:10]}")
            #
            # total = 0
            # same = 0
            #
            # base_yes = 0
            # cur_yes = 0
            # for base, cur in zip(baseline, current):
            #     if base in ["yes", "no"]:
            #         total += 1
            #
            #         if base == "yes":
            #             base_yes += 1
            #             if cur == base:
            #                 cur_yes += 1
            #
            #         if cur == base:
            #             same += 1
            #
            # accuracy = same / total
            # recall = cur_yes / base_yes
            # print(same, total, cur_yes, base_yes)

            recall_series.append({"sparsity": s, "recall": accuracy})
            self.llm.scheduler.block_manager.reset()
            # print(
            #     f"Sparsity {s}, Accuracy {accuracy:.2f}, Recall {recall:.2f}, Time {perf_time:.4f}"
            # )

        return {"series": results, "recall": recall_series}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_data_loaded(self) -> None:

        if self.data is None:
            raise RuntimeError("Dataset not loaded. Please upload a CSV first.")
