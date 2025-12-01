import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

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
        self.current_sparsity: float | None = None
        self.index_limit: int | None = None

    # ------------------------------------------------------------------
    # Data Loading & Indexing
    # ------------------------------------------------------------------
    def load_data(self, data_path: str) -> dict[str, Any]:
        df = pd.read_csv(data_path)
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
        limit: int | None = 1000,
        no_pruning=False,
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
        subset = self.data[[field]]  # type: ignore[index]
        if limit is not None:
            subset = subset.head(limit)

        samples: list[tuple[int, str]] = []
        for idx, row in subset.iterrows():
            text_id = idx
            text = str(row[field])
            prompt = f"{text}\n "
            samples.append((text_id, prompt))  # type: ignore

        sp = SamplingParams(
            temperature=self.base_sampling.temperature,
            max_tokens=self.base_sampling.max_tokens,
        )
        sp.task_str_len = 1

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
        m = re.search(r"LLM\((?P<prompt>.*?)\)\s*==\s*(?P<target>.+)$", query, flags=re.DOTALL)
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

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def query(
        self, query: str, use_index: bool, limit: int | None = 1000
    ) -> dict[str, Any]:
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

        # 2. Run LLM
        set_all_seeds(42)
        start_time = time.perf_counter()
        effective_index = bool(use_index and self.index_ready)
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
        sp.task_str_len = len(base_prompt)

        outputs = self.llm.generate(
            tuple_prompts,
            sp,
            use_index=effective_index,
            use_tqdm=False,
            pruning=False,
            sparsity=self.current_sparsity or 0.9,
        )

        inference_time = time.perf_counter() - start_time

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
                start = (item["start_time"] - min_time) * 1000  # ms
                end = (item["end_time"] - min_time) * 1000  # ms
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

    def analyse(self, query: str, limit: int | None = 1000) -> dict[str, Any]:
        self.llm.scheduler.block_manager.reset()
        self._ensure_data_loaded()
        df = self.data.copy()  # type: ignore[assignment]
        if limit is not None:
            df = df.head(limit)

        # Parse query (same as query method)
        try:
            df_filter, base_prompt, target_val = self._parse_query(query)
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
            return {"error": f"Pandas query error: {e}"}

        if df.empty:
            return {"error": "Query returned no data"}
        end = time.time()
        print(f"df filter time: {( end - start ):.2f} s")

        # Prepare prompts
        tuple_prompts: list[tuple[int, str]] = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            context_value = (
                str(row_dict.get(self.text_field, "")) if self.text_field else ""
            )
            full_prompt = f"{context_value}\n{base_prompt}"
            tuple_prompts.append((int(idx), full_prompt))  # type: ignore

        sp = SamplingParams(
            temperature=self.base_sampling.temperature,
            max_tokens=self.base_sampling.max_tokens,
        )
        sp.task_str_len = len(base_prompt)

        results = []
        sorted_ids = sorted([p[0] for p in tuple_prompts])

        # Run 1: No Index
        print("=========No Index==========")
        set_all_seeds(42)
        start_time = time.perf_counter()
        outputs_no_index = self.llm.generate(
            tuple_prompts,
            sp,
            use_index=False,
            use_tqdm=False,
            pruning=False,
            sparsity=0.9,
            optimize=False,
        )
        time_no_index = time.perf_counter() - start_time
        results.append({"name": "No Index", "value": time_no_index})
        self.llm.scheduler.block_manager.reset()
        print(f"========={time_no_index:.2f}s==========")

        baseline_indices = set()
        for idx, out in zip(sorted_ids, outputs_no_index):
            if out.get("text", "").strip() == target_val:
                baseline_indices.add(idx)
        optimized_generated = [output["text"] for output in outputs_no_index]
        print(f"{optimized_generated[:10]}")

        # Run 2: Pruned Index (Async/Optimize=True)
        print("=========Pruned Index==========")
        set_all_seeds(42)
        start_time = time.perf_counter()
        self.llm.generate(
            tuple_prompts,
            sp,
            use_index=True,
            use_tqdm=False,
            pruning=True,
            sparsity=self.current_sparsity or 0.9,
            optimize=True,
        )
        time_pruned = time.perf_counter() - start_time
        self.llm.scheduler.block_manager.reset()
        print(f"========={time_pruned:.2f}s==========")

        # Run 3: Full Index (Sync/Optimize=False)
        print("=========Full Index==========")
        self.build_index(
            self.current_sparsity, self.text_field, limit, True  # type: ignore
        )
        set_all_seeds(42)
        start_time = time.perf_counter()
        self.llm.generate(
            tuple_prompts,
            sp,
            use_index=True,
            use_tqdm=False,
            pruning=False,
            sparsity=self.current_sparsity or 0.9,
            optimize=False,
        )
        time_full = time.perf_counter() - start_time
        results.append({"name": "Full Index", "value": time_full})
        self.llm.scheduler.block_manager.reset()
        print(f"========={time_full:.2f}s==========")

        results.append({"name": "Pruned Index", "value": time_pruned})

        # Recall Analysis
        print("=========Recall Analysis==========")
        recall_series = []

        for s in [0.6, 0.7, 0.8, 0.9, 0.99]:
            set_all_seeds(42)
            self.build_index(
                s, self.text_field, limit, False  # type: ignore
            )
            outputs_s = self.llm.generate(
                tuple_prompts,
                sp,
                use_index=True,
                use_tqdm=False,
                pruning=False,
                sparsity=s,
                optimize=True,
            )
            
            current_indices = set()
            for idx, out in zip(sorted_ids, outputs_s):
                if out.get("text", "").strip() == target_val:
                    current_indices.add(idx)
            
            if len(baseline_indices) > 0:
                recall = len(baseline_indices.intersection(current_indices)) / len(baseline_indices)
            else:
                recall = 1.0
            print(len(baseline_indices.intersection(current_indices)), len(baseline_indices))
            
            recall_series.append({"sparsity": s, "recall": recall})
            self.llm.scheduler.block_manager.reset()
            print(f"Sparsity {s}: Recall {recall:.2f}")
            optimized_generated = [output["text"] for output in outputs_s]
            print(f"{optimized_generated[:10]}")

        return {"series": results, "recall": recall_series}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_data_loaded(self) -> None:

        if self.data is None:
            raise RuntimeError("Dataset not loaded. Please upload a CSV first.")
