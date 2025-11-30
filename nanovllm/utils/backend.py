import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


class BackendAPI:
    def __init__(self) -> None:
        print("init backend")
        path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
        self.llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
        self.base_sampling = SamplingParams(temperature=1, max_tokens=1)
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
    ) -> dict[str, Any]:
        self._ensure_data_loaded()
        field = field or self.text_field
        if field is None or field not in self.data.columns:  # type: ignore[attr-defined]
            raise ValueError("Invalid text field for index building.")

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
            pruning=True,
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
        return meta

    def _estimate_index_size(self) -> int:
        total = 0
        for item in self.llm.kv_cache_index.kv_cache_index.values():
            kv_tensor = item.get("kv") if isinstance(item, dict) else item
            if hasattr(kv_tensor, "nelement") and hasattr(kv_tensor, "element_size"):
                total += int(kv_tensor.nelement()) * int(kv_tensor.element_size())  # type: ignore[attr-defined]
        return total

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def query(
        self, query: str, use_index: bool, limit: int | None = 50
    ) -> dict[str, Any]:
        self._ensure_data_loaded()
        df = self.data.copy()  # type: ignore[assignment]

        # Parse query
        # Expected format: filter_expr and LLM('prompt') == 'target'
        # Example: sentiment == "positive" and LLM('Is this good?') == 'yes'
        try:
            filter_part, llm_part = query.split(" and LLM(", 1)
            
            # Use rfind to locate the closing parenthesis and operator, 
            # allowing quotes inside the prompt string.
            split_marker = ") == "
            split_idx = llm_part.rfind(split_marker)
            
            if split_idx == -1:
                 raise ValueError("Invalid format")

            prompt_part = llm_part[:split_idx].strip()
            target_part = llm_part[split_idx + len(split_marker):].strip()
            
            # Remove outer quotes from prompt
            if len(prompt_part) >= 2 and prompt_part[0] in ("'", '"') and prompt_part[0] == prompt_part[-1]:
                base_prompt = prompt_part[1:-1]
            else:
                base_prompt = prompt_part.strip("'\"")

            target_val = target_part.strip().strip("'\"")
            
            df_filter = filter_part.strip()
        except ValueError:
             return {"results": [], "metadata": {}, "error": "Invalid query format. Expected: filter and LLM('prompt') == 'target'"}

        # 1. Apply pandas filter
        try:
            df = df.query(df_filter)
        except Exception as e:
             return {"results": [], "metadata": {}, "error": f"Pandas query error: {e}"}

        if limit is not None:
            df = df.head(limit)

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
                "inference_time": "0.0000 seconds"
            }

        # print(base_prompt)
        # print(df_filter)
        # print(target_val)

        # 2. Run LLM
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
        output_map = {idx: out.get("text", "").strip() for idx, out in zip(order, outputs)}
        df["llm_output"] = df.index.map(output_map)
        
        # Filter where output matches target
        df_final = df[df["llm_output"] == target_val]

        results = df_final.to_dict(orient="records")

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
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_data_loaded(self) -> None:

        if self.data is None:
            raise RuntimeError("Dataset not loaded. Please upload a CSV first.")
