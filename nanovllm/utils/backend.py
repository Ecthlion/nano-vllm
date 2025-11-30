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

        if limit is not None:
            df = df.head(limit)

        # Query example:
        # sentiment == "positive" and LLM('Given the above film review, answer whether it contains names. Respond ONLY with \"yes\" or \"no\", in all lower case.\n') == 'yes'

        # Simplify parser, this is a example, you should correct the logic
        # don't consider any other situation

        df_filter = 'sentiment=="positive"'  # only get filter before 'and'
        base_prompt = 'Given the above film review, answer whether it contains names. Respond ONLY with "yes" or "no", in all lower case.\n'  # only get base_prompt inside LLM('')
        target = "yes"  # only get target after LLM('') ==

        df = df.query(df_filter)
        self.base_sampling.task_str_len = len(base_prompt)

        effective_index = bool(use_index and self.index_ready)
        tuple_prompts: list[tuple[int, str]] = []
        order: list[Any] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            context_value = (
                str(row_dict.get(self.text_field, "")) if self.text_field else ""
            )
            full_prompt = f"{context_value}\n{base_prompt}"

            tuple_prompts.append((idx, full_prompt))  # type: ignore
            order.append(idx)

        outputs = self.llm.generate(
            tuple_prompts,
            self.base_sampling,
            use_index=effective_index,
            use_tqdm=False,
            pruning=False,
            sparsity=self.current_sparsity or 0.9,
        )

        # FIX: should filter according to the target
        output_map = {idx: out.get("text", "") for idx, out in zip(order, outputs)}
        df["llm_output"] = df.index.map(output_map)
        results = df.to_dict(orient="records")

        metadata = {
            "total_rows": len(self.data) if self.data is not None else 0,
            "returned_rows": len(df),
            "use_index": effective_index,
            "text_field": self.text_field,
            "limit": limit,
        }

        return {
            "results": results,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_data_loaded(self) -> None:

        if self.data is None:
            raise RuntimeError("Dataset not loaded. Please upload a CSV first.")
