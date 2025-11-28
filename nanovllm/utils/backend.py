import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font  # type: ignore[import]

import pandas as pd

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


def _font_supports_chars(font_path: str, sample: str) -> bool:
    try:
        font = ft2font.FT2Font(font_path)
    except OSError:
        return False
    charmap = font.get_charmap()
    targets = {ord(char) for char in sample if char.strip()}
    return targets.issubset(charmap.keys())


def _configure_matplotlib_fonts() -> None:
    sample_text = "使用索引未稀疏度查询构建结果漂移"
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "SimHei",
    ]

    def pick_font() -> tuple[str | None, str | None]:
        for font_name in preferred_fonts:
            try:
                font_path = font_manager.findfont(font_name, fallback_to_default=False)
            except ValueError:
                continue
            if font_path and _font_supports_chars(font_path, sample_text):
                return font_name, font_path
        for entry in font_manager.fontManager.ttflist:
            font_path = getattr(entry, "fname", None)
            if not font_path or not os.path.exists(font_path):
                continue
            try:
                if _font_supports_chars(font_path, sample_text):
                    font_name = getattr(entry, "name", None) or os.path.basename(font_path)
                    return font_name, font_path
            except OSError:
                continue
        return None, None

    font_name, font_path = pick_font()
    if font_name:
        matplotlib.rcParams["font.family"] = [font_name, "sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        print(f"[backend] Matplotlib will render charts using '{font_name}' ({font_path}).")
    else:
        print("[backend] Warning: No CJK-capable font found; charts may show missing glyphs.")
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()


@dataclass
class Clause:
    kind: str
    value: str
    operator: str | None = None
    expected: str | None = None


class BackendAPI:
    CLAUSE_SPLIT_PATTERN = re.compile(r"\s+(and|or|&&|\|\|)\s+", re.IGNORECASE)
    LLM_PATTERN = re.compile(
        r"LLM\((?P<quote>['\"])(?P<prompt>.*?)\1\)\s*(?P<op>==|!=)\s*(?P<label_quote>['\"])(?P<label>.*?)\4",
        re.IGNORECASE,
    )
    TEMPLATE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PICS_DIR = os.path.join(PROJECT_ROOT, "pics")
    CHART_FILES = {
        "latency": "analytics_latency.png",
        "recall": "analytics_recall.png",
        "trace": "analytics_trace.png",
    }

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
        self.analytics: dict[str, list[dict[str, Any]]] = {"indexes": [], "queries": []}
        os.makedirs(self.PICS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Data Loading & Indexing
    # ------------------------------------------------------------------
    def load_data(self, data_path: str) -> dict[str, Any]:
        df = pd.read_csv(data_path)
        df["__id"] = range(len(df))
        self.data = df
        self.data_path = data_path
        self.index_ready = False
        self.text_field = self._infer_text_field(df)
        self.analytics["queries"].clear()
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "text_field": self.text_field,
        }

    def _infer_text_field(self, df: pd.DataFrame) -> str:
        for column in df.columns:
            if column == "__id":
                continue
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
        subset = self.data[["__id", field]]  # type: ignore[index]
        if limit is not None:
            subset = subset.head(limit)

        samples: list[tuple[int, str]] = []
        for _, row in subset.iterrows():
            text_id = int(row["__id"])
            text = str(row[field])
            prompt = f"{text}\n "
            samples.append((text_id, prompt))

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
        self._append_index_meta(meta)
        return meta

    def _estimate_index_size(self) -> int:
        total = 0
        for item in self.llm.kv_cache_index.kv_cache_index.values():
            kv_tensor = item.get("kv") if isinstance(item, dict) else item
            if hasattr(kv_tensor, "nelement") and hasattr(kv_tensor, "element_size"):
                total += int(kv_tensor.nelement()) * int(kv_tensor.element_size())  # type: ignore[attr-defined]
        return total

    def _append_index_meta(self, meta: dict[str, Any]) -> None:
        self.analytics["indexes"].append({
            "timestamp": time.time(),
            **meta,
        })
        self.analytics["indexes"] = self.analytics["indexes"][-20:]
        # self._update_charts()

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def query(self, query: str, use_index: bool, limit: int | None = 50) -> dict[str, Any]:
        self._ensure_data_loaded()
        df = self.data.copy()  # type: ignore[assignment]
        clauses, connectors = self._parse_query(query)

        clause_masks: list[pd.Series] = []
        llm_metrics: list[dict[str, Any]] = []
        expr_metrics: list[dict[str, Any]] = []

        if not clauses:
            clause_masks.append(pd.Series([True] * len(df), index=df.index))

        for clause in clauses:
            if clause.kind == "llm":
                mask, metrics = self._execute_llm_clause(df, clause, use_index)
                clause_masks.append(mask)
                llm_metrics.append(metrics)
            else:
                mask, metrics = self._execute_expr_clause(df, clause.value)
                clause_masks.append(mask)
                expr_metrics.append(metrics)

        final_mask = clause_masks[0]
        for connector, mask in zip(connectors, clause_masks[1:]):
            if connector in {"and", "&&"}:
                final_mask = final_mask & mask
            else:
                final_mask = final_mask | mask

        final_mask = final_mask.fillna(False)
        filtered = df.loc[final_mask]
        total_matches = len(filtered)
        preview = filtered if limit is None else filtered.head(limit)

        inference_time = sum(m.get("latency_s", 0.0) for m in llm_metrics) + sum(
            m.get("latency_s", 0.0) for m in expr_metrics
        )

        trace = self._build_trace_events(llm_metrics, expr_metrics)
        preview_for_user = preview.copy()
        if "__id" in preview_for_user.columns:
            preview_for_user = preview_for_user.rename(columns={"__id": "row_id"})
        results = preview_for_user.to_dict(orient="records")
        metadata = {
            "total_rows": len(df),
            "matched_rows": total_matches,
            "returned_rows": len(preview),
            "use_index": any(m.get("use_index") for m in llm_metrics),
            "text_field": self.text_field,
            "limit": limit,
        }

        signature = hashlib.sha1(query.encode("utf-8")).hexdigest()
        self._append_query_log(
            query,
            signature,
            llm_metrics,
            filtered["__id"].tolist() if "__id" in filtered.columns else [],
        )
        # self._update_charts()

        return {
            "results": results,
            "metadata": metadata,
            "inference_time": f"{inference_time:.4f} seconds",
            "trace_data": trace,
        }

    def _execute_llm_clause(
        self, df: pd.DataFrame, clause: Clause, use_index: bool
    ) -> tuple[pd.Series, dict[str, Any]]:
        start = time.perf_counter()
        if df.empty:
            return pd.Series([], dtype=bool), {
                "latency_s": 0.0,
                "transfer_ms": 0.0,
                "compute_ms": 0.0,
                "use_index": False,
                "prompt": clause.value,
            }

        effective_index = bool(use_index and self.index_ready)
        tuple_prompts: list[tuple[int, str]] = []
        string_prompts: list[str] = []
        params: list[SamplingParams] = []
        order: list[Any] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            context_value = str(row_dict.get(self.text_field, "")) if self.text_field else ""
            rendered_prompt = self._render_prompt_template(clause.value, row_dict)
            # Guarantee at least one character for the task tail
            task_suffix = rendered_prompt or "Answer:"
            full_prompt = f"{context_value}\n\n{task_suffix}"
            if effective_index:
                tuple_prompts.append((int(row_dict.get("__id", idx)), full_prompt))
            else:
                string_prompts.append(full_prompt)
            sp = SamplingParams(
                temperature=self.base_sampling.temperature,
                max_tokens=self.base_sampling.max_tokens,
            )
            sp.task_str_len = max(len(task_suffix), 1)
            params.append(sp)
            order.append(idx)

        prompt_payload: list[str] | list[tuple[int, str]] = (
            tuple_prompts if effective_index else string_prompts
        )
        outputs = self.llm.generate(
            prompt_payload,
            params,
            use_index=effective_index,
            use_tqdm=False,
            pruning=False,
            sparsity=self.current_sparsity or 0.9,
        )

        latency = time.perf_counter() - start
        stats = getattr(self.llm, "last_run_stats", None) or {}
        transfer_ms = stats.get("avg_transfer_ms", 0.0)
        compute_ms = stats.get("avg_compute_ms", latency * 1000)

        normalized_expected = self._normalize_prediction(clause.expected or "")
        series_data = []
        for idx, output in zip(order, outputs):
            prediction = self._normalize_prediction(output.get("text", ""))
            if clause.operator == "!=":
                series_data.append(prediction != normalized_expected)
            else:
                series_data.append(prediction == normalized_expected)

        mask = pd.Series(series_data, index=order)
        accuracy = sum(series_data) / len(series_data) if series_data else 0.0
        metrics = {
            "latency_s": latency,
            "transfer_ms": transfer_ms,
            "compute_ms": compute_ms,
            "use_index": effective_index,
            "prompt": clause.value,
            "accuracy": accuracy,
        }
        return mask, metrics

    def _execute_expr_clause(self, df: pd.DataFrame, expr: str) -> tuple[pd.Series, dict[str, Any]]:
        start = time.perf_counter()
        result = df.eval(expr, engine="python")
        duration = time.perf_counter() - start
        if isinstance(result, pd.Series):
            mask = result.reindex(df.index).fillna(False).astype(bool)
        else:
            mask = pd.Series(bool(result), index=df.index)
        metrics = {
            "latency_s": duration,
            "label": expr,
        }
        return mask, metrics

    def _parse_query(self, query: str) -> tuple[list[Clause], list[str]]:
        parts = self.CLAUSE_SPLIT_PATTERN.split(query)
        clauses: list[Clause] = []
        connectors: list[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                clause = part.strip().strip("()")
                if not clause:
                    continue
                match = self.LLM_PATTERN.match(clause)
                if match:
                    clauses.append(
                        Clause(
                            kind="llm",
                            value=match.group("prompt"),
                            operator=match.group("op"),
                            expected=match.group("label"),
                        )
                    )
                else:
                    clauses.append(Clause(kind="expr", value=clause))
            else:
                connectors.append(part.strip().lower())
        return clauses, connectors

    def _render_prompt_template(self, template: str, row: dict[str, Any]) -> str:
        row_with_defaults = {**row}
        if self.text_field and self.text_field not in row_with_defaults:
            row_with_defaults[self.text_field] = ""
        if self.text_field:
            default_text = str(row_with_defaults.get(self.text_field, ""))
        else:
            default_text = ""
        row_with_defaults.setdefault("text", default_text)

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(row_with_defaults.get(key, ""))

        return self.TEMPLATE_PATTERN.sub(repl, template)

    def _normalize_prediction(self, text: str) -> str:
        return text.strip().strip("\n").lower()

    def _build_trace_events(
        self,
        llm_metrics: Iterable[dict[str, Any]],
        expr_metrics: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        events = []
        ts = 0
        pid = 1
        tid_transfer = 1
        tid_compute = 2
        tid_filter = 3

        for metric in llm_metrics:
            transfer = max(metric.get("transfer_ms", 0.0), 0.0) * 1000
            compute = max(metric.get("compute_ms", 0.0), 0.0) * 1000
            if transfer:
                events.append(
                    {
                        "ph": "X",
                        "name": "KV Cache Transfer",
                        "ts": ts,
                        "dur": transfer,
                        "pid": pid,
                        "tid": tid_transfer,
                        "args": {"prompt": metric.get("prompt")},
                    }
                )
                ts += transfer
            if compute:
                events.append(
                    {
                        "ph": "X",
                        "name": "Computation",
                        "ts": ts,
                        "dur": compute,
                        "pid": pid,
                        "tid": tid_compute,
                        "args": {"prompt": metric.get("prompt")},
                    }
                )
                ts += compute

        for metric in expr_metrics:
            dur = metric.get("latency_s", 0.0) * 1_000_000
            if dur == 0:
                continue
            events.append(
                {
                    "ph": "X",
                    "name": "Pandas Filter",
                    "ts": ts,
                    "dur": dur,
                    "pid": pid,
                    "tid": tid_filter,
                    "args": {"expr": metric.get("label")},
                }
            )
            ts += dur

        return {"traceEvents": events}

    def _append_query_log(
        self,
        query: str,
        signature: str,
        llm_metrics: list[dict[str, Any]],
        result_ids: list[int],
    ) -> None:
        latency = sum(metric.get("latency_s", 0.0) for metric in llm_metrics)
        accuracies = [m.get("accuracy", 0.0) for m in llm_metrics if "accuracy" in m]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None

        entry = {
            "timestamp": time.time(),
            "query": query,
            "signature": signature,
            "use_index": any(m.get("use_index") for m in llm_metrics),
            "latency_ms": latency * 1000,
            "transfer_ms": sum(m.get("transfer_ms", 0.0) for m in llm_metrics),
            "compute_ms": sum(m.get("compute_ms", 0.0) for m in llm_metrics),
            "result_ids": result_ids,
            "sparsity": self.current_sparsity,
            "accuracy": avg_accuracy,
            "llm_metrics": llm_metrics,  # Store detailed metrics for trace reconstruction
        }
        self.analytics["queries"].append(entry)
        self.analytics["queries"] = self.analytics["queries"][-50:]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def analyse(self, _data: Any | None = None) -> dict[str, Any]:
        # Just return the current state and image URLs.
        # Images are updated by _update_charts() called in query/build_index.
        queries = self.analytics["queries"]
        indexes = self.analytics["indexes"]
        
        summary = {
            "index_builds": len(indexes),
            "query_runs": len(queries),
            "generated_at": int(time.time()),
        }
        
        # Construct payload pointing to existing images
        version = int(time.time())
        images = {}
        for key, filename in self.CHART_FILES.items():
            # Check if file exists to set has_data flag roughly
            filepath = os.path.join(self.PICS_DIR, filename)
            has_data = os.path.exists(filepath)
            images[key] = {
                "url": f"/pics/{filename}",
                "filename": filename,
                "version": version,
                "has_data": has_data,
                "message": "" if has_data else "No data available yet.",
                "alt": key.capitalize(),
            }

        return {
            "images": images,
            "summary": summary,
            "metrics": {}, # Metrics are less important for the frontend now if images are pre-rendered
        }

    def _update_charts(self) -> None:
        queries = self.analytics["queries"]
        indexes = self.analytics["indexes"]
        latency_with = [q["latency_ms"] for q in queries if q.get("use_index")]
        latency_without = [q["latency_ms"] for q in queries if not q.get("use_index")]

        latency_bar = {
            "with_index": sum(latency_with) / len(latency_with) if latency_with else 0,
            "without_index": sum(latency_without) / len(latency_without)
            if latency_without
            else 0,
        }

        recall_map: dict[float, list[float]] = {}
        for q in queries:
            if q.get("accuracy") is not None and q.get("sparsity") is not None:
                s = float(q["sparsity"])
                acc = float(q["accuracy"])
                if s not in recall_map:
                    recall_map[s] = []
                recall_map[s].append(acc)

        recall_curve = []
        for s in sorted(recall_map.keys()):
            avg_acc = sum(recall_map[s]) / len(recall_map[s])
            recall_curve.append({"sparsity": s, "recall": avg_acc})

        sparsity_curve = [
            {
                "timestamp": item["timestamp"],
                "sparsity": item.get("sparsity"),
                "index_size_kb": (item.get("index_size_bytes", 0) / 1024),
            }
            for item in indexes
        ]

        last_query = queries[-1] if queries else None
        trace_segments = []
        if last_query:
            metrics = last_query.get("llm_metrics", [])
            current_time = 0.0
            for m in metrics:
                t = m.get("transfer_ms", 0.0)
                c = m.get("compute_ms", 0.0)
                if t > 0:
                    trace_segments.append(("Transfer", current_time, t))
                    current_time += t
                if c > 0:
                    trace_segments.append(("Compute", current_time, c))
                    current_time += c

        self._render_analytics_images(
            latency_bar=latency_bar,
            recall_curve=recall_curve,
            sparsity_curve=sparsity_curve,
            trace_segments=trace_segments,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_analytics_images(
        self,
        *,
        latency_bar: dict[str, float],
        recall_curve: list[dict[str, float]],
        sparsity_curve: list[dict[str, Any]],
        trace_segments: list[tuple[str, float, float]],
    ) -> dict[str, dict[str, Any]]:
        os.makedirs(self.PICS_DIR, exist_ok=True)
        version = int(time.time())
        statuses = {
            "latency": self._plot_latency_chart(latency_bar),
            "recall": self._plot_recall_chart(recall_curve),
            "sparsity": self._plot_sparsity_chart(sparsity_curve),
            "trace": self._plot_trace_chart(trace_segments),
        }
        payload: dict[str, dict[str, Any]] = {}
        for key, status in statuses.items():
            filename = self.CHART_FILES[key]
            payload[key] = {
                "url": f"/pics/{filename}",
                "filename": filename,
                "version": version,
                "has_data": status.get("has_data", False),
                "message": status.get("message", ""),
                "alt": status.get("alt"),
            }
        return payload

    def _chart_path(self, key: str) -> str:
        filename = self.CHART_FILES[key]
        return os.path.join(self.PICS_DIR, filename)

    def _finalize_chart(self, fig, filepath: str) -> None:
        fig.tight_layout()
        fig.savefig(filepath, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _draw_placeholder(self, ax, message: str) -> None:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=11,
            color="#475569",
            wrap=True,
        )

    def _plot_latency_chart(self, latency_bar: dict[str, float]) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        labels = ["Indexed", "Baseline"]
        values = [float(latency_bar.get("with_index", 0)), float(latency_bar.get("without_index", 0))]
        has_data = any(value > 0 for value in values)
        if has_data:
            colors = ["#16a34a", "#f97316"]
            ax.bar(labels, values, color=colors, alpha=0.9)
            ax.set_ylabel("Avg Latency (ms)")
            ax.set_title("Index vs Baseline")
            for idx, value in enumerate(values):
                ax.text(idx, value + max(values) * 0.02, f"{value:.1f}", ha="center", fontsize=11)
        else:
            self._draw_placeholder(ax, "Run one indexed and one baseline query to compare latency.")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        self._finalize_chart(fig, self._chart_path("latency"))
        return {
            "has_data": has_data,
            "message": "Run one indexed and one baseline query to compare latency.",
            "alt": "Latency comparison",
        }

    def _plot_recall_chart(self, recall_curve: list[dict[str, float]]) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        if recall_curve:
            xs = [item["sparsity"] for item in recall_curve]
            ys = [item["recall"] for item in recall_curve]
            ax.plot(xs, ys, marker="o", linestyle="-", color="#f97316", linewidth=2)
            ax.set_xlabel("Sparsity")
            ax.set_ylabel("Recall (Accuracy)")
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 1.0)
            ax.set_title("Sparsity vs Recall")
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=9)
        else:
            self._draw_placeholder(ax, "Run queries with different sparsity to generate recall curve.")
        ax.grid(True, linestyle="--", alpha=0.25)
        self._finalize_chart(fig, self._chart_path("recall"))
        return {
            "has_data": bool(recall_curve),
            "message": "Run queries with different sparsity to generate recall curve.",
            "alt": "Recall vs Sparsity",
        }

    def _plot_sparsity_chart(self, sparsity_curve: list[dict[str, Any]]) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        if sparsity_curve:
            points = [
                (
                    datetime.fromtimestamp(item.get("timestamp", 0)),
                    float(item.get("sparsity") or 0),
                )
                for item in sparsity_curve
                if item.get("timestamp")
            ]
            points.sort(key=lambda pair: pair[0])
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, color="#0f172a", linewidth=2.2, marker="o")
                ax.set_ylim(0, 1)
                ax.set_ylabel("Sparsity")
                ax.set_xlabel("Build Time")
                ax.set_title("Index Sparsity Trend")
                fig.autofmt_xdate(rotation=20)
            else:
                self._draw_placeholder(ax, "Build at least one index to see sparsity trend.")
        else:
            self._draw_placeholder(ax, "Build at least one index to see sparsity trend.")
        ax.grid(True, linestyle="--", alpha=0.25)
        self._finalize_chart(fig, self._chart_path("sparsity"))
        return {
            "has_data": bool(sparsity_curve),
            "message": "Build at least one index to see sparsity trend.",
            "alt": "Sparsity timeline",
        }

    def _plot_trace_chart(self, trace_segments: list[tuple[str, float, float]]) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        if trace_segments:
            transfers = [
                (start, dur) for label, start, dur in trace_segments if label == "Transfer"
            ]
            computes = [
                (start, dur) for label, start, dur in trace_segments if label == "Compute"
            ]

            if transfers:
                ax.broken_barh(transfers, (10, 9), facecolors="#2563eb", label="Transfer")
            if computes:
                ax.broken_barh(computes, (20, 9), facecolors="#facc15", label="Compute")

            ax.set_ylim(5, 35)
            ax.set_yticks([14.5, 24.5])
            ax.set_yticklabels(["Transfer", "Compute"])
            ax.set_xlabel("Time (ms)")
            ax.set_title("KV Transfer vs Compute Ratio (Last Query)")
            ax.legend(loc="upper right")

            total_transfer = sum(d for _, d in transfers)
            total_compute = sum(d for _, d in computes)
            total = total_transfer + total_compute
            if total > 0:
                ratio_text = f"Transfer: {total_transfer/total:.1%}\nCompute: {total_compute/total:.1%}"
                ax.text(
                    0.98,
                    0.02,
                    ratio_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        else:
            self._draw_placeholder(ax, "Run a query to see KV transfer vs compute ratio.")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        self._finalize_chart(fig, self._chart_path("trace"))
        return {
            "has_data": bool(trace_segments),
            "message": "Run a query to see KV transfer vs compute ratio.",
            "alt": "Trace Gantt Chart",
        }

    def _ensure_data_loaded(self) -> None:
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Please upload a CSV first.")

