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
        "diff": "analytics_diff.png",
        "sparsity": "analytics_sparsity.png",
        "kv": "analytics_kv.png",
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
        metrics = {
            "latency_s": latency,
            "transfer_ms": transfer_ms,
            "compute_ms": compute_ms,
            "use_index": effective_index,
            "prompt": clause.value,
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
        }
        self.analytics["queries"].append(entry)
        self.analytics["queries"] = self.analytics["queries"][-50:]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def analyse(self, _data: Any | None = None) -> dict[str, Any]:
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

        diff_points = []
        grouped: dict[str, dict[bool, dict[str, Any]]] = {}
        for entry in queries:
            grouped.setdefault(entry["signature"], {})[bool(entry["use_index"])] = entry
        for group in grouped.values():
            if True in group and False in group:
                idx_set = set(group[True]["result_ids"])
                base_set = set(group[False]["result_ids"])
                union = idx_set | base_set
                if not union:
                    continue
                diff = 1 - (len(idx_set & base_set) / len(union))
                diff_points.append(
                    {
                        "diff_ratio": diff,
                        "sparsity": group[True].get("sparsity"),
                        "query": group[True]["query"][:80],
                    }
                )

        sparsity_curve = [
            {
                "timestamp": item["timestamp"],
                "sparsity": item.get("sparsity"),
                "index_size_kb": (item.get("index_size_bytes", 0) / 1024),
            }
            for item in indexes
        ]

        kv_timeline = [
            {
                "timestamp": q["timestamp"],
                "transfer_ms": q.get("transfer_ms"),
                "compute_ms": q.get("compute_ms"),
                "use_index": q.get("use_index"),
            }
            for q in queries
            if q.get("transfer_ms") or q.get("compute_ms")
        ]
        charts = self._render_analytics_images(
            latency_bar=latency_bar,
            diff_points=diff_points,
            sparsity_curve=sparsity_curve,
            kv_timeline=kv_timeline,
        )
        summary = {
            "index_builds": len(indexes),
            "query_runs": len(queries),
            "generated_at": int(time.time()),
        }
        metrics = {
            "latency_bar": latency_bar,
            "diff_points": diff_points,
            "sparsity_curve": sparsity_curve,
            "kv_timeline": kv_timeline,
        }
        return {
            "images": charts,
            "summary": summary,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_analytics_images(
        self,
        *,
        latency_bar: dict[str, float],
        diff_points: list[dict[str, Any]],
        sparsity_curve: list[dict[str, Any]],
        kv_timeline: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        os.makedirs(self.PICS_DIR, exist_ok=True)
        version = int(time.time())
        statuses = {
            "latency": self._plot_latency_chart(latency_bar),
            "diff": self._plot_diff_chart(diff_points),
            "sparsity": self._plot_sparsity_chart(sparsity_curve),
            "kv": self._plot_kv_chart(kv_timeline),
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
        labels = ["使用索引", "未使用索引"]
        values = [float(latency_bar.get("with_index", 0)), float(latency_bar.get("without_index", 0))]
        has_data = any(value > 0 for value in values)
        if has_data:
            colors = ["#16a34a", "#f97316"]
            ax.bar(labels, values, color=colors, alpha=0.9)
            ax.set_ylabel("平均延迟 (ms)")
            ax.set_title("索引 VS 基准")
            for idx, value in enumerate(values):
                ax.text(idx, value + max(values) * 0.02, f"{value:.1f}", ha="center", fontsize=11)
        else:
            self._draw_placeholder(ax, "运行一次索引查询和一次基准查询以比较延迟。")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        self._finalize_chart(fig, self._chart_path("latency"))
        return {
            "has_data": has_data,
            "message": "运行一次索引查询与一次基准查询以比较延迟。",
            "alt": "Latency comparison",
        }

    def _plot_diff_chart(self, diff_points: list[dict[str, Any]]) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        if diff_points:
            xs = [max(0.0, min(1.0, float(point.get("diff_ratio") or 0))) for point in diff_points]
            ys = [float(point.get("sparsity") or 0) for point in diff_points]
            ax.scatter(xs, ys, s=70, c="#f97316", edgecolors="#c2410c", alpha=0.85)
            ax.set_xlabel("结果差异比例")
            ax.set_ylabel("稀疏度")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(1, max(ys + [0])))
            ax.set_title("索引漂移")
        else:
            self._draw_placeholder(ax, "运行索引与非索引查询以比较结果漂移。")
        ax.grid(True, linestyle="--", alpha=0.25)
        self._finalize_chart(fig, self._chart_path("diff"))
        return {
            "has_data": bool(diff_points),
            "message": "运行索引与非索引查询以比较结果漂移。",
            "alt": "Result drift scatter",
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
                ax.set_ylabel("稀疏度")
                ax.set_xlabel("构建时间")
                ax.set_title("索引稀疏度趋势")
                fig.autofmt_xdate(rotation=20)
            else:
                self._draw_placeholder(ax, "构建至少一次索引以查看稀疏度趋势。")
        else:
            self._draw_placeholder(ax, "构建至少一次索引以查看稀疏度趋势。")
        ax.grid(True, linestyle="--", alpha=0.25)
        self._finalize_chart(fig, self._chart_path("sparsity"))
        return {
            "has_data": bool(sparsity_curve),
            "message": "构建至少一次索引以查看稀疏度趋势。",
            "alt": "Sparsity timeline",
        }

    def _plot_kv_chart(self, kv_timeline: list[dict[str, Any]]) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        if kv_timeline:
            rows = [
                (
                    datetime.fromtimestamp(item.get("timestamp", 0)),
                    float(item.get("transfer_ms") or 0),
                    float(item.get("compute_ms") or 0),
                )
                for item in kv_timeline
                if item.get("timestamp")
            ]
            rows.sort(key=lambda row: row[0])
            if rows:
                xs = [row[0] for row in rows]
                transfer = [row[1] for row in rows]
                compute = [row[2] for row in rows]
                ax.plot(xs, transfer, label="传输", color="#2563eb", linewidth=2)
                ax.plot(xs, compute, label="计算", color="#facc15", linewidth=2)
                ax.set_ylabel("时间 (ms)")
                ax.set_xlabel("查询时间")
                ax.legend()
                ax.set_title("KV Cache 时间线")
                fig.autofmt_xdate(rotation=20)
            else:
                self._draw_placeholder(ax, "运行查询以收集 KV 传输与计算时间。")
        else:
            self._draw_placeholder(ax, "运行查询以收集 KV 传输与计算时间。")
        ax.grid(True, linestyle="--", alpha=0.25)
        self._finalize_chart(fig, self._chart_path("kv"))
        return {
            "has_data": bool(kv_timeline),
            "message": "运行查询以收集 KV 传输与计算时间。",
            "alt": "KV cache timeline",
        }

    def _ensure_data_loaded(self) -> None:
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Please upload a CSV first.")

