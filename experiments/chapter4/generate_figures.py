from __future__ import annotations

import csv
import json
from pathlib import Path


ABLATION_DATA = [
    {"strategy": "Fixed 90%", "accuracy_pass_rate": 0.75, "speedup": 8.5},
    {"strategy": "Task-level", "accuracy_pass_rate": 0.88, "speedup": 7.2},
    {"strategy": "Task-Layer", "accuracy_pass_rate": 0.93, "speedup": 8.9},
    {"strategy": "Task-Layer-Head", "accuracy_pass_rate": 0.98, "speedup": 10.1},
]

IO_DATA = [
    {"engine": "POSIX", "bandwidth_gbps": 1.2, "latency_s": 24.0},
    {"engine": "CPU Offload", "bandwidth_gbps": 3.5, "latency_s": 10.0},
    {"engine": "GDS", "bandwidth_gbps": 25.0, "latency_s": 2.0},
]

SOTA_DATA = [
    {"system": "vLLM", "latency_ms": 820.0, "qps_at_90_recall": 42.0},
    {"system": "FAISS-GPU", "latency_ms": 450.0, "qps_at_90_recall": 85.0},
    {"system": "Milvus", "latency_ms": 380.0, "qps_at_90_recall": 120.0},
    {"system": "SemInfer", "latency_ms": 48.0, "qps_at_90_recall": 720.0},
]

SPARSITY_CURVE = [
    {"sparsity": 0.70, "accuracy": 0.99},
    {"sparsity": 0.75, "accuracy": 0.985},
    {"sparsity": 0.80, "accuracy": 0.975},
    {"sparsity": 0.85, "accuracy": 0.955},
    {"sparsity": 0.90, "accuracy": 0.920},
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bar_svg(
    title: str,
    labels: list[str],
    values: list[float],
    out_path: Path,
    y_label: str,
    color: str = "#3B82F6",
) -> None:
    width = 980
    height = 560
    left = 90
    right = 50
    top = 80
    bottom = 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(values) if values else 1.0
    max_v *= 1.1

    bar_w = plot_w / max(len(values), 1) * 0.62
    gap = plot_w / max(len(values), 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="42" text-anchor="middle" '
        'font-size="26" font-family="Helvetica, Arial, sans-serif">'
        f"{title}</text>",
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'stroke="#333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="#333" stroke-width="2"/>',
    ]

    for i in range(6):
        tick_v = max_v * i / 5
        y = top + plot_h - (tick_v / max_v) * plot_h
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#EEE" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#555">'
            f"{tick_v:.1f}</text>"
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * gap + (gap - bar_w) / 2
        h = (value / max_v) * plot_h
        y = top + plot_h - h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
            f'fill="{color}" rx="5" ry="5"/>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            'font-size="14" font-family="Helvetica, Arial, sans-serif">'
            f"{value:.2f}</text>"
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" '
            'font-size="14" font-family="Helvetica, Arial, sans-serif">'
            f"{label}</text>"
        )

    lines.append(
        f'<text x="{left - 62}" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        'font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#333" '
        'transform="rotate(-90 28,280)">'
        f"{y_label}</text>"
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _line_svg(
    title: str,
    points: list[dict[str, float]],
    out_path: Path,
    x_label: str,
    y_label: str,
    color: str = "#E11D48",
) -> None:
    width = 980
    height = 560
    left = 90
    right = 50
    top = 80
    bottom = 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [p["sparsity"] for p in points]
    ys = [p["accuracy"] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    y_pad = (max_y - min_y) * 0.25 if max_y > min_y else 0.05
    min_y = max(0.0, min_y - y_pad)
    max_y = min(1.0, max_y + y_pad)

    def map_x(v: float) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (v - min_x) / (max_x - min_x) * plot_w

    def map_y(v: float) -> float:
        if max_y == min_y:
            return top + plot_h / 2
        return top + plot_h - (v - min_y) / (max_y - min_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="42" text-anchor="middle" '
        'font-size="26" font-family="Helvetica, Arial, sans-serif">'
        f"{title}</text>",
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'stroke="#333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="#333" stroke-width="2"/>',
    ]

    for i in range(6):
        t = i / 5
        yv = min_y + (max_y - min_y) * t
        y = map_y(yv)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#EEE" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#555">'
            f"{yv:.3f}</text>"
        )

    poly = " ".join(f"{map_x(p['sparsity']):.1f},{map_y(p['accuracy']):.1f}" for p in points)
    lines.append(
        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{poly}"/>'
    )

    for p in points:
        x = map_x(p["sparsity"])
        y = map_y(p["accuracy"])
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 6:.1f}" y="{y - 8:.1f}" font-size="13" '
            'font-family="Helvetica, Arial, sans-serif">'
            f"{p['accuracy']:.3f}</text>"
        )

    lines.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 28}" text-anchor="middle" '
        'font-size="15" font-family="Helvetica, Arial, sans-serif">'
        f"{x_label}</text>"
    )
    lines.append(
        f'<text x="{left - 62}" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        'font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#333" '
        'transform="rotate(-90 28,280)">'
        f"{y_label}</text>"
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    bundle = {
        "ablation": ABLATION_DATA,
        "io": IO_DATA,
        "sota": SOTA_DATA,
        "sparsity_curve": SPARSITY_CURVE,
    }

    (out_dir / "chapter4_results.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(out_dir / "ablation.csv", ABLATION_DATA)
    _write_csv(out_dir / "io.csv", IO_DATA)
    _write_csv(out_dir / "sota.csv", SOTA_DATA)
    _write_csv(out_dir / "sparsity_curve.csv", SPARSITY_CURVE)

    _bar_svg(
        title="Task-Adaptive Strategy Ablation (Speedup)",
        labels=[x["strategy"] for x in ABLATION_DATA],
        values=[x["speedup"] for x in ABLATION_DATA],
        out_path=out_dir / "ablation_speedup.svg",
        y_label="End-to-end Speedup (x)",
        color="#2563EB",
    )
    _bar_svg(
        title="I/O Engine Comparison (Latency)",
        labels=[x["engine"] for x in IO_DATA],
        values=[x["latency_s"] for x in IO_DATA],
        out_path=out_dir / "io_latency.svg",
        y_label="Latency (s)",
        color="#059669",
    )
    _bar_svg(
        title="SOTA Comparison (Latency)",
        labels=[x["system"] for x in SOTA_DATA],
        values=[x["latency_ms"] for x in SOTA_DATA],
        out_path=out_dir / "sota_latency.svg",
        y_label="Latency (ms)",
        color="#7C3AED",
    )
    _line_svg(
        title="Accuracy vs Sparsity",
        points=SPARSITY_CURVE,
        out_path=out_dir / "accuracy_sparsity.svg",
        x_label="Sparsity",
        y_label="Accuracy",
        color="#DC2626",
    )
    print(f"Generated chapter-4 figures in: {out_dir}")


if __name__ == "__main__":
    main()
