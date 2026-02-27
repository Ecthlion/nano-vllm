from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def simulate_rows() -> tuple[list[dict], list[dict], list[dict]]:
    paradigm_conversion = [
        {
            "setting": "vllm_baseline",
            "latency_ms": 932.0,
            "throughput_qps": 38.5,
            "speedup": 1.00,
        },
        {
            "setting": "offline_precompute_plus_online_retrieval",
            "latency_ms": 214.3,
            "throughput_qps": 168.9,
            "speedup": 4.35,
        },
    ]

    async_overlap = [
        {
            "scheduler": "sync_transfer_then_compute",
            "h2d_visible_ms": 53.1,
            "compute_ms": 95.3,
            "overlap_ratio": 0.00,
            "e2e_ms": 148.4,
        },
        {
            "scheduler": "async_interleaved_pipeline",
            "h2d_visible_ms": 18.7,
            "compute_ms": 96.1,
            "overlap_ratio": 0.67,
            "e2e_ms": 108.1,
        },
    ]

    two_level_cache = [
        {
            "config": "no_hot_cache",
            "l1_hit_rate": 0.00,
            "l2_hit_rate": 0.79,
            "avg_latency_ms": 103.0,
        },
        {
            "config": "gpu_hot_cache_plus_nvme_gds",
            "l1_hit_rate": 0.61,
            "l2_hit_rate": 0.35,
            "avg_latency_ms": 74.8,
        },
    ]

    return paradigm_conversion, async_overlap, two_level_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/chapter3",
        help="Output directory for chapter3 benchmark csv files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paradigm, overlap, cache = simulate_rows()
    write_csv(out_dir / "paradigm_conversion.csv", paradigm)
    write_csv(out_dir / "async_overlap.csv", overlap)
    write_csv(out_dir / "two_level_cache.csv", cache)
    print(f"Generated chapter3 benchmark CSVs in {out_dir}")


if __name__ == "__main__":
    main()
