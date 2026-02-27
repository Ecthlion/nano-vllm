from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _style_axes(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("#e6e6e6")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax.tick_params(axis="both", which="both", width=1.2, length=5, colors="black")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)


def _set_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _save_fig(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_point1_pipeline(data_dir: Path, out_dir: Path, tag: str) -> None:
    path = data_dir / "point1_pipeline_vs_vllm.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    x = np.arange(len(df))
    latency = df["latency_ms"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(
        x,
        latency,
        width=0.66,
        edgecolor="#0b1a8f",
        linewidth=1.8,
        color="white",
        hatch="x",
    )
    for b in bars:
        b.set_facecolor("#f8f8f8")

    ax.set_xticks(x)
    ax.set_xticklabels(df["stage"].tolist(), rotation=12, ha="right")
    _style_axes(ax, xlabel="方案", ylabel="延迟 (ms)")
    ax.set_title(f"({tag}) 范式与GDS分阶段延迟", fontsize=14)

    _save_fig(fig, out_dir / "point1_pipeline_latency")


def plot_point1_transfer(data_dir: Path, out_dir: Path, tag: str) -> None:
    path = data_dir / "point1_transfer_breakdown.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    x = np.arange(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(
        x - width,
        df["disk_to_cpu_ms"].to_numpy(),
        width=width,
        edgecolor="#0b1a8f",
        linewidth=1.6,
        color="white",
        hatch="x",
        label="Disk→CPU",
    )
    ax.bar(
        x,
        df["cpu_to_gpu_ms"].to_numpy(),
        width=width,
        edgecolor="#a3543d",
        linewidth=1.6,
        color="white",
        hatch="\\",
        label="CPU→GPU",
    )
    ax.bar(
        x + width,
        df["sync_overhead_ms"].to_numpy(),
        width=width,
        edgecolor="#666666",
        linewidth=1.6,
        color="white",
        hatch="/",
        label="同步开销",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["stage"].tolist(), rotation=12, ha="right")
    _style_axes(ax, xlabel="方案", ylabel="时间 (ms)")
    ax.set_title(f"({tag}) 传输链路分解", fontsize=14)
    ax.legend(frameon=False, loc="upper right")

    _save_fig(fig, out_dir / "point1_transfer_breakdown")


def plot_point2_task_adaptive(data_dir: Path, out_dir: Path, tag: str) -> None:
    path = data_dir / "point2_task_adaptive_sampling.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    x = np.arange(len(df))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax = axes[0]
    ax.bar(
        x - width / 2,
        df["accuracy_loss_static_pct"].to_numpy(),
        width=width,
        edgecolor="#0b1a8f",
        linewidth=1.6,
        color="white",
        hatch="x",
        label="静态稀疏",
    )
    ax.bar(
        x + width / 2,
        df["accuracy_loss_adapted_pct"].to_numpy(),
        width=width,
        edgecolor="#a3543d",
        linewidth=1.6,
        color="white",
        hatch="\\",
        label="采样自适应",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["task_type"].tolist(), rotation=20, ha="right")
    _style_axes(ax, xlabel="任务", ylabel="精度损失 (%)")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.bar(
        x - width / 2,
        df["transfer_gb_static"].to_numpy(),
        width=width,
        edgecolor="#0b1a8f",
        linewidth=1.6,
        color="white",
        hatch="x",
        label="静态稀疏",
    )
    ax.bar(
        x + width / 2,
        df["transfer_gb_adapted"].to_numpy(),
        width=width,
        edgecolor="#a3543d",
        linewidth=1.6,
        color="white",
        hatch="\\",
        label="采样自适应",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["task_type"].tolist(), rotation=20, ha="right")
    _style_axes(ax, xlabel="任务", ylabel="传输量 (GB)")

    fig.suptitle(f"({tag}) 任务感知动态稀疏效果", fontsize=14)
    _save_fig(fig, out_dir / "point2_task_adaptive")


def plot_point2_write_amp(data_dir: Path, out_dir: Path, tag: str) -> None:
    path = data_dir / "point2_csr_kv_write_amplification.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    grouped = (
        df.groupby("sparsity")[["write_amplification_dense", "write_amplification_csr_kv"]]
        .mean()
        .reset_index()
        .sort_values("sparsity")
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        grouped["sparsity"],
        grouped["write_amplification_dense"],
        color="#0b1a8f",
        linewidth=2.2,
        marker="o",
        markersize=6,
        label="Dense 存储",
    )
    ax.plot(
        grouped["sparsity"],
        grouped["write_amplification_csr_kv"],
        color="#a3543d",
        linewidth=2.2,
        marker="s",
        markersize=6,
        label="CSR-KV",
    )

    _style_axes(ax, xlabel="稀疏度", ylabel="写放大倍数")
    ax.set_title(f"({tag}) CSR-KV 抑制写放大", fontsize=14)
    ax.legend(frameon=False, loc="upper left")

    _save_fig(fig, out_dir / "point2_write_amplification")


def plot_overall_summary(data_dir: Path, out_dir: Path, tag: str) -> None:
    path = data_dir / "overall_stack_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(
        x,
        df["speedup_vs_vllm"].to_numpy(),
        width=0.62,
        edgecolor="#0b1a8f",
        linewidth=1.8,
        color="white",
        hatch="x",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["system"].tolist())
    _style_axes(ax, xlabel="系统", ylabel="相对加速比")
    ax.set_title(f"({tag}) 全栈加速对比", fontsize=14)

    _save_fig(fig, out_dir / "overall_speedup")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="实验")
    args = parser.parse_args()

    _set_font()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    plot_point1_pipeline(data_dir, out_dir, args.tag)
    plot_point1_transfer(data_dir, out_dir, args.tag)
    plot_point2_task_adaptive(data_dir, out_dir, args.tag)
    plot_point2_write_amp(data_dir, out_dir, args.tag)
    plot_overall_summary(data_dir, out_dir, args.tag)

    print(f"Figures generated in {out_dir}")


if __name__ == "__main__":
    main()
