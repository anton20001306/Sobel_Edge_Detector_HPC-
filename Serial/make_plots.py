#!/usr/bin/env python3
"""
make_plots.py
Generate comparison plots from one or more results_report.txt files
produced by run_pipeline.sh.

USAGE:
    python3 make_plots.py results_small.txt results_medium.txt results_large.txt

PRODUCES (in plots/):
    01_speedup_vs_workers.png/.pdf         Speedup curves, one subplot per image
    02_efficiency_vs_workers.png/.pdf      Efficiency curves, one subplot per image
    03_time_vs_image_size.png/.pdf         Execution time vs problem size (log-log)
    04_best_speedup_vs_size.png/.pdf       Best speedup vs problem size
    05_implementation_comparison.png/.pdf  Bar chart: best time per implementation

REQUIREMENTS:
    pip install matplotlib numpy --break-system-packages
"""

import os
import re
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required.")
    print("Install: pip install matplotlib numpy --break-system-packages")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_results(path):
    """Parse a results_report.txt file → dict of measurements."""
    with open(path) as f:
        text = f.read()

    r = {"path": path, "omp": {}, "mpi": {}, "hybrid": None}

    # Image dimensions
    m = re.search(r"Processed as\s*:\s*\S+\s*\[(\d+)x(\d+)\]", text)
    if not m:
        raise ValueError(f"Could not find image dimensions in {path}")
    w, h = int(m.group(1)), int(m.group(2))
    r["width"], r["height"] = w, h
    r["pixels"] = w * h
    r["megapixels"] = (w * h) / 1e6

    # Serial baseline
    m = re.search(r"Serial baseline\s*:\s*([\d.]+)", text)
    if not m:
        raise ValueError(f"Could not find serial baseline in {path}")
    r["serial"] = float(m.group(1))

    # OpenMP rows
    for m in re.finditer(
        r"OpenMP\s+(\d+)\s+threads?\s+([\d.]+)\s+([\d.]+)\s*x\s+([\d.]+)%", text
    ):
        n = int(m.group(1))
        r["omp"][n] = {
            "time": float(m.group(2)),
            "speedup": float(m.group(3)),
            "efficiency": float(m.group(4)),
        }

    # MPI rows  (NB: 'MPI+CUDA' contains no space, so this won't match the hybrid line)
    for m in re.finditer(
        r"MPI\s+(\d+)\s+processes?\s+([\d.]+)\s+([\d.]+)\s*x\s+([\d.]+)%", text
    ):
        n = int(m.group(1))
        r["mpi"][n] = {
            "time": float(m.group(2)),
            "speedup": float(m.group(3)),
            "efficiency": float(m.group(4)),
        }

    # Hybrid
    m = re.search(
        r"MPI\+CUDA Hybrid\s+(\d+)\s+ranks?\s+([\d.]+)\s+([\d.]+)\s*x\s+([\d.]+)%",
        text,
    )
    if m:
        r["hybrid"] = {
            "ranks": int(m.group(1)),
            "time": float(m.group(2)),
            "speedup": float(m.group(3)),
            "efficiency": float(m.group(4)),
        }
    return r


def label(r):
    return f"{r['width']}×{r['height']} ({r['megapixels']:.1f} MP)"


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, outdir, name):
    png = os.path.join(outdir, name + ".png")
    pdf = os.path.join(outdir, name + ".pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png}  +  {pdf}")


def plot_speedup_vs_workers(results, outdir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        omp_x = sorted(r["omp"].keys())
        omp_y = [r["omp"][k]["speedup"] for k in omp_x]
        mpi_x = sorted(r["mpi"].keys())
        mpi_y = [r["mpi"][k]["speedup"] for k in mpi_x]
        all_x = sorted(set(omp_x + mpi_x))

        ax.plot(all_x, all_x, "k--", alpha=0.4, label="Ideal (linear)")
        ax.plot(omp_x, omp_y, "o-", label="OpenMP",
                linewidth=2, markersize=8, color="#1f77b4")
        ax.plot(mpi_x, mpi_y, "s-", label="MPI",
                linewidth=2, markersize=8, color="#2ca02c")
        if r["hybrid"]:
            ax.axhline(r["hybrid"]["speedup"], color="#d62728", linestyle=":",
                       linewidth=2, label=f"Hybrid (4 ranks + GPU)")

        ax.set_xscale("log", base=2)
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_x)
        ax.set_xlabel("Workers (threads / processes)")
        if ax is axes[0]:
            ax.set_ylabel("Speedup vs Serial")
        ax.set_title(label(r))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Speedup vs Worker Count", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "01_speedup_vs_workers")


def plot_efficiency_vs_workers(results, outdir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        omp_x = sorted(r["omp"].keys())
        omp_y = [r["omp"][k]["efficiency"] for k in omp_x]
        mpi_x = sorted(r["mpi"].keys())
        mpi_y = [r["mpi"][k]["efficiency"] for k in mpi_x]
        all_x = sorted(set(omp_x + mpi_x))

        ax.axhline(100, color="k", linestyle="--", alpha=0.4, label="Ideal (100%)")
        ax.plot(omp_x, omp_y, "o-", label="OpenMP",
                linewidth=2, markersize=8, color="#1f77b4")
        ax.plot(mpi_x, mpi_y, "s-", label="MPI",
                linewidth=2, markersize=8, color="#2ca02c")

        ax.set_xscale("log", base=2)
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_x)
        ax.set_xlabel("Workers")
        if ax is axes[0]:
            ax.set_ylabel("Parallel Efficiency (%)")
        ax.set_title(label(r))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_ylim(0, 135)

    fig.suptitle("Parallel Efficiency vs Worker Count",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "02_efficiency_vs_workers")


def plot_time_vs_size(results, outdir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sizes = [r["megapixels"] for r in results]

    ax.plot(sizes, [r["serial"] for r in results],
            "D-", label="Serial", linewidth=2, markersize=8, color="#7f7f7f")

    omp_best = [min(v["time"] for v in r["omp"].values()) for r in results]
    ax.plot(sizes, omp_best, "o-", label="OpenMP (best)",
            linewidth=2, markersize=8, color="#1f77b4")

    mpi_best = [min(v["time"] for v in r["mpi"].values()) for r in results]
    ax.plot(sizes, mpi_best, "s-", label="MPI (best)",
            linewidth=2, markersize=8, color="#2ca02c")

    if all(r["hybrid"] for r in results):
        ax.plot(sizes, [r["hybrid"]["time"] for r in results],
                "*-", label="MPI+CUDA Hybrid",
                linewidth=2, markersize=14, color="#d62728")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Image Size (megapixels)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Execution Time vs Image Size (best configuration per implementation)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    _save(fig, outdir, "03_time_vs_image_size")


def plot_best_speedup_vs_size(results, outdir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sizes = [r["megapixels"] for r in results]

    omp_sp = [max(v["speedup"] for v in r["omp"].values()) for r in results]
    mpi_sp = [max(v["speedup"] for v in r["mpi"].values()) for r in results]

    ax.axhline(1, color="k", linestyle="--", alpha=0.4, label="Serial (1×)")
    ax.plot(sizes, omp_sp, "o-", label="OpenMP (best)",
            linewidth=2, markersize=8, color="#1f77b4")
    ax.plot(sizes, mpi_sp, "s-", label="MPI (best)",
            linewidth=2, markersize=8, color="#2ca02c")
    if all(r["hybrid"] for r in results):
        ax.plot(sizes, [r["hybrid"]["speedup"] for r in results],
                "*-", label="MPI+CUDA Hybrid",
                linewidth=2, markersize=14, color="#d62728")

    ax.set_xscale("log")
    ax.set_xlabel("Image Size (megapixels)")
    ax.set_ylabel("Best Speedup vs Serial")
    ax.set_title("Best Speedup vs Image Size", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    _save(fig, outdir, "04_best_speedup_vs_size")


def plot_implementation_comparison(results, outdir):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(results)
    x = np.arange(n)
    bw = 0.2

    serial = [r["serial"] for r in results]
    omp = [min(v["time"] for v in r["omp"].values()) for r in results]
    mpi = [min(v["time"] for v in r["mpi"].values()) for r in results]
    has_hybrid = all(r["hybrid"] for r in results)

    ax.bar(x - 1.5*bw, serial, bw, label="Serial", color="#7f7f7f")
    ax.bar(x - 0.5*bw, omp,    bw, label="OpenMP (best)",  color="#1f77b4")
    ax.bar(x + 0.5*bw, mpi,    bw, label="MPI (best)",     color="#2ca02c")
    if has_hybrid:
        hyb = [r["hybrid"]["time"] for r in results]
        ax.bar(x + 1.5*bw, hyb, bw, label="MPI+CUDA Hybrid", color="#d62728")

    # Annotate each bar with its value
    def annotate(xs, vals):
        for xi, v in zip(xs, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    annotate(x - 1.5*bw, serial)
    annotate(x - 0.5*bw, omp)
    annotate(x + 0.5*bw, mpi)
    if has_hybrid:
        annotate(x + 1.5*bw, hyb)

    ax.set_xticks(x)
    ax.set_xticklabels([label(r) for r in results])
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Best Time per Implementation, by Image Size",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    _save(fig, outdir, "05_implementation_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = sys.argv[1:]
    print(f"Parsing {len(paths)} results file(s)...")

    results = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  WARN: {p} not found, skipping.")
            continue
        try:
            r = parse_results(p)
            results.append(r)
            print(f"  Parsed: {p} → {label(r)}, "
                  f"serial={r['serial']:.4f}s, "
                  f"omp={len(r['omp'])} runs, mpi={len(r['mpi'])} runs, "
                  f"hybrid={'yes' if r['hybrid'] else 'no'}")
        except Exception as e:
            print(f"  ERROR parsing {p}: {e}")

    if not results:
        print("No valid results files. Exiting.")
        sys.exit(1)

    # Sort by image size (smallest first)
    results.sort(key=lambda r: r["pixels"])

    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    print(f"\nGenerating plots in '{outdir}/' ...")
    plot_speedup_vs_workers(results, outdir)
    plot_efficiency_vs_workers(results, outdir)
    if len(results) >= 2:
        plot_time_vs_size(results, outdir)
        plot_best_speedup_vs_size(results, outdir)
        plot_implementation_comparison(results, outdir)
    else:
        print("  (Skipping size-comparison plots: only 1 results file given.)")

    print(f"\nDone. All plots in {outdir}/")


if __name__ == "__main__":
    main()
