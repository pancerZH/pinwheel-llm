"""Mini scaling law: fit val BPB against training compute, taking the best learning rate per model size.

    uv run python scripts/scaling.py [-o runs/scaling.png]

Reads the finished runs in runs/results.tsv (the 50M runs 1c and 1e-v0-lr2e-3, and side-scaling-*) and their config.json.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

from plot import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402

LRS = [1e-3, 2e-3, 4e-3, 8e-3]  # fixed color slot per learning rate


def load_runs():
    rows = csv.DictReader(open("runs/results.tsv"), delimiter="\t")
    runs = []
    for r in rows:
        if not (r["run_id"] in ("1c-v0-fineweb-1b", "1e-v0-lr2e-3") or r["run_id"].startswith("side-scaling-")):
            continue
        cfg = json.loads((Path("runs") / r["run_id"] / "config.json").read_text())
        n, m, t = cfg["params"], cfg["model"], cfg["train"]
        tokens = int(t["max_tokens"] // t["batch_tokens"]) * t["batch_tokens"]
        flops_per_token = 6 * n + 12 * m["n_layer"] * m["d_model"] * t["seq_len"]  # as model.flops_per_token
        runs.append(dict(name=r["run_id"], params=n, lr=t["lr"], tokens=tokens,
                         compute=flops_per_token * tokens, bpb=float(r["val_bpb"])))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="runs/scaling.png")
    args = p.parse_args()

    by_size = defaultdict(list)
    for r in load_runs():
        by_size[r["params"]].append(r)
    best = [min(rs, key=lambda r: r["bpb"]) for _, rs in sorted(by_size.items())]

    print(f"{'params':>8} {'tokens':>7} {'compute':>9}  " + "  ".join(f"lr {lr:.0e}" for lr in LRS))
    for n, rs in sorted(by_size.items()):
        bpb = {r["lr"]: r["bpb"] for r in rs}
        cells = "  ".join(f"{bpb[lr]:8.4f}" if lr in bpb else f"{'-':>8}" for lr in LRS)
        print(f"{n / 1e6:7.2f}M {rs[0]['tokens'] / 1e6:6.0f}M {rs[0]['compute']:9.2e}  {cells}")

    # Pure power law BPB = a * C^-b, a straight line in log-log space.
    c, y = np.array([r["compute"] for r in best]), np.array([r["bpb"] for r in best])
    slope, intercept = np.polyfit(np.log(c), np.log(y), 1)
    a, b = np.exp(intercept), -slope
    fit = a * c ** -b
    print(f"\nfit on the best LR per size: BPB = {a:.3g} * C^-{b:.4f}")
    print(f"each 2x compute multiplies BPB by {2 ** -b:.4f} ({(1 - 2 ** -b) * 100:.1f}% lower)")
    print("residuals: " + "  ".join(f"{r['params'] / 1e6:.1f}M {yy - ff:+.4f}" for r, yy, ff in zip(best, y, fit)))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    xs = np.geomspace(c.min() / 1.5, c.max() * 1.5, 50)
    ax.plot(xs, a * xs ** -b, color=INK_2, linewidth=2, linestyle=(0, (4, 3)),
            label=f"fit: BPB = {a:.3g} · C^-{b:.3f}")
    runs = [r for rs in by_size.values() for r in rs]
    for lr in sorted({r["lr"] for r in runs}):
        pts = [r for r in runs if r["lr"] == lr]
        ax.scatter([r["compute"] for r in pts], [r["bpb"] for r in pts], s=64, zorder=3,
                   color=SERIES[LRS.index(lr)], edgecolors=SURFACE, linewidths=2, label=f"lr {lr:.0e}")
    for n, rs in by_size.items():  # size label above each column of points, clear of the fit line
        top = max(rs, key=lambda r: r["bpb"])
        ax.annotate(f"{n / 1e6:.1f}M", (top["compute"], top["bpb"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.margins(y=0.1)  # room for the size labels under the title
    ax.yaxis.set_major_locator(FixedLocator(np.arange(1.1, 1.65, 0.1)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_facecolor(SURFACE)
    ax.set_title("val BPB vs training compute, FineWeb-Edu, ~20 tokens per parameter",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlabel("training compute (FLOPs)", color=MUTED, fontsize=9)
    ax.set_ylabel("val bits per byte", color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(args.out)


if __name__ == "__main__":
    main()
