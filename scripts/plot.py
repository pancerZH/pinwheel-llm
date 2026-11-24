"""Plot train loss and val BPB against tokens for one or more runs.

    uv run python scripts/plot.py runs/<a> runs/<b> [-o runs/plot.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated palette); a run keeps its slot by position.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("-o", "--out", default="runs/plot.png")
    args = p.parse_args()
    assert len(args.runs) <= len(SERIES), f"at most {len(SERIES)} runs per plot"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE)
    panels = [("loss", "train loss (nats/token)"), ("val_bpb", "val bits per byte")]
    for run, color in zip(args.runs, SERIES):
        rows = [json.loads(line) for line in open(Path(run) / "log.jsonl")]
        for ax, (key, _) in zip(axes, panels):
            pts = [(r["tokens"] / 1e6, r[key]) for r in rows if key in r]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=1.5, label=Path(run).name)
            if len(args.runs) <= 4:  # direct label at the line's end, in ink, not series color
                ax.annotate(f"{Path(run).name} {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK_2)

    for ax, (_, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("tokens (M)", color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        if len(args.runs) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    if len(args.runs) == 1:
        fig.suptitle(Path(args.runs[0]).name, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(args.out)


if __name__ == "__main__":
    main()
