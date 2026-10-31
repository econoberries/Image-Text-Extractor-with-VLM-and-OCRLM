#!/usr/bin/env python3
"""Accuracy / latency / memory trade-off plot for the KIE pipelines.

One point per pipeline configuration (model_name in the CSV):
  x      = mean processing time per sample (inference_seconds = OCR + LM), s
  y      = mean KIE score
  colour = model family (Qwen3.5, Qwen2.5, MiniCPM)
  shape  = square for OCR-based pipelines (OCR/layout parser -> LLM),
           circle for end-to-end VLMs
  area   = max peak GPU memory (GB) over all samples of that configuration

Usage:
    python scripts/vis_accuracyvsEFF.py data/aggregated_predictions_scored.csv --out=figures/acc_eff_trade.pdf --annotate
    python plot_kie_tradeoff.py data.csv -o fig.pdf --annotate
"""
import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- settings
COL_WIDTH_IN = 3.4        # single column of a two-column paper (~3.25-3.5 in)
FIG_HEIGHT_IN = 2.7
FONT_SIZE = 8             # final printed size; figure is not rescaled in the paper
AREA_PER_GB = 6.0         # marker area (pt^2) per GB of peak GPU memory

FAMILIES = {              # substring in model_name -> (legend label, colour)
    "qwen3.5": ("Qwen3.5-9B", "#0072B2"),
    "qwen2.5": ("Qwen2.5-14B", "#E69F00"),
    "minicpm": ("MiniCPM-V-4.6", "#009E73"),
}

LABEL_OVERRIDES = {
    "Qwen3.5-9B_la_ocr_lm": "la",
    "Qwen3.5-9B_ppstructure_lm": "pp",
    "Qwen3.5-9B_spatial_ocr_lm": "sp",
    "Qwen3.5-9B_ocr_lm": "Q3.5-9B_ocr", 
    "Qwen2.5-14B-Instruct-1M_ocr_lm": "Q2.5-14B_ocr", 
    "MiniCPM-V-4.6_vlm": "MiniCPM_VLM", 
    "Qwen3.5-9B_vlm": "Q3.5-9B_vlm", 

}

LABEL_OFFSETS = {          # label -> (dx, dy) in points; default is (0, 9)
    "pp": (6, 9),
    "sp": (0, -9),    # below the marker
    "pp": (0,-9),
    "la": (0,-9),
}

def label_offset(label: str):
    return LABEL_OFFSETS.get(label, (0, 9))

def point_label(name: str) -> str:
    return LABEL_OVERRIDES.get(name, name)


def family_of(name: str):
    low = name.lower()
    for key, val in FAMILIES.items():
        if key in low:
            return val
    raise ValueError(f"Unknown model family in model_name: {name!r}")


def is_ocr_pipeline(name: str) -> bool:
    """Everything that is not an end-to-end VLM (ocr_lm, la_ocr_lm, ppstructure_lm)."""
    return not name.lower().endswith("_vlm")


def parse_gpu_bytes(cell) -> float:
    """peak_allocated_gpu_bytes is a stringified list, e.g. '[19017312768]'."""
    vals = ast.literal_eval(cell) if isinstance(cell, str) else cell
    return float(max(vals) if isinstance(vals, (list, tuple)) else vals)


def short_variant(name: str) -> str:
    """Short tag for optional point annotations."""
    if name.endswith("_vlm"):
        return "VLM"
    if "ppstructure" in name:
        return "PP-Struct."
    if "_la_ocr" in name:
        return "LA+OCR"
    return "OCR"


# ---------------------------------------------------------------- data
def aggregate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["gpu_gb"] = df["peak_allocated_gpu_bytes"].map(parse_gpu_bytes) / 1e9
    agg = (
        df.groupby("model_name")
        .agg(score=("score", "mean"),
             time=("inference_seconds", "mean"),
             gpu_gb=("gpu_gb", "max"))
        .reset_index()
    )
    agg["family_label"], agg["color"] = zip(*agg["model_name"].map(family_of))
    agg["ocr"] = agg["model_name"].map(is_ocr_pipeline)
    return agg


# ---------------------------------------------------------------- plot
def make_plot(agg: pd.DataFrame, out_path: str, annotate: bool = False, dpi: int = 300):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,   # embed TrueType fonts (journal-friendly)
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, FIG_HEIGHT_IN))

    for _, r in agg.iterrows():
        ax.scatter(
            r["time"], r["score"],
            s=AREA_PER_GB * r["gpu_gb"],
            marker="s" if r["ocr"] else "o",
            c=r["color"], edgecolors="black", linewidths=0.6,
            alpha=0.9, zorder=3,
        )
        if annotate:
            lab = point_label(r["model_name"])
            dx, dy = label_offset(lab)
            ax.annotate(lab, (r["time"], r["score"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="center", va="top" if dy < 0 else "bottom",
                        fontsize=FONT_SIZE - 2)

    # axes
    ax.set_xlabel("Mean processing time per sample (s)")
    ax.set_ylabel("Mean KIE score")
    xpad = 0.25 * (agg["time"].max() - agg["time"].min())
    ypad = 0.15 * (agg["score"].max() - agg["score"].min())
    ax.set_xlim(agg["time"].min() - xpad, agg["time"].max() + xpad)
    ax.set_ylim(agg["score"].min() - ypad, agg["score"].max() + ypad)
    ax.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
    ax.tick_params(length=2.5, width=0.6, pad=2)

    # legend 1: colour = model, shape = pipeline type
    present = [(lab, col) for lab, col in FAMILIES.values() if lab in set(agg["family_label"])]
    handles = [Line2D([], [], marker="o", ls="", markersize=5.5, markerfacecolor=c,
                      markeredgecolor="black", markeredgewidth=0.5, label=lab)
               for lab, c in present]
    handles += [
        Line2D([], [], marker="s", ls="", markersize=5.5, markerfacecolor="0.75",
               markeredgecolor="black", markeredgewidth=0.5, label="OCR + LLM"),
        Line2D([], [], marker="o", ls="", markersize=5.5, markerfacecolor="0.75",
               markeredgecolor="black", markeredgewidth=0.5, label="VLM"),
    ]
    leg1 = ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
                     edgecolor="0.7", fancybox=False, borderpad=0.4, labelspacing=0.3,
                     handletextpad=0.2, borderaxespad=0.3)
    leg1.get_frame().set_linewidth(0.5)
    ax.add_artist(leg1)

    # legend 2: marker area = max GPU memory
    gmin, gmax = agg["gpu_gb"].min(), agg["gpu_gb"].max()
    ref = np.unique(np.round(np.linspace(gmin, gmax, 3) / 5) * 5)   # ~5 GB steps
    ref = [g for g in ref if g > 0] or [gmax]

    diam = [np.sqrt(AREA_PER_GB * g) for g in ref]     # marker diameters in pt
    leg_fs = FONT_SIZE - 1
    size_handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="0.75", markeredgecolor="black",
               markeredgewidth=0.5, markersize=d, label=f"{g:.0f} GB")
        for g, d in zip(ref, diam)
    ]
    leg2 = ax.legend(
        handles=size_handles, title="Max GPU mem.", loc="upper left",
        frameon=True, framealpha=0.9, edgecolor="0.7", fancybox=False,
        fontsize=leg_fs, title_fontsize=leg_fs,
        borderpad=0.6, borderaxespad=0.3,
        handleheight=max(diam) / leg_fs,     # row tall enough for the biggest marker
        handlelength=max(diam) / leg_fs,     # handle box wide enough, so text isn't overlapped
        labelspacing=0.35,                   # small gap on top of the enlarged rows
        handletextpad=0.4,
    )
    leg2.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="aggregated_predictions_scored.csv")
    ap.add_argument("-o", "--out", default="kie_tradeoff.pdf",
                    help="output file; extension picks the format (pdf, png, svg)")
    ap.add_argument("--annotate", action="store_true",
                    help="label each point with its pipeline variant")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    agg = aggregate(args.csv)
    print(agg.round(3).to_string(index=False))
    make_plot(agg, args.out, annotate=args.annotate, dpi=args.dpi)


if __name__ == "__main__":
    main()
