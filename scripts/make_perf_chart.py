"""Render TTFC + RTF comparison bars against brief targets.

Outputs: docs/img/perf_vs_brief.png  (1600x900, dark theme)
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path("/Users/pratham/Documents/Repositories/e3-megakernel-tts")
OUT_PNG = REPO / "docs" / "img" / "perf_vs_brief.png"

# Measured values
CONFIGS = ["Config A\n(stub)", "Config B\n(real codec, cold)"]
TTFC_MS = [17.2, 694.0]
RTF = [0.209, 0.347]

# Brief targets
TTFC_TARGETS = [
    ("Step 4 (50 ms)", 50.0, "#34d399"),  # green
    ("Performance Targets (60 ms)", 60.0, "#60a5fa"),  # blue
    ("Deliverables (90 ms)", 90.0, "#fbbf24"),  # amber
]
RTF_TARGETS = [
    ("0.10", 0.10, "#34d399"),
    ("0.15", 0.15, "#60a5fa"),
    ("0.30", 0.30, "#fbbf24"),
]


def color_for_ttfc(v: float) -> str:
    if v <= TTFC_TARGETS[0][1]:
        return "#22c55e"  # green - under all three
    if v <= TTFC_TARGETS[2][1]:
        return "#f59e0b"  # amber - under deliverables only
    return "#ef4444"  # red - missing all


def color_for_rtf(v: float) -> str:
    if v <= RTF_TARGETS[0][1]:
        return "#22c55e"
    if v <= RTF_TARGETS[2][1]:
        return "#f59e0b"
    return "#ef4444"


def style_axis(ax) -> None:
    ax.set_facecolor("#0f1216")
    for spine in ax.spines.values():
        spine.set_color("#2a2f37")
    ax.tick_params(colors="#c7ccd1")
    ax.yaxis.label.set_color("#c7ccd1")
    ax.xaxis.label.set_color("#c7ccd1")
    ax.title.set_color("#e6e9ee")
    ax.grid(True, axis="y", color="#1a1f26", linewidth=0.5)


def bar_panel(ax, values, targets, ylabel, title, color_fn, value_fmt):
    x = np.arange(len(values))
    colors = [color_fn(v) for v in values]
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="#1a1f26", linewidth=1.0)

    # target lines
    for label, y, c in targets:
        ax.axhline(y, color=c, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.text(
            len(values) - 0.5,
            y,
            f"  {label}",
            color=c,
            fontsize=9,
            va="center",
            ha="left",
        )

    # bar value labels
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f" {value_fmt(v)}",
            color="#f5f7fa",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, color="#c7ccd1", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=12, fontsize=12, fontweight="bold")

    # give room for labels + the target legend overflow
    ymax = max(max(values) * 1.18, max(t[1] for t in targets) * 1.2)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(values) - 0.4 + 1.2)  # extra room for target labels


def main() -> None:
    plt.style.use("dark_background")
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#0b0d10")

    style_axis(ax_l)
    style_axis(ax_r)

    bar_panel(
        ax_l,
        TTFC_MS,
        TTFC_TARGETS,
        ylabel="TTFC (ms)",
        title="Time to First Chunk vs brief targets",
        color_fn=color_for_ttfc,
        value_fmt=lambda v: f"{v:.1f} ms",
    )
    bar_panel(
        ax_r,
        RTF,
        RTF_TARGETS,
        ylabel="RTF (audio_time / wall_time wait)",
        title="Real-Time Factor vs brief thresholds",
        color_fn=color_for_rtf,
        value_fmt=lambda v: f"{v:.3f}",
    )

    fig.suptitle(
        "Qwen3-TTS megakernel — measured vs brief",
        color="#f5f7fa",
        fontsize=15,
        y=0.97,
        x=0.04,
        ha="left",
        fontweight="bold",
    )

    fig.text(
        0.04,
        0.93,
        "Config A: streaming stub (no decode).   Config B: real Qwen3 codec, cold cache.",
        color="#9aa1a8",
        fontsize=10,
        ha="left",
    )

    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.07, wspace=0.22)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
