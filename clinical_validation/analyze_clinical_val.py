"""
Clinical Validation Results Analysis
=====================================
This script reads 20260424-Clinical_Val_Results.csv, groups patients by their
"Score" column (0–3), and computes descriptive statistics (count, mean, SD, min,
Q1, median, Q3, max) for the "Values" (Contrast Ratio) column within each group.
It then runs a Kruskal-Wallis test with Dunn's post-hoc and Holm correction
(via kruskal_dunn.py) for pairwise comparisons between consecutive severity
score groups (0 vs 1, 1 vs 2, 2 vs 3), and generates a colorblind-accessible
box plot with Holm-corrected p-value annotations.

Score labels:
  0 = Normal
  1 = Degenerated
  2 = Partial Thickness Tear
  3 = Full Thickness Tear

How to run:
  1. Ensure the following packages are installed:
       pip install pandas matplotlib scipy
  2. Place this script in the same directory as:
       20260424-Clinical_Val_Results.csv
       kruskal_dunn.py
     (or update CSV_PATH below).
  3. Run:
       python analyze_clinical_val.py
       python analyze_clinical_val.py --values-column Predicted_Contrast_Ratio
  4. The box plot will be saved as clinical_val_boxplot.png (or a column-specific
     filename) in the same directory and displayed on screen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np

_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_ROOT))

from latepi_paths import CLINICAL_RESULTS, CLINICAL_RESULTS_CSV, FIGURE_4
from kruskal_dunn import kruskal_dunn

# ── Font: Times New Roman (falls back to Liberation Serif if not installed) ───
# Liberation Serif is metrically identical to Times New Roman and is the
# standard Linux substitute.  Install ttf-mscorefonts-installer to get the
# actual Times New Roman font and then delete matplotlib's font cache
# (~/.cache/matplotlib/) so the change is picked up.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"]  = ["Times New Roman", "Liberation Serif",
                                       "DejaVu Serif", "serif"]

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_CSV_PATH = CLINICAL_RESULTS_CSV
DEFAULT_VALUES_COLUMN = "Values"
DEFAULT_OUT_PATH = CLINICAL_RESULTS / "clinical_val_boxplot_reference_values.png"
PREDICTED_VALUES_COLUMN = "Predicted_Contrast_Ratio"
PREDICTED_OUT_PATH = FIGURE_4

SCORE_LABELS = {
    0: "Normal",
    1: "Degenerated",
    2: "Partial Thickness Tear",
    3: "Full Thickness Tear",
}

# Okabe–Ito colorblind-safe palette (distinguishable for deuteranopia,
# protanopia, and tritanopia)
COLORS = {
    0: "#56B4E9",  # sky blue
    1: "#E69F00",  # orange
    2: "#009E73",  # bluish green
    3: "#D55E00",  # vermillion
}

Y_MIN, Y_MAX = 0.0, 1.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze clinical validation contrast ratios by severity score."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Clinical validation CSV (default: {DEFAULT_CSV_PATH.name})",
    )
    parser.add_argument(
        "--values-column",
        default=DEFAULT_VALUES_COLUMN,
        help=f"Contrast ratio column to analyze (default: {DEFAULT_VALUES_COLUMN})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Box plot output path (default: clinical_val_boxplot.png or _predicted.png)",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to save the group summary table as CSV",
    )
    return parser.parse_args()


def default_output_path(values_column: str) -> Path:
    if values_column == PREDICTED_VALUES_COLUMN:
        return PREDICTED_OUT_PATH
    if values_column == DEFAULT_VALUES_COLUMN:
        return DEFAULT_OUT_PATH
    slug = values_column.lower().replace(" ", "_")
    return CLINICAL_RESULTS / f"clinical_val_boxplot_{slug}.png"


def default_summary_csv(values_column: str) -> Path:
    slug = values_column.lower().replace(" ", "_")
    return CLINICAL_RESULTS / f"20260614-clinical_val_summary_{slug}.csv"


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path
    values_column = args.values_column
    out_path = args.output or default_output_path(values_column)
    summary_csv = args.summary_csv or default_summary_csv(values_column)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if values_column not in df.columns:
        raise ValueError(
            f"Column '{values_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df["Score"] = df["Score"].astype(int)
    df = df.dropna(subset=[values_column]).copy()

    # ── Group and collect per-score lists ─────────────────────────────────────
    groups: dict[int, pd.DataFrame] = {}
    data_lists: dict[int, list[float]] = {}
    stats_rows = []

    for score in sorted(SCORE_LABELS.keys()):
        subset = df[df["Score"] == score].copy()
        groups[score] = subset

        values = subset[values_column].to_numpy(dtype=float)
        data_lists[score] = values.tolist()

        q1, median, q3 = np.percentile(values, [25, 50, 75]) if len(values) else (np.nan,) * 3

        stats_rows.append(
            {
                "Score": score,
                "Label": SCORE_LABELS[score],
                "Count": len(values),
                "Mean": values.mean() if len(values) else np.nan,
                "SD": values.std(ddof=1) if len(values) > 1 else np.nan,
                "Min": values.min() if len(values) else np.nan,
                "Q1": q1,
                "Median": median,
                "Q3": q3,
                "Max": values.max() if len(values) else np.nan,
            }
        )

    stats_df = pd.DataFrame(stats_rows).set_index("Score")
    stats_df.to_csv(summary_csv)

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n=== Group Summary ({values_column}) ===\n")
    print(stats_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Patients per Score ===")
    for score in sorted(SCORE_LABELS.keys()):
        patients = groups[score]["Patient"].tolist()
        print(f"\n  Score {score} – {SCORE_LABELS[score]} (n={len(patients)}):")
        print(f"    {patients}")

    # ── Kruskal-Wallis + Dunn's post-hoc with Holm correction ─────────────────
    PAIRS = [(0, 1), (1, 2), (2, 3)]

    print(f"\n=== Kruskal-Wallis + Dunn Post-Hoc (Holm correction) [{values_column}] ===")
    kw_stat, kw_p, dunn_results = kruskal_dunn(
        data=df[values_column].to_numpy(dtype=float),
        groups=df["Score"].to_numpy(dtype=int),
        pairs=PAIRS,
    )
    print(f"Kruskal-Wallis statistic: {kw_stat:.4f}")
    print(f"Kruskal-Wallis p-value  : {kw_p:.4f}")
    for pair, result in dunn_results.items():
        print(
            f"  {pair[0]} vs {pair[1]}: "
            f"p_raw={result['p_raw']:.4f}, p_holm={result['p_holm']:.4f}"
        )

    # ── Box plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    score_keys = sorted(SCORE_LABELS.keys())
    plot_data = [data_lists[s] for s in score_keys]
    positions = list(range(len(score_keys)))

    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, linestyle="none"),
    )

    for patch, score in zip(bp["boxes"], score_keys):
        patch.set_facecolor(COLORS[score])
        patch.set_alpha(0.75)

    for flier, score in zip(bp["fliers"], score_keys):
        flier.set(markerfacecolor=COLORS[score], markeredgecolor=COLORS[score])

    rng = np.random.default_rng(42)
    for pos, score in zip(positions, score_keys):
        vals = np.array(data_lists[score])
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color=COLORS[score],
            edgecolors="white",
            linewidths=0.6,
            s=40,
            zorder=3,
            alpha=0.85,
        )

    def p_label(p: float) -> str:
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    gap = 0.05
    bracket_height = 0.012

    for s1, s2 in PAIRS:
        p_val = dunn_results[(s1, s2)]["p_holm"]
        x1, x2 = positions[s1], positions[s2]

        pair_max = max(max(data_lists[s1]), max(data_lists[s2]))
        y = pair_max + gap

        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_height, y + bracket_height, y],
            lw=1.2,
            color="black",
        )
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height + 0.005,
            p_label(p_val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(s) for s in score_keys], fontsize=13)
    ax.set_xlabel("Severity Score", fontsize=13, labelpad=8)
    ax.set_ylabel("Contrast Ratio", fontsize=13, labelpad=8)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(
            facecolor=COLORS[s],
            edgecolor="grey",
            alpha=0.75,
            label=f"{s} = {SCORE_LABELS[s]}",
        )
        for s in score_keys
    ]
    ax.legend(
        handles=legend_patches,
        title="Severity Score",
        title_fontsize=9,
        fontsize=8,
        loc="upper left",
        frameon=True,
        framealpha=0.85,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSummary CSV saved to: {summary_csv}")
    print(f"Box plot saved to:    {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
