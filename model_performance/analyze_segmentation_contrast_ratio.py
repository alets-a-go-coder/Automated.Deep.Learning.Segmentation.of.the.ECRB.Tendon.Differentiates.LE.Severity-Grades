"""
Analyze contrast ratios from ground-truth and predicted segmentations.

Contrast ratio is defined as:

    mean signal intensity in label 1 (ECRB tendon)
    -----------------------------------------------
    mean signal intensity in label 2 (CE muscle)

For every matched subject, this script computes:
  - Ground-truth contrast ratio
  - Predicted contrast ratio
  - Difference (ground truth - prediction)

It then:
  - saves a per-subject CSV
  - saves a summary/statistics CSV
  - generates a box plot using the same font and overall styling as
    analyze_clinical_val.py
  - runs a Mann-Whitney U test between ground-truth and predicted ratios

Usage:
  python analyze_segmentation_contrast_ratio.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_ROOT))

from latepi_paths import (
    CONTRAST_RATIO_DIR,
    FIGURE_3,
    LEGACY_SEGMENTATION_CR_BOXPLOT,
    TEST_GROUND_TRUTH,
    TEST_IMAGES,
    TEST_PREDICTIONS_ERODED,
)

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from evaluate_segmentation_metrics import (
    build_subject_map,
    load_nifti_array,
    to_integer_labels,
)


# ── Font: Times New Roman (falls back to Liberation Serif if not installed) ───
# Liberation Serif is metrically identical to Times New Roman and is the
# standard Linux substitute. Install ttf-mscorefonts-installer to get the
# actual Times New Roman font and then delete matplotlib's font cache
# (~/.cache/matplotlib/) so the change is picked up.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
    "serif",
]


IMAGES_DIR = TEST_IMAGES
GROUND_TRUTH_DIR = TEST_GROUND_TRUTH
PREDICTIONS_DIR = TEST_PREDICTIONS_ERODED
OUTPUT_DIR = CONTRAST_RATIO_DIR
PER_SUBJECT_CSV = OUTPUT_DIR / "segmentation_contrast_ratios.csv"
SUMMARY_CSV = OUTPUT_DIR / "segmentation_contrast_ratio_summary.csv"
BOXPLOT_PATH = FIGURE_3

LABEL_1 = 1
LABEL_2 = 2
LABEL_1_NAME = "ECRB Tendon"
LABEL_2_NAME = "CE Muscle"

GROUP_LABELS = {
    0: "Ground Truth",
    1: "Prediction",
}

COLORS = {
    0: "#4A4A4A",  # dark grey — ground truth
    1: "#E377A1",  # pink — prediction
}


def compute_contrast_ratio(image: np.ndarray, segmentation: np.ndarray) -> tuple[float, float, float]:
    label_1_mask = segmentation == LABEL_1
    label_2_mask = segmentation == LABEL_2

    if not np.any(label_1_mask):
        raise ValueError(f"Segmentation is missing label {LABEL_1} ({LABEL_1_NAME}).")
    if not np.any(label_2_mask):
        raise ValueError(f"Segmentation is missing label {LABEL_2} ({LABEL_2_NAME}).")

    label_1_mean = float(np.mean(image[label_1_mask]))
    label_2_mean = float(np.mean(image[label_2_mask]))

    if np.isclose(label_2_mean, 0.0):
        raise ValueError(
            f"Mean signal intensity for label {LABEL_2} ({LABEL_2_NAME}) is zero."
        )

    contrast_ratio = label_1_mean / label_2_mean
    return label_1_mean, label_2_mean, contrast_ratio


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "min": float(np.min(values)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(values)),
    }


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def main() -> None:
    image_map = build_subject_map(IMAGES_DIR)
    gt_map = build_subject_map(GROUND_TRUTH_DIR)
    pred_map = build_subject_map(PREDICTIONS_DIR)

    matched_subjects = sorted(set(image_map) & set(gt_map) & set(pred_map))
    if not matched_subjects:
        raise RuntimeError("No matched subjects were found across the three folders.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject_id in matched_subjects:
        image_data, _ = load_nifti_array(image_map[subject_id])
        gt_data, _ = load_nifti_array(gt_map[subject_id])
        pred_data, _ = load_nifti_array(pred_map[subject_id])

        if image_data.shape != gt_data.shape or gt_data.shape != pred_data.shape:
            raise ValueError(
                f"Shape mismatch for subject {subject_id}: "
                f"image={image_data.shape}, gt={gt_data.shape}, pred={pred_data.shape}"
            )

        gt_labels = to_integer_labels(gt_data, gt_map[subject_id])
        pred_labels = to_integer_labels(pred_data, pred_map[subject_id])

        gt_label_1_mean, gt_label_2_mean, gt_ratio = compute_contrast_ratio(
            image_data, gt_labels
        )
        pred_label_1_mean, pred_label_2_mean, pred_ratio = compute_contrast_ratio(
            image_data, pred_labels
        )

        rows.append(
            {
                "subject_id": subject_id,
                "image_file": image_map[subject_id].name,
                "ground_truth_file": gt_map[subject_id].name,
                "prediction_file": pred_map[subject_id].name,
                "gt_label_1_mean_signal": gt_label_1_mean,
                "gt_label_2_mean_signal": gt_label_2_mean,
                "gt_contrast_ratio": gt_ratio,
                "pred_label_1_mean_signal": pred_label_1_mean,
                "pred_label_2_mean_signal": pred_label_2_mean,
                "pred_contrast_ratio": pred_ratio,
                "difference_gt_minus_pred": gt_ratio - pred_ratio,
                "absolute_difference": abs(gt_ratio - pred_ratio),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("subject_id")
    results_df.to_csv(PER_SUBJECT_CSV, index=False)

    gt_values = results_df["gt_contrast_ratio"].to_numpy(dtype=float)
    pred_values = results_df["pred_contrast_ratio"].to_numpy(dtype=float)
    differences = results_df["difference_gt_minus_pred"].to_numpy(dtype=float)

    mw_stat, mw_p = stats.mannwhitneyu(gt_values, pred_values, alternative="two-sided")

    summary_rows = []
    gt_summary = summarize(gt_values)
    gt_summary["group"] = "Ground Truth"
    summary_rows.append(gt_summary)

    pred_summary = summarize(pred_values)
    pred_summary["group"] = "Prediction"
    summary_rows.append(pred_summary)

    diff_summary = summarize(differences)
    diff_summary["group"] = "Difference (GT - Pred)"
    summary_rows.append(diff_summary)

    summary_df = pd.DataFrame(summary_rows)[
        ["group", "count", "mean", "sd", "min", "q1", "median", "q3", "max"]
    ]
    summary_df["mann_whitney_u_statistic"] = np.nan
    summary_df["mann_whitney_u_p_value"] = np.nan
    summary_df["mean_difference_gt_minus_pred"] = np.nan
    summary_df.loc[0, "mann_whitney_u_statistic"] = float(mw_stat)
    summary_df.loc[0, "mann_whitney_u_p_value"] = float(mw_p)
    summary_df.loc[0, "mean_difference_gt_minus_pred"] = float(np.mean(differences))
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== Contrast Ratio Summary ===\n")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Mann-Whitney U Test ===")
    print(f"  Statistic: {mw_stat:.4f}")
    print(f"  p-value  : {mw_p:.4f}")
    print(f"  Mean difference (GT - Pred): {np.mean(differences):.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))

    plot_data = [gt_values.tolist(), pred_values.tolist()]
    positions = [0, 1]

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

    for patch, group_id in zip(bp["boxes"], positions):
        patch.set_facecolor(COLORS[group_id])
        patch.set_alpha(0.75)

    for flier, group_id in zip(bp["fliers"], positions):
        flier.set(markerfacecolor=COLORS[group_id], markeredgecolor=COLORS[group_id])

    rng = np.random.default_rng(42)
    for pos, values, group_id in zip(positions, plot_data, positions):
        vals = np.array(values, dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color=COLORS[group_id],
            edgecolors="white",
            linewidths=0.6,
            s=40,
            zorder=3,
            alpha=0.85,
        )

    gap = max(0.05, 0.05 * max(np.max(gt_values), np.max(pred_values)))
    bracket_height = max(0.01, 0.012 * max(np.max(gt_values), np.max(pred_values)))
    y = max(np.max(gt_values), np.max(pred_values)) + gap

    ax.plot(
        [positions[0], positions[0], positions[1], positions[1]],
        [y, y + bracket_height, y + bracket_height, y],
        lw=1.2,
        color="black",
    )
    ax.text(
        sum(positions) / 2,
        y + bracket_height + 0.01,
        p_label(mw_p),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [GROUP_LABELS[pos] for pos in positions],
        fontsize=13,
    )
    ax.set_xlabel("Segmentation Source", fontsize=13, labelpad=8)
    ax.set_ylabel("Contrast Ratio", fontsize=13, labelpad=8)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    y_min = min(np.min(gt_values), np.min(pred_values))
    y_max = y + bracket_height + 0.06
    lower_margin = max(0.02, 0.05 * (y_max - y_min))
    upper_margin = max(0.02, 0.05 * (y_max - y_min))
    ax.set_ylim(y_min - lower_margin, y_max + upper_margin)

    legend_patches = [
        mpatches.Patch(
            facecolor=COLORS[group_id],
            edgecolor="grey",
            alpha=0.75,
            label=GROUP_LABELS[group_id],
        )
        for group_id in positions
    ]
    ax.legend(
        handles=legend_patches,
        title="Segmentation Source",
        title_fontsize=9,
        fontsize=8,
        loc="upper left",
        frameon=True,
        framealpha=0.85,
    )

    plt.tight_layout()
    FIGURES_DIR = BOXPLOT_PATH.parent
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(BOXPLOT_PATH, dpi=150, bbox_inches="tight")
    shutil.copy2(BOXPLOT_PATH, LEGACY_SEGMENTATION_CR_BOXPLOT)
    print(f"\nPer-subject CSV saved to: {PER_SUBJECT_CSV}")
    print(f"Summary CSV saved to:     {SUMMARY_CSV}")
    print(f"Box plot saved to:        {BOXPLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
