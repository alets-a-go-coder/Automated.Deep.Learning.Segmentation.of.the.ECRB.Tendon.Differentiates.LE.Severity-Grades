"""
Create a review figure showing matched segmentation cases.

For each case, the figure includes five columns:
  1. Vertical case label
  2. Original image
  3. Ground-truth segmentation overlay on the image
  4. Predicted segmentation overlay on the image
  5. Label 1 metrics

The script uses the same serif font configuration as analyze_clinical_val.py.

Usage:
  python make_segmentation_case_grid.py

Optional:
  python make_segmentation_case_grid.py --rows 5
  python make_segmentation_case_grid.py --subjects s0004 s0009 s0010
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_ROOT))

from latepi_paths import (
    FIGURE_2,
    SEGMENTATION_METRICS_DIR,
    TEST_GROUND_TRUTH,
    TEST_IMAGES,
    TEST_PREDICTIONS_ERODED,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from evaluate_segmentation_metrics import build_subject_map, load_nifti_array


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


DEFAULT_IMAGES_DIR = TEST_IMAGES
DEFAULT_GROUND_TRUTH_DIR = TEST_GROUND_TRUTH
DEFAULT_PREDICTIONS_DIR = TEST_PREDICTIONS_ERODED
DEFAULT_METRICS_CSV = SEGMENTATION_METRICS_DIR / "segmentation_metrics_all_labels.csv"
DEFAULT_OUTPUT_PATH = FIGURE_2

LABEL_COLORS = {
    1: "#D62728",  # red
    2: "#2CA02C",  # green
    3: "#1F77B4",  # blue
    4: "#F1C40F",  # yellow
    5: "#FF7F0E",  # orange
}

LABEL_NAMES = {
    1: "ECRB Tendon",
    2: "CE Muscle",
    3: "Humerus",
    4: "Radius",
    5: "Ulna",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a figure showing image, GT overlay, prediction overlay, and label 1 metrics."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing source images (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help=f"Directory containing ground truth segmentations (default: {DEFAULT_GROUND_TRUTH_DIR})",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory containing predicted segmentations (default: {DEFAULT_PREDICTIONS_DIR})",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help=f"Combined metrics CSV from evaluate_segmentation_metrics.py (default: {DEFAULT_METRICS_CSV})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the figure (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--case-count",
        "--rows",
        "--n-cases",
        type=int,
        default=5,
        dest="case_count",
        help="Number of cases/rows to display when --subjects is not provided.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional explicit list of subject IDs to plot.",
    )
    return parser.parse_args()


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    lo, hi = np.percentile(image, [2, 98])
    if hi <= lo:
        lo = float(np.min(image))
        hi = float(np.max(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=float)
    scaled = (image - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def build_overlay(segmentation: np.ndarray) -> np.ndarray:
    overlay = np.zeros(segmentation.shape + (4,), dtype=float)
    for label, color in LABEL_COLORS.items():
        mask = segmentation == label
        if not np.any(mask):
            continue
        rgba = matplotlib.colors.to_rgba(color, alpha=0.45)
        overlay[mask] = rgba
    return overlay


def rotate_ccw(array: np.ndarray) -> np.ndarray:
    return np.rot90(array, k=1)


def select_subjects(
    image_map: dict[str, Path],
    gt_map: dict[str, Path],
    pred_map: dict[str, Path],
    requested_subjects: list[str] | None,
    case_count: int,
) -> list[str]:
    matched = sorted(set(image_map) & set(gt_map) & set(pred_map))
    if requested_subjects:
        missing = [subject for subject in requested_subjects if subject not in matched]
        if missing:
            raise ValueError(f"Requested subjects not found in all folders: {missing}")
        return requested_subjects
    return matched[:case_count]


def format_metric(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{decimals}f}"


def plot_case_row(
    axes_row,
    case_label: str,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    label1_metrics: pd.Series,
) -> None:
    case_ax, image_ax, gt_ax, pred_ax, text_ax = axes_row

    display_image = rotate_ccw(normalize_image(image))
    gt_overlay = rotate_ccw(build_overlay(gt))
    pred_overlay = rotate_ccw(build_overlay(pred))

    case_ax.text(
        0.82,
        0.5,
        case_label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=16,
        transform=case_ax.transAxes,
    )

    image_ax.imshow(display_image, cmap="gray", interpolation="nearest")
    image_ax.set_title("Image", fontsize=13, pad=8)

    gt_ax.imshow(display_image, cmap="gray", interpolation="nearest")
    gt_ax.imshow(gt_overlay, interpolation="nearest")
    gt_ax.set_title("Ground Truth", fontsize=13, pad=8)

    pred_ax.imshow(display_image, cmap="gray", interpolation="nearest")
    pred_ax.imshow(pred_overlay, interpolation="nearest")
    pred_ax.set_title("Prediction", fontsize=13, pad=8)

    metrics_text = "\n".join(
        [
            "ECRB Tendon",
            "",
            f"Dice: {format_metric(label1_metrics['dice_score'])}",
            f"Jaccard: {format_metric(label1_metrics['jaccard_index'])}",
            f"Hausdorff: {format_metric(label1_metrics['hausdorff_distance'], 2)}",
            f"Vol overlap: {format_metric(label1_metrics['volumetric_overlap'])}",
            "",
            f"GT voxels: {int(label1_metrics['gt_voxels'])}",
            f"Pred voxels: {int(label1_metrics['pred_voxels'])}",
        ]
    )
    text_ax.text(
        0.02,
        0.5,
        metrics_text,
        ha="left",
        va="center",
        fontsize=15,
        transform=text_ax.transAxes,
    )

    case_ax.axis("off")
    for ax in (image_ax, gt_ax, pred_ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    text_ax.axis("off")


def main() -> None:
    args = parse_args()

    image_map = build_subject_map(args.images_dir)
    gt_map = build_subject_map(args.ground_truth_dir)
    pred_map = build_subject_map(args.predictions_dir)
    metrics_df = pd.read_csv(args.metrics_csv)

    selected_subjects = select_subjects(
        image_map=image_map,
        gt_map=gt_map,
        pred_map=pred_map,
        requested_subjects=args.subjects,
        case_count=args.case_count,
    )

    if not selected_subjects:
        raise RuntimeError("No subjects selected for plotting.")

    n_rows = len(selected_subjects)
    fig, axes = plt.subplots(
        n_rows,
        5,
        figsize=(14, 2.9 * n_rows + 0.8),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [0.12, 1.0, 1.0, 1.0, 1.05]},
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.97,
        bottom=0.08,
        wspace=0.18,
        hspace=0.14,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, subject_id in enumerate(selected_subjects):
        image, _ = load_nifti_array(image_map[subject_id])
        gt, _ = load_nifti_array(gt_map[subject_id])
        pred, _ = load_nifti_array(pred_map[subject_id])
        case_label = f"Case {row_idx + 1}"

        label1_rows = metrics_df[
            (metrics_df["subject_id"] == subject_id) & (metrics_df["label"] == 1)
        ]
        if label1_rows.empty:
            raise ValueError(f"No label 1 metrics found for subject {subject_id}")

        plot_case_row(
            axes[row_idx],
            case_label=case_label,
            image=image,
            gt=gt,
            pred=pred,
            label1_metrics=label1_rows.iloc[0],
        )

    fig.suptitle(
        " ",
        fontsize=18,
        y=1.02,
    )

    legend_handles = [
        Patch(facecolor=LABEL_COLORS[label], edgecolor="black", label=LABEL_NAMES[label])
        for label in sorted(LABEL_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.03),
        handlelength=1.2,
        handleheight=1.2,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved figure to: {args.output_path}")


if __name__ == "__main__":
    main()
