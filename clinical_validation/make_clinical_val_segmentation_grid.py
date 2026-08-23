"""
Create a grid figure of clinical validation predicted segmentations.

By default, cases are grouped into severity-score columns using
20260424-Clinical_Val_Results.csv (4 cases per score). Each case shows:
  - Left: full image with label 1 overlay and a black crop box
  - Right: zoomed crop with label 1 overlay

Usage:
  python make_clinical_val_segmentation_grid.py
  python make_clinical_val_segmentation_grid.py --per-score-count 4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
for path in (
    _SUBMISSION_ROOT,
    _SUBMISSION_ROOT / "2026-06-14-Segmentation_Results",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from latepi_paths import (
    CLINICAL_IMAGES,
    CLINICAL_PREDICTIONS_ERODED,
    CLINICAL_RESULTS_CSV,
    FIGURE_5,
)
from evaluate_segmentation_metrics import load_nifti_array, to_integer_labels


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
    "serif",
]

DEFAULT_IMAGES_DIR = CLINICAL_IMAGES
DEFAULT_PREDICTIONS_DIR = CLINICAL_PREDICTIONS_ERODED
DEFAULT_RESULTS_CSV = CLINICAL_RESULTS_CSV
DEFAULT_OUTPUT_PATH = FIGURE_5

SCORE_LABELS = {
    0: "Normal",
    1: "Degenerated",
    2: "Partial Thickness Tear",
    3: "Full Thickness Tear",
}

SCORE_COLORS = {
    0: "#56B4E9",
    1: "#E69F00",
    2: "#009E73",
    3: "#D55E00",
}

SUBJECT_PATTERN = re.compile(r"^(?P<subject>.+)_AutoSeg\.nii(?:\.gz)?$", re.IGNORECASE)

LABEL_1 = 1
LABEL_1_COLOR = "#D62728"
LABEL_1_NAME = "ECRB Tendon"
FULL_OVERLAY_ALPHA = 0.50
CROP_OVERLAY_ALPHA = 0.32

# Curated cases per severity score (order = top-to-bottom within each column).
CURATED_SCORE_SUBJECTS: dict[int, list[str]] = {
    0: ["P16", "P23", "P24", "P33"],
    1: ["P10", "P34", "P26", "P27"],
    2: ["P1", "P11", "P13", "P14"],
    3: ["P21", "P28", "R14", "R42"],
}


def score_column_title(score: int) -> str:
    return f"{SCORE_LABELS[score]} ({score})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grid of clinical validation label-1 segmentation overlays."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing source images (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory containing predicted segmentations (default: {DEFAULT_PREDICTIONS_DIR})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the figure (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Clinical validation CSV with Patient and Score columns (default: {DEFAULT_RESULTS_CSV.name})",
    )
    parser.add_argument(
        "--layout",
        choices=("by-score", "flat"),
        default="by-score",
        help="Grid layout: group by severity score columns or a flat grid (default: by-score)",
    )
    parser.add_argument(
        "--per-score-count",
        type=int,
        default=4,
        help="Number of cases to show in each score column for by-score layout (default: 4)",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.20,
        help="Padding around label 1 bbox as a fraction of bbox size (default: 0.20)",
    )
    parser.add_argument(
        "--min-crop-padding",
        type=int,
        default=12,
        help="Minimum crop padding in pixels on each side (default: 12)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of cases to display for flat layout when --subjects is not provided (default: 20)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in the flat layout (default: 5)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional explicit list of subject IDs to plot (flat layout only).",
    )
    return parser.parse_args()


def squeeze_volume(data: np.ndarray) -> np.ndarray:
    return np.squeeze(data)


def subject_from_prediction(path: Path) -> str:
    match = SUBJECT_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Could not parse subject ID from prediction filename: {path.name}")
    return match.group("subject")


def build_prediction_map(predictions_dir: Path) -> dict[str, Path]:
    prediction_map: dict[str, Path] = {}
    for path in sorted(predictions_dir.iterdir()):
        if not path.is_file():
            continue
        if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            continue
        subject_id = subject_from_prediction(path)
        prediction_map[subject_id] = path
    return prediction_map


def image_path_for_subject(images_dir: Path, subject_id: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = images_dir / f"{subject_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No matching image found for subject '{subject_id}' in {images_dir}"
    )


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


def build_label1_overlay(segmentation: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    overlay = np.zeros(segmentation.shape + (4,), dtype=float)
    mask = segmentation == LABEL_1
    if np.any(mask):
        overlay[mask] = matplotlib.colors.to_rgba(LABEL_1_COLOR, alpha=alpha)
    return overlay


def rotate_ccw(array: np.ndarray) -> np.ndarray:
    return np.rot90(array, k=1)


def crop_bbox(
    mask: np.ndarray,
    padding_fraction: float,
    min_padding: int,
) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("Label 1 mask is empty; cannot compute crop box.")

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    rmin, rmax = int(row_indices[0]), int(row_indices[-1])
    cmin, cmax = int(col_indices[0]), int(col_indices[-1])

    height = rmax - rmin + 1
    width = cmax - cmin + 1
    pad_r = max(min_padding, int(round(height * padding_fraction)))
    pad_c = max(min_padding, int(round(width * padding_fraction)))

    r0 = max(0, rmin - pad_r)
    r1 = min(mask.shape[0], rmax + pad_r + 1)
    c0 = max(0, cmin - pad_c)
    c1 = min(mask.shape[1], cmax + pad_c + 1)
    return r0, r1, c0, c1


def has_case_files(
    subject_id: str,
    prediction_map: dict[str, Path],
    images_dir: Path,
) -> bool:
    if subject_id not in prediction_map:
        return False
    try:
        image_path_for_subject(images_dir, subject_id)
    except FileNotFoundError:
        return False
    return True


def select_subjects_by_score(
    results_csv: Path,
    prediction_map: dict[str, Path],
    images_dir: Path,
    per_score_count: int,
) -> dict[int, list[str]]:
    if all(len(CURATED_SCORE_SUBJECTS.get(score, [])) >= per_score_count for score in SCORE_LABELS):
        score_subjects: dict[int, list[str]] = {}
        for score in SCORE_LABELS:
            subjects = CURATED_SCORE_SUBJECTS[score][:per_score_count]
            missing = [
                subject_id
                for subject_id in subjects
                if not has_case_files(subject_id, prediction_map, images_dir)
            ]
            if missing:
                raise RuntimeError(
                    f"Curated score {score} subject(s) missing image/prediction: {missing}"
                )
            score_subjects[score] = subjects
        return score_subjects

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    if "Patient" not in df.columns or "Score" not in df.columns:
        raise ValueError(f"Expected Patient and Score columns in {results_csv}")

    score_subjects: dict[int, list[str]] = {score: [] for score in SCORE_LABELS}
    for _, row in df.sort_values("Patient").iterrows():
        subject_id = str(row["Patient"])
        score = int(row["Score"])
        if score not in score_subjects:
            continue
        if len(score_subjects[score]) >= per_score_count:
            continue
        if not has_case_files(subject_id, prediction_map, images_dir):
            continue
        score_subjects[score].append(subject_id)

    for score, subjects in score_subjects.items():
        if len(subjects) < per_score_count:
            raise RuntimeError(
                f"Score {score} ({SCORE_LABELS[score]}) only has {len(subjects)} "
                f"plottable case(s); need {per_score_count}."
            )

    return score_subjects


def select_subjects(
    prediction_map: dict[str, Path],
    images_dir: Path,
    requested_subjects: list[str] | None,
    count: int,
) -> list[str]:
    available = []
    for subject_id in sorted(prediction_map):
        try:
            image_path_for_subject(images_dir, subject_id)
        except FileNotFoundError:
            continue
        available.append(subject_id)

    if requested_subjects:
        missing = [subject for subject in requested_subjects if subject not in available]
        if missing:
            raise ValueError(f"Requested subjects not found with matching image/prediction: {missing}")
        return requested_subjects

    return available[:count]


def load_case(
    subject_id: str,
    images_dir: Path,
    prediction_map: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    image_path = image_path_for_subject(images_dir, subject_id)
    prediction_path = prediction_map[subject_id]

    image, _ = load_nifti_array(image_path)
    segmentation, _ = load_nifti_array(prediction_path)
    image = squeeze_volume(image)
    segmentation = squeeze_volume(to_integer_labels(segmentation, prediction_path))

    if image.shape != segmentation.shape:
        raise ValueError(
            f"Shape mismatch for {subject_id}: image={image.shape}, pred={segmentation.shape}"
        )
    if not np.any(segmentation == LABEL_1):
        raise ValueError(f"Subject {subject_id} has no label 1 in the prediction.")
    return image, segmentation


def style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_crop_axis(ax, edgecolor: str, linewidth: float = 1.8) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(linewidth)


def add_corner_connectors(
    fig,
    full_ax,
    crop_ax,
    c0: int,
    r0: int,
    r1: int,
    edgecolor: str,
    linewidth: float = 1.2,
) -> None:
    left_corner_pairs = [
        ((c0, r0), (0.0, 1.0)),
        ((c0, r1), (0.0, 0.0)),
    ]
    for (xA, yA), (xB, yB) in left_corner_pairs:
        connector = ConnectionPatch(
            xyA=(xA, yA),
            coordsA=full_ax.transData,
            xyB=(xB, yB),
            coordsB=crop_ax.transAxes,
            color=edgecolor,
            linewidth=linewidth,
            clip_on=False,
        )
        fig.add_artist(connector)


def plot_case_pair(
    fig,
    full_ax,
    crop_ax,
    case_label: str,
    image: np.ndarray,
    segmentation: np.ndarray,
    score_color: str,
    padding_fraction: float,
    min_padding: int,
) -> None:
    display_image = rotate_ccw(normalize_image(image))
    full_overlay = rotate_ccw(build_label1_overlay(segmentation, alpha=FULL_OVERLAY_ALPHA))
    crop_overlay = rotate_ccw(build_label1_overlay(segmentation, alpha=CROP_OVERLAY_ALPHA))
    label_mask = full_overlay[..., 3] > 0

    r0, r1, c0, c1 = crop_bbox(label_mask, padding_fraction, min_padding)
    crop_image = display_image[r0:r1, c0:c1]
    crop_overlay = crop_overlay[r0:r1, c0:c1]

    full_ax.imshow(display_image, cmap="gray", interpolation="nearest")
    full_ax.imshow(full_overlay, interpolation="nearest")
    full_ax.add_patch(
        Rectangle(
            (c0, r0),
            c1 - c0,
            r1 - r0,
            linewidth=1.8,
            edgecolor=score_color,
            facecolor="none",
        )
    )
    style_axis(full_ax)

    crop_ax.imshow(crop_image, cmap="gray", interpolation="nearest")
    crop_ax.imshow(crop_overlay, interpolation="nearest")
    style_crop_axis(crop_ax, edgecolor=score_color)
    add_corner_connectors(fig, full_ax, crop_ax, c0, r0, r1, edgecolor=score_color)

    full_ax.text(
        -0.07,
        0.5,
        case_label,
        transform=full_ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def add_legend(fig) -> None:
    fig.legend(
        handles=[
            Patch(facecolor=LABEL_1_COLOR, edgecolor="black", label=LABEL_1_NAME),
        ],
        loc="lower center",
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=1.2,
        handleheight=1.2,
    )


def build_score_gridspec(n_scores: int, n_rows: int, fig: plt.Figure) -> tuple[GridSpec, list[float]]:
    width_ratios: list[float] = []
    for score_idx in range(n_scores):
        width_ratios.extend([1.0, 1.0])
        if score_idx < n_scores - 1:
            width_ratios.append(0.45)
    gs = GridSpec(
        n_rows,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.05,
        hspace=0.18,
    )
    return gs, width_ratios


def score_column_indices(score_idx: int) -> tuple[int, int]:
    return score_idx * 3, score_idx * 3 + 1


def plot_by_score_grid(
    score_subjects: dict[int, list[str]],
    images_dir: Path,
    prediction_map: dict[str, Path],
    output_path: Path,
    per_score_count: int,
    padding_fraction: float,
    min_padding: int,
) -> None:
    score_keys = sorted(SCORE_LABELS.keys())
    n_rows = per_score_count
    n_scores = len(score_keys)
    n_panel_cols = n_scores * 2
    n_spacers = max(0, n_scores - 1)

    fig = plt.figure(
        figsize=(2.05 * n_panel_cols + 0.85 * n_spacers, 2.35 * n_rows + 0.95),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.97,
        top=0.94,
        bottom=0.07,
    )
    gs, _ = build_score_gridspec(n_scores, n_rows, fig)

    for score_idx, score in enumerate(score_keys):
        full_col, crop_col = score_column_indices(score_idx)
        header_axes: list[plt.Axes] = []

        for row_idx, subject_id in enumerate(score_subjects[score]):
            full_ax = fig.add_subplot(gs[row_idx, full_col])
            crop_ax = fig.add_subplot(gs[row_idx, crop_col])
            image, segmentation = load_case(subject_id, images_dir, prediction_map)
            case_num = score_idx * n_rows + row_idx + 1
            plot_case_pair(
                fig,
                full_ax,
                crop_ax,
                f"Case {case_num}",
                image,
                segmentation,
                SCORE_COLORS[score],
                padding_fraction,
                min_padding,
            )
            if row_idx == 0:
                header_axes.extend([full_ax, crop_ax])

        for row_idx in range(len(score_subjects[score]), n_rows):
            fig.add_subplot(gs[row_idx, full_col]).axis("off")
            fig.add_subplot(gs[row_idx, crop_col]).axis("off")

        if header_axes:
            header_x = (
                header_axes[0].get_position().x0 + header_axes[-1].get_position().x1
            ) / 2
            header_offset = 15 / (fig.get_figheight() * fig.dpi)
            header_y = header_axes[0].get_position().y1 + 0.008 + header_offset
            fig.text(
                header_x,
                header_y,
                score_column_title(score),
                ha="center",
                va="bottom",
                fontsize=20,
                fontweight="bold",
                color=SCORE_COLORS[score],
            )

    add_legend(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.12)

    total_cases = sum(len(subjects) for subjects in score_subjects.values())
    print(f"Saved by-score figure with {total_cases} cases to: {output_path}")
    for score in score_keys:
        print(f"  Score {score} ({SCORE_LABELS[score]}): {', '.join(score_subjects[score])}")


def plot_flat_grid(
    selected_subjects: list[str],
    images_dir: Path,
    prediction_map: dict[str, Path],
    output_path: Path,
    n_cols: int,
    padding_fraction: float,
    min_padding: int,
) -> None:
    n_cases = len(selected_subjects)
    n_pair_cols = max(1, n_cols)
    n_rows = int(np.ceil(n_cases / n_pair_cols))
    n_axes_cols = n_pair_cols * 2

    fig, axes = plt.subplots(
        n_rows,
        n_axes_cols,
        figsize=(2.05 * n_axes_cols, 2.35 * n_rows + 0.5),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.96,
        bottom=0.08,
        wspace=0.10,
        hspace=0.18,
    )

    axes_array = np.atleast_1d(axes).reshape(n_rows, n_axes_cols)

    for idx, subject_id in enumerate(selected_subjects):
        row, pair_col = divmod(idx, n_pair_cols)
        full_col = pair_col * 2
        crop_col = full_col + 1
        image, segmentation = load_case(subject_id, images_dir, prediction_map)
        plot_case_pair(
            fig,
            axes_array[row, full_col],
            axes_array[row, crop_col],
            f"Case {idx + 1}",
            image,
            segmentation,
            SCORE_COLORS[0],
            padding_fraction,
            min_padding,
        )

    for idx in range(n_cases, n_rows * n_pair_cols):
        row, pair_col = divmod(idx, n_pair_cols)
        axes_array[row, pair_col * 2].axis("off")
        axes_array[row, pair_col * 2 + 1].axis("off")

    add_legend(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    print(f"Saved flat figure with {n_cases} cases to: {output_path}")


def main() -> None:
    args = parse_args()

    prediction_map = build_prediction_map(args.predictions_dir)
    if not prediction_map:
        raise RuntimeError(f"No prediction files found in {args.predictions_dir}")

    if args.layout == "by-score":
        if args.subjects:
            raise ValueError("--subjects is only supported with --layout flat")
        score_subjects = select_subjects_by_score(
            results_csv=args.results_csv,
            prediction_map=prediction_map,
            images_dir=args.images_dir,
            per_score_count=args.per_score_count,
        )
        plot_by_score_grid(
            score_subjects=score_subjects,
            images_dir=args.images_dir,
            prediction_map=prediction_map,
            output_path=args.output_path,
            per_score_count=args.per_score_count,
            padding_fraction=args.crop_padding,
            min_padding=args.min_crop_padding,
        )
        return

    selected_subjects = select_subjects(
        prediction_map=prediction_map,
        images_dir=args.images_dir,
        requested_subjects=args.subjects,
        count=args.count,
    )
    if not selected_subjects:
        raise RuntimeError("No subjects selected for plotting.")

    plot_flat_grid(
        selected_subjects=selected_subjects,
        images_dir=args.images_dir,
        prediction_map=prediction_map,
        output_path=args.output_path,
        n_cols=args.cols,
        padding_fraction=args.crop_padding,
        min_padding=args.min_crop_padding,
    )


if __name__ == "__main__":
    main()
