"""
Evaluate segmentation performance for matched NIfTI image/label triplets.

This script:
  1. Finds matching image, ground-truth, and prediction files by subject ID,
     defined as the first 5 characters of each filename.
  2. Computes per-label segmentation metrics for every matched subject.
  3. Writes one CSV per subject, one combined CSV across all subjects, and a
     grouped summary CSV.

Metrics reported per label:
  - Dice score
  - Jaccard index
  - Hausdorff distance
  - Volumetric overlap

The volumetric overlap reported here is a volume-agreement score:

    1 - abs(pred_volume - gt_volume) / (pred_volume + gt_volume)

This keeps it distinct from the Jaccard index while still quantifying how well
the predicted and reference label volumes agree.

Default folder layout:
  LatEpi/
    evaluate_segmentation_metrics.py
    images/
    ground_truth/
    predictions/

Requirements:
  pip install nibabel pandas scipy numpy

Usage:
  python evaluate_segmentation_metrics.py

Optional:
  python evaluate_segmentation_metrics.py --include-background
  python evaluate_segmentation_metrics.py --output-dir custom_results
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_ROOT))

from latepi_paths import SUBMISSION_ROOT

import numpy as np
import pandas as pd
from scipy import ndimage

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit(
        "This script requires nibabel. Install it with:\n"
        "  python -m pip install nibabel"
    ) from exc


warnings.filterwarnings(
    "ignore",
    message=r"pixdim\[1,2,3\] should be non-zero; setting 0 dims to 1",
)
nib.imageglobals.logger.setLevel(logging.ERROR)


DEFAULT_IMAGES_DIR = SUBMISSION_ROOT / "Model_Development" / "Test_Set_Images"
DEFAULT_GROUND_TRUTH_DIR = (
    SUBMISSION_ROOT / "Model_Development" / "Test_Set_Ground_Truth_Segmentations"
)
DEFAULT_PREDICTIONS_DIR = (
    SUBMISSION_ROOT
    / "Model_Development"
    / "Test_Set_AutoSeg_Predictions_Eroded_FOR_MANUSCRIPT"
)
DEFAULT_OUTPUT_DIR = SUBMISSION_ROOT / "2026-06-14-Segmentation_Results" / "segmentation_metrics"

ANATOMY_LABELS = {
    1: "ECRB Tendon",
    2: "CE Muscle",
    3: "Humerus",
    4: "Radius",
    5: "Ulna",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-label segmentation metrics for matched NIfTI files."
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
        help=(
            "Directory containing ground-truth segmentations "
            f"(default: {DEFAULT_GROUND_TRUTH_DIR})"
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=(
            "Directory containing predicted segmentations "
            f"(default: {DEFAULT_PREDICTIONS_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include label 0 in the evaluation.",
    )
    return parser.parse_args()


def subject_id_from_name(path: Path) -> str:
    return path.name[:5]


def build_subject_map(directory: Path) -> Dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {directory}")

    file_paths = sorted(p for p in directory.iterdir() if p.is_file())
    subject_map: Dict[str, Path] = {}

    for path in file_paths:
        subject_id = subject_id_from_name(path)
        if len(subject_id) < 5:
            raise ValueError(f"Filename is too short to extract subject ID: {path.name}")
        if subject_id in subject_map:
            raise ValueError(
                f"Duplicate subject ID '{subject_id}' found in {directory}: "
                f"{subject_map[subject_id].name} and {path.name}"
            )
        subject_map[subject_id] = path

    return subject_map


def clean_spacing(zooms: Sequence[float], ndim: int) -> np.ndarray:
    spacing = np.asarray(zooms[:ndim], dtype=float)
    if spacing.size != ndim:
        spacing = np.ones(ndim, dtype=float)
    spacing[~np.isfinite(spacing)] = 1.0
    spacing[spacing <= 0] = 1.0
    return spacing


def load_nifti_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    spacing = clean_spacing(img.header.get_zooms(), data.ndim)
    return data, spacing


def to_integer_labels(data: np.ndarray, path: Path) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.int64, copy=False)

    rounded = np.rint(data)
    if not np.allclose(data, rounded):
        raise ValueError(
            f"Segmentation contains non-integer label values: {path.name}"
        )
    return rounded.astype(np.int64, copy=False)


def dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt_count = int(gt_mask.sum())
    pred_count = int(pred_mask.sum())
    total = gt_count + pred_count
    if total == 0:
        return 1.0

    intersection = int(np.logical_and(gt_mask, pred_mask).sum())
    return (2.0 * intersection) / total


def jaccard_index(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    if union == 0:
        return 1.0

    intersection = int(np.logical_and(gt_mask, pred_mask).sum())
    return intersection / union


def volumetric_overlap(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt_count = int(gt_mask.sum())
    pred_count = int(pred_mask.sum())
    total = gt_count + pred_count
    if total == 0:
        return 1.0
    return 1.0 - (abs(pred_count - gt_count) / total)


def extract_surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def hausdorff_distance(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    spacing: np.ndarray,
) -> float:
    gt_present = bool(gt_mask.any())
    pred_present = bool(pred_mask.any())

    if not gt_present and not pred_present:
        return 0.0
    if not gt_present or not pred_present:
        return float("nan")

    gt_surface = extract_surface(gt_mask)
    pred_surface = extract_surface(pred_mask)

    # Distance transforms let us compute the symmetric Hausdorff distance
    # efficiently without explicitly forming all surface point pairs.
    dt_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_gt = float(dt_to_gt[pred_surface].max()) if pred_surface.any() else 0.0
    gt_to_pred = float(dt_to_pred[gt_surface].max()) if gt_surface.any() else 0.0
    return max(pred_to_gt, gt_to_pred)


def compute_label_metrics(
    subject_id: str,
    image_path: Path,
    gt_path: Path,
    pred_path: Path,
    include_background: bool,
) -> List[dict]:
    image_data, image_spacing = load_nifti_array(image_path)
    gt_data, gt_spacing = load_nifti_array(gt_path)
    pred_data, pred_spacing = load_nifti_array(pred_path)

    if image_data.shape != gt_data.shape or gt_data.shape != pred_data.shape:
        raise ValueError(
            f"Shape mismatch for subject {subject_id}: "
            f"image={image_data.shape}, gt={gt_data.shape}, pred={pred_data.shape}"
        )

    gt_labels = to_integer_labels(gt_data, gt_path)
    pred_labels = to_integer_labels(pred_data, pred_path)

    spacing = image_spacing if image_spacing.size == gt_labels.ndim else gt_spacing
    if pred_spacing.size == gt_labels.ndim and not np.allclose(pred_spacing, spacing):
        print(
            f"Warning: spacing mismatch for {subject_id}; "
            f"using image spacing {tuple(spacing)}."
        )

    all_labels = sorted(set(np.unique(gt_labels)).union(np.unique(pred_labels)))
    if not include_background:
        all_labels = [label for label in all_labels if label != 0]

    rows: List[dict] = []
    for label in all_labels:
        gt_mask = gt_labels == label
        pred_mask = pred_labels == label

        gt_voxels = int(gt_mask.sum())
        pred_voxels = int(pred_mask.sum())
        voxel_difference = abs(pred_voxels - gt_voxels)
        intersection_voxels = int(np.logical_and(gt_mask, pred_mask).sum())
        union_voxels = int(np.logical_or(gt_mask, pred_mask).sum())

        rows.append(
            {
                "subject_id": subject_id,
                "image_file": image_path.name,
                "ground_truth_file": gt_path.name,
                "prediction_file": pred_path.name,
                "label": int(label),
                "image_shape": "x".join(str(dim) for dim in image_data.shape),
                "spacing": "x".join(f"{value:g}" for value in spacing),
                "gt_voxels": gt_voxels,
                "pred_voxels": pred_voxels,
                "voxel_difference": voxel_difference,
                "intersection_voxels": intersection_voxels,
                "union_voxels": union_voxels,
                "dice_score": dice_score(gt_mask, pred_mask),
                "jaccard_index": jaccard_index(gt_mask, pred_mask),
                "hausdorff_distance": hausdorff_distance(gt_mask, pred_mask, spacing),
                "volumetric_overlap": volumetric_overlap(gt_mask, pred_mask),
            }
        )

    return rows


def write_summary_csv(results_df: pd.DataFrame, output_dir: Path) -> Path:
    metric_columns = [
        "dice_score",
        "jaccard_index",
        "hausdorff_distance",
        "volumetric_overlap",
        "gt_voxels",
        "pred_voxels",
    ]

    summary_df = (
        results_df.groupby("label")[metric_columns]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .sort_index()
    )
    summary_df.columns = [
        f"{metric}_{stat}" for metric, stat in summary_df.columns.to_flat_index()
    ]
    summary_df = summary_df.reset_index()

    summary_path = output_dir / "segmentation_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def write_paper_summary_csv(results_df: pd.DataFrame, output_dir: Path) -> Path:
    paper_summary_df = (
        results_df.groupby("label", as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            dice_mean=("dice_score", "mean"),
            dice_sd=("dice_score", "std"),
            jaccard_mean=("jaccard_index", "mean"),
            jaccard_sd=("jaccard_index", "std"),
            hausdorff_mean=("hausdorff_distance", "mean"),
            hausdorff_sd=("hausdorff_distance", "std"),
            volumetric_overlap_mean=("volumetric_overlap", "mean"),
            volumetric_overlap_sd=("volumetric_overlap", "std"),
        )
        .sort_values("label")
    )

    paper_summary_path = output_dir / "segmentation_metrics_paper_summary.csv"
    paper_summary_df.to_csv(paper_summary_path, index=False)
    return paper_summary_path


def format_mean_sd(series: pd.Series, decimals: int = 3) -> str:
    mean = float(series.mean())
    sd = float(series.std(ddof=1))
    if np.isnan(sd):
        sd = 0.0
    return f"{mean:.{decimals}f} \u00b1 {sd:.{decimals}f}"


def write_manuscript_table_csv(results_df: pd.DataFrame, output_dir: Path) -> Path:
    table_rows = []

    for label, label_df in results_df.groupby("label", sort=True):
        anatomy = ANATOMY_LABELS.get(int(label), f"Label {int(label)}")
        table_rows.append(
            {
                "Anatomy": anatomy,
                "Subjects (n)": int(label_df["subject_id"].nunique()),
                "DICE": format_mean_sd(label_df["dice_score"]),
                "Jaccard Index": format_mean_sd(label_df["jaccard_index"]),
                "Hausdorff Distance": format_mean_sd(label_df["hausdorff_distance"]),
                "Voxel Difference": format_mean_sd(label_df["voxel_difference"], decimals=1),
                "Volume Overlap": format_mean_sd(label_df["volumetric_overlap"]),
            }
        )

    manuscript_df = pd.DataFrame(table_rows)
    manuscript_path = output_dir / "segmentation_metrics_manuscript_table.csv"
    manuscript_df.to_csv(manuscript_path, index=False)
    return manuscript_path


def print_subject_report(
    subject_ids: Iterable[str],
    image_map: Dict[str, Path],
    gt_map: Dict[str, Path],
    pred_map: Dict[str, Path],
) -> None:
    print("Matched subjects:")
    for subject_id in subject_ids:
        print(
            f"  {subject_id}: "
            f"{image_map[subject_id].name} | "
            f"{gt_map[subject_id].name} | "
            f"{pred_map[subject_id].name}"
        )


def main() -> None:
    args = parse_args()

    image_map = build_subject_map(args.images_dir)
    gt_map = build_subject_map(args.ground_truth_dir)
    pred_map = build_subject_map(args.predictions_dir)

    matched_subjects = sorted(set(image_map) & set(gt_map) & set(pred_map))
    missing_images = sorted((set(gt_map) | set(pred_map)) - set(image_map))
    missing_gt = sorted((set(image_map) | set(pred_map)) - set(gt_map))
    missing_pred = sorted((set(image_map) | set(gt_map)) - set(pred_map))

    if not matched_subjects:
        raise RuntimeError("No matched subjects were found across the three folders.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_subject_dir = args.output_dir / "per_subject_csvs"
    per_subject_dir.mkdir(parents=True, exist_ok=True)

    print_subject_report(matched_subjects, image_map, gt_map, pred_map)
    if missing_images:
        print(f"\nSubjects missing images: {missing_images}")
    if missing_gt:
        print(f"Subjects missing ground truth: {missing_gt}")
    if missing_pred:
        print(f"Subjects missing predictions: {missing_pred}")

    all_rows: List[dict] = []
    for subject_id in matched_subjects:
        rows = compute_label_metrics(
            subject_id=subject_id,
            image_path=image_map[subject_id],
            gt_path=gt_map[subject_id],
            pred_path=pred_map[subject_id],
            include_background=args.include_background,
        )
        subject_df = pd.DataFrame(rows).sort_values(["subject_id", "label"])
        subject_csv_path = per_subject_dir / f"{subject_id}_segmentation_metrics.csv"
        subject_df.to_csv(subject_csv_path, index=False)
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows).sort_values(["subject_id", "label"])
    combined_csv_path = args.output_dir / "segmentation_metrics_all_labels.csv"
    results_df.to_csv(combined_csv_path, index=False)

    summary_csv_path = write_summary_csv(results_df, args.output_dir)
    paper_summary_csv_path = write_paper_summary_csv(results_df, args.output_dir)
    manuscript_table_csv_path = write_manuscript_table_csv(results_df, args.output_dir)

    table_xlsx_path = args.output_dir / "segmentation_metrics_table.xlsx"
    try:
        from format_segmentation_metrics_table import (
            FOOTNOTE,
            TABLE_TITLE,
            build_table_rows,
            write_excel_table,
        )

        table_rows = build_table_rows(results_df)
        write_excel_table(table_rows, table_xlsx_path, TABLE_TITLE, FOOTNOTE)
    except ImportError:
        print(
            "\nSkipping Excel table export (install openpyxl to enable): "
            "pip install openpyxl"
        )
        table_xlsx_path = None

    print(f"\nProcessed {len(matched_subjects)} matched subjects.")
    print(f"Combined CSV: {combined_csv_path}")
    print(f"Summary CSV:  {summary_csv_path}")
    print(f"Paper CSV:    {paper_summary_csv_path}")
    print(f"Table CSV:    {manuscript_table_csv_path}")
    if table_xlsx_path is not None:
        print(f"Table XLSX:   {table_xlsx_path}")
    print(f"Per-subject CSVs: {per_subject_dir}")


if __name__ == "__main__":
    main()
