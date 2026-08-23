"""
Compute predicted contrast ratios for the clinical validation set and append
them to 20260424-Clinical_Val_Results.csv.

Contrast ratio is defined as:

    mean signal intensity in label 1 (ECRB tendon)
    -----------------------------------------------
    mean signal intensity in label 2 (CE muscle)

Predictions are read from clinical_val_pred_eroded/ (subject parsed from
``{Subject}_AutoSeg.nii.gz``). Matching images are read from
clinical_set/ImgWithSeg/{Subject}.nii.gz.

Usage:
  python add_clinical_val_contrast_ratio.py
  python add_clinical_val_contrast_ratio.py --column-name Predicted_Contrast_Ratio
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    CLINICAL_RESULTS,
    CLINICAL_RESULTS_CSV,
)
from analyze_segmentation_contrast_ratio import compute_contrast_ratio
from evaluate_segmentation_metrics import load_nifti_array, to_integer_labels


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_DIR = CLINICAL_PREDICTIONS_ERODED
DEFAULT_IMAGES_DIR = CLINICAL_IMAGES
DEFAULT_RESULTS_CSV = CLINICAL_RESULTS_CSV
DEFAULT_OUTPUT_CSV = CLINICAL_RESULTS_CSV
DEFAULT_DETAILS_CSV = CLINICAL_RESULTS / "20260614-clinical_val_contrast_ratios.csv"

SUBJECT_PATTERN = re.compile(r"^(?P<subject>.+)_AutoSeg\.nii(?:\.gz)?$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute clinical-validation contrast ratios and update the results CSV."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory with eroded AutoSeg predictions (default: {DEFAULT_PREDICTIONS_DIR})",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory with source images (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Clinical validation results CSV to update (default: {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: overwrite {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=DEFAULT_DETAILS_CSV,
        help=f"Per-subject details CSV (default: {DEFAULT_DETAILS_CSV})",
    )
    parser.add_argument(
        "--column-name",
        default="Predicted_Contrast_Ratio",
        help="New column name to add to the clinical validation CSV",
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
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    prediction_map: dict[str, Path] = {}
    for path in sorted(predictions_dir.iterdir()):
        if not path.is_file():
            continue
        if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            continue
        subject_id = subject_from_prediction(path)
        if subject_id in prediction_map:
            raise ValueError(
                f"Duplicate prediction subject ID '{subject_id}' in {predictions_dir}"
            )
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


def compute_subject_contrast_ratio(image_path: Path, prediction_path: Path) -> dict[str, object]:
    image_data, _ = load_nifti_array(image_path)
    pred_data, _ = load_nifti_array(prediction_path)

    image_data = squeeze_volume(image_data)
    pred_data = squeeze_volume(to_integer_labels(pred_data, prediction_path))

    if image_data.shape != pred_data.shape:
        raise ValueError(
            f"Shape mismatch for {prediction_path.name}: "
            f"image={image_data.shape}, prediction={pred_data.shape}"
        )

    label_1_mean, label_2_mean, contrast_ratio = compute_contrast_ratio(
        image_data, pred_data
    )
    return {
        "subject_id": subject_from_prediction(prediction_path),
        "image_file": image_path.name,
        "prediction_file": prediction_path.name,
        "pred_label_1_mean_signal": label_1_mean,
        "pred_label_2_mean_signal": label_2_mean,
        "predicted_contrast_ratio": contrast_ratio,
    }


def main() -> None:
    args = parse_args()

    results_df = pd.read_csv(args.results_csv)
    results_df.columns = results_df.columns.str.strip()
    if "Patient" not in results_df.columns:
        raise ValueError(f"Expected a 'Patient' column in {args.results_csv}")

    prediction_map = build_prediction_map(args.predictions_dir)
    if not prediction_map:
        raise RuntimeError(f"No prediction files found in {args.predictions_dir}")

    detail_rows = []
    failures: list[str] = []

    for subject_id in sorted(prediction_map):
        prediction_path = prediction_map[subject_id]
        try:
            image_path = image_path_for_subject(args.images_dir, subject_id)
            detail_rows.append(
                compute_subject_contrast_ratio(image_path, prediction_path)
            )
        except Exception as exc:
            failures.append(f"{subject_id}: {exc}")
            logger.warning("Skipping %s: %s", subject_id, exc)

    if not detail_rows:
        raise RuntimeError(
            "No contrast ratios could be computed. "
            "Predicted segmentations must contain labels 1 and 2."
        )

    details_df = pd.DataFrame(detail_rows).sort_values("subject_id")
    args.details_csv.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(args.details_csv, index=False)

    ratio_map = dict(
        zip(details_df["subject_id"], details_df["predicted_contrast_ratio"])
    )

    updated_df = results_df.copy()
    updated_df[args.column_name] = updated_df["Patient"].map(ratio_map)

    missing_in_results = sorted(set(ratio_map) - set(updated_df["Patient"]))
    missing_ratios = sorted(set(updated_df["Patient"]) - set(ratio_map))

    updated_df.to_csv(args.output_csv, index=False)

    computed = updated_df[args.column_name].notna().sum()
    logger.info("Computed contrast ratios for %d subject(s)", len(detail_rows))
    logger.info("Updated %d row(s) in %s", computed, args.output_csv)
    logger.info("Saved per-subject details to %s", args.details_csv)

    if missing_in_results:
        logger.warning(
            "Predictions without a matching CSV row: %s",
            ", ".join(missing_in_results),
        )
    if missing_ratios:
        logger.warning(
            "CSV patients without a computed contrast ratio: %s",
            ", ".join(missing_ratios),
        )
    if failures:
        logger.warning("Failed subjects (%d):", len(failures))
        for message in failures:
            logger.warning("  %s", message)

    print("\n=== Clinical Validation Contrast Ratios ===\n")
    display_columns = [
        col for col in ("Patient", "Values", args.column_name, "Score")
        if col in updated_df.columns
    ]
    print(
        updated_df[display_columns]
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )


if __name__ == "__main__":
    main()
