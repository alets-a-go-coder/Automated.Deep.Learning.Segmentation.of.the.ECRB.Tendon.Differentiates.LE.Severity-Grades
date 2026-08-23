"""
Clean predicted segmentations and adaptively erode label 1 (ECRB tendon).

For each NIfTI in the predictions folder:
  1. Keep only the largest connected component for labels 1 and 2 (removes islands).
  2. Adaptively erode label 1 using a distance-transform-based inward trim:
     - Thick masks: keep voxels at least --pixels from the boundary.
     - Thin masks: reduce erosion so the mask is not destroyed.
     - Safety floor: retain at least --min-voxels label-1 voxels when possible.

Label 2 is not eroded.

Requirements:
  pip install nibabel numpy scipy

Usage:
    python erode_label1_predictions.py
    python erode_label1_predictions.py --input-dir clinical_val_pred --output-dir clinical_val_pred_eroded --pixels 6
    python erode_label1_predictions.py --input-dir predictions --output-dir predictions_eroded --pixels 6
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit(
        "This script requires nibabel. Install it with:\n"
        "  python -m pip install nibabel"
    ) from exc

_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(_SUBMISSION_ROOT / "2026-06-14-Segmentation_Results") not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_ROOT / "2026-06-14-Segmentation_Results"))
if str(_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_ROOT))

from latepi_paths import (
    CLINICAL_PREDICTIONS,
    CLINICAL_PREDICTIONS_ERODED,
    setup_import_paths,
)

setup_import_paths()
from evaluate_segmentation_metrics import to_integer_labels


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = CLINICAL_PREDICTIONS
DEFAULT_OUTPUT_DIR = CLINICAL_PREDICTIONS_ERODED

LABEL_1 = 1
LABEL_2 = 2


@dataclass(frozen=True)
class ErosionResult:
    mask: np.ndarray
    effective_pixels: int
    max_edt: float
    strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove label 1/2 islands, then adaptively erode label 1."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input predictions directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pixels",
        type=int,
        default=10,
        help="Target label-1 erosion depth in pixels (default: 10)",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=800,
        help="Minimum label-1 voxels to retain when possible (default: 800)",
    )
    parser.add_argument(
        "--thin-margin",
        type=int,
        default=2,
        help="Pixels to leave when mask is thinner than --pixels (default: 2)",
    )
    return parser.parse_args()


def connectivity_structure(ndim: int) -> np.ndarray:
    return ndimage.generate_binary_structure(ndim, 1)


def keep_largest_component(data: np.ndarray, label_value: int) -> tuple[np.ndarray, int, int]:
    mask = data == label_value
    if not mask.any():
        return data, 0, 0

    structure = connectivity_structure(data.ndim)
    component_ids, num_components = ndimage.label(mask, structure=structure)
    if num_components <= 1:
        return data, 0, num_components

    component_sizes = np.bincount(component_ids.ravel())
    component_sizes[0] = 0
    largest_id = int(component_sizes.argmax())
    largest_mask = component_ids == largest_id

    out = data.copy()
    removed_mask = mask & ~largest_mask
    out[removed_mask] = 0
    return out, int(removed_mask.sum()), num_components


def remove_islands(data: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Keep largest component for labels 1 and 2."""
    out = data
    removed: dict[int, int] = {}
    components: dict[int, int] = {}

    for label_value in (LABEL_1, LABEL_2):
        out, n_removed, n_components = keep_largest_component(out, label_value)
        removed[label_value] = n_removed
        components[label_value] = n_components

    return out, removed, components


def keep_top_edt_voxels(mask: np.ndarray, edt: np.ndarray, n_keep: int) -> np.ndarray:
    n_keep = min(n_keep, int(mask.sum()))
    if n_keep <= 0:
        return np.zeros_like(mask, dtype=bool)

    edt_values = edt[mask]
    if n_keep >= len(edt_values):
        return mask.copy()

    threshold = np.partition(edt_values, len(edt_values) - n_keep)[len(edt_values) - n_keep]
    return mask & (edt >= threshold)


def adaptive_erode_label1(
    mask: np.ndarray,
    target_pixels: int,
    min_voxels: int,
    thin_margin: int,
) -> ErosionResult:
    if target_pixels <= 0 or not mask.any():
        return ErosionResult(
            mask=mask.copy(),
            effective_pixels=0,
            max_edt=0.0,
            strategy="none",
        )

    edt = ndimage.distance_transform_edt(mask)
    max_edt = float(edt.max())

    if max_edt >= target_pixels:
        effective_pixels = target_pixels
        strategy = "target"
    else:
        effective_pixels = max(1, int(np.floor(max_edt)) - thin_margin)
        strategy = "thin"

    eroded = edt >= effective_pixels

    if int(eroded.sum()) < min_voxels:
        relaxed_pixels = max(1, int(np.floor(max_edt)) - 1)
        relaxed = edt >= relaxed_pixels
        if int(relaxed.sum()) >= int(eroded.sum()):
            eroded = relaxed
            effective_pixels = relaxed_pixels
            strategy = f"{strategy}+relaxed"

    if int(eroded.sum()) < min_voxels:
        eroded = keep_top_edt_voxels(mask, edt, min_voxels)
        effective_pixels = int(np.min(edt[eroded])) if eroded.any() else 0
        strategy = f"{strategy}+floor"

    structure = connectivity_structure(mask.ndim)
    eroded_labels, n_components = ndimage.label(eroded, structure=structure)
    if n_components > 1:
        component_sizes = np.bincount(eroded_labels.ravel())
        component_sizes[0] = 0
        largest_id = int(component_sizes.argmax())
        eroded = eroded_labels == largest_id
        strategy = f"{strategy}+lcc"

    return ErosionResult(
        mask=eroded,
        effective_pixels=effective_pixels,
        max_edt=max_edt,
        strategy=strategy,
    )


def apply_label1_erosion(data: np.ndarray, erosion: ErosionResult) -> np.ndarray:
    label1_mask = data == LABEL_1
    if not label1_mask.any():
        return data

    out = data.copy()
    out[label1_mask] = 0
    out[erosion.mask] = LABEL_1
    return out


def squeeze_for_morphology(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Process 2D in-plane labels even when stored as a singleton 3D slice."""
    original_shape = data.shape
    if data.ndim == 3 and 1 in original_shape:
        return np.squeeze(data), original_shape
    return data, original_shape


def restore_volume_shape(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    if data.shape == original_shape:
        return data
    return np.reshape(data, original_shape)


def process_file(
    input_path: Path,
    output_path: Path,
    target_pixels: int,
    min_voxels: int,
    thin_margin: int,
) -> None:
    img = nib.load(str(input_path))
    data = to_integer_labels(np.asarray(img.dataobj), input_path)
    working, original_shape = squeeze_for_morphology(data)

    label1_before = int((working == LABEL_1).sum())
    label2_before = int((working == LABEL_2).sum())

    cleaned, removed, components = remove_islands(working)
    label1_after_cleanup = int((cleaned == LABEL_1).sum())
    label2_after_cleanup = int((cleaned == LABEL_2).sum())

    label1_mask = cleaned == LABEL_1
    erosion = adaptive_erode_label1(
        label1_mask,
        target_pixels=target_pixels,
        min_voxels=min_voxels,
        thin_margin=thin_margin,
    )
    processed = restore_volume_shape(
        apply_label1_erosion(cleaned, erosion),
        original_shape,
    )
    processed_2d, _ = squeeze_for_morphology(processed)
    label1_final = int((processed_2d == LABEL_1).sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(processed.astype(data.dtype), img.affine, img.header),
        str(output_path),
    )

    logger.info(
        "%s: L1 %d -> %d -> %d (islands=%d, comps=%d, target=%d px, "
        "effective=%d px, max_edt=%.1f, strategy=%s), "
        "L2 %d -> %d (islands=%d, comps=%d)",
        input_path.name,
        label1_before,
        label1_after_cleanup,
        label1_final,
        removed[LABEL_1],
        components[LABEL_1],
        target_pixels,
        erosion.effective_pixels,
        erosion.max_edt,
        erosion.strategy,
        label2_before,
        label2_after_cleanup,
        removed[LABEL_2],
        components[LABEL_2],
    )


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.pixels < 0:
        raise ValueError("--pixels must be >= 0")
    if args.min_voxels < 1:
        raise ValueError("--min-voxels must be >= 1")
    if args.thin_margin < 0:
        raise ValueError("--thin-margin must be >= 0")

    nifti_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    )
    if not nifti_paths:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in {args.input_dir}")

    logger.info(
        "Processing %d file(s): remove islands (labels 1, 2), adaptive erode label 1 "
        "(target=%d px, min_voxels=%d, thin_margin=%d)",
        len(nifti_paths),
        args.pixels,
        args.min_voxels,
        args.thin_margin,
    )
    logger.info("Input:  %s", args.input_dir.resolve())
    logger.info("Output: %s", args.output_dir.resolve())

    for input_path in nifti_paths:
        process_file(
            input_path,
            args.output_dir / input_path.name,
            target_pixels=args.pixels,
            min_voxels=args.min_voxels,
            thin_margin=args.thin_margin,
        )

    logger.info("Done. Wrote %d file(s) to %s", len(nifti_paths), args.output_dir)


if __name__ == "__main__":
    main()
