"""
Shared paths for the Tomography submission analysis pipeline.

All scripts under Tomography_Submission/ import from this module so data
folders and manuscript figure names stay consistent.
"""

from __future__ import annotations

from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parent

# ── Model development (held-out test set) ───────────────────────────────────
MODEL_DEVELOPMENT = SUBMISSION_ROOT / "Model_Development"
TEST_IMAGES = MODEL_DEVELOPMENT / "Test_Set_Images"
TEST_GROUND_TRUTH = MODEL_DEVELOPMENT / "Test_Set_Ground_Truth_Segmentations"
TEST_PREDICTIONS = MODEL_DEVELOPMENT / "Test_Set_AutoSeg_Predictions"
TEST_PREDICTIONS_ERODED = (
    MODEL_DEVELOPMENT / "Test_Set_AutoSeg_Predictions_Eroded_FOR_MANUSCRIPT"
)
TRAINING_IMAGES = MODEL_DEVELOPMENT / "Training_Set_Images"
TRAINING_SEGMENTATIONS = MODEL_DEVELOPMENT / "Training_Set_Segmentations"

# ── Clinical validation ───────────────────────────────────────────────────────
CLINICAL_VALIDATION = SUBMISSION_ROOT / "Clinical_Validation"
CLINICAL_INFERENCE = CLINICAL_VALIDATION / "Clinical_Validation_Inference"
CLINICAL_PREDICTIONS = CLINICAL_INFERENCE / "clinical_val_pred"
CLINICAL_IMAGES = CLINICAL_INFERENCE / "ImgWithSeg"
CLINICAL_GROUND_TRUTH = CLINICAL_INFERENCE / "Ground_Truth_Seg_2024_Analysis"
CLINICAL_PREDICTIONS_ERODED = (
    CLINICAL_VALIDATION / "Clinical_Validation_Inference_Eroded_FOR_MANUSCRIPT"
)
ORIGINAL_CLINICAL_IMAGES = (
    CLINICAL_VALIDATION / "Original_Clinical_Validation_Images"
)

# ── Analysis outputs ─────────────────────────────────────────────────────────
SEGMENTATION_RESULTS = SUBMISSION_ROOT / "2026-06-14-Segmentation_Results"
SEGMENTATION_METRICS_DIR = SEGMENTATION_RESULTS / "segmentation_metrics"
CONTRAST_RATIO_DIR = SEGMENTATION_RESULTS / "contrast_ratio_analysis"

CLINICAL_RESULTS = SUBMISSION_ROOT / "2026-06-14-Clinical_Validation_Results"
CLINICAL_RESULTS_CSV = CLINICAL_RESULTS / "20260614-Clinical_Val_Results.csv"

# ── Script locations ─────────────────────────────────────────────────────────
POST_PROCESSING_DIR = SUBMISSION_ROOT / "Segmentation_Post_Processing_Scripts"
ERODE_SCRIPT = POST_PROCESSING_DIR / "erode_label1_predictions.py"
INFERENCE_DIR = SUBMISSION_ROOT / "Model_Inferencing_Scripts"
SEGMENTATION_SCRIPTS_DIR = SEGMENTATION_RESULTS
CLINICAL_SCRIPTS_DIR = CLINICAL_VALIDATION

# ── Manuscript figures (Tomography Resubmission/) ───────────────────────────
FIGURES_DIR = SUBMISSION_ROOT / "Tomography Resubmission"
FIGURE_PREFIX = "Tan et al. Tomography 2026 - Figure"

# Figure 2: test-set segmentation review grid (image, GT, prediction, metrics)
FIGURE_2 = FIGURES_DIR / f"{FIGURE_PREFIX} 2.png"
# Figure 3: test-set ground-truth vs predicted contrast-ratio boxplot
FIGURE_3 = FIGURES_DIR / f"{FIGURE_PREFIX} 3.png"
# Figure 4: clinical-validation predicted contrast-ratio boxplot by severity
FIGURE_4 = FIGURES_DIR / f"{FIGURE_PREFIX} 4.png"
# Figure 5: clinical-validation segmentation grid by severity score
FIGURE_5 = FIGURES_DIR / f"{FIGURE_PREFIX} 5.png"

# Legacy / intermediate figure names (also written beside manuscript copies)
LEGACY_SEGMENTATION_CASE_GRID = SEGMENTATION_RESULTS / "segmentation_case_grid.png"
LEGACY_SEGMENTATION_CR_BOXPLOT = (
    CONTRAST_RATIO_DIR / "segmentation_contrast_ratio_boxplot.png"
)
LEGACY_CLINICAL_VAL_GRID = CLINICAL_VALIDATION / "clinical_val_segmentation_grid.png"
LEGACY_CLINICAL_VAL_BOXPLOT = (
    CLINICAL_RESULTS / "clinical_val_boxplot_predicted_contrast_ratio.png"
)

# ── Erosion defaults (same adaptive settings for test + clinical) ─────────────
EROSION_TARGET_PX = 6
EROSION_MIN_VOXELS = 800
EROSION_THIN_MARGIN = 2


def setup_import_paths() -> None:
    """Ensure submission root and segmentation scripts are importable."""
    import sys

    for path in (SUBMISSION_ROOT, SEGMENTATION_RESULTS):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
