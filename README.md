# Tomography Submission — LatEpi Deep Learning Pipeline

This folder contains the code, data, and analysis outputs for the manuscript:

**Automated Deep Learning Segmentation of the Extensor Carpi Radialis Brevis Tendon Can Differentiate Between Severity Grades** (Tan et al., *Tomography*, 2026)

**Raw clinical validation MRI volumes** are not included in the repository. The folder `Clinical_Validation/Original_Clinical_Validation_Images/` (~78 subject folders, full-volume coronal PD NIfTI) is kept on local HSS storage only. All downstream outputs derived from those images — single-slice exports (`Clinical_Validation_Inference/`), predictions, eroded masks, contrast-ratio tables, and figures — are included in the repo.

## Study overview

Lateral epicondylitis (LE, “tennis elbow”) is assessed clinically from coronal proton density (PD) MRI of the elbow. Radiologists grade ECRB tendon severity on an ordinal scale:

| Score | Label |
|-------|-------|
| 0 | Normal |
| 1 | Degenerated |
| 2 | Partial thickness tear |
| 3 | Full thickness tear |

This project trains a **DeepLabv3+** semantic segmentation network (MATLAB Deep Learning Toolbox) to segment five elbow structures from coronal PD MRI:

| Label | Structure |
|-------|-----------|
| 1 | ECRB tendon |
| 2 | Common extensor (CE) muscle |
| 3 | Humerus |
| 4 | Radius |
| 5 | Ulna |

The clinical endpoint is **contrast ratio**:

```
Contrast ratio = mean PD signal in label 1 (ECRB) / mean PD signal in label 2 (CE muscle)
```

After automated segmentation, contrast ratios are compared across severity grades to test whether quantitative signal from the DL pipeline tracks clinical LE diagnosis.

### Cohorts

| Cohort | n | Purpose |
|--------|---|---------|
| Model development | 200 | Train / validate / test the segmentation network |
| Held-out test set | 20 | Quantitative overlap metrics (Dice, Jaccard, Hausdorff, volume overlap) |
| Clinical validation | 61 | Severity-graded independent cohort; contrast-ratio analysis |

---

## End-to-end workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. MODEL DEVELOPMENT (MATLAB)                                               │
│    Training_Set_Images/ + Training_Set_Segmentations/                       │
│    → ROISegmentTrain.m → MR_LatEpi_Seg_*.mat                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. INFERENCE (MATLAB + optional Python prep)                                │
│    Raw NIfTI volumes → ROIAutoSegmenter_SegModOnly.m → *_AutoSeg.nii(.gz)   │
│    Clinical volumes → ROINiiSeparateAndSort_LatEpi.m → single-slice         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. POST-PROCESSING (Python)                                                 │
│    erode_label1_predictions.py — island removal + adaptive label-1 erosion  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. ANALYSIS (Python) — model_performance/ and clinical_validation/          │
│    Test set: metrics, tables, contrast-ratio boxplot, review grid           │
│    Clinical validation: contrast ratios, statistics, segmentation grid      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
                    Tomography Resubmission/  (Figures 2–5, Tables)
```

---

## Directory structure

```
Tomography_Submission/
├── README.md                          ← this file
├── latepi_paths.py                    ← path file, used for analysis scripts (must be updated accordingly)
│
├── Model/
│   ├── ROISegmentTrain.m              ← network training
│   ├── MR_LatEpi_Seg_091825d2.mat     ← trained model weights
│   ├── ROIAutoSegmenter_SegModOnly.m  ← run inference on NIfTI volumes
│   ├── ROINiiSeparateAndSort_LatEpi.m ← MATLAB: batch slice export (clinical)
│   ├── ROINiiSeparater_LatEpi.m
│
├── post_processing/  ← on predictions
│   └── erode_label1_predictions.py
│
├── model_performance/   ← test-set analysis scripts + outputs
│   ├── evaluate_segmentation_metrics.py
│   ├── format_segmentation_metrics_table.py
│   ├── analyze_segmentation_contrast_ratio.py
│   ├── make_segmentation_case_grid.py
│
├── Clinical_Validation/
│   ├── Original_Clinical_Validation_Images/   ← 78 raw subject folders (local only; not in git)
│   ├── add_clinical_val_contrast_ratio.py
│   ├── analyze_clinical_val.py
│   ├── kruskal_dunn.py
│   └── make_clinical_val_segmentation_grid.py
```

All Python analysis scripts read paths from `latepi_paths.py`. If folders are renamed, update that file once.

---

## Requirements

### MATLAB (training + inference)

- MATLAB with **Deep Learning Toolbox** and **Image Processing Toolbox**
- Helper functions referenced in `ROISegmentTrain.m` (e.g. `ImgRead512`, `MaskRead512`, `partitionImdsPxds`, `augmentImageLabel`, `modelLoss`) must be on the MATLAB path
- Trained model file: `Model_Inferencing_Scripts/MR_LatEpi_Seg_091825d2.mat`

### Python (post-processing + analysis)

```bash
conda activate ZTEAutoAngles   # or any env with the packages below
pip install nibabel numpy scipy pandas matplotlib openpyxl
```

Tested with Python 3.9+.

---

## 1. Model development

### `Model/ROISegmentTrain.m`

Trains a **DeepLabv3+** network (`deeplabv3plus`, ResNet-50 backbone) for 512×512 coronal PD slices with **5 foreground labels** (+ background).

**Key steps inside the script:**

1. Load paired image/segmentation NIfTI datastores from `Training_Set_Images/` and `Training_Set_Segmentations/`
2. Compute **inverse-frequency class weights** from pixel counts
3. Split 80 / 10 / 10 into train / validation / test (`partitionImdsPxds`)
4. Apply on-the-fly augmentation (translation, rotation, scale, shear)
5. Train with SGDM (50 epochs, mini-batch size 1, piecewise learning-rate decay, validation patience 15)
6. Save `SegNet` and `pixelLabelID` to a dated `.mat` file (e.g. `MR_LatEpi_Seg_091825d2.mat`)

**To retrain (MATLAB):**

```matlab
% Edit user inputs at top of ROISegmentTrain.m, then run:
ROISegmentTrain
```

Point `pROIImg` and `pROISeg` at the training folders. The exported model used for inference is stored in `Model/`.

**Data in this submission (stored locally):**

| Folder | Files | Description |
|--------|-------|-------------|
| `Training_Set_Images/` | 180 | Training MRI slices |
| `Training_Set_Segmentations/` | 180 | Manual reference segmentations |
| `Test_Set_Images/` | 20 | Held-out test images |
| `Test_Set_Ground_Truth_Segmentations/` | 20 | Test reference segmentations |
| `training_split.xlsx` | — | Subject-level train/val/test assignments |

Hyperparameter optimization (batch size, class weighting, label subsets) described in the manuscript was performed interactively by editing training options and re-running this script.

---

## 2. Model inference

### `Model/ROIAutoSegmenter_SegModOnly.m`

Applies the saved `SegNet` to NIfTI volumes slice-by-slice and writes `{filename}_AutoSeg.nii` or `.nii.gz` with integer labels 0–5.

**Usage (MATLAB):**

```matlab
% Interactive:
ROIAutoSegmenter_SegModOnly

% Or programmatic:
ROIAutoSegmenter_SegModOnly(pwd, 'MR_LatEpi_Seg_091825d2.mat', fullfile(imageDir, 'P1.nii.gz'))
```

**Outputs in this submission (stored locally):**

- Test set → `Model_Development/Test_Set_AutoSeg_Predictions/` (20 files)
- Clinical validation → `Clinical_Validation/Clinical_Validation_Inference/clinical_val_pred/` (65 files)

### Clinical slice export (Python or MATLAB)

Full clinical volumes live under `Original_Clinical_Validation_Images/{Subject}/` (stored locally). The separater scripts find the slice containing label 1, reorient to LPS, and export single-slice NIfTI pairs for inference:

```bash
python roi_nii_separate_and_sort_latepi.py \
  --ext ../Clinical_Validation/Original_Clinical_Validation_Images \
  --seg-file-indexer Sgmnttn.nii.gz
```

This populates `ImgWithSeg/` and related folders under each subject, then feeds `ROIAutoSegmenter_SegModOnly.m`.

| Script | Language | Role |
|--------|----------|------|
| `roi_nii_separate_and_sort_latepi.py` | Python | Recursively find segmentations, build separation tables, call separater |
| `roi_nii_separater_latepi.py` | Python | Export `{Subject}.nii.gz` + `{Subject}_Seg.nii.gz` for labeled slices |
| `ROINiiSeparateAndSort_LatEpi.m` | MATLAB | Original version of sort script |
| `ROINiiSeparater_LatEpi.m` | MATLAB | Original version of separater |

---

## 3. Post-processing

### `post_processing/erode_label1_predictions.py`

Cleans raw AutoSeg outputs before contrast-ratio and metric calculation:

1. **Island removal** — keep largest connected component for labels 1 and 2
2. **Adaptive erosion of label 1** — distance-transform inward trim (same algorithm and parameters for **both** the test set and clinical validation):
   - **Target depth:** 6 px from the boundary for thick ECRB masks
   - **Thin masks:** erosion depth is reduced automatically so the tendon is not eliminated
   - **Safety floor:** retain ≥ 800 label-1 voxels when possible (keeps highest-interior voxels if needed)

Label 2 (CE muscle) is not eroded.

```bash
# Test set — adaptive erosion, 6 px target
python erode_label1_predictions.py \
  --input-dir ../Model_Development/Test_Set_AutoSeg_Predictions \
  --output-dir ../Model_Development/Test_Set_AutoSeg_Predictions_Eroded_FOR_MANUSCRIPT \
  --pixels 6

# Clinical validation — same adaptive erosion, 6 px target
python erode_label1_predictions.py \
  --input-dir ../Clinical_Validation/Clinical_Validation_Inference/clinical_val_pred \
  --output-dir ../Clinical_Validation/Clinical_Validation_Inference_Eroded_FOR_MANUSCRIPT \
  --pixels 6
```

---

## 4. Analysis pipeline

### Quick start — run everything (run locally, orchestrator script not within repo)

From `Tomography_Submission/`:

```bash
./run_latepi_analysis_pipeline.sh
```

This executes:

| Step | Script | Output |
|------|--------|--------|
| 1 | `erode_label1_predictions.py` | Eroded test predictions |
| 2 | `erode_label1_predictions.py` | Eroded clinical predictions |
| 3 | `evaluate_segmentation_metrics.py` | Per-subject + summary CSVs |
| 4 | `format_segmentation_metrics_table.py` | Excel table (Table 3) |
| 5 | `analyze_segmentation_contrast_ratio.py` | **Figure 3** + contrast-ratio CSVs |
| 6 | `make_segmentation_case_grid.py` | **Figure 2** |
| 7 | `add_clinical_val_contrast_ratio.py` | Updates clinical results CSV |
| 8 | `analyze_clinical_val.py` | **Figure 4** + summary CSV |
| 9 | `make_clinical_val_segmentation_grid.py` | **Figure 5** |

Optional environment overrides: `EROSION_TARGET_PX` (default 6), `MIN_VOXELS`, `THIN_MARGIN`.

---

## Script reference

### Test-set analysis (`2026-06-14-Segmentation_Results/`)

#### `evaluate_segmentation_metrics.py`

Matches test images, ground truth, and eroded predictions by subject ID (first 5 characters of filename). Computes per-label **Dice**, **Jaccard**, **Hausdorff distance**, and **volume overlap**:

```
Volume overlap = 1 − |V_pred − V_gt| / (V_pred + V_gt)
```

**Reads:** `Test_Set_Images/`, `Test_Set_Ground_Truth_Segmentations/`, `Test_Set_AutoSeg_Predictions_Eroded_FOR_MANUSCRIPT/`

**Writes:** `segmentation_metrics/` (combined CSV, summary CSVs, per-subject CSVs, Excel table)

```bash
python evaluate_segmentation_metrics.py
```

#### `format_segmentation_metrics_table.py`

Formats `segmentation_metrics_all_labels.csv` into a manuscript-style Excel table (`mean ± SD [min, max]`).

```bash
python format_segmentation_metrics_table.py
```

#### `analyze_segmentation_contrast_ratio.py`

For each test subject, computes contrast ratio from ground-truth and predicted segmentations, then:

- Saves per-subject and summary CSVs
- Runs **Mann–Whitney U** test (GT vs prediction)
- Generates **Figure 3** (boxplot)

```bash
python analyze_segmentation_contrast_ratio.py
```

#### `make_segmentation_case_grid.py`

Builds **Figure 2**: five columns per case (case label, source image, GT overlay, prediction overlay, label-1 metrics). Default: 5 test cases.

```bash
python make_segmentation_case_grid.py
python make_segmentation_case_grid.py --subjects s0004 s0009 s0010 s0012 s0013
```

---

### Clinical validation analysis (`Clinical_Validation/`)

#### `add_clinical_val_contrast_ratio.py`

Computes predicted contrast ratio for each subject in `clinical_val_pred_eroded/` using the matching image in `ImgWithSeg/`. Appends `Predicted_Contrast_Ratio` to `20260614-Clinical_Val_Results.csv`.

```bash
python add_clinical_val_contrast_ratio.py
```

**Note:** The clinical CSV contains `Patient`, `Score`, and `Predicted_Contrast_Ratio` only (legacy manual `Values` column was removed).

#### `analyze_clinical_val.py`

Groups patients by severity score (0–3), prints descriptive statistics, runs **Kruskal–Wallis**, then **Dunn post-hoc** with **Holm correction** for adjacent pairs (0 vs 1, 1 vs 2, 2 vs 3). Saves summary CSV and boxplot.

```bash
python analyze_clinical_val.py --values-column Predicted_Contrast_Ratio
```

Generates **Figure 4**.

#### `kruskal_dunn.py`

Statistical helper module (not run directly). Imported by `analyze_clinical_val.py`.

#### `make_clinical_val_segmentation_grid.py`

Builds **Figure 5**: 4 severity columns × 4 curated cases. Each case shows the full image with ECRB overlay + crop box, and a zoomed inset with connectors.

```bash
python make_clinical_val_segmentation_grid.py
```

Curated case lists are defined in the script (`CURATED_SCORE_SUBJECTS`).

---

## Manuscript outputs

| Manuscript item | Generated by | Path |
|-----------------|--------------|------|
| **Figure 1** | Manual (workflow schematic) | `Tomography Resubmission/...Figure 1.png` |
| **Figure 2** | `make_segmentation_case_grid.py` | `Tomography Resubmission/...Figure 2.png` |
| **Figure 3** | `analyze_segmentation_contrast_ratio.py` | `Tomography Resubmission/...Figure 3.png` |
| **Figure 4** | `analyze_clinical_val.py` | `Tomography Resubmission/...Figure 4.png` |
| **Figure 5** | `make_clinical_val_segmentation_grid.py` | `Tomography Resubmission/...Figure 5.png` |
| **Table 3** | `format_segmentation_metrics_table.py` | `segmentation_metrics/segmentation_metrics_table.xlsx` |

Intermediate CSVs are kept alongside figures for reproducibility.

---

## Key results (2026-06-14 run)

### Test set (n = 20)

| Structure | Dice | Jaccard | Hausdorff (px) | Volume overlap |
|-----------|------|---------|----------------|----------------|
| ECRB tendon | 0.527 ± 0.168 | 0.375 ± 0.162 | 45.9 ± 33.3 | 0.858 ± 0.111 |
| CE muscle | 0.902 ± 0.057 | 0.827 ± 0.088 | 64.1 ± 56.7 | 0.914 ± 0.056 |

Predicted contrast ratios were higher than ground truth (Mann–Whitney **p = 0.013**).

### Clinical validation (n = 61)

Predicted contrast ratios increased with severity (Kruskal–Wallis **p < 0.001**). Adjacent-pair testing: significant difference between **degeneration vs partial thickness tear** (Holm-corrected **p = 0.013**).

---

## Data notes

- **Raw clinical validation images** (`Clinical_Validation/Original_Clinical_Validation_Images/`) are excluded from the GitHub repository and retained on local HSS storage. Clone the repo and place that folder locally if you need to re-run slice export or inference from scratch; otherwise use the included `Clinical_Validation_Inference/` outputs.
- **65 clinical predictions** vs **61 CSV rows** — P12, P17, P3, P35 have predictions but no severity score in the validation spreadsheet.
- **78 original clinical folders** vs **65 inferred cases** — 13 subjects lack a usable AutoSeg export.
- Erosion uses the **same adaptive algorithm** for test and clinical cohorts (6 px target depth, 800-voxel floor). Effective depth per case is logged by `erode_label1_predictions.py`.

---

## Citation

If you use this code or data, please cite the Tomography manuscript (Tan et al., 2026) and acknowledge IRB protocol **2022-1920** (Hospital for Special Surgery).

---

## Contact

Jack Consolini / Alexa H. Tan — Department of Radiology and Imaging, Hospital for Special Surgery
# Automated.Deep.Learning.Segmentation.of.the.ECRB.Tendon.Differentiates.LE.Severity-Grades
