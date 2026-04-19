# Deforestation Detection — TUMAI Makeathon 26

Detect post-2020 deforestation across the tropics using 6-year AlphaEarth Foundation
(AEF) embeddings and the best available ground truth mask.

## Pipeline

```
Step 1 — Label fusion
  fuse_labels.py
  RADD + GLAD-L + GLAD-S2 + Hansen (majority vote + forest mask)
  → outputs/fused_labels/{tile}_fused.tif

Step 2 — Forest classifier (improves label quality)
  train_model1.ipynb  [conda: deep_learning, GPU]
  AEF embeddings + ESA WorldCover 2020
  → outputs/model1/improved_labels/{tile}_improved_label.tif
  → outputs/model1/forest_probs/{tile}_forest_prob_2020.tif

Step 3 — Build training dataset
  python build_dataset.py --labels combined
  combined = (fused OR model1_improved) AND was_forest_2020
  → outputs/X_train_combined.npy  (N, 384)  — 6 years × 64 AEF dims
  → outputs/y_train_combined.npy  (N,)

Step 4 — Train detector
  detector.ipynb
  sklearn MLP (384 → 256 → 128 → 64 → 1)
  → outputs/model2_sklearn.pkl

Step 5 — Generate submission
  python inference.py
  → submission/submission.geojson   ← upload this
```

## Setup

**Steps 1, 3, 5 — project venv:**

```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2 — needs GPU, use the conda env:**

```bash
conda activate deep_learning
# Open train_model1.ipynb with kernel: "Python (deep_learning CUDA)"
```

**Step 4 — detector.ipynb works in either environment.**

## Running the Pipeline

```bash
# Step 1 — fuse weak labels
python fuse_labels.py

# Step 2 — train forest classifier (optional but recommended)
# open train_model1.ipynb → Run All

# Step 3 — build dataset (combined labels, 6-year AEF)
python build_dataset.py --labels combined

# Step 4 — open detector.ipynb → Run All
# (saves outputs/model2_sklearn.pkl automatically)

# Step 5 — generate submission
python inference.py
```

Shortcut if you skip Model 1:

```bash
python fuse_labels.py
python build_dataset.py --labels fused
# open detector.ipynb → Run All
python inference.py
```

## Repository Structure

### Core pipeline

| File | Purpose |
|---|---|
| `config.py` | All shared paths and constants. **Do not edit without telling team.** |
| `loader.py` | Load Sentinel-1/2 and AEF data. Handles reprojection and AEF dequantization. |
| `fuse_labels.py` | Step 1 — majority-vote fusion of RADD+GLAD-L+GLAD-S2+Hansen → binary mask. |
| `train_model1.ipynb` | Step 2 — PyTorch MLP: AEF → forest classifier (WorldCover GT) → improved labels. |
| `build_dataset.py` | Step 3 — assembles 6-year AEF features + combined labels → `.npy` arrays. |
| `detector.ipynb` | Step 4 — sklearn MLP classifier, loads `.npy` arrays, saves trained model. |
| `inference.py` | Step 5 — loads saved model, runs all test tiles, writes `submission.geojson`. |
| `submission_utils.py` | Converts binary prediction rasters to valid GeoJSON polygons. |

### Scripts & Tools

| File | Purpose |
|---|---|
| `scripts/download_data.py` | One-time S3 challenge data download. |
| `scripts/download_worldcover.py` | Downloads ESA WorldCover tiles (needed for Step 2). |
| `scripts/visualize_tile.py` | 3×3 visualisation: S2 RGB, weak labels, fused mask, Model 1 forest probs. |

## Outputs

| Path | Contents |
|---|---|
| `outputs/fused_labels/` | Majority-vote binary masks (16 training tiles) |
| `outputs/model1/forest_probs/` | Per-year forest probability maps, 2020–2025 |
| `outputs/model1/improved_labels/` | Model 1 deforestation labels (16 training tiles) |
| `outputs/model1/forest_mlp.pt` | Trained Model 1 weights (PyTorch) |
| `outputs/X_train_{source}.npy` | 6-year AEF features (N, 384) |
| `outputs/y_train_{source}.npy` | Binary labels (N,) |
| `outputs/model2_sklearn.pkl` | Trained detector (sklearn Pipeline) |
| `outputs/figures/` | Visualisation PNGs |
| `submission/submission.geojson` | **Final submission — upload this** |

## Label Quality Hierarchy

| Source | Coverage | Quality |
|---|---|---|
| `fused` | 16 tiles | Good — 4-source majority vote + forest mask |
| `model1` | 16 tiles | Good — WorldCover-validated forest transitions |
| `combined` | 16 tiles | **Best** — union of both, filtered to confirmed forest pixels |

## Important Rules

- **NEVER** commit `data/` or `outputs/`. They are in `.gitignore`.
- **NEVER** hardcode a path. Always import from `config.py`.
- **NEVER** include Hansen `lossyear` as a model input feature — target leakage.
- **NEVER** optimise for Accuracy. The dataset is imbalanced — track Precision, Recall, F1.
