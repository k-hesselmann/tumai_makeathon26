# Deforestation Detection — TUMAI Makeathon 26

Detect post-2020 deforestation across the tropics using 6-year AlphaEarth Foundation (AEF) embeddings.

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        GROUND TRUTH                             │
│                                                                 │
│  Step 1: fuse_labels.py          Step 2: train_model1.ipynb     │
│  ──────────────────────          ────────────────────────────   │
│  Majority-vote fusion of         PyTorch MLP trained on AEF     │
│  RADD + GLAD-L + GLAD-S2 +       embeddings + ESA WorldCover    │
│  Hansen into a single binary     to classify forest pixels and  │
│  deforestation mask per tile.    produce improved defor labels. │
│                                                                 │
│   [fused_label]                   [model1_label + forest_prob]  │
│        │                                      │                 │
│        └──────────────┬───────────────────────┘                 │
│                       ▼                                         │
│             --labels fused      (Step 1 only, best so far)      │
│             --labels model1     (Step 2 only)                   │
│             --labels combined   (Step 1 ∪ Step 2, filtered)     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 3: build_dataset.py                                     │
│  ────────────────────────                                     │
│  Loads 6 years of AEF embeddings (64 dims/year = 384 total)   │
│  per pixel, aligns them with the chosen label mask, and       │
│  saves balanced X_train / y_train arrays as .npy files.       │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 4: detector.ipynb                                       │
│  ──────────────────────                                       │
│  LightGBM classifier trained on the 384-dim AEF features.    │
│  Runs inference on all test tiles and vectorises predictions  │
│  into submission/submission.geojson  ← upload this.          │
└───────────────────────────────────────────────────────────────┘
```

## Quick Start

**Steps 1, 3 — project venv:**
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2 — GPU, use conda:**
```bash
conda activate deep_learning
# Open train_model1.ipynb → Run All
```

**Steps 3, 4 — conda env (rasterio):**
```bash
C:\Users\khess\anaconda3\envs\deep_learning\python.exe build_dataset.py --labels fused
# Open detector.ipynb → Run All  →  submission/submission.geojson
```

## Label Sources

| Source | Description | Score |
|---|---|---|
| `fused` | RADD + GLAD-L + GLAD-S2 + Hansen majority vote | **34%** |
| `combined` | fused ∪ model1, filtered to confirmed forest pixels | 25% |

Use `--labels fused` (best results so far).

## Files

| File | Purpose |
|---|---|
| `config.py` | All shared paths — do not hardcode paths elsewhere |
| `fuse_labels.py` | Step 1 — majority-vote label fusion |
| `train_model1.ipynb` | Step 2 — PyTorch forest classifier (GPU) |
| `build_dataset.py` | Step 3 — assembles AEF features + labels into .npy |
| `detector.ipynb` | Step 4 — LightGBM classifier + submission generation |
| `submission_utils.py` | Vectorises binary prediction rasters to GeoJSON |
| `loader.py` | Shared raster I/O used by fuse_labels and visualize_tile |
| `scripts/download_data.py` | One-time S3 data download |
| `scripts/download_worldcover.py` | One-time WorldCover download (Step 2) |
| `scripts/visualize_tile.py` | Debug: S2 RGB + labels side-by-side |

## Rules

- **NEVER** commit `data/` or `outputs/` (gitignored)
- **NEVER** hardcode paths — import from `config.py`
- **NEVER** use Hansen `lossyear` as a model feature (target leakage)
- **NEVER** optimise for Accuracy — track F1/Precision/Recall (imbalanced dataset)