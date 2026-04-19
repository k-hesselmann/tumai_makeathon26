# Deforestation Detection — TUMAI Makeathon 26

Submission for the [TUMAI Makeathon 26]([https://www.tum-ai.com/]) challenge: detect post-2020 tropical deforestation using satellite imagery and machine learning.

The core idea is to combine multiple noisy, publicly available deforestation alert systems into a high-quality ground truth mask, then train a gradient boosting classifier on 6-year temporal embeddings from the AlphaEarth Foundation (AEF) model — a self-supervised geospatial foundation model producing 64-dimensional patch embeddings per year.

---

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
│  LightGBM classifier trained on the 384-dim AEF features.     │
│  Runs inference on all test tiles and vectorises predictions  │
│  into submission/submission.geojson.                          │
└───────────────────────────────────────────────────────────────┘
```

---

## Setup

**Requirements:** Python 3.9+, a GPU-capable conda environment for Step 2 (PyTorch), and a standard venv for everything else.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Step 2 (`train_model1.ipynb`) requires PyTorch with CUDA. Use a conda environment with GPU support and open the notebook with that kernel.

---

## Running the Pipeline

```bash
# Step 1 — fuse weak deforestation labels
python fuse_labels.py

# Step 2 — train forest classifier (optional, improves label quality)
# Open train_model1.ipynb and run all cells with a GPU kernel

# Step 3 — build training dataset from AEF embeddings
python build_dataset.py --labels fused   # or: model1 | combined

# Step 4 — train detector and generate submission
# Open detector.ipynb and run all cells
# → submission/submission.geojson
```

---

## Label Sources

Three label strategies are available via `--labels`:

| Strategy   | Description                                                | Leaderboard |
| ---------- | ---------------------------------------------------------- | ----------- |
| `fused`    | Majority vote across RADD, GLAD-L, GLAD-S2, Hansen         | **34%**     |
| `combined` | `fused ∪ model1`, filtered to confirmed 2020 forest pixels | 25%         |
| `model1`   | Forest classifier output only                              | 17%         |

---

## Repository Structure

| File                             | Purpose                                                      |
| -------------------------------- | ------------------------------------------------------------ |
| `config.py`                      | Central config for all data paths and constants              |
| `fuse_labels.py`                 | Step 1 — majority-vote label fusion across alert systems     |
| `train_model1.ipynb`             | Step 2 — PyTorch forest pixel classifier (GPU)               |
| `build_dataset.py`               | Step 3 — assembles 384-dim AEF features + labels into `.npy` |
| `detector.ipynb`                 | Step 4 — LightGBM classifier, inference, submission export   |
| `submission_utils.py`            | Converts binary prediction rasters to GeoJSON polygons       |
| `loader.py`                      | Shared raster I/O utilities                                  |
| `scripts/download_data.py`       | One-time download of challenge data from S3                  |
| `scripts/download_worldcover.py` | One-time download of ESA WorldCover tiles                    |
| `scripts/visualize_tile.py`      | Generates presentation figures for a given tile              |
