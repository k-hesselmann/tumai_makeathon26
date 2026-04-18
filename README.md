# Deforestation Detection Hackathon

Detect post-2020 deforestation using multimodal satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical) and 64-dimensional AlphaEarth foundation model embeddings.

## Setup

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
python scripts/download_data.py
```

Expected layout (paths configured in `config.py`):

```text
data/makeathon-challenge/
  ├── sentinel-2/       # Optical, 12 bands, monthly, UTM
  ├── sentinel-1/       # Radar, VV polarisation, monthly, UTM
  ├── aef-embeddings/   # AlphaEarth 64-dim, annual, EPSG:4326 (needs reprojection!)
  ├── labels/           # RADD + GLAD-L + GLAD-S2 weak labels (train only)
  └── metadata/         # train_tiles.geojson, test_tiles.geojson
```

## Pipeline

```text
┌──────────────────┐     ┌──────────────────┐
│  fuse_labels.py  │     │ ndvi_features.py │
│  RADD+GLAD →     │     │ NDVI drop map    │
│  binary mask     │     │ (optional extra) │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼  (optional)
┌────────────────────────────────────────────┐
│           build_dataset.py                 │
│  AEF embeddings + fused labels             │
│   → outputs/X_train.npy  (N, 64)           │
│   → outputs/y_train.npy  (N,)              │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│            Classifier (MLP)                │
│  Train on X_train / y_train                │
│  → predict on test tiles → submission      │
└────────────────────────────────────────────┘
```

## Repository Structure

### Core pipeline

| File               | Purpose                                                               |
| ------------------ | --------------------------------------------------------------------- |
| `config.py`        | All shared paths and constants. **Do not edit without telling team.** |
| `loader.py`        | Load Sentinel-2, Sentinel-1, and AEF data. Handles CRS reprojection.  |
| `fuse_labels.py`   | Majority-vote fusion of 3 weak label sources → clean binary mask.     |
| `build_dataset.py` | Assembles AEF + fused labels → `X_train.npy` / `y_train.npy`.         |

### Feature extraction (optional extras for the classifier)

| File                | Purpose                                                               |
| ------------------- | --------------------------------------------------------------------- |
| `ndvi_features.py`  | NDVI change detection — baseline (2020) vs recent vegetation drop.    |
| `radar_features.py` | Sentinel-1 radar confidence layer (stub — implement if time permits). |

### Auxiliary

| File / Dir                 | Purpose                    |
| -------------------------- | -------------------------- |
| `scripts/download_data.py` | One-time S3 data download. |
| `requirements.txt`         | Python dependencies.       |

## Important Rules

- **NEVER** commit `data/` or `outputs/`. They are in `.gitignore`.
- **NEVER** hardcode a path. Always import from `config.py`.
- **NEVER** train on raw weak labels. Always use the fused output from `fuse_labels.py`.
- **NEVER** rely purely on NDVI thresholds. The 64D AEF embeddings are the primary signal.
- **NEVER** optimize for Accuracy. The dataset is massively imbalanced — use Precision and Recall.
