# Deforestation Detection Hackathon

Detect post-2020 deforestation using multimodal satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical) and 64-dimensional AlphaEarth foundation model embeddings.

## What We Are Building

We are building a machine learning pipeline that avoids "target leakage" while maximizing label quality:

1. **Ground Truth Generation**: We fuse 3 noisy weak label sources (RADD, GLAD-L, GLAD-S2) using a majority vote, then add Hansen Global Forest Change as a 4th voter and use its `treecover2000` layer as an absolute forest mask to filter out false positives.
2. **Feature Extraction**: We extract 64-dimensional AlphaEarth foundation model embeddings (AEF) for every pixel. These are the **only** inputs to the model.
3. **Model Training**: A neural network learns to map those 64 numbers to our high-quality fused labels. Hansen is **never** seen as an input feature — only as a label source.
4. **Inference**: The trained model predicts deforestation on the test set and exports a GeoJSON submission.

## How Hansen Data Is Used

### Spatial alignment
Hansen Global Forest Change tiles are 10×10 degree global granules at 30 m resolution. Our challenge tiles are small ~10×10 km patches at 10 m resolution in various UTM projections. The script `data/p5_hansen.py` clips and reprojects each Hansen tile to **exactly** match the challenge tile's CRS, transform, width, and height. After clipping, every Hansen pixel is perfectly aligned with the corresponding Sentinel-2 / AEF pixel — same grid, same dimensions.

### Temporal encoding
Hansen does not produce monthly time series like Sentinel. Instead it provides two layers:
- **`treecover2000`** — A single snapshot: canopy cover percentage in the year 2000 (0–100%). We threshold at ≥30% to create a binary "was this ever forest?" mask.
- **`lossyear`** — A single raster where the pixel *value* encodes the year that forest loss was detected. Value `21` = loss in 2021, `22` = 2022, `23` = 2023, `24` = 2024. Value `0` = no loss detected.

These two layers are enough to answer: "Was this pixel forest, and did it lose that forest after 2020?"

### Role in label fusion (`fuse_labels.py`)
Hansen participates in the pipeline in two ways:
1. **4th voter**: If Hansen detected loss in 2021–2024 at a pixel, it casts a vote alongside RADD, GLAD-L, and GLAD-S2. A pixel needs ≥2 votes out of 4 to be labelled as deforestation.
2. **Absolute forest mask**: After voting, any pixel where `treecover2000 < 30%` is forced to 0 regardless of votes. This eliminates false positives in areas that were never forest (agriculture, shrublands, etc.).

## The Neural Network

A teammate is building a neural network that takes the **64-dimensional AEF embeddings** as input and learns to predict deforestation. The key design principle:

- **Inputs**: Only the 64 AEF numbers per pixel. These come from AlphaEarth's foundation model, which was pre-trained on satellite imagery to produce rich, general-purpose feature vectors. The network never sees Hansen, RADD, GLAD, or any label source as an input.
- **Labels**: The fused binary mask from `fuse_labels.py` (which uses Hansen + RADD + GLAD internally to create clean ground truth).
- **What it learns**: "Given these 64 numbers describing a pixel, does it look like deforestation happened?"

### Why this is not circular
Hansen is itself a deforestation detection model (built by University of Maryland using decades of Landsat data). If we fed Hansen's output *into* the network as a feature, the network would just learn to copy it — producing a worse version of something that already exists. Instead, we use Hansen only to create the best possible training labels, and we train the network on **completely different signals** (the AEF embeddings, derived from Sentinel-2 imagery that Hansen never used). This means the network can potentially:
- Detect deforestation **faster** (monthly Sentinel-2 vs annual Landsat)
- Detect deforestation in areas where **Landsat had cloud cover**
- Generalise to regions and time periods **beyond Hansen's coverage**

## Setup

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Data Download & Prep

1. **Challenge Data** (Sentinel & AEF embeddings):
   ```bash
   python scripts/download_data.py
   ```
2. **Hansen Global Forest Change Data** (For label filtering):
   ```bash
   python data/download_hansen.py
   python data/p5_hansen.py
   ```

## Pipeline

```text
┌─────────────────────────┐        ┌─────────────────────────┐
│     GROUND TRUTH        │        │        FEATURES         │
│     (fuse_labels.py)    │        │       (loader.py)       │
│ RADD+GLAD+Hansen -> Mask│        │ AlphaEarth 64D Embeds   │
└────────────┬────────────┘        └────────────┬────────────┘
             │                                  │
             │     ┌──────────────────┐         │
             └────>│ build_dataset.py │<────────┘
                   │ Flattens arrays  │
                   └─────────┬────────┘
                             │
                             v
              ┌─────────────────────────────┐
              │      Classifier (MLP)       │
              │ Inputs: X_train (N, 64) AEF │
              │ Target: y_train (N,) labels │
              │ -> Learns 64D relationship  │
              └──────────────┬──────────────┘
                             │
                             v
              ┌─────────────────────────────┐
              │        inference.py         │
              │ Predicts Test Set -> GeoJSON│
              └─────────────────────────────┘
```

## Repository Structure

### Core pipeline

| File               | Purpose                                                               |
| ------------------ | --------------------------------------------------------------------- |
| `config.py`        | All shared paths and constants. **Do not edit without telling team.** |
| `loader.py`        | Load Sentinel-2, Sentinel-1, and AEF data. Handles CRS reprojection.  |
| `fuse_labels.py`   | Majority-vote fusion of weak labels + Hansen filtering -> clean mask. |
| `build_dataset.py` | Assembles AEF + fused labels -> `X_train.npy` / `y_train.npy`.        |
| `inference.py`     | Runs model on test data, generates final `submission.geojson`.        |
| `submission_utils.py`| Helper functions to convert raster predictions to valid GeoJSON.    |

### Feature extraction (optional extras for the classifier)

| File                | Purpose                                                               |
| ------------------- | --------------------------------------------------------------------- |
| `ndvi_features.py`  | NDVI change detection — baseline (2020) vs recent vegetation drop.    |
| `radar_features.py` | Sentinel-1 radar confidence layer (stub — implement if time permits). |

### Auxiliary / Data Tools

| File / Dir                 | Purpose                                   |
| -------------------------- | ----------------------------------------- |
| `scripts/download_data.py` | One-time S3 challenge data download.      |
| `data/download_hansen.py`  | Downloads global Hansen Forest data.      |
| `data/p5_hansen.py`        | Clips Hansen data to challenge tile grids.|
| `requirements.txt`         | Python dependencies.                      |

## Important Rules

- **NEVER** commit `data/` or `outputs/`. They are in `.gitignore`.
- **NEVER** hardcode a path. Always import from `config.py`.
- **NEVER** train on raw weak labels. Always use the fused output from `fuse_labels.py`.
- **NEVER** include Hansen `lossyear` as an input feature (target leakage!).
- **NEVER** optimize for Accuracy. The dataset is massively imbalanced — use Precision and Recall.
