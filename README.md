# Deforestation Detection Hackathon

Deforestation detection hackathon project to identify post-2020 deforestation using multimodal satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical) and 64-dimensional foundation model embeddings.

## Setup

Ensure everyone is using the exact same environment to avoid library conflicts.

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

## Data

Place the downloaded dataset into the `./data/makeathon-challenge/` directory as configured in `config.py`. The directory structure should look like this:

```text
data/makeathon-challenge/
  ├── sentinel-1/
  ├── sentinel-2/
  ├── aef-embeddings/
  ├── labels/
  └── metadata/
```

## File Ownership Table

| File                | Owner    | Description                                                                                              |
| ------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `config.py`         | Person 1 | Shared configuration, do not edit without telling everyone                                               |
| `p1_loader.py`      | Person 1 | Data loading. **Crucial:** Resize 64D embeddings to match Sentinel resolution using `scipy.ndimage.zoom` |
| `p2_fuse_labels.py` | Person 2 | Label fusion logic. Fuses RADD, GLAD-L, GLAD-S2 into a single clean binary label per tile                |
| `p3_detect.py`      | Person 3 | NDVI Change Detection. Detects deforestation by finding NDVI drops after 2020                            |
| `p4_sentinel1.py`   | Person 4 | Sentinel-1 Radar Confidence Layer + Visualisation                                                        |

## Important Rules

- **NEVER** commit the `data/` folder. Ensure it is in `.gitignore`.
- **NEVER** hardcode a path. Always import from `config.py`.
- **NEVER** train on raw, unfiltered weak labels. Always pass them through `p2_fuse_labels.py` first.
- **NEVER** rely purely on manual NDVI thresholds. The 64D embeddings are the "cheat code" and must be used in the simple MLP/SVM.
- **NEVER** optimize for Accuracy. The dataset is massively imbalanced; optimize strictly for Precision and Recall.
