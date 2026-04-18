"""
config.py — Shared Configuration (DO NOT EDIT without telling everyone)

All paths and constants used across scripts live here.
Import what you need explicitly:
    from config import S2_TRAIN, FUSED_LABELS_DIR, TILE_SIZE
"""

# ── Raw data paths (populated by download_data.py) ──────────────
DATA_ROOT = "./data/makeathon-challenge"

S2_TRAIN = f"{DATA_ROOT}/sentinel-2/train"   # Sentinel-2 optical (12 bands, 10 m)
S2_TEST = f"{DATA_ROOT}/sentinel-2/test"

S1_TRAIN = f"{DATA_ROOT}/sentinel-1/train"   # Sentinel-1 radar (VV, 10 m, RTC)
S1_TEST = f"{DATA_ROOT}/sentinel-1/test"

AEF_TRAIN = f"{DATA_ROOT}/aef-embeddings/train"  # AlphaEarth 64-dim embeddings
AEF_TEST = f"{DATA_ROOT}/aef-embeddings/test"     # (EPSG:4326 — needs reprojection!)

# ── Weak-label directories ──────────────────────────────────────
LABELS_ROOT = f"{DATA_ROOT}/labels/train"
RADD_DIR = f"{LABELS_ROOT}/radd"       # Radar-based alerts
GLADL_DIR = f"{LABELS_ROOT}/gladl"     # Landsat-based alerts (per year)
GLADS2_DIR = f"{LABELS_ROOT}/glads2"   # Sentinel-2-based alerts (all years)

# ── Metadata ────────────────────────────────────────────────────
TRAIN_META = f"{DATA_ROOT}/metadata/train_tiles.geojson"
TEST_META = f"{DATA_ROOT}/metadata/test_tiles.geojson"

# ── Output directories (created automatically) ──────────────────
FUSED_LABELS_DIR = "./outputs/fused_labels"   # p2_fuse_labels.py writes here
PREDICTIONS_DIR = "./outputs/predictions"     # p3_detect.py writes here
SUBMISSION_DIR = "./submission"               # final submission GeoJSON

# ── Constants ───────────────────────────────────────────────────
TILE_SIZE = 1002         # pixels per tile edge
PIXEL_M = 10             # ground resolution in metres
DEFOR_START_YEAR = 2020  # deforestation is defined as post-2020 tree loss

