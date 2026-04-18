"""
ndvi_features.py — NDVI Change Detection

PURPOSE:
    Produces a per-pixel "vegetation drop" map by comparing the NDVI
    (Normalized Difference Vegetation Index) before and after 2020.

HOW IT WORKS:
    NDVI = (NIR - Red) / (NIR + Red)
    Healthy vegetation → NDVI close to +1
    Bare soil / urban   → NDVI close to  0

    For each tile we compute:
      baseline = max NDVI across all months in 2020
      recent   = min NDVI across all months after 2020
      drop     = baseline - recent

    A large positive drop means that pixel went from green to bare,
    which is a strong indicator of deforestation.

OUTPUT:
    ./outputs/predictions/{tile_id}_ndvi_drop.tif  — float32 drop map

NOTE:
    This is a feature map, NOT a final prediction.  The classifier
    (Person 3's MLP) may optionally use this as an extra input
    alongside the 64D AEF embeddings.

USAGE:
    python ndvi_features.py
"""

import pathlib
import numpy as np
from config import DEFOR_START_YEAR, PREDICTIONS_DIR, S2_TRAIN
from loader import load_s2, list_available_months, save_raster

def compute_ndvi(bands):
    # Sentinel-2 bands: B04 (Red) is index 3, B08 (NIR) is index 7
    red = bands[3]
    nir = bands[7]
    denominator = (nir + red)
    ndvi = np.divide((nir - red), denominator, out=np.zeros_like(red), where=(denominator != 0))
    return ndvi

def detect_deforestation(tile_id):
    months = list_available_months(tile_id, data_split="train")
    if not months:
        print(f"No Sentinel-2 data available for {tile_id}")
        return

    baseline_ndvis = []
    post_2020_ndvis = []
    reference_meta = None

    for year, month in months:
        bands, meta = load_s2(tile_id, year, month, data_split="train")
        if bands is None:
            continue
            
        if reference_meta is None:
            reference_meta = meta
            
        ndvi = compute_ndvi(bands)
        
        if year == DEFOR_START_YEAR:
            baseline_ndvis.append(ndvi)
        elif year > DEFOR_START_YEAR:
            post_2020_ndvis.append(ndvi)
            
    if not baseline_ndvis or not post_2020_ndvis:
        print(f"Skipping {tile_id} - insufficient data across years for drop calculation.")
        return
        
    # Max NDVI in the baseline year
    baseline_max = np.max(np.array(baseline_ndvis), axis=0)
    # Min NDVI in the post-baseline years
    post_2020_min = np.min(np.array(post_2020_ndvis), axis=0)
    
    # Positive drop indicates vegetation loss
    ndvi_drop = baseline_max - post_2020_min
    
    out_path = f"{PREDICTIONS_DIR}/{tile_id}_ndvi_drop.tif"
    save_raster(ndvi_drop, reference_meta, out_path)
    print(f"Saved NDVI drop map for {tile_id} to {out_path}")

if __name__ == "__main__":
    s2_train_path = pathlib.Path(S2_TRAIN)
    if s2_train_path.exists():
        for tile_dir in s2_train_path.iterdir():
            if tile_dir.is_dir() and tile_dir.name.endswith("__s2_l2a"):
                tile_id = tile_dir.name.replace("__s2_l2a", "")
                print(f"Processing {tile_id}...")
                detect_deforestation(tile_id)
    else:
        print(f"Data directory not found: {S2_TRAIN}. Wait for the download to finish.")
