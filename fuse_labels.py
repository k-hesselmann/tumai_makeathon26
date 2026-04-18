"""
fuse_labels.py — Label Fusion

PURPOSE:
    The challenge provides 3 independent "weak label" sources for
    deforestation.  Each one is noisy and incomplete on its own.
    This script fuses them with a majority vote to produce a single,
    higher-confidence binary mask per tile.

LABEL SOURCES:
    1. RADD  — Radar-based (Sentinel-1).  One file per tile.
       Encoding: 0 = no alert, leading digit 2 = low conf, 3 = high conf,
       remaining digits = days since 2014-12-31.
    2. GLAD-L — Landsat optical.  One file per year (alert21.tif, etc.).
       Encoding: 0 = no loss, 2 = probable, 3 = confirmed.
    3. GLAD-S2 — Sentinel-2 optical.  One file per tile (all years).
       Encoding: 0 = no loss, 1-4 = increasing confidence.

FUSION LOGIC:
    A pixel is labelled as "deforestation" if >= 2 of the 3 sources
    agree that there was an alert at that pixel (any confidence level).

OUTPUT:
    ./outputs/fused_labels/{tile_id}_fused.tif  — binary mask (0 or 1)

USAGE:
    python fuse_labels.py
"""

import pathlib
import rasterio
import numpy as np
from config import RADD_DIR, GLADL_DIR, GLADS2_DIR, FUSED_LABELS_DIR, S2_TRAIN
from loader import save_raster, load_s2, list_available_months

def load_label(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1)

def fuse_labels(tile_id):
    # We need reference_meta to save the output properly aligned.
    # Load the first available Sentinel-2 tile metadata for this tile_id.
    months = list_available_months(tile_id, data_split="train")
    if not months:
        print(f"No Sentinel-2 data to derive metadata for {tile_id}")
        return
    
    year, month = months[0]
    _, reference_meta = load_s2(tile_id, year, month, data_split="train")
    
    height, width = reference_meta['height'], reference_meta['width']
    
    # 1. RADD
    radd_path = f"{RADD_DIR}/radd_{tile_id}_labels.tif"
    radd_data = load_label(radd_path)
    if radd_data is not None:
        radd_mask = (radd_data > 0).astype(np.uint8)
    else:
        radd_mask = np.zeros((height, width), dtype=np.uint8)

    # 2. GLAD-L (Post-2020)
    gladl_mask = np.zeros((height, width), dtype=np.uint8)
    for yy in ["21", "22", "23", "24", "25"]:
        gladl_path = f"{GLADL_DIR}/gladl_{tile_id}_alert{yy}.tif"
        data = load_label(gladl_path)
        if data is not None:
            gladl_mask |= (data > 0).astype(np.uint8)

    # 3. GLAD-S2
    glads2_path = f"{GLADS2_DIR}/glads2_{tile_id}_alert.tif"
    glads2_data = load_label(glads2_path)
    if glads2_data is not None:
        glads2_mask = (glads2_data > 0).astype(np.uint8)
    else:
        glads2_mask = np.zeros((height, width), dtype=np.uint8)

    # Majority vote: >= 2 sources
    vote_sum = radd_mask + gladl_mask + glads2_mask
    fused_mask = (vote_sum >= 2).astype(np.uint8)

    # Save to FUSED_LABELS_DIR
    out_path = f"{FUSED_LABELS_DIR}/{tile_id}_fused.tif"
    save_raster(fused_mask, reference_meta, out_path)
    print(f"Saved fused labels for {tile_id} to {out_path}")

if __name__ == "__main__":
    # Iterate over all training tiles and fuse labels
    s2_train_path = pathlib.Path(S2_TRAIN)
    if s2_train_path.exists():
        for tile_dir in s2_train_path.iterdir():
            if tile_dir.is_dir() and tile_dir.name.endswith("__s2_l2a"):
                tile_id = tile_dir.name.replace("__s2_l2a", "")
                print(f"Processing {tile_id}...")
                fuse_labels(tile_id)
    else:
        print(f"Data directory not found: {S2_TRAIN}. Wait for the download to finish.")
