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
    4. Hansen — Global Forest Change (lossyear + treecover2000).
       Used as 4th voter AND as a forest mask.

FUSION LOGIC:
    A pixel is labelled as "deforestation" if >= 2 of the 4 sources
    agree, AND the pixel had >= 30% tree cover in 2000 (Hansen mask).

NOTE:
    Label sources come at different resolutions. RADD/GLAD-S2 are
    ~905x899, GLAD-L is ~362x360, while S2/Hansen are 1002x1002.
    All labels are resampled to the S2 reference grid (1002x1002)
    using nearest-neighbour interpolation before voting.

OUTPUT:
    ./outputs/fused_labels/{tile_id}_fused.tif  -- binary mask (0 or 1)

USAGE:
    python fuse_labels.py
"""

import pathlib
import rasterio
import numpy as np
from scipy.ndimage import zoom
from config import RADD_DIR, GLADL_DIR, GLADS2_DIR, FUSED_LABELS_DIR, S2_TRAIN

import sys
sys.path.append("./data")
from p5_hansen import load_hansen_for_tile
from loader import save_raster, load_s2, list_available_months


def load_label(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1)


def resize_to(arr, target_h, target_w):
    """Resize a 2D array to (target_h, target_w) using nearest-neighbour."""
    if arr is None:
        return None
    if arr.shape == (target_h, target_w):
        return arr
    return zoom(
        arr.astype(np.float32),
        (target_h / arr.shape[0], target_w / arr.shape[1]),
        order=0,  # nearest-neighbour for categorical/binary data
    ).astype(arr.dtype)


def fuse_labels(tile_id):
    # We need reference_meta to save the output properly aligned.
    # Load the first available Sentinel-2 tile metadata for this tile_id.
    months = list_available_months(tile_id, data_split="train")
    if not months:
        print(f"  [skip] No Sentinel-2 data to derive metadata for {tile_id}")
        return

    year, month = months[0]
    _, reference_meta = load_s2(tile_id, year, month, data_split="train")

    height, width = reference_meta['height'], reference_meta['width']

    # 1. RADD
    radd_path = f"{RADD_DIR}/radd_{tile_id}_labels.tif"
    radd_data = load_label(radd_path)
    if radd_data is not None:
        radd_binary = (radd_data > 0).astype(np.uint8)
        radd_mask = resize_to(radd_binary, height, width)
    else:
        radd_mask = np.zeros((height, width), dtype=np.uint8)

    # 2. GLAD-L (Post-2020)
    gladl_mask = np.zeros((height, width), dtype=np.uint8)
    for yy in ["21", "22", "23", "24", "25"]:
        gladl_path = f"{GLADL_DIR}/gladl_{tile_id}_alert{yy}.tif"
        data = load_label(gladl_path)
        if data is not None:
            binary = (data > 0).astype(np.uint8)
            resized = resize_to(binary, height, width)
            gladl_mask |= resized

    # 3. GLAD-S2
    glads2_path = f"{GLADS2_DIR}/glads2_{tile_id}_alert.tif"
    glads2_data = load_label(glads2_path)
    if glads2_data is not None:
        glads2_binary = (glads2_data > 0).astype(np.uint8)
        glads2_mask = resize_to(glads2_binary, height, width)
    else:
        glads2_mask = np.zeros((height, width), dtype=np.uint8)

    # 4. Hansen Global Forest Change
    hansen_loss, treecover = load_hansen_for_tile(tile_id)
    if hansen_loss is not None:
        # Vote 1 if Hansen detected loss between 2021 and 2024
        hansen_mask = ((hansen_loss >= 21) & (hansen_loss <= 24)).astype(np.uint8)
    else:
        hansen_mask = np.zeros((height, width), dtype=np.uint8)
        treecover = np.ones((height, width), dtype=np.uint8) * 100  # assume forest if missing

    # Majority vote: >= 2 sources out of 4
    vote_sum = radd_mask + gladl_mask + glads2_mask + hansen_mask
    fused_mask = (vote_sum >= 2).astype(np.uint8)

    # Absolute constraint: deforestation can only happen where there was forest!
    # Mask out any votes in areas with < 30% tree canopy cover in year 2000
    was_forest = (treecover >= 30).astype(np.uint8)
    fused_mask = (fused_mask & was_forest)

    # Save to FUSED_LABELS_DIR
    out_path = f"{FUSED_LABELS_DIR}/{tile_id}_fused.tif"
    save_raster(fused_mask, reference_meta, out_path)

    n_defor = int(fused_mask.sum())
    pct = 100.0 * n_defor / fused_mask.size
    print(f"  Saved {tile_id}: {n_defor:,} deforestation pixels ({pct:.1f}%) -> {out_path}")


if __name__ == "__main__":
    # Iterate over all training tiles and fuse labels
    s2_train_path = pathlib.Path(S2_TRAIN)
    if s2_train_path.exists():
        tiles = sorted([
            d.name.replace("__s2_l2a", "")
            for d in s2_train_path.iterdir()
            if d.is_dir() and d.name.endswith("__s2_l2a")
        ])
        print(f"Fusing labels for {len(tiles)} training tiles...\n")
        for tile_id in tiles:
            print(f"Processing {tile_id}...")
            fuse_labels(tile_id)
        print(f"\nDone. Fused labels saved to: {FUSED_LABELS_DIR}")
    else:
        print(f"Data directory not found: {S2_TRAIN}. Wait for the download to finish.")
