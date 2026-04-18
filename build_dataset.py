"""
build_dataset.py — Dataset Builder

PURPOSE:
    Converts the raw per-tile raster data into flat numpy arrays that
    Person 3's classifier can directly consume.

WHAT IT DOES:
    For every training tile it:
      1. Loads the fused binary label mask (output of fuse_labels.py)
      2. Loads the 64-dim AlphaEarth embeddings and reprojects them
         to match the Sentinel-2 grid (via loader.load_aef)
      3. Flattens both into 1D arrays:
           - X row = one pixel's 64 embedding dimensions  → shape (N, 64)
           - y row = that pixel's binary label (0 or 1)    → shape (N,)
      4. Saves X_train.npy and y_train.npy into ./outputs/

USAGE:
    python build_dataset.py

    After running, Person 3 just does:
        X = np.load("outputs/X_train.npy")   # (N, 64) float32
        y = np.load("outputs/y_train.npy")   # (N,)    uint8

DEPENDENCIES:
    - Fused labels must exist in ./outputs/fused_labels/
      (run fuse_labels.py first)
    - AEF embeddings must be downloaded in ./data/makeathon-challenge/
"""

import pathlib
import numpy as np
import rasterio
from config import (
    AEF_TRAIN, AEF_TEST, S2_TRAIN, S2_TEST,
    FUSED_LABELS_DIR, PREDICTIONS_DIR,
)
from loader import load_s2, load_aef, list_available_months


# ──────────────────────────────────────────────────────────────────
# Which AEF year to use.  The workshop shows embeddings are annual;
# we pick the most recent year that overlaps the deforestation window.
# ──────────────────────────────────────────────────────────────────
AEF_YEAR = 2022


def get_reference_meta(tile_id, data_split="train"):
    """
    Grab the CRS / transform / dimensions from the first available
    Sentinel-2 scene for this tile.  We need this so load_aef() can
    reproject the EPSG:4326 embeddings into the local UTM grid.
    """
    months = list_available_months(tile_id, data_split=data_split)
    if not months:
        return None
    year, month = months[0]
    _, meta = load_s2(tile_id, year, month, data_split=data_split)
    return meta


def build_tile_arrays(tile_id, aef_year=AEF_YEAR):
    """
    For a single tile, return (X, y) where:
        X : ndarray (H*W, 64)  — AEF embedding per pixel
        y : ndarray (H*W,)     — binary label per pixel

    Returns (None, None) if any input is missing.
    """

    # --- 1. Load the fused label mask produced by p2_fuse_labels.py ---
    label_path = pathlib.Path(f"{FUSED_LABELS_DIR}/{tile_id}_fused.tif")
    if not label_path.exists():
        print(f"  [SKIP] No fused label for {tile_id}. Run fuse_labels.py first.")
        return None, None

    with rasterio.open(label_path) as src:
        # shape: (H, W), values 0 or 1
        label_mask = src.read(1)

    # --- 2. Load & reproject AEF embeddings to match Sentinel-2 grid ---
    ref_meta = get_reference_meta(tile_id, data_split="train")
    if ref_meta is None:
        print(f"  [SKIP] No Sentinel-2 metadata for {tile_id}.")
        return None, None

    # aef_bands shape: (64, H, W) after reprojection
    aef_bands = load_aef(tile_id, aef_year, ref_meta, data_split="train")
    if aef_bands is None:
        print(f"  [SKIP] No AEF embedding for {tile_id} year {aef_year}.")
        return None, None

    num_channels, h, w = aef_bands.shape

    # --- 3. Sanity-check that shapes match ---
    if label_mask.shape != (h, w):
        print(
            f"  [WARN] Shape mismatch for {tile_id}: "
            f"labels {label_mask.shape} vs AEF ({h}, {w}). Skipping."
        )
        return None, None

    # --- 4. Flatten to (N, 64) and (N,) ---
    # Transpose (64, H, W) → (H, W, 64) then reshape to (H*W, 64)
    X = aef_bands.transpose(1, 2, 0).reshape(-1, num_channels)
    y = label_mask.flatten().astype(np.uint8)

    return X, y


def build_full_dataset(output_dir="./outputs"):
    """
    Iterate over every training tile, stack all pixels into one big
    X_train.npy / y_train.npy, and save them.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_X = []
    all_y = []

    # Discover tiles from the fused labels directory
    fused_dir = pathlib.Path(FUSED_LABELS_DIR)
    if not fused_dir.exists():
        print(f"ERROR: {FUSED_LABELS_DIR} does not exist.")
        print("Run fuse_labels.py first to generate fused label masks.")
        return

    tile_files = sorted(fused_dir.glob("*_fused.tif"))
    if not tile_files:
        print(f"No fused label files found in {FUSED_LABELS_DIR}.")
        return

    print(f"Found {len(tile_files)} tiles with fused labels.\n")

    for tile_file in tile_files:
        # Extract tile_id from filename like "18NWG_6_6_fused.tif"
        tile_id = tile_file.stem.replace("_fused", "")
        print(f"Processing tile: {tile_id}")

        X, y = build_tile_arrays(tile_id)
        if X is None:
            continue

        # Report class balance for this tile
        n_pos = int(y.sum())
        n_total = len(y)
        pct = 100.0 * n_pos / n_total if n_total > 0 else 0
        print(f"  → {n_total:,} pixels, {n_pos:,} deforestation ({pct:.2f}%)")

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        print("\nNo data collected. Check that fused labels and AEF data exist.")
        return

    # --- Stack everything ---
    X_train = np.concatenate(all_X, axis=0)  # (N_total, 64)
    y_train = np.concatenate(all_y, axis=0)  # (N_total,)

    # --- Summary ---
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"\n{'='*50}")
    print(f"Dataset built successfully!")
    print(f"  Total pixels : {len(y_train):,}")
    print(f"  Deforestation: {n_pos:,}  ({100*n_pos/len(y_train):.2f}%)")
    print(f"  No change    : {n_neg:,}  ({100*n_neg/len(y_train):.2f}%)")
    print(f"  Feature dims : {X_train.shape[1]}")
    print(f"{'='*50}")

    # --- Save ---
    x_path = output_path / "X_train.npy"
    y_path = output_path / "y_train.npy"
    np.save(x_path, X_train)
    np.save(y_path, y_train)
    print(f"\nSaved: {x_path}  shape={X_train.shape}")
    print(f"Saved: {y_path}  shape={y_train.shape}")


if __name__ == "__main__":
    build_full_dataset()
