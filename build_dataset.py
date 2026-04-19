"""
build_dataset.py — Dataset Builder

USAGE:
    python build_dataset.py --labels combined   # recommended
    python build_dataset.py --labels fused      # if Model 1 not yet trained
    python build_dataset.py --labels model1     # Model 1 only

OUTPUT:
    outputs/X_train_{source}.npy  (N, 384)  float32  6-year AEF temporal stack
    outputs/y_train_{source}.npy  (N,)      uint8    binary deforestation label
"""

import argparse
import pathlib
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from config import AEF_TRAIN, FUSED_LABELS_DIR, MODEL1_DIR

AEF_YEARS        = [2020, 2021, 2022, 2023, 2024, 2025]
MAX_POS_PER_TILE = 50_000
NEG_MULT         = 3

MODEL1_LABELS_DIR = pathlib.Path(MODEL1_DIR) / "improved_labels"
MODEL1_PROBS_DIR  = pathlib.Path(MODEL1_DIR) / "forest_probs"
AEF_DIR           = pathlib.Path(AEF_TRAIN)


def _dequantize(raw: np.ndarray) -> np.ndarray:
    """uint8 AEF → float32 via workshop power-law transform."""
    out = raw.astype(np.float32)
    zero_pixel = np.all(out == 0, axis=0, keepdims=True)
    out = np.where(zero_pixel, np.nan, out)
    out = (out - 127.0) / 127.5
    neg = out < 0
    out = np.abs(out) ** 2.0
    out[neg] *= -1.0
    return out


def load_aef_stack(tile_id):
    """Load 6-year AEF in native CRS — no reprojection, fast direct read.

    Returns (384, H, W) float32, transform, crs, (H, W) — or (None,)*4.
    """
    bands, transform, crs, shape = [], None, None, None
    for year in AEF_YEARS:
        p = AEF_DIR / f"{tile_id}_{year}.tiff"
        if not p.exists():
            print(f"  [SKIP] Missing AEF {year}.")
            return None, None, None, None
        with rasterio.open(p) as src:
            raw = src.read()  # (64, H, W) uint8 — fast, no reproject
            if transform is None:
                transform, crs, shape = src.transform, src.crs, src.shape
        bands.append(_dequantize(raw))  # (64, H, W) float32
    return np.concatenate(bands, axis=0), transform, crs, shape  # (384, H, W)


def reproject_label(label_path, aef_transform, aef_crs, aef_shape, dtype=np.uint8):
    """Reproject a single-band label raster to the AEF grid (fast — 1 band)."""
    dst = np.zeros(aef_shape, dtype=dtype)
    with rasterio.open(label_path) as src:
        reproject(
            source=src.read(1).astype(dtype),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=aef_transform,
            dst_crs=aef_crs,
            resampling=Resampling.nearest,
        )
    return dst


def load_label_mask(tile_id, aef_transform, aef_crs, aef_shape, label_source):
    fused_p = pathlib.Path(f"{FUSED_LABELS_DIR}/{tile_id}_fused.tif")
    m1_p    = MODEL1_LABELS_DIR / f"{tile_id}_improved_label.tif"
    prob_p  = MODEL1_PROBS_DIR  / f"{tile_id}_forest_prob_2020.tif"

    if label_source == "fused":
        if not fused_p.exists():
            print(f"  [SKIP] No fused label. Run fuse_labels.py first.")
            return None
        return reproject_label(fused_p, aef_transform, aef_crs, aef_shape)

    elif label_source == "model1":
        if not m1_p.exists():
            print(f"  [SKIP] No Model 1 label. Run train_model1.ipynb first.")
            return None
        return reproject_label(m1_p, aef_transform, aef_crs, aef_shape)

    elif label_source == "combined":
        if not fused_p.exists():
            print(f"  [SKIP] No fused label. Run fuse_labels.py first.")
            return None
        fused = reproject_label(fused_p, aef_transform, aef_crs, aef_shape)
        if m1_p.exists() and prob_p.exists():
            m1    = reproject_label(m1_p,   aef_transform, aef_crs, aef_shape)
            prob  = reproject_label(prob_p, aef_transform, aef_crs, aef_shape,
                                    dtype=np.float32)
            return ((fused | m1) & (prob >= 0.5)).astype(np.uint8)
        else:
            print(f"  [WARN] Model 1 missing — using fused only.")
            return fused

    raise ValueError(f"Unknown label_source: {label_source!r}")


def build_tile_arrays(tile_id, label_source):
    aef, transform, crs, shape = load_aef_stack(tile_id)
    if aef is None:
        return None, None

    mask = load_label_mask(tile_id, transform, crs, shape, label_source)
    if mask is None:
        return None, None

    X_full = aef.reshape(aef.shape[0], -1).T   # (H*W, 384)
    y_full = mask.flatten()

    valid  = np.isfinite(X_full).all(axis=1)
    X_full = X_full[valid]
    y_full = y_full[valid]

    rng     = np.random.default_rng(42)
    pos_idx = np.where(y_full == 1)[0]
    neg_idx = np.where(y_full == 0)[0]

    if len(pos_idx) == 0:
        print(f"  [SKIP] No deforestation pixels.")
        return None, None

    if len(pos_idx) > MAX_POS_PER_TILE:
        pos_idx = rng.choice(pos_idx, MAX_POS_PER_TILE, replace=False)

    n_neg   = min(len(neg_idx), len(pos_idx) * NEG_MULT)
    neg_idx = rng.choice(neg_idx, n_neg, replace=False)

    idx = np.concatenate([pos_idx, neg_idx])
    return X_full[idx].astype(np.float32), y_full[idx]


def build_full_dataset(output_dir="./outputs", label_source="combined"):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tile_ids = sorted(set(
        p.stem.rsplit("_", 1)[0] for p in AEF_DIR.glob("*.tiff")
    ))
    print(f"Building dataset  label_source='{label_source}'  ({len(tile_ids)} tiles)\n")

    all_X, all_y = [], []
    for tile_id in tile_ids:
        print(f"  {tile_id} ...", end=" ", flush=True)
        X, y = build_tile_arrays(tile_id, label_source)
        if X is None:
            continue
        n_pos = int(y.sum())
        print(f"{len(y):,} px  {n_pos:,} pos ({100*n_pos/len(y):.1f}%)")
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        print("\nNo data collected.")
        return

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0).astype(np.uint8)

    n_pos = int(y_train.sum())
    print(f"\n{'='*50}")
    print(f"Dataset [{label_source}]  {len(y_train):,} px  "
          f"{n_pos:,} pos ({100*n_pos/len(y_train):.1f}%)  "
          f"dims={X_train.shape[1]}")
    print(f"{'='*50}")

    x_path = output_path / f"X_train_{label_source}.npy"
    y_path = output_path / f"y_train_{label_source}.npy"
    np.save(x_path, X_train)
    np.save(y_path, y_train)
    print(f"Saved: {x_path}  {X_train.shape}")
    print(f"Saved: {y_path}  {y_train.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", choices=["combined", "fused", "model1"],
                        default="combined")
    args = parser.parse_args()
    build_full_dataset(label_source=args.labels)
