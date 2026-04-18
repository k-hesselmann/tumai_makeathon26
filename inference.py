"""
inference.py — Run trained model on test tiles and produce submission GeoJSON

PURPOSE:
    This is the FINAL step in the pipeline.  It takes a trained model,
    runs it on every test tile's AEF embeddings, produces a binary
    prediction raster per tile, then converts each raster to GeoJSON
    using submission_utils.py.

WORKFLOW:
    1. Load the trained model (expects a .predict() or forward pass
       that takes (N, 64) → (N,) binary predictions)
    2. For each test tile:
       a. Load AEF embeddings via loader.load_aef()
       b. Flatten to (H*W, 64), run model, reshape to (H, W)
       c. Save binary prediction raster as GeoTIFF
       d. Convert to GeoJSON via submission_utils.raster_to_geojson()
    3. Merge all per-tile GeoJSONs into one final submission file

USAGE:
    python inference.py

OUTPUT:
    submission/submission.geojson  — upload this to the leaderboard
"""

import json
import pathlib
import numpy as np
import rasterio

from config import (
    AEF_TEST, S2_TEST, PREDICTIONS_DIR, SUBMISSION_DIR,
)
from loader import load_s2, load_aef, list_available_months
from submission_utils import raster_to_geojson


# ── Configuration ───────────────────────────────────────────────
AEF_YEAR = 2022          # Which AEF embedding year to use for inference
THRESHOLD = 0.5          # Probability threshold for binarisation
MIN_AREA_HA = 0.5        # Minimum polygon area (hectares) for submission


def load_model(model_path="outputs/model.pkl"):
    """
    Load the trained model.  Supports both scikit-learn (pickle)
    and PyTorch (.pt) formats.

    Returns an object with a .predict(X) method that takes
    (N, 64) float32 → (N,) binary predictions.
    """
    p = pathlib.Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Train the model first and save it there."
        )

    if p.suffix == ".pkl":
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)
    elif p.suffix == ".pt":
        import torch
        model = torch.load(p, map_location="cpu")
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model format: {p.suffix}")


def discover_test_tiles():
    """Find all test tile IDs from the Sentinel-2 test directory."""
    test_dir = pathlib.Path(S2_TEST)
    if not test_dir.exists():
        # Fall back: try AEF test directory
        test_dir = pathlib.Path(AEF_TEST)
    if not test_dir.exists():
        print(f"ERROR: No test data found in {S2_TEST} or {AEF_TEST}")
        return []

    tile_ids = set()
    for item in test_dir.iterdir():
        if item.is_dir() and item.name.endswith("__s2_l2a"):
            tile_ids.add(item.name.replace("__s2_l2a", ""))
        elif item.is_file() and item.suffix == ".tiff":
            # AEF files: {tile_id}_{year}.tiff
            parts = item.stem.rsplit("_", 1)
            if len(parts) == 2:
                tile_ids.add(parts[0])

    return sorted(tile_ids)


def get_test_reference_meta(tile_id):
    """Get CRS/transform from the first available Sentinel-2 test scene."""
    months = list_available_months(tile_id, data_split="test")
    if not months:
        return None
    year, month = months[0]
    _, meta = load_s2(tile_id, year, month, data_split="test")
    return meta


def predict_tile(model, tile_id, aef_year=AEF_YEAR):
    """
    Run the model on a single test tile.

    Returns:
        prediction: ndarray (H, W) uint8, binary 0/1
        meta: rasterio metadata dict for saving as GeoTIFF
    """
    ref_meta = get_test_reference_meta(tile_id)
    if ref_meta is None:
        print(f"  [SKIP] No Sentinel-2 metadata for test tile {tile_id}")
        return None, None

    # Load & reproject AEF embeddings to match Sentinel-2 grid
    aef_bands = load_aef(tile_id, aef_year, ref_meta, data_split="test")
    if aef_bands is None:
        print(f"  [SKIP] No AEF embedding for test tile {tile_id}")
        return None, None

    num_channels, h, w = aef_bands.shape

    # Flatten (64, H, W) → (H*W, 64) for the model
    X = aef_bands.transpose(1, 2, 0).reshape(-1, num_channels)

    # Run model
    predictions = model.predict(X)

    # Binarise (in case model returns probabilities)
    if predictions.dtype in (np.float32, np.float64):
        predictions = (predictions >= THRESHOLD).astype(np.uint8)
    else:
        predictions = predictions.astype(np.uint8)

    # Reshape back to (H, W)
    prediction_map = predictions.reshape(h, w)

    return prediction_map, ref_meta


def save_prediction_raster(prediction, meta, output_path):
    """Save a binary prediction array as a single-band GeoTIFF."""
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out_meta = meta.copy()
    out_meta.update({
        "count": 1,
        "dtype": "uint8",
        "height": prediction.shape[0],
        "width": prediction.shape[1],
    })

    with rasterio.open(p, "w", **out_meta) as dst:
        dst.write(prediction, 1)


def run_inference():
    """Main entry point: load model, predict all test tiles, produce submission."""
    # 1. Load the trained model
    model = load_model()
    print(f"Model loaded successfully.\n")

    # 2. Discover test tiles
    tile_ids = discover_test_tiles()
    if not tile_ids:
        return
    print(f"Found {len(tile_ids)} test tiles.\n")

    # 3. Predict each tile and convert to GeoJSON
    all_features = []
    pred_dir = pathlib.Path(PREDICTIONS_DIR) / "test"
    sub_dir = pathlib.Path(SUBMISSION_DIR)

    for tile_id in tile_ids:
        print(f"Predicting: {tile_id}")

        prediction, meta = predict_tile(model, tile_id)
        if prediction is None:
            continue

        n_defor = int(prediction.sum())
        pct = 100.0 * n_defor / prediction.size
        print(f"  → {n_defor:,} deforestation pixels ({pct:.2f}%)")

        if n_defor == 0:
            print(f"  → No deforestation detected, skipping.")
            continue

        # Save binary raster
        raster_path = pred_dir / f"{tile_id}_prediction.tif"
        save_prediction_raster(prediction, meta, raster_path)

        # Convert to GeoJSON
        try:
            geojson = raster_to_geojson(
                raster_path=raster_path,
                min_area_ha=MIN_AREA_HA,
            )
            all_features.extend(geojson["features"])
            print(f"  → {len(geojson['features'])} polygons")
        except ValueError as e:
            print(f"  → Warning: {e}")

    # 4. Merge all features into one submission file
    if not all_features:
        print("\nNo deforestation detected in any test tile!")
        return

    submission = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    sub_dir.mkdir(parents=True, exist_ok=True)
    submission_path = sub_dir / "submission.geojson"
    with open(submission_path, "w") as f:
        json.dump(submission, f)

    print(f"\n{'='*50}")
    print(f"Submission saved: {submission_path}")
    print(f"Total polygons: {len(all_features)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_inference()
