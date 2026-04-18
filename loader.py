"""
loader.py — Shared Data Loading Utilities

PURPOSE:
    Provides functions to load all three data modalities
    (Sentinel-2, Sentinel-1, AlphaEarth embeddings) and a helper to
    save raster outputs.  Every other script imports from here.

KEY DETAIL — CRS MISMATCH:
    Sentinel-1 and Sentinel-2 tiles are in a local UTM projection,
    but AlphaEarth (AEF) embeddings are in EPSG:4326 (lat/lon).
    load_aef() handles the reprojection automatically so downstream
    code always gets arrays that are pixel-aligned with Sentinel data.

BAND ORDERING (Sentinel-2, after load_s2):
    Index 0  → B01  Aerosol        (60 m, 443 nm)
    Index 1  → B02  Blue           (10 m, 490 nm)
    Index 2  → B03  Green          (10 m, 560 nm)
    Index 3  → B04  Red            (10 m, 665 nm)   ← used for NDVI
    Index 4  → B05  Red Edge 1     (20 m, 705 nm)
    Index 5  → B06  Red Edge 2     (20 m, 740 nm)
    Index 6  → B07  Red Edge 3     (20 m, 783 nm)
    Index 7  → B08  NIR            (10 m, 842 nm)   ← used for NDVI
    Index 8  → B8A  Narrow NIR     (20 m, 865 nm)
    Index 9  → B09  Water Vapour   (60 m, 945 nm)
    Index 10 → B10  Cirrus         (60 m, 1375 nm)
    Index 11 → B11  SWIR 1         (20 m, 1610 nm)
    Index 12 → B12  SWIR 2         (20 m, 2190 nm)
    (all upsampled to 10 m in the provided dataset)
"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pathlib
from config import S2_TRAIN, S2_TEST, S1_TRAIN, S1_TEST, AEF_TRAIN, AEF_TEST


def load_s2(tile_id, year, month, data_split="train"):
    """
    Load a single Sentinel-2 scene.

    Args:
        tile_id:    e.g. "18NWG_6_6"
        year:       e.g. 2022
        month:      e.g. 1  (will be zero-padded to "01")
        data_split: "train" or "test"

    Returns:
        bands: ndarray (12, H, W) float32, reflectance scaled to [0, 1]
        meta:  rasterio metadata dict (contains CRS, transform, etc.)
        — or (None, None) if the file doesn't exist.
    """
    base_dir = S2_TRAIN if data_split == "train" else S2_TEST
    path = f"{base_dir}/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{year}_{int(month)}.tif"
    p = pathlib.Path(path)
    if not p.exists():
        print(f"Warning: File does not exist: {path}")
        return None, None

    with rasterio.open(p) as src:
        bands = src.read().astype(np.float32) / 10000.0
        meta = src.meta.copy()

    return bands, meta


def load_s1(tile_id, year, month, orbit="ascending", data_split="train"):
    """
    Load a single Sentinel-1 radar scene (VV polarisation, RTC corrected).

    Args:
        tile_id:    e.g. "18NWG_6_6"
        year:       e.g. 2022
        month:      e.g. 1
        orbit:      "ascending" or "descending"
        data_split: "train" or "test"

    Returns:
        bands: ndarray (1, H, W) float32, backscatter values
        meta:  rasterio metadata dict
        — or (None, None) if the file doesn't exist.
    """
    base_dir = S1_TRAIN if data_split == "train" else S1_TEST
    path = f"{base_dir}/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{year}_{int(month)}_{orbit}.tif"
    p = pathlib.Path(path)
    if not p.exists():
        print(f"Warning: File does not exist: {path}")
        return None, None

    with rasterio.open(p) as src:
        bands = src.read().astype(np.float32)
        meta = src.meta.copy()

    return bands, meta


def list_available_months(tile_id, data_split="train"):
    """
    Scan the Sentinel-2 directory for a given tile and return a sorted
    list of (year, month) tuples that have data on disk.

    Example return: [(2020, 1), (2020, 3), (2021, 6), ...]
    """
    base_dir = S2_TRAIN if data_split == "train" else S2_TEST
    tile_dir = pathlib.Path(f"{base_dir}/{tile_id}__s2_l2a")

    if not tile_dir.exists():
        return []

    months = []
    for file_path in tile_dir.glob("*.tif"):
        # Expected format: {tile_id}__s2_l2a_{year}_{month}.tif
        parts = file_path.stem.split('_')
        try:
            year = int(parts[-2])
            month = int(parts[-1])
            months.append((year, month))
        except (ValueError, IndexError):
            continue

    return sorted(months)


def load_aef(tile_id, year, reference_meta, data_split="train"):
    """
    Load AlphaEarth Foundations embeddings and reproject them to match
    the Sentinel-2 grid.

    WHY THIS IS NEEDED:
        AEF tiles are in EPSG:4326 (geographic lat/lon) while Sentinel
        tiles are in a local UTM projection.  This function uses
        rasterio.warp.reproject to resample the 64-band AEF raster
        into the exact same pixel grid as the Sentinel data.

    Args:
        tile_id:        e.g. "18NWG_6_6"
        year:           e.g. 2022
        reference_meta: rasterio meta dict from a Sentinel-2 scene
                        (provides target CRS, transform, height, width)
        data_split:     "train" or "test"

    Returns:
        ndarray (64, H, W) float32, pixel-aligned with Sentinel data
        — or None if the file doesn't exist.
    """
    base_dir = AEF_TRAIN if data_split == "train" else AEF_TEST
    path = f"{base_dir}/{tile_id}_{year}.tiff"
    p = pathlib.Path(path)
    if not p.exists():
        print(f"Warning: File does not exist: {path}")
        return None

    with rasterio.open(p) as src:
        source_bands = src.read().astype(np.float32)
        source_crs = src.crs
        source_transform = src.transform

    dst_crs = reference_meta['crs']
    dst_transform = reference_meta['transform']
    dst_height = reference_meta['height']
    dst_width = reference_meta['width']

    dst_bands = np.zeros(
        (source_bands.shape[0], dst_height, dst_width), dtype=np.float32
    )

    reproject(
        source=source_bands,
        destination=dst_bands,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    return dst_bands


def save_raster(array, reference_meta, output_path):
    """
    Write a numpy array to a GeoTIFF, inheriting geospatial info from
    reference_meta.

    Args:
        array:          2D (H, W) or 3D (C, H, W) numpy array
        reference_meta: rasterio meta dict (CRS, transform, driver, …)
        output_path:    where to write the .tif file
    """
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    meta = reference_meta.copy()

    if len(array.shape) == 2:
        count = 1
        height, width = array.shape
    else:
        count, height, width = array.shape

    meta.update({
        "count": count,
        "height": height,
        "width": width,
        "dtype": array.dtype.name,
    })

    with rasterio.open(p, "w", **meta) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            dst.write(array)

