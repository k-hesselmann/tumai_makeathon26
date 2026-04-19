"""
Download ESA WorldCover 2020 tiles for all training and test tiles.

WorldCover is a 10m global land cover map produced by ESA.
We use it as clean ground truth for Model 1 (AEF → forest classifier).
The output is one binary forest mask per challenge tile (class 10 = tree cover).

Usage:
    python scripts/download_worldcover.py
    python scripts/download_worldcover.py --year 2021  # for v200/2021
"""

import argparse
import logging
import math
import pathlib
import sys

import boto3
import geopandas as gpd
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import S2_TRAIN, S2_TEST, TRAIN_META, TEST_META

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ESA WorldCover public S3 bucket (anonymous access, no AWS credentials needed)
WC_BUCKET = "esa-worldcover"
WC_VERSIONS = {2020: "v100", 2021: "v200"}

# WorldCover class 10 = Tree cover (the only class we care about)
TREE_COVER_CLASS = 10

OUTPUT_DIR = pathlib.Path("./data/worldcover")
CACHE_DIR = OUTPUT_DIR / "_cache"   # raw downloaded 3°×3° WorldCover tiles


def wc_s3_prefix(year: int) -> str:
    version = WC_VERSIONS[year]
    return f"{version}/{year}/map"


def wc_tile_code(lat: float, lon: float) -> str:
    """Return WorldCover 3°×3° tile code whose SW corner covers (lat, lon)."""
    lat_sw = math.floor(lat / 3) * 3
    lon_sw = math.floor(lon / 3) * 3
    lat_str = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{abs(lat_sw):02d}"
    lon_str = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{abs(lon_sw):03d}"
    return f"{lat_str}{lon_str}"


def wc_tiles_for_bbox(minx: float, miny: float, maxx: float, maxy: float) -> set:
    """All WorldCover tile codes that overlap a WGS84 bounding box."""
    codes = set()
    for lat in (miny, maxy):
        for lon in (minx, maxx):
            codes.add(wc_tile_code(lat, lon))
    return codes


def wc_filename(code: str, year: int) -> str:
    version = WC_VERSIONS[year]
    return f"ESA_WorldCover_10m_{year}_{version}_{code}_Map.tif"


def download_wc_tile(code: str, year: int, s3) -> pathlib.Path:
    """Download a raw WorldCover tile to cache; skip if already present."""
    dest = CACHE_DIR / wc_filename(code, year)
    if dest.exists():
        logger.info(f"  [cache] {dest.name}")
        return dest

    key = f"{wc_s3_prefix(year)}/{wc_filename(code, year)}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  [download] s3://{WC_BUCKET}/{key}")
    try:
        s3.download_file(WC_BUCKET, key, str(dest))
    except ClientError as e:
        logger.error(f"  Failed to download {key}: {e}")
        return None
    return dest


def get_s2_reference_meta(tile_id: str, data_split: str) -> dict | None:
    """Load rasterio meta from the first available Sentinel-2 file for a tile."""
    base = pathlib.Path(S2_TRAIN if data_split == "train" else S2_TEST)
    tile_dir = base / f"{tile_id}__s2_l2a"
    if not tile_dir.exists():
        logger.warning(f"  S2 directory not found: {tile_dir}")
        return None
    tifs = sorted(tile_dir.glob("*.tif"))
    if not tifs:
        logger.warning(f"  No S2 files in {tile_dir}")
        return None
    with rasterio.open(tifs[0]) as src:
        return src.meta.copy()


def clip_worldcover_to_tile(
    wc_paths: list[pathlib.Path],
    ref_meta: dict,
    out_path: pathlib.Path,
) -> None:
    """
    Reproject WorldCover tile(s) to the challenge tile's S2 grid and save
    a binary forest mask (1 = tree cover, 0 = everything else).
    """
    dst_crs = ref_meta["crs"]
    dst_transform = ref_meta["transform"]
    dst_h = ref_meta["height"]
    dst_w = ref_meta["width"]

    # Mosaic the raw WorldCover tiles (almost always just 1 tile per ~10km AOI)
    if len(wc_paths) == 1:
        with rasterio.open(wc_paths[0]) as src:
            raw_data = src.read(1)
            raw_transform = src.transform
            raw_crs = src.crs
    else:
        datasets = [rasterio.open(p) for p in wc_paths]
        mosaic, raw_transform = merge(datasets)
        raw_data = mosaic[0]
        raw_crs = datasets[0].crs
        for ds in datasets:
            ds.close()

    # Reproject to challenge tile UTM grid
    wc_reprojected = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=raw_data,
        destination=wc_reprojected,
        src_transform=raw_transform,
        src_crs=raw_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    forest_mask = (wc_reprojected == TREE_COVER_CLASS).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta = ref_meta.copy()
    out_meta.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(forest_mask, 1)

    forest_pct = forest_mask.mean() * 100
    logger.info(f"  -> {out_path.name}  ({forest_pct:.1f}% forest)")


def load_tile_bboxes(geojson_path: str) -> list[dict]:
    """Read tile names and WGS84 bounding boxes from a tile metadata GeoJSON."""
    gdf = gpd.read_file(geojson_path)
    return [
        {"name": row["name"], "bbox": row.geometry.bounds}
        for _, row in gdf.iterrows()
    ]


def main(year: int = 2020) -> None:
    if year not in WC_VERSIONS:
        raise ValueError(f"year must be one of {list(WC_VERSIONS)}, got {year}")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    train_tiles = load_tile_bboxes(TRAIN_META)
    test_tiles = load_tile_bboxes(TEST_META)
    all_tiles = [(t, "train") for t in train_tiles] + [(t, "test") for t in test_tiles]

    logger.info(f"Processing {len(all_tiles)} tiles  (WorldCover {year})")

    for tile_info, split in all_tiles:
        tile_id = tile_info["name"]
        bbox = tile_info["bbox"]
        out_path = OUTPUT_DIR / f"{tile_id}_worldcover_{year}.tif"

        logger.info(f"\n[{tile_id}] ({split})")

        # 1. Determine which WorldCover 3°×3° source tiles are needed
        wc_codes = wc_tiles_for_bbox(*bbox)
        logger.info(f"  WorldCover tiles: {wc_codes}")

        # 2. Download source tiles (cached after first run)
        wc_paths = []
        for code in wc_codes:
            p = download_wc_tile(code, year, s3)
            if p is not None:
                wc_paths.append(p)

        if not wc_paths:
            logger.error(f"  No WorldCover data downloaded for {tile_id}, skipping.")
            continue

        # 3. Get S2 reference grid to align output
        ref_meta = get_s2_reference_meta(tile_id, split)
        if ref_meta is None:
            logger.error(f"  No S2 reference found for {tile_id}, skipping.")
            continue

        # 4. Reproject + save binary forest mask
        if out_path.exists():
            logger.info(f"  [exists] {out_path.name}, skipping.")
            continue

        clip_worldcover_to_tile(wc_paths, ref_meta, out_path)

    logger.info("\nDone. Forest masks saved to ./data/worldcover/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and clip ESA WorldCover data")
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        choices=[2020, 2021],
        help="WorldCover edition year (2020=v100, 2021=v200)",
    )
    args = parser.parse_args()
    main(args.year)