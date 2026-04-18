import rasterio
import numpy as np
import pathlib
from config import *

def load_s2(tile_id, year, month, data_split="train"):
    base_dir = S2_TRAIN if data_split == "train" else S2_TEST
    path = f"{base_dir}/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{year}_{month}.tif"
    p = pathlib.Path(path)
    if not p.exists():
        print(f"Warning: File does not exist: {path}")
        return None, None
    
    with rasterio.open(p) as src:
        bands = src.read().astype(np.float32) / 10000.0
        meta = src.meta.copy()
        
    return bands, meta

def load_s1(tile_id, year, month, orbit="ascending", data_split="train"):
    base_dir = S1_TRAIN if data_split == "train" else S1_TEST
    path = f"{base_dir}/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{year}_{month}_{orbit}.tif"
    p = pathlib.Path(path)
    if not p.exists():
        print(f"Warning: File does not exist: {path}")
        return None, None
        
    with rasterio.open(p) as src:
        bands = src.read().astype(np.float32)
        meta = src.meta.copy()
        
    return bands, meta

def list_available_months(tile_id, data_split="train"):
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

def save_raster(array, reference_meta, output_path):
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
        "dtype": array.dtype.name
    })
    
    with rasterio.open(p, "w", **meta) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            dst.write(array)
