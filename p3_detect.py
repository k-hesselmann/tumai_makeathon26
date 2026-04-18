# Person 3 — NDVI Change Detection
# Detects deforestation by finding NDVI drops after 2020.
# Run: python p3_detect.py

from config import *
from p1_loader import load_s2, list_available_months, save_raster

def compute_ndvi(bands):
    # TODO: Person 3 implements this
    # bands is (12, H, W) float array from load_s2
    # Return (H, W) NDVI array
    pass

def detect_deforestation(tile_id):
    # TODO: Person 3 implements this
    pass

if __name__ == "__main__":
    pass
