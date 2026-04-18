"""
radar_features.py — Sentinel-1 Radar Confidence Layer

PURPOSE:
    Sentinel-1 radar can see through clouds, so it provides a
    complementary signal to optical (Sentinel-2) data.  This script
    computes a per-pixel "radar confidence" score that indicates how
    much the backscatter changed between baseline (2020) and recent
    years — a drop in VV backscatter often correlates with tree removal.

    This is an OPTIONAL secondary feature.  The primary features for
    the classifier are the 64D AEF embeddings (via build_dataset.py).

STATUS: Stub — implement if time permits.

USAGE:
    python radar_features.py
"""

import pathlib
import numpy as np
from config import S1_TRAIN, DEFOR_START_YEAR, PREDICTIONS_DIR
from loader import load_s1, load_s2, list_available_months


def radar_confidence(tile_id):
    """
    Compute a per-pixel radar change score between the baseline year
    and post-baseline years.  Higher values = more likely deforestation.

    TODO: Implement this if time permits.
    """
    pass


def visualise_prediction(tile_id, prediction_array):
    """
    Overlay a binary prediction mask on a Sentinel-2 RGB composite
    for visual sanity checking during development.

    TODO: Implement this for the final jury pitch.
    """
    pass


if __name__ == "__main__":
    pass

