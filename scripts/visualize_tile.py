"""
visualize_tile.py -- Generate pitch-ready visualizations for a single tile.

Produces a multi-panel figure showing:
  1. Sentinel-2 true-colour RGB composite
  2. Individual weak label sources (RADD, GLAD-L, GLAD-S2, Hansen lossyear)
  3. The fused label mask (majority vote + forest mask)
  4. The fused deforestation polygons overlaid on the RGB

Usage (from project root, with venv activated):
    python scripts/visualize_tile.py                        # uses default tile
    python scripts/visualize_tile.py --tile 18NWG_6_6       # specific tile
    python scripts/visualize_tile.py --tile 18NWG_6_6 --year 2023 --month 6
"""

import sys
import argparse
from pathlib import Path

# Allow imports from the project root
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import (
    S2_TRAIN, RADD_DIR, GLADL_DIR, GLADS2_DIR, FUSED_LABELS_DIR,
)
from loader import load_s2, list_available_months


HANSEN_CLIPPED = Path("./data/hansen/clipped")
OUTPUT_DIR = Path("./outputs/figures")


# ── Helper: load a single-band label raster ─────────────────────────────────

def load_label(path):
    p = Path(path)
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1)


# ── Panel 1: Sentinel-2 RGB ─────────────────────────────────────────────────

def make_rgb(tile_id, year, month):
    """Load S2 scene and build a true-colour RGB (bands 4, 3, 2)."""
    bands, meta = load_s2(tile_id, year, month, data_split="train")
    if bands is None:
        return None
    # B04=Red (idx 3), B03=Green (idx 2), B02=Blue (idx 1)
    rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1)
    # Stretch for visibility
    rgb = np.clip(rgb * 3.5, 0, 1)
    return rgb


# ── Panel 2-5: Individual label sources ──────────────────────────────────────

def load_all_labels(tile_id):
    """Return a dict of label-name -> binary mask (H, W) uint8."""
    labels = {}

    # RADD
    radd = load_label(f"{RADD_DIR}/radd_{tile_id}_labels.tif")
    labels["RADD"] = (radd > 0).astype(np.uint8) if radd is not None else None

    # GLAD-L (union of all years)
    gladl_union = None
    for yy in ["21", "22", "23", "24", "25"]:
        d = load_label(f"{GLADL_DIR}/gladl_{tile_id}_alert{yy}.tif")
        if d is not None:
            mask = (d > 0).astype(np.uint8)
            gladl_union = mask if gladl_union is None else (gladl_union | mask)
    labels["GLAD-L"] = gladl_union

    # GLAD-S2
    glads2 = load_label(f"{GLADS2_DIR}/glads2_{tile_id}_alert.tif")
    labels["GLAD-S2"] = (glads2 > 0).astype(np.uint8) if glads2 is not None else None

    # Hansen lossyear (post-2020 only)
    hansen_path = HANSEN_CLIPPED / f"{tile_id}_lossyear.tif"
    hansen = load_label(hansen_path)
    if hansen is not None:
        labels["Hansen"] = ((hansen >= 21) & (hansen <= 24)).astype(np.uint8)
    else:
        labels["Hansen"] = None

    return labels


# ── Panel 6: Fused mask ─────────────────────────────────────────────────────

def load_or_compute_fused(tile_id, labels):
    """Load pre-computed fused mask, or compute it on the fly."""
    fused_path = Path(f"{FUSED_LABELS_DIR}/{tile_id}_fused.tif")
    if fused_path.exists():
        return load_label(fused_path)

    # Compute inline: majority vote + forest mask
    available = [v for v in labels.values() if v is not None]
    if not available:
        return None

    h, w = available[0].shape
    vote_sum = np.zeros((h, w), dtype=np.uint8)
    for mask in labels.values():
        if mask is not None:
            # Handle shape mismatch gracefully
            if mask.shape == (h, w):
                vote_sum += mask

    fused = (vote_sum >= 2).astype(np.uint8)

    # Apply treecover mask if available
    tc_path = HANSEN_CLIPPED / f"{tile_id}_treecover2000.tif"
    tc = load_label(tc_path)
    if tc is not None and tc.shape == (h, w):
        fused = fused & (tc >= 30).astype(np.uint8)

    return fused


# ── Main figure ──────────────────────────────────────────────────────────────

def generate_figure(tile_id, year, month):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Sentinel-2 RGB for {tile_id} ({year}-{month})...")
    rgb = make_rgb(tile_id, year, month)
    if rgb is None:
        print("ERROR: Could not load Sentinel-2 data.")
        return

    print("Loading label sources...")
    labels = load_all_labels(tile_id)

    print("Computing fused mask...")
    fused = load_or_compute_fused(tile_id, labels)

    # ── Build 2x3 figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    fig.suptitle(
        f"Deforestation Detection Pipeline  --  Tile: {tile_id}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Deforestation colormap: transparent for 0, red for 1
    defor_cmap = ListedColormap(["#00000000", "#FF3333"])

    # 1) Sentinel-2 RGB
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("Sentinel-2 RGB\n(True colour composite)", fontsize=11)
    ax.axis("off")

    # 2-5) Individual label sources
    label_names = ["RADD", "GLAD-L", "GLAD-S2", "Hansen"]
    label_colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71"]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]

    for (name, color, pos) in zip(label_names, label_colors, positions):
        ax = axes[pos]
        mask = labels.get(name)
        if mask is not None:
            cmap = ListedColormap(["#1a1a2e", color])
            ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            n_pixels = int(mask.sum())
            pct = 100.0 * n_pixels / mask.size
            ax.set_title(f"{name}\n({n_pixels:,} pixels, {pct:.1f}%)", fontsize=11)
        else:
            ax.text(0.5, 0.5, f"{name}\n(not available)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_facecolor("#1a1a2e")
            ax.set_title(name, fontsize=11)
        ax.axis("off")

    # 6) Fused mask overlaid on RGB
    ax = axes[1, 2]
    ax.imshow(rgb)
    if fused is not None:
        overlay = np.zeros((*fused.shape, 4))
        overlay[fused == 1] = [1.0, 0.15, 0.15, 0.6]  # semi-transparent red
        ax.imshow(overlay, interpolation="nearest")
        n_fused = int(fused.sum())
        pct = 100.0 * n_fused / fused.size
        ax.set_title(
            f"Fused Labels on RGB\n(>= 2 votes + forest mask, {n_fused:,} px, {pct:.1f}%)",
            fontsize=11,
        )
        ax.legend(
            handles=[Patch(facecolor="#FF2626", alpha=0.6, label="Deforestation")],
            loc="lower right", fontsize=9,
        )
    else:
        ax.set_title("Fused Labels\n(not computed yet)", fontsize=11)
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUTPUT_DIR / f"tile_{tile_id}_overview.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {out_path.resolve()}")

    # ── Second figure: just the polygon view ─────────────────────────────
    if fused is not None and fused.sum() > 0:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10), dpi=150)

        ax2.imshow(rgb)

        # Draw contours around deforestation patches
        from matplotlib.colors import to_rgba
        contours = ax2.contour(fused, levels=[0.5], colors=["#FF2626"], linewidths=1.5)
        overlay = np.zeros((*fused.shape, 4))
        overlay[fused == 1] = [1.0, 0.15, 0.15, 0.45]
        ax2.imshow(overlay, interpolation="nearest")

        ax2.set_title(
            f"Deforestation Polygons -- {tile_id}\n"
            f"({int(fused.sum()):,} deforested pixels)",
            fontsize=14, fontweight="bold",
        )
        ax2.axis("off")
        ax2.legend(
            handles=[Patch(facecolor="#FF2626", alpha=0.6, edgecolor="#FF2626", label="Deforestation polygon")],
            loc="lower right", fontsize=10,
        )

        poly_path = OUTPUT_DIR / f"tile_{tile_id}_polygons.png"
        fig2.savefig(poly_path, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"Saved -> {poly_path.resolve()}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a tile for the pitch.")
    parser.add_argument("--tile", default="18NWG_6_6", help="Tile ID (default: 18NWG_6_6)")
    parser.add_argument("--year", type=int, default=None, help="S2 scene year (default: auto)")
    parser.add_argument("--month", type=int, default=None, help="S2 scene month (default: auto)")
    args = parser.parse_args()

    tile_id = args.tile

    if args.year and args.month:
        year, month = args.year, args.month
    else:
        # Pick a recent scene automatically
        months = list_available_months(tile_id, data_split="train")
        if not months:
            print(f"No Sentinel-2 data found for tile {tile_id}")
            sys.exit(1)
        # Prefer a mid-year scene (less cloud)
        mid_year = [m for m in months if m[1] in (5, 6, 7, 8)]
        year, month = mid_year[-1] if mid_year else months[-1]

    print(f"Tile: {tile_id}  |  Scene: {year}-{month:02d}")
    print("=" * 50)
    generate_figure(tile_id, year, month)
