"""
visualize_tile.py -- Presentation figures for a single tile.

Produces two figures:
  1. label_sources  -- S2 RGB + 4 weak label sources + fused ground truth
  2. gt_evolution   -- Fused vs Model 1 vs Combined label masks on RGB
  + optional 3rd figure if LightGBM model is found:
  3. prediction     -- Final model prediction overlaid on S2 RGB

Usage (from project root, with deep_learning conda env):
    python scripts/visualize_tile.py                        # default tile
    python scripts/visualize_tile.py --tile 18NWG_6_6
    python scripts/visualize_tile.py --tile 18NWG_6_6 --year 2023 --month 6
"""

import sys
import argparse
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import (
    S2_TRAIN, RADD_DIR, GLADL_DIR, GLADS2_DIR,
    FUSED_LABELS_DIR, MODEL1_DIR, AEF_TRAIN,
)
from loader import load_s2, list_available_months

HANSEN_CLIPPED = Path("./data/hansen/clipped")
MODEL1_PROBS   = Path(MODEL1_DIR) / "forest_probs"
MODEL1_LABELS  = Path(MODEL1_DIR) / "improved_labels"
MODEL2_PATH    = Path("./outputs/model2_sklearn.pkl")
AEF_YEARS      = ["2020", "2021", "2022", "2023", "2024", "2025"]
OUTPUT_DIR     = Path("./outputs/figures")

DARK_BG  = "#111111"
RED      = "#E84040"
ORANGE   = "#E67E22"
YELLOW   = "#F1C40F"
GREEN    = "#2ECC71"
BLUE     = "#3498DB"
PURPLE   = "#9B59B6"

TITLE_FS   = 13
CAPTION_FS = 10


def _load_band(path):
    p = Path(path)
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1)


def _overlay(mask, color_hex, alpha=0.55):
    """Return an RGBA overlay array for the given binary mask."""
    r = int(color_hex[1:3], 16) / 255
    g = int(color_hex[3:5], 16) / 255
    b = int(color_hex[5:7], 16) / 255
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    out[mask == 1] = [r, g, b, alpha]
    return out


def _pct(mask):
    return 100.0 * mask.sum() / mask.size if mask is not None else 0.0


def make_rgb(tile_id, year, month):
    bands, _ = load_s2(tile_id, year, month, data_split="train")
    if bands is None:
        return None
    rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1)
    return np.clip(rgb * 3.5, 0, 1)


def load_labels(tile_id):
    labels = {}

    radd = _load_band(f"{RADD_DIR}/radd_{tile_id}_labels.tif")
    labels["RADD"] = (radd > 0).astype(np.uint8) if radd is not None else None

    gladl = None
    for yy in ["21", "22", "23", "24", "25"]:
        d = _load_band(f"{GLADL_DIR}/gladl_{tile_id}_alert{yy}.tif")
        if d is not None:
            m = (d > 0).astype(np.uint8)
            gladl = m if gladl is None else (gladl | m)
    labels["GLAD-L"] = gladl

    gs = _load_band(f"{GLADS2_DIR}/glads2_{tile_id}_alert.tif")
    labels["GLAD-S2"] = (gs > 0).astype(np.uint8) if gs is not None else None

    hansen = _load_band(HANSEN_CLIPPED / f"{tile_id}_lossyear.tif")
    labels["Hansen"] = ((hansen >= 21) & (hansen <= 24)).astype(np.uint8) if hansen is not None else None

    fused_p = Path(f"{FUSED_LABELS_DIR}/{tile_id}_fused.tif")
    labels["Fused"] = _load_band(fused_p)

    m1_p = MODEL1_LABELS / f"{tile_id}_improved_label.tif"
    labels["Model 1"] = _load_band(m1_p)

    prob_p = MODEL1_PROBS / f"{tile_id}_forest_prob_2020.tif"
    prob = _load_band(prob_p)
    if labels["Fused"] is not None and labels["Model 1"] is not None and prob is not None:
        labels["Combined"] = ((labels["Fused"] | labels["Model 1"]) & (prob >= 0.5)).astype(np.uint8)
    else:
        labels["Combined"] = None

    return labels


def load_model2_prediction(tile_id, rgb_shape):
    if not MODEL2_PATH.exists():
        return None
    aef_dir = Path(AEF_TRAIN)
    stack, aef_shape = [], None
    for year in AEF_YEARS:
        p = aef_dir / f"{tile_id}_{year}.tiff"
        if not p.exists():
            return None
        with rasterio.open(p) as src:
            raw = src.read().astype(np.float32)
            if aef_shape is None:
                aef_shape = src.shape  # (H, W) from the AEF raster
        zero = np.all(raw == 0, axis=0, keepdims=True)
        raw  = np.where(zero, np.nan, raw)
        raw  = (raw - 127.0) / 127.5
        neg  = raw < 0
        raw  = np.abs(raw) ** 2.0
        raw[neg] *= -1.0
        stack.append(raw)
    aef  = np.concatenate(stack, axis=0)   # (384, H, W)
    flat = aef.reshape(384, -1).T          # (H*W, 384)
    valid = np.isfinite(flat).all(axis=1)
    proba = np.zeros(flat.shape[0], dtype=np.float32)
    with open(MODEL2_PATH, "rb") as f:
        model = pickle.load(f)
    proba[valid] = model.predict_proba(flat[valid])[:, 1]
    pred = (proba.reshape(aef_shape) > 0.5).astype(np.uint8)
    # Resize to RGB grid if shapes differ (nearest-neighbour via zoom)
    if aef_shape != rgb_shape:
        from scipy.ndimage import zoom
        factors = (rgb_shape[0] / aef_shape[0], rgb_shape[1] / aef_shape[1])
        pred = (zoom(pred.astype(np.float32), factors, order=0) > 0.5).astype(np.uint8)
    return pred


def _blank_panel(ax, label):
    ax.set_facecolor(DARK_BG)
    ax.text(0.5, 0.5, f"{label}\n(not available)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=CAPTION_FS, color="white")
    ax.axis("off")


# ── Figure 1: Weak label sources → fused ground truth ───────────────────────

def fig_label_sources(tile_id, rgb, labels):
    """
    1×6 grid: S2 RGB | RADD | GLAD-L | GLAD-S2 | Hansen | Fused (on RGB)
    Shows the 4 weak label sources and the majority-vote fusion result.
    """
    sources = [
        ("RADD",    RED,    "Radar-based deforestation alerts\n(JAXA ALOS-2)"),
        ("GLAD-L",  ORANGE, "Landsat-based forest disturbance\n(Hansen/UMD, 2021–2025)"),
        ("GLAD-S2", YELLOW, "Sentinel-2 forest disturbance\n(Hansen/UMD)"),
        ("Hansen",  GREEN,  "Hansen GFC lossyear\n(post-2020 only)"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(24, 5), dpi=150)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Weak Label Sources → Fused Ground Truth  |  Tile {tile_id}",
        fontsize=TITLE_FS + 1, fontweight="bold", y=1.01,
    )

    # Panel 0: S2 RGB reference
    ax = axes[0]
    ax.imshow(rgb)
    ax.set_title("Sentinel-2 RGB\nTrue-colour reference", fontsize=TITLE_FS, pad=6)
    ax.axis("off")

    # Panels 1–4: individual sources
    for i, (name, color, caption) in enumerate(sources):
        ax = axes[i + 1]
        mask = labels.get(name)
        if mask is not None:
            bg = np.zeros((*mask.shape, 3))
            ax.imshow(bg)
            ax.imshow(_overlay(mask, color, alpha=0.9), interpolation="nearest")
            ax.set_title(
                f"{name}\n{caption}\n"
                f"{int(mask.sum()):,} px  ({_pct(mask):.1f}%)",
                fontsize=CAPTION_FS, pad=6,
            )
            ax.legend(
                handles=[Patch(facecolor=color, label="Deforestation")],
                loc="lower right", fontsize=8, framealpha=0.7,
            )
        else:
            _blank_panel(ax, name)
            ax.set_title(f"{name}\n{caption}", fontsize=CAPTION_FS, pad=6)
        ax.axis("off")

    # Panel 5: Fused result on RGB
    ax = axes[5]
    fused = labels.get("Fused")
    ax.imshow(rgb)
    if fused is not None:
        ax.imshow(_overlay(fused, RED, alpha=0.6), interpolation="nearest")
        ax.set_title(
            f"Fused Ground Truth\nMajority vote (≥ 2/4 sources) + forest mask\n"
            f"{int(fused.sum()):,} px  ({_pct(fused):.1f}%)",
            fontsize=CAPTION_FS, pad=6,
        )
        ax.legend(
            handles=[Patch(facecolor=RED, alpha=0.6, label="Deforestation")],
            loc="lower right", fontsize=8, framealpha=0.7,
        )
    else:
        ax.set_title("Fused Ground Truth\n(run fuse_labels.py first)", fontsize=CAPTION_FS, pad=6)
    ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / f"{tile_id}_label_sources.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Figure 2: Ground truth evolution ────────────────────────────────────────

def fig_gt_evolution(tile_id, rgb, labels):
    """
    1×3 grid: Fused | Model 1 | Combined
    Shows how ground truth quality improves through the pipeline.
    """
    panels = [
        ("Fused",    RED,    "Majority-vote fusion\n(RADD + GLAD-L + GLAD-S2 + Hansen)"),
        ("Model 1",  BLUE,   "Forest classifier labels\n(AEF embeddings + ESA WorldCover)"),
        ("Combined", PURPLE, "Combined (Fused ∪ Model 1)\nfiltered to confirmed forest pixels"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=150)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Ground Truth Refinement Pipeline  |  Tile {tile_id}",
        fontsize=TITLE_FS + 1, fontweight="bold", y=1.01,
    )

    for ax, (name, color, caption) in zip(axes, panels):
        mask = labels.get(name)
        ax.imshow(rgb)
        if mask is not None:
            ax.imshow(_overlay(mask, color, alpha=0.6), interpolation="nearest")
            ax.set_title(
                f"{name}\n{caption}\n"
                f"{int(mask.sum()):,} px  ({_pct(mask):.1f}%)",
                fontsize=CAPTION_FS + 1, pad=8,
            )
            ax.legend(
                handles=[Patch(facecolor=color, alpha=0.6, label="Deforestation")],
                loc="lower right", fontsize=9, framealpha=0.7,
            )
        else:
            ax.set_title(f"{name}\n{caption}\n(not available)", fontsize=CAPTION_FS + 1, pad=8)
        ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / f"{tile_id}_gt_evolution.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Figure 3: Final model prediction (optional) ──────────────────────────────

def fig_prediction(tile_id, rgb, labels, pred):
    """
    1×2 grid: Best ground truth (fused) | LightGBM prediction
    Side-by-side comparison for the presentation punchline.
    """
    fused = labels.get("Fused")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Ground Truth vs. Model Prediction  |  Tile {tile_id}",
        fontsize=TITLE_FS + 1, fontweight="bold", y=1.01,
    )

    # Left: fused ground truth
    ax = axes[0]
    ax.imshow(rgb)
    if fused is not None:
        ax.imshow(_overlay(fused, RED, alpha=0.6), interpolation="nearest")
        ax.set_title(
            f"Fused Ground Truth\n{int(fused.sum()):,} deforested pixels  ({_pct(fused):.1f}%)",
            fontsize=TITLE_FS, pad=8,
        )
        ax.legend(handles=[Patch(facecolor=RED, alpha=0.6, label="True deforestation")],
                  loc="lower right", fontsize=9, framealpha=0.7)
    else:
        ax.set_title("Fused Ground Truth\n(not available)", fontsize=TITLE_FS, pad=8)
    ax.axis("off")

    # Right: LightGBM prediction
    ax = axes[1]
    ax.imshow(rgb)
    if pred is not None:
        ax.imshow(_overlay(pred, BLUE, alpha=0.6), interpolation="nearest")
        ax.set_title(
            f"LightGBM Prediction\n{int(pred.sum()):,} deforested pixels  ({_pct(pred):.1f}%)",
            fontsize=TITLE_FS, pad=8,
        )
        ax.legend(handles=[Patch(facecolor=BLUE, alpha=0.6, label="Predicted deforestation")],
                  loc="lower right", fontsize=9, framealpha=0.7)
    else:
        ax.set_title("LightGBM Prediction\n(model not found)", fontsize=TITLE_FS, pad=8)
    ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / f"{tile_id}_prediction.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile",  default="18NWG_6_6")
    parser.add_argument("--year",  type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    args = parser.parse_args()

    tile_id = args.tile
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.year and args.month:
        year, month = args.year, args.month
    else:
        months = list_available_months(tile_id, data_split="train")
        if not months:
            print(f"No Sentinel-2 data found for tile {tile_id}")
            sys.exit(1)
        mid = [m for m in months if m[1] in (5, 6, 7, 8)]
        year, month = (mid[-1] if mid else months[-1])

    print(f"Tile: {tile_id}  |  Scene: {year}-{month:02d}\n")

    print("Loading S2 RGB...")
    rgb = make_rgb(tile_id, year, month)
    if rgb is None:
        print("ERROR: Could not load Sentinel-2 data.")
        sys.exit(1)

    print("Loading labels...")
    labels = load_labels(tile_id)

    print("Figure 1: label sources...")
    fig_label_sources(tile_id, rgb, labels)

    print("Figure 2: ground truth evolution...")
    fig_gt_evolution(tile_id, rgb, labels)

    print("Figure 3: model prediction...")
    pred = load_model2_prediction(tile_id, rgb.shape[:2])
    fig_prediction(tile_id, rgb, labels, pred)

    print(f"\nAll figures saved to {OUTPUT_DIR.resolve()}")