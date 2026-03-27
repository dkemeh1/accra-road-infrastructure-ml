"""
Accra Road Surface Mapping Pipeline (Sentinel-2 Only)
=====================================================

"Mapping Road Surface Conditions in Accra Using Multi-Sensor Satellite Data"


Pipeline steps:
1. Extract OSM road surface tags
2. Create weak training labels
3. Segment road network into 100 m segments
4. Extract Sentinel-2 features
5. Train models using spatial cross-validation
6. Predict road surface classes
7. Detect road surface changes
8. Compare Sentinel-2 model outputs

Author: Desmond Kemeh
Institution: Ariel University

Expected project structure:
project_root/
├── snapshots/
│   ├── snapshot_2018.geojson
│   └── snapshot_2024.geojson
├── imagery/
│   ├── S2_2018.tif
│   └── S2_2024.tif
└── outputs/

Usage example:
    python accra_sentinel2_road_surface_pipeline.py --project-root ./project_root
"""

from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

YEAR_A = "2018"
YEAR_B = "2024"

METRIC_CRS = "EPSG:32630"   # UTM Zone 30N
SEG_LEN_M = 100.0
TILE_SIZE_M = 2000
MAX_MATCH_DIST_M = 15

FEATURE_CLIP_QLOW = 0.01
FEATURE_CLIP_QHIGH = 0.99
THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
MIN_POS_RATE = 0.05
MAX_POS_RATE = 0.95

MODEL_TYPES = ["RF", "XGB", "LGBM"]

RF_PARAMS = dict(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_features=0.5,
    min_samples_leaf=1,
    min_samples_split=4,
)

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
)

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

PAVED_SET = {"asphalt", "concrete", "paved", "paving_stones", "sett", "concrete:plates"}
UNPAVED_SET = {"unpaved", "pebblestone", "grass", "metal", "wood", "rock",
               "gravel", "dirt", "earth", "ground", "sand", "mud", "fine_gravel", "compacted"}

LAYER_BY_CLASS = {
    "Upgrade_unpaved_to_paved": "upgrade_unpaved_to_paved",
    "Downgrade_paved_to_unpaved": "downgrade_paved_to_unpaved",
    "Stable_paved": "stable_paved",
    "Stable_unpaved": "stable_unpaved",
    "Unmatched_or_unknown": "unmatched_or_unknown",
}


# ============================================================
# COMMON HELPERS
# ============================================================

def ensure_crs_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure a GeoDataFrame has WGS84 CRS."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def get_surface_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Extract OSM surface tags from direct fields or nested tags."""
    direct = gdf["surface"] if "surface" in gdf.columns else pd.Series([None] * len(gdf), index=gdf.index)
    if "tags" in gdf.columns:
        nested = gdf["tags"].apply(lambda x: x.get("surface") if isinstance(x, dict) else None)
    else:
        nested = pd.Series([None] * len(gdf), index=gdf.index)
    return direct.where(direct.notna(), nested)


def get_highway_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Extract OSM highway tags from direct fields or nested tags."""
    direct = gdf["highway"] if "highway" in gdf.columns else pd.Series([None] * len(gdf), index=gdf.index)
    if "tags" in gdf.columns:
        nested = gdf["tags"].apply(lambda x: x.get("highway") if isinstance(x, dict) else None)
    else:
        nested = pd.Series([None] * len(gdf), index=gdf.index)
    return direct.where(direct.notna(), nested)


def extract_osm_id(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Extract a usable OSM identifier from common possible fields."""
    for c in ["@id", "id", "osm_id", "osmid", "osmId", "@osmId"]:
        if c in gdf.columns:
            return gdf[c]
    if "tags" in gdf.columns:
        return gdf["tags"].apply(lambda x: x.get("@id") if isinstance(x, dict) else None)
    return pd.Series([None] * len(gdf), index=gdf.index)


def to_lines(geom):
    """Convert supported geometry to a LineString where possible."""
    if geom is None:
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
    return None


def split_line_into_segments(line: LineString, seg_len: float) -> list[LineString]:
    """Split a line into approximately equal-length segments."""
    length = line.length
    if length <= 0:
        return []

    n = max(1, int(math.ceil(length / seg_len)))
    segs = []

    for i in range(n):
        start_d = i * seg_len
        end_d = min((i + 1) * seg_len, length)

        if end_d - start_d < 1.0:
            continue

        steps = max(2, int((end_d - start_d) / 5))
        pts = [line.interpolate(start_d + (end_d - start_d) * (k / steps)) for k in range(steps + 1)]
        segs.append(LineString(pts))

    return segs


def make_model(model_type: str):
    """Instantiate a classifier by name."""
    model_type = model_type.upper()

    if model_type == "RF":
        return RandomForestClassifier(**RF_PARAMS)
    if model_type == "XGB":
        return XGBClassifier(**XGB_PARAMS)
    if model_type == "LGBM":
        return LGBMClassifier(**LGBM_PARAMS)

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# PATHS
# ============================================================

def make_paths(project_root: Path) -> dict:
    """Create and return standard project paths."""
    snapshots_dir = project_root / "snapshots"
    imagery_dir = project_root / "imagery"
    outputs_dir = project_root / "outputs"

    global_dir = outputs_dir / "GLOBAL_preprocessing"
    step5_dir = outputs_dir / "step5_models"
    step6_dir = outputs_dir / "step6_predictions"
    step7_dir = outputs_dir / "step7_change_detection"
    step8_dir = outputs_dir / "_COMPARISON_STEP8"

    for d in [outputs_dir, global_dir, step5_dir, step6_dir, step7_dir, step8_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "snap_a": snapshots_dir / f"snapshot_{YEAR_A}.geojson",
        "snap_b": snapshots_dir / f"snapshot_{YEAR_B}.geojson",
        "s2_a": imagery_dir / f"S2_{YEAR_A}.tif",
        "s2_b": imagery_dir / f"S2_{YEAR_B}.tif",
        "global_dir": global_dir,
        "step5_dir": step5_dir,
        "step6_dir": step6_dir,
        "step7_dir": step7_dir,
        "step8_dir": step8_dir,
    }


# ============================================================
# STEP 1 — UNIQUE SURFACE TAGS
# ============================================================

def step1_extract_unique_surfaces(snapshot_path: Path, year_label: str, out_dir: Path) -> None:
    """Save unique OSM surface tags to a text file."""
    print(f"\nSTEP 1 — UNIQUE SURFACE TAG VALUES: {year_label}")

    gdf = ensure_crs_wgs84(gpd.read_file(snapshot_path))
    surface = get_surface_series(gdf)

    unique_vals = (
        surface.dropna().astype(str).str.strip()
        .replace("", pd.NA).dropna().unique().tolist()
    )
    unique_vals = sorted(set(unique_vals), key=lambda s: s.lower())

    out_txt = out_dir / f"surface_tags_{year_label}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for v in unique_vals:
            f.write(v + "\n")

    print(f"Saved: {out_txt}")


# ============================================================
# STEP 2 — WEAK LABELS
# ============================================================

def step2_label_snapshot(in_path: Path, year_label: str, out_dir: Path) -> Path:
    """Create weak training labels from OSM surface tags."""
    print(f"\nSTEP 2 — TRAINING LABELS (PAVED=1 / UNPAVED=0): {year_label}")

    gdf = ensure_crs_wgs84(gpd.read_file(in_path))
    gdf["highway_extracted"] = get_highway_series(gdf)
    gdf["osm_id"] = extract_osm_id(gdf)

    roads = gdf[gdf["highway_extracted"].notna()].copy()
    roads["surface_extracted"] = get_surface_series(roads).astype(str).str.strip().str.lower()
    roads.loc[roads["surface_extracted"].isin(["none", "nan", ""]), "surface_extracted"] = pd.NA
    roads = roads[roads["surface_extracted"].notna()].copy()

    roads["label"] = pd.NA
    roads.loc[roads["surface_extracted"].isin(PAVED_SET), "label"] = 1
    roads.loc[roads["surface_extracted"].isin(UNPAVED_SET), "label"] = 0

    train = roads[roads["label"].notna()].copy()
    train["label"] = train["label"].astype(int)
    train["year"] = int(year_label)

    out_gpkg = out_dir / f"train_{year_label}.gpkg"
    train.to_file(out_gpkg, layer="train", driver="GPKG")

    print(f"Saved: {out_gpkg} | rows={len(train):,}")
    return out_gpkg


# ============================================================
# STEP 3 — SEGMENT ROADS
# ============================================================

def step3_segment_all_roads(snapshot_path: Path, year_label: str, out_dir: Path) -> Path:
    """Segment all road geometries into 100 m segments."""
    print(f"\nSTEP 3A — SEGMENT ALL ROADS: {year_label}")

    gdf = ensure_crs_wgs84(gpd.read_file(snapshot_path))
    gdf["highway"] = get_highway_series(gdf)
    gdf["surface"] = get_surface_series(gdf)
    gdf["osm_id"] = extract_osm_id(gdf)

    roads = gdf[gdf["highway"].notna()].copy()
    roads = roads[~roads.geometry.isna()].copy()
    roads = roads.to_crs(METRIC_CRS)
    roads["geometry"] = roads["geometry"].apply(to_lines)
    roads = roads[roads["geometry"].notna()].copy()

    rows = []
    for idx, r in roads.iterrows():
        segs = split_line_into_segments(r.geometry, SEG_LEN_M)
        for j, seg in enumerate(segs):
            rows.append({
                "seg_id": f"{year_label}_{idx}_{j}",
                "year": int(year_label),
                "osm_id": r.get("osm_id"),
                "highway": str(r.get("highway")),
                "surface": r.get("surface"),
                "geometry": seg,
            })

    seg_gdf = gpd.GeoDataFrame(rows, crs=METRIC_CRS)
    out_all = out_dir / f"segments_{year_label}_all.gpkg"
    seg_gdf.to_file(out_all, layer="segments_all", driver="GPKG")

    print(f"Saved ALL segments: {len(seg_gdf):,} -> {out_all}")
    return out_all


def step3_make_train_segments(seg_all_gpkg: Path, train_gpkg: Path, year_label: str, out_dir: Path) -> Path:
    """Create training segments by joining all segments with weak labels via osm_id."""
    print(f"\nSTEP 3B — BUILD TRAIN SEGMENTS: {year_label}")

    seg_all = gpd.read_file(seg_all_gpkg, layer="segments_all").to_crs(METRIC_CRS)
    train = ensure_crs_wgs84(gpd.read_file(train_gpkg, layer="train")).to_crs(METRIC_CRS)

    train_small = train[["osm_id", "label"]].dropna(subset=["osm_id"]).copy()
    joined = seg_all.merge(train_small, on="osm_id", how="inner")

    out_train = out_dir / f"segments_{year_label}_train.gpkg"
    joined.to_file(out_train, layer="segments_train", driver="GPKG")

    print(f"Saved TRAIN segments: {len(joined):,} -> {out_train}")
    return out_train


# ============================================================
# STEP 4 — SENTINEL-2 FEATURE EXTRACTION
# ============================================================

def build_s2_from_export(raster_path: Path):
    """
    Build the final Sentinel-2 feature stack.
    Expected bands:
    1 BLUE
    2 GREEN
    3 RED
    4 NIR
    5 SWIR1
    6 SWIR2
    Optional:
    7 NDVI
    8 NDBI
    """
    with rasterio.open(raster_path) as src:
        stack = src.read().astype("float32")
        profile = src.profile

    blue = stack[0]
    green = stack[1]
    red = stack[2]
    nir = stack[3]
    swir1 = stack[4]
    swir2 = stack[5]

    if stack.shape[0] >= 8:
        ndvi = stack[6]
        ndbi = stack[7]
    else:
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)

    bright = (blue + green + red) / 3.0

    final_stack = np.stack([
        blue, green, red, nir,
        swir1, swir2,
        ndvi, ndbi, bright
    ])

    feature_names = [
        "BLUE", "GREEN", "RED", "NIR",
        "SWIR1", "SWIR2",
        "NDVI", "NDBI", "BRIGHT"
    ]

    return final_stack, profile, feature_names


def sample_points_along_line(line: LineString, n: int = 5):
    """Sample evenly spaced points along a line."""
    if line is None or line.length == 0:
        return []
    distances = np.linspace(0, line.length, n)
    return [line.interpolate(d) for d in distances]


def step4_extract_s2_features(seg_all_gpkg: Path, raster_path: Path, out_csv: Path) -> None:
    """Extract Sentinel-2 features for each road segment."""
    print(f"\nSTEP 4 — EXTRACT SENTINEL-2 FEATURES: {raster_path.name}")

    seg = gpd.read_file(seg_all_gpkg, layer="segments_all")
    if seg.crs != METRIC_CRS:
        seg = seg.to_crs(METRIC_CRS)

    with rasterio.open(raster_path) as src:
        if src.crs != METRIC_CRS:
            seg = seg.to_crs(src.crs)

        stack, _, feature_names = build_s2_from_export(raster_path)
        rows = []

        for geom in seg.geometry:
            pts = sample_points_along_line(geom, n=5)
            values = []

            for pt in pts:
                x, y = pt.x, pt.y
                row, col = src.index(x, y)

                if 0 <= row < stack.shape[1] and 0 <= col < stack.shape[2]:
                    values.append(stack[:, row, col])
                else:
                    values.append(np.full(stack.shape[0], np.nan))

            vals = np.array(values, dtype="float32")
            if np.isnan(vals).all():
                pixel_vals = np.full(stack.shape[0], np.nan)
            else:
                pixel_vals = np.nanmean(vals, axis=0)

            rows.append(pixel_vals)

    df = pd.DataFrame(rows, columns=feature_names)
    df["seg_id"] = seg["seg_id"].values
    df["year"] = seg["year"].values
    df["highway"] = seg["highway"].values
    df["surface"] = seg["surface"].values

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


# ============================================================
# STEP 5 — TRAIN + VALIDATE
# ============================================================

def add_tile_group(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign spatial tile IDs for GroupKFold."""
    cent = gdf.geometry.centroid
    gdf["tx"] = (cent.x // TILE_SIZE_M).astype(int)
    gdf["ty"] = (cent.y // TILE_SIZE_M).astype(int)
    gdf["tile_id"] = gdf["tx"].astype(str) + "_" + gdf["ty"].astype(str)
    return gdf


def load_features(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load feature CSV and return numeric feature columns."""
    df = pd.read_csv(csv_path)
    keep_non_num = {"seg_id", "year", "highway", "surface"}
    feat_cols = [c for c in df.columns if c not in keep_non_num and pd.api.types.is_numeric_dtype(df[c])]
    return df, feat_cols


def _compute_clip_bounds(X: pd.DataFrame, qlow: float = FEATURE_CLIP_QLOW, qhigh: float = FEATURE_CLIP_QHIGH) -> dict:
    """Compute per-feature clipping bounds from training data only."""
    bounds = {}
    for c in X.columns:
        s = X[c].dropna()
        if len(s) == 0:
            continue
        lo = float(s.quantile(qlow))
        hi = float(s.quantile(qhigh))
        if lo > hi:
            lo, hi = hi, lo
        bounds[c] = (lo, hi)
    return bounds


def _clip_with_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Apply precomputed clip bounds to features."""
    Xc = X.copy()
    for c, (lo, hi) in bounds.items():
        if c in Xc.columns:
            Xc[c] = Xc[c].clip(lower=lo, upper=hi)
    return Xc


def _passes_guardrail(pred: np.ndarray) -> bool:
    """Reject thresholds that collapse predictions into one class."""
    pos_rate = float(np.mean(pred == 1))
    return MIN_POS_RATE <= pos_rate <= MAX_POS_RATE


def _mcc_from_cm(cm: np.ndarray) -> float:
    """Compute MCC from a 2x2 confusion matrix."""
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    fn, tp = float(cm[1, 0]), float(cm[1, 1])

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0

    return (tp * tn - fp * fn) / denom


def _cm_row_normalized(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix."""
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def _metrics_from_pred(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """Compute model metrics from predictions."""
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, pred))
    balacc = float(balanced_accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(_mcc_from_cm(cm))
    pos_rate = float(np.mean(pred == 1))
    cm_norm = _cm_row_normalized(cm)

    return {
        "acc": acc,
        "balacc": balacc,
        "precision": prec,
        "recall": rec,
        "tpr": rec,
        "tnr": tnr,
        "mcc": mcc,
        "pos_rate": pos_rate,
        "cm": cm,
        "cm_norm": cm_norm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _search_thresholds(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Find the best threshold based on balanced accuracy."""
    best_score = -1.0
    best_row = None

    for t in THRESH_GRID:
        pred = (proba >= t).astype(int)
        if not _passes_guardrail(pred):
            continue

        m = _metrics_from_pred(y_true, pred)
        if m["balacc"] > best_score:
            best_score = m["balacc"]
            best_row = {"t": float(t), **m}

    if best_row is None:
        t = 0.50
        pred = (proba >= t).astype(int)
        m = _metrics_from_pred(y_true, pred)
        best_row = {"t": float(t), **m}

    return best_row


def step5_train_model_with_validation(
    year_label: str,
    train_seg_gpkg: Path,
    feat_csv: Path,
    model_out: Path,
    eval_out_csv: Path,
    model_type: str,
) -> None:
    """Train a model with spatial CV and save final fitted model bundle."""
    print(f"\nSTEP 5 — TRAIN + VALIDATE: {year_label} | {model_type}")

    seg_train = gpd.read_file(train_seg_gpkg, layer="segments_train").to_crs(METRIC_CRS)
    seg_train = add_tile_group(seg_train)

    Xdf, feat_cols = load_features(feat_csv)
    df = seg_train[["seg_id", "label", "tile_id"]].merge(Xdf, on="seg_id", how="inner").copy()

    if df.empty:
        raise RuntimeError(f"[{year_label}] No training rows after merge.")

    y = df["label"].astype(int).values
    groups = df["tile_id"].astype(str).values
    Xraw = df[feat_cols].copy()

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Not enough spatial groups for GroupKFold.")

    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.full(shape=(len(Xraw),), fill_value=np.nan, dtype=float)

    for fold, (tr, te) in enumerate(gkf.split(Xraw, y, groups), start=1):
        X_tr = Xraw.iloc[tr].copy()
        X_te = Xraw.iloc[te].copy()

        med_tr = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med_tr)
        X_te = X_te.fillna(med_tr)

        clip_bounds_tr = _compute_clip_bounds(X_tr)
        X_tr = _clip_with_bounds(X_tr, clip_bounds_tr)
        X_te = _clip_with_bounds(X_te, clip_bounds_tr)

        clf = make_model(model_type)
        clf.fit(X_tr, y[tr])

        proba = clf.predict_proba(X_te)[:, 1]
        oof_proba[te] = proba

        pred50 = (proba >= 0.50).astype(int)
        m50 = _metrics_from_pred(y[te], pred50)

        print(f"Fold {fold}: balacc@0.50={m50['balacc']:.3f} | mcc@0.50={m50['mcc']:.3f}")

    mask = ~np.isnan(oof_proba)
    y_oof = y[mask]
    p_oof = oof_proba[mask]
    best = _search_thresholds(y_oof, p_oof)

    cm_norm = best["cm_norm"]
    summary = pd.DataFrame([{
        "year": int(year_label),
        "model": model_type,
        "best_threshold": float(best["t"]),
        "balanced_accuracy": float(best["balacc"]),
        "mcc": float(best["mcc"]),
        "tn": int(best["tn"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "tp": int(best["tp"]),
        "tn_pct": round(float(cm_norm[0, 0] * 100), 2),
        "fp_pct": round(float(cm_norm[0, 1] * 100), 2),
        "fn_pct": round(float(cm_norm[1, 0] * 100), 2),
        "tp_pct": round(float(cm_norm[1, 1] * 100), 2),
        "n_train_rows": int(len(df)),
        "n_features": int(len(feat_cols)),
    }])

    summary.to_csv(eval_out_csv, index=False)
    print(f"Saved eval CSV: {eval_out_csv}")

    med_full = Xraw.median(numeric_only=True)
    X_full = Xraw.fillna(med_full)
    clip_bounds_full = _compute_clip_bounds(X_full)
    X_full = _clip_with_bounds(X_full, clip_bounds_full)

    final_model = make_model(model_type)
    final_model.fit(X_full, y)

    joblib.dump({
        "model": final_model,
        "features": feat_cols,
        "train_medians": med_full.to_dict(),
        "clip_bounds": clip_bounds_full,
        "threshold_used": float(best["t"]),
        "threshold_mode": "BALACC",
        "provenance": "Sentinel-2 only paper version",
    }, model_out)

    print(f"Saved model bundle: {model_out}")


# ============================================================
# STEP 6 — PREDICT ALL
# ============================================================

def _ensure_feature_columns(df: pd.DataFrame, feat_cols: list[str], fill_values: dict) -> pd.DataFrame:
    """Ensure all required feature columns exist and fill missing columns."""
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = np.nan
        if c in fill_values:
            out[c] = out[c].fillna(fill_values[c])
    return out


def step6_predict_all_with_qc(
    year_label: str,
    seg_all_gpkg: Path,
    feat_csv: Path,
    model_path: Path,
    out_gpkg: Path,
    qc_out_csv: Path,
) -> None:
    """Predict road surface for all segments and save QC outputs."""
    print(f"\nSTEP 6 — PREDICT ALL + QC: {year_label}")

    seg_all = gpd.read_file(seg_all_gpkg, layer="segments_all").to_crs(METRIC_CRS)
    Xdf, _ = load_features(feat_csv)

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    feat_cols = bundle["features"]
    clip_bounds = bundle.get("clip_bounds", {})
    train_medians = bundle.get("train_medians", {})
    threshold = float(bundle.get("threshold_used", 0.50))

    df = seg_all.merge(Xdf, on="seg_id", how="left")
    miss_feat = float(df[feat_cols].isna().mean().mean()) if all(c in df.columns for c in feat_cols) else np.nan

    df = _ensure_feature_columns(df, feat_cols, train_medians)
    Xmat = df[feat_cols].copy()

    for c in feat_cols:
        if c in train_medians:
            Xmat[c] = Xmat[c].fillna(train_medians[c])

    if clip_bounds:
        Xmat = _clip_with_bounds(Xmat, clip_bounds)

    proba = clf.predict_proba(Xmat)[:, 1]
    pred = (proba >= threshold).astype(int)

    df["p_paved"] = proba
    df["pred_label"] = pred
    df["pred_surface"] = df["pred_label"].map({1: "paved", 0: "unpaved"})
    df["threshold_used"] = threshold
    df["threshold_mode"] = "BALACC"

    out = gpd.GeoDataFrame(df, crs=METRIC_CRS)
    out.to_file(out_gpkg, layer="predicted", driver="GPKG")
    print(f"Saved predictions: {out_gpkg}")

    qc = pd.DataFrame([{
        "year": int(year_label),
        "threshold_used": threshold,
        "threshold_mode": "BALACC",
        "n_segments": int(len(df)),
        "mean_missing_feature_rate": float(miss_feat) if pd.notna(miss_feat) else np.nan,
        "p_paved_mean": float(np.mean(proba)),
        "p_paved_std": float(np.std(proba)),
        "p_paved_min": float(np.min(proba)),
        "p_paved_max": float(np.max(proba)),
        "pct_pred_paved": float(np.mean(pred == 1) * 100.0),
        "pct_pred_unpaved": float(np.mean(pred == 0) * 100.0),
    }])

    qc.to_csv(qc_out_csv, index=False)
    print(f"Saved QC CSV: {qc_out_csv}")


# ============================================================
# STEP 7 — CHANGE DETECTION
# ============================================================

def step7_change_class(s_a: str, s_b: str) -> str:
    """Assign a surface change class between two years."""
    if s_a == "unpaved" and s_b == "paved":
        return "Upgrade_unpaved_to_paved"
    if s_a == "paved" and s_b == "unpaved":
        return "Downgrade_paved_to_unpaved"
    if s_a == "paved" and s_b == "paved":
        return "Stable_paved"
    if s_a == "unpaved" and s_b == "unpaved":
        return "Stable_unpaved"
    return "Unmatched_or_unknown"


def step7_run_change_detection_for_model(step6_dir: Path, step7_dir: Path, model_type: str) -> None:
    """Run nearest-neighbour change detection between prediction years for one model."""
    print(f"\nSTEP 7 — CHANGE DETECTION ({YEAR_A} → {YEAR_B}) | {model_type}")

    pred_a = step6_dir / f"segments_{YEAR_A}_predicted_{model_type}.gpkg"
    pred_b = step6_dir / f"segments_{YEAR_B}_predicted_{model_type}.gpkg"

    if not pred_a.exists() or not pred_b.exists():
        print(f"Change detection skipped for {model_type}: prediction files not found.")
        return

    g_a = gpd.read_file(pred_a, layer="predicted").to_crs(METRIC_CRS)
    g_b = gpd.read_file(pred_b, layer="predicted").to_crs(METRIC_CRS)

    g_b_small = g_b[["seg_id", "pred_surface", "p_paved", "geometry"]].copy().rename(columns={
        "seg_id": f"seg_id_{YEAR_B}",
        "pred_surface": f"surface_{YEAR_B}",
        "p_paved": f"p_paved_{YEAR_B}",
    })

    joined = gpd.sjoin_nearest(
        g_a,
        g_b_small,
        how="left",
        max_distance=MAX_MATCH_DIST_M,
        distance_col="match_dist_m"
    )

    joined = joined.rename(columns={
        "pred_surface": f"surface_{YEAR_A}",
        "p_paved": f"p_paved_{YEAR_A}",
    })

    joined["change_class"] = joined.apply(
        lambda r: step7_change_class(r.get(f"surface_{YEAR_A}"), r.get(f"surface_{YEAR_B}")),
        axis=1
    )
    joined["len_km"] = joined.geometry.length / 1000.0
    joined["model"] = model_type

    out_csv = step7_dir / f"surface_change_summary_{model_type}_{YEAR_A}_{YEAR_B}.csv"
    summary = (
        joined.groupby("change_class")["len_km"]
        .sum()
        .reset_index()
        .sort_values("len_km", ascending=False)
    )
    summary["len_km"] = summary["len_km"].round(3)
    summary["model"] = model_type
    summary.to_csv(out_csv, index=False)

    out_gpkg = step7_dir / f"surface_change_{model_type}_{YEAR_A}_{YEAR_B}.gpkg"
    joined.to_file(out_gpkg, layer="change_all", driver="GPKG")

    for cls, layer_name in LAYER_BY_CLASS.items():
        sub = joined[joined["change_class"] == cls].copy()
        if len(sub) > 0:
            sub.to_file(out_gpkg, layer=layer_name, driver="GPKG")

    print(f"Saved change GPKG: {out_gpkg}")
    print(f"Saved summary CSV: {out_csv}")


def step7_run_change_detection(step6_dir: Path, step7_dir: Path) -> None:
    """Run change detection for all model types."""
    for model_type in MODEL_TYPES:
        step7_run_change_detection_for_model(step6_dir, step7_dir, model_type)


# ============================================================
# STEP 8 — COMPARE SENTINEL-2 MODELS
# ============================================================

def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Save CSV safely, creating a timestamped alternative if needed."""
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{ts}" + path.suffix)
        df.to_csv(alt, index=False)
        return alt


def read_eval_for_report(eval_csv: Path) -> dict:
    """Read model evaluation summary."""
    if not eval_csv.exists():
        return {}
    df = pd.read_csv(eval_csv)
    if df.empty:
        return {}
    row = df.iloc[0]
    return row.to_dict()


def read_change_summary(summary_csv: Path) -> dict:
    """Read change summary and compute unmatched percentage."""
    if not summary_csv.exists():
        return {}

    df = pd.read_csv(summary_csv)
    if df.empty or "change_class" not in df.columns or "len_km" not in df.columns:
        return {}

    total = float(df["len_km"].sum())
    unmatched = float(df.loc[df["change_class"] == "Unmatched_or_unknown", "len_km"].sum()) \
        if "Unmatched_or_unknown" in set(df["change_class"]) else 0.0

    return {
        "total_change_km": total,
        "unmatched_pct_step7": (unmatched / total * 100.0) if total > 0 else np.nan
    }


def step8_compare_and_report(step5_dir: Path, step7_dir: Path, step8_dir: Path) -> None:
    """Compare RF, XGB, and LGBM for Sentinel-2."""
    rows = []

    for model_type in MODEL_TYPES:
        eval_a = step5_dir / f"eval_{model_type}_{YEAR_A}.csv"
        eval_b = step5_dir / f"eval_{model_type}_{YEAR_B}.csv"
        change_csv = step7_dir / f"surface_change_summary_{model_type}_{YEAR_A}_{YEAR_B}.csv"

        e_a = read_eval_for_report(eval_a)
        e_b = read_eval_for_report(eval_b)
        c_s = read_change_summary(change_csv)

        rows.append({
            "experiment": f"S2_{model_type}",
            "threshold_2018": e_a.get("best_threshold", np.nan),
            "balacc_2018": e_a.get("balanced_accuracy", np.nan),
            "mcc_2018": e_a.get("mcc", np.nan),
            "tn_pct_2018": e_a.get("tn_pct", np.nan),
            "fp_pct_2018": e_a.get("fp_pct", np.nan),
            "fn_pct_2018": e_a.get("fn_pct", np.nan),
            "tp_pct_2018": e_a.get("tp_pct", np.nan),
            "threshold_2024": e_b.get("best_threshold", np.nan),
            "balacc_2024": e_b.get("balanced_accuracy", np.nan),
            "mcc_2024": e_b.get("mcc", np.nan),
            "tn_pct_2024": e_b.get("tn_pct", np.nan),
            "fp_pct_2024": e_b.get("fp_pct", np.nan),
            "fn_pct_2024": e_b.get("fn_pct", np.nan),
            "tp_pct_2024": e_b.get("tp_pct", np.nan),
            "total_change_km": c_s.get("total_change_km", np.nan),
            "unmatched_pct_step7": c_s.get("unmatched_pct_step7", np.nan),
        })

    comp = pd.DataFrame(rows)
    out_csv = step8_dir / f"comparison_step8_{YEAR_A}_{YEAR_B}.csv"
    saved_path = safe_to_csv(comp, out_csv)

    print("\nSTEP 8 — EXPERIMENT COMPARISON")
    print(comp.to_string(index=False))
    print(f"\nSaved comparison CSV: {saved_path}")
# ============================================================
# MAIN WORKFLOW
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sentinel-2 road surface mapping pipeline.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory. Default is the current folder."
    )
    parser.add_argument("--run-step-1", action="store_true", help="Run Step 1.")
    parser.add_argument("--run-step-2", action="store_true", help="Run Step 2.")
    parser.add_argument("--run-step-3", action="store_true", help="Run Step 3.")
    parser.add_argument("--run-step-4", action="store_true", help="Run Step 4.")
    parser.add_argument("--run-step-5-6", action="store_true", help="Run Steps 5 and 6.")
    parser.add_argument("--run-step-7", action="store_true", help="Run Step 7.")
    parser.add_argument("--run-step-8", action="store_true", help="Run Step 8.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    paths = make_paths(Path(args.project_root))

    snap_a = paths["snap_a"]
    snap_b = paths["snap_b"]
    s2_a = paths["s2_a"]
    s2_b = paths["s2_b"]

    global_dir = paths["global_dir"]
    step5_dir = paths["step5_dir"]
    step6_dir = paths["step6_dir"]
    step7_dir = paths["step7_dir"]
    step8_dir = paths["step8_dir"]

    if args.run_step_1:
        step1_extract_unique_surfaces(snap_a, YEAR_A, global_dir)
        step1_extract_unique_surfaces(snap_b, YEAR_B, global_dir)

    if args.run_step_2:
        step2_label_snapshot(snap_a, YEAR_A, global_dir)
        step2_label_snapshot(snap_b, YEAR_B, global_dir)

    if args.run_step_3:
        seg_all_a = step3_segment_all_roads(snap_a, YEAR_A, global_dir)
        seg_all_b = step3_segment_all_roads(snap_b, YEAR_B, global_dir)

        train_a = global_dir / f"train_{YEAR_A}.gpkg"
        train_b = global_dir / f"train_{YEAR_B}.gpkg"

        step3_make_train_segments(seg_all_a, train_a, YEAR_A, global_dir)
        step3_make_train_segments(seg_all_b, train_b, YEAR_B, global_dir)

    seg_all_a = global_dir / f"segments_{YEAR_A}_all.gpkg"
    seg_all_b = global_dir / f"segments_{YEAR_B}_all.gpkg"
    seg_train_a = global_dir / f"segments_{YEAR_A}_train.gpkg"
    seg_train_b = global_dir / f"segments_{YEAR_B}_train.gpkg"

    feat_a = global_dir / f"X_S2_{YEAR_A}.csv"
    feat_b = global_dir / f"X_S2_{YEAR_B}.csv"

    if args.run_step_4:
        step4_extract_s2_features(seg_all_a, s2_a, feat_a)
        step4_extract_s2_features(seg_all_b, s2_b, feat_b)

    if args.run_step_5_6:
        for required in [seg_all_a, seg_all_b, seg_train_a, seg_train_b, feat_a, feat_b]:
            if not required.exists():
                raise FileNotFoundError(f"Missing required file:\n{required}")

        for model_type in MODEL_TYPES:
            model_a = step5_dir / f"model_{model_type}_{YEAR_A}.joblib"
            model_b = step5_dir / f"model_{model_type}_{YEAR_B}.joblib"

            eval_a = step5_dir / f"eval_{model_type}_{YEAR_A}.csv"
            eval_b = step5_dir / f"eval_{model_type}_{YEAR_B}.csv"

            step5_train_model_with_validation(YEAR_A, seg_train_a, feat_a, model_a, eval_a, model_type)
            step5_train_model_with_validation(YEAR_B, seg_train_b, feat_b, model_b, eval_b, model_type)

            step6_predict_all_with_qc(
                YEAR_A,
                seg_all_a,
                feat_a,
                model_a,
                step6_dir / f"segments_{YEAR_A}_predicted_{model_type}.gpkg",
                step6_dir / f"qc_{YEAR_A}_{model_type}.csv",
            )
            step6_predict_all_with_qc(
                YEAR_B,
                seg_all_b,
                feat_b,
                model_b,
                step6_dir / f"segments_{YEAR_B}_predicted_{model_type}.gpkg",
                step6_dir / f"qc_{YEAR_B}_{model_type}.csv",
            )

    if args.run_step_7:
        step7_run_change_detection(step6_dir, step7_dir)
        
    if args.run_step_8:
        step8_compare_and_report(step5_dir, step7_dir, step8_dir)


if __name__ == "__main__":
    main()