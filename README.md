# Accra Road Surface Mapping Pipeline (Sentinel-2 Only)

##  Overview

This project provides a reproducible pipeline for mapping road surface conditions (paved vs. unpaved) in Accra using **Sentinel-2 satellite imagery** and **OpenStreetMap (OSM) data**.

The workflow integrates geospatial data processing and machine learning to identify infrastructure conditions and detect changes over time.

---

##  What This Pipeline Does

The pipeline performs the following steps:

1. Extracts road surface tags from OSM
2. Generates weak training labels (paved vs. unpaved)
3. Segments the road network into 100-meter segments
4. Extracts Sentinel-2 spectral features
5. Trains machine learning models with spatial cross-validation
6. Predicts road surface classes
7. Detects road surface changes between years
8. Compares model performance

---

##  Required Folder Structure

⚠️ The folder structure must be EXACTLY as shown below:

```
project_root/
│
├── accra_sentinel2_road_surface_pipeline.py
├── README.md
├── install_requirements.bat
├── run_pipeline.bat
│
├── snapshots/
│   ├── snapshot_2018.geojson
│   └── snapshot_2024.geojson
│
├── imagery/
│   ├── S2_2018.tif
│   └── S2_2024.tif
│
└── outputs/
```

* The `outputs` folder can be empty.
* Do NOT rename any files.

---

## ⚙️ Requirements

* Python 3.10 or higher
* Internet connection (for installing packages)

---

## 🧩 Installation (VERY IMPORTANT)

1. Install Python from:
   https://www.python.org/downloads/

2. During installation, **tick this option**:

   ```
   ✔ Add Python to PATH
   ```

---

## ▶️ How to Run (Easiest Method)

### Step 1: Install dependencies

Double-click:

```
install_requirements.bat
```

Wait until installation finishes.

---

### Step 2: Run the pipeline

Double-click:

```
run_pipeline.bat
```

That’s it. No coding required.

---

## ⏳ What to Expect

A black command window will open and show progress such as:

```
STEP 1 — UNIQUE SURFACE TAG VALUES
STEP 2 — TRAINING LABELS
STEP 3 — SEGMENT ROADS
STEP 4 — FEATURE EXTRACTION
STEP 5 — MODEL TRAINING
STEP 6 — PREDICTION
...
PIPELINE FINISHED
```

---

## 📊 Output Results

All results will be saved automatically inside:

```
outputs/
```

Including:

* `GLOBAL_preprocessing/` → intermediate data
* `step5_models/` → trained models
* `step6_predictions/` → predictions
* `step7_change_detection/` → detected changes
* `_COMPARISON_STEP8/` → model comparison

---

## ⚠️ Important Notes

* Do NOT rename or move files
* Do NOT change folder structure
* Make sure all required files exist before running

---

## ❗ Troubleshooting

### Error: “python is not recognized”

➡️ Python was not added to PATH
✔ Reinstall Python and tick “Add to PATH”

---

### Error: missing files

✔ Ensure these files exist:

```
snapshots/snapshot_2018.geojson
snapshots/snapshot_2024.geojson
imagery/S2_2018.tif
imagery/S2_2024.tif
```

---

### Nothing happens when clicking .bat files

✔ Right-click → “Run as administrator”
OR
✔ Open Command Prompt and run manually:

```
install_requirements.bat
run_pipeline.bat
```

---

## 👤 Author

Desmond Kemeh
Ariel University

---

## 📌 Version Note

This version is:

* Sentinel-2 only (no multi-sensor integration)
* Designed for reproducibility and ease of use
* Suitable for research and academic publication

---

## 🚀 Quick Start (Advanced Users)

If you prefer command line:

```
pip install numpy pandas geopandas rasterio shapely scikit-learn xgboost lightgbm joblib
python accra_sentinel2_road_surface_pipeline.py --project-root . --run-step-1 --run-step-2 --run-step-3 --run-step-4 --run-step-5-6 --run-step-8
```

---

## ✅ Summary

1. Install Python
2. Double-click `install_requirements.bat`
3. Double-click `run_pipeline.bat`
4. Check results in `outputs/`

---

Enjoy using the pipeline 🚀
