Accra Road Surface Mapping Pipeline (Sentinel-2 Only)

Overview

This project provides a fully reproducible workflow for analysing transport infrastructure inequality in Accra, Ghana.

The workflow is divided into three integrated parts:

- Part 1 — Road Surface Mapping (Python)
- Part 2 — Spatial Analysis (QGIS)
- Part 3 — Statistical Analysis (Python → Excel)

Together, these steps identify blind spots in road infrastructure, assess population exposure, and evaluate transport accessibility inequalities.

--------------------------------------------------

PART 1 — ROAD SURFACE MAPPING (PYTHON)

What This Part Does

This pipeline uses Sentinel-2 imagery and OpenStreetMap (OSM) data to:

1. Extract road surface tags from OSM
2. Generate weak training labels (paved vs unpaved)
3. Segment the road network into 100 m segments
4. Extract Sentinel-2 spectral features
5. Train machine learning models using spatial cross-validation
6. Predict road surface classes
7. Detect road surface changes between years
8. Compare model performance

Data Download

https://drive.google.com/drive/folders/1keBSAerh1B_OoTAXU1Oo2cH3SQ-T2OLC

Place files into:

project_root/
├── snapshots/
├── imagery/
├── QGIS-PART 2/

How to Run (Part 1)

Step 1 — Install dependencies
install_requirements.bat

Step 2 — Run pipeline
run_pipeline.bat

Output (Part 1)

outputs/

Includes:
- GLOBAL_preprocessing/
- step5_models/
- step6_predictions/
- step7_change_detection/
- _COMPARISON_STEP8/

--------------------------------------------------

PART 2 — QGIS ANALYSIS PACKAGE

Overview

This section provides the full QGIS environment used to:

- Visualise spatial patterns
- Identify blind spots
- Analyse accessibility
- Generate manuscript maps
- Export datasets for statistical analysis

Contents of QGIS folder:

- Accra_QGIS_Project.qgz
- GHSL population raster
- Blind spot polygons
- Stable unpaved roads
- OSM roads
- District boundaries

How to Use

1. Open:
QGIS/Qgis.qgz

2. Fix missing layers if needed

3. Ensure CRS:
EPSG:32630

Maps produced:
- Study area
- Blind spots
- Accessibility
- Road hierarchy

Export for Part 3:
Blindspots_final.csv
Non_blind_final.csv
Stable_unpaved_roads_fixed.csv
Blindspots_districts.csv

--------------------------------------------------

PART 3 — STATISTICAL ANALYSIS (PYTHON)

Overview

Performs statistical analysis and outputs Excel results.

Required files in project_root:
- Blindspots_final.csv
- Non_blind_final.csv
- Stable_unpaved_roads_fixed.csv
- Blindspots_districts.csv

Run:
run_part3_analysis.bat

Output:
outputs_part3/
transport_exclusion_results.xlsx

Includes:
- Population exposure
- Accessibility
- Road hierarchy
- District analysis
- Statistical tests
- Charts

--------------------------------------------------

PROJECT STRUCTURE

project_root/
├── accra_sentinel2_road_surface_pipeline.py
├── transport_exclusion_analysis.py
├── README.md
├── install_requirements.bat
├── run_pipeline.bat
├── run_part3_analysis.bat
├── snapshots/
├── imagery/
├── QGIS-PART 2/
├── outputs/
├── outputs_part3/

--------------------------------------------------

REQUIREMENTS

- Python 3.10+
- QGIS

--------------------------------------------------

AUTHOR

Desmond Kemeh
Ariel University

--------------------------------------------------

SUMMARY

1. Run Part 1
2. Use QGIS (Part 2)
3. Run Part 3

Complete pipeline: Satellite → Spatial → Statistical

