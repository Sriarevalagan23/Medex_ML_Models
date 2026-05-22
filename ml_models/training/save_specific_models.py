#!/usr/bin/env python3
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC_MODELS = ROOT / 'models'
REPO_ROOT = Path(__file__).resolve().parents[2]
DEST = REPO_ROOT / 'medex_ml_api' / 'models'
DEST.mkdir(parents=True, exist_ok=True)

# mapping from source filename (without path) to destination filename
MAPPING = {
    'heart_xgboost.joblib': 'heart_model.joblib',
    'diabetes_logistic_regression.joblib': 'diabetes_model.joblib',
    'bp_random_forest.joblib': 'bp_model.joblib',
}

for src_name, dest_name in MAPPING.items():
    src_path = SRC_MODELS / src_name
    if not src_path.exists():
        print(f"Source model not found: {src_path}")
        continue
    dest_path = DEST / dest_name
    try:
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_path} -> {dest_path}")
    except Exception as e:
        print(f"Failed to copy {src_path} -> {dest_path}: {e}")

print('Done')
