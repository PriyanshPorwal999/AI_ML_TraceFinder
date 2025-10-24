"""
config.py
Stores all global paths and configuration constants for the project.
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Input paths
FLATFIELD_RESIDUALS_PATH = os.path.join(ARTIFACTS_DIR, "flatfield_residuals.pkl")
OW_RESIDUALS_PATH = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl")

# Output paths
FP_OUT_PATH = os.path.join(ARTIFACTS_DIR, "scanner_fingerprints.pkl") # Fingerprints are the same
ORDER_NPY = os.path.join(ARTIFACTS_DIR, "fp_keys.npy") # Key order is the same (should be 11)
# NEW Output path for the 27-dimension features
FEATURES_27DIM_OUT = os.path.join(ARTIFACTS_DIR, "features_27dim.pkl")

EXPECTED_FEATURE_DIM = 27
EXPECTED_FP_KEYS = 11

# Feature extraction parameters
FFT_BINS = 6      # K=6
LBP_POINTS = 8  # P=8
LBP_RADIUS = 1.0  # R=1.0
LBP_BINS = LBP_POINTS + 2 # 10 bins for 'uniform' method