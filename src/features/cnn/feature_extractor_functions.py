"""
feature_extractor_functions.py
Contains all core logic and functions for fingerprinting and feature extraction.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import time

try:
    from skimage.feature import local_binary_pattern as sk_lbp
except ImportError:
    print("ERROR: scikit-image not found. Please install it: pip install scikit-image")
    sk_lbp = None
try:
    from scipy.fft import fft2, fftshift
except ImportError:
    print("ERROR: scipy not found. Please install it: pip install scipy")
    fft2, fftshift = None, None

# Import all constants and paths from config file
import config


# Helper functions
def corr2d(a, b):
    """Calculates the 2D correlation between two numpy arrays."""
    if a is None or b is None or not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.shape != b.shape:
        return 0.0
    try:
        a_f = a.astype(np.float32).ravel(); b_f = b.astype(np.float32).ravel()
        a_mean, b_mean = np.mean(a_f), np.mean(b_f); a_std, b_std = np.std(a_f), np.std(b_f)
        if a_std < 1e-9 or b_std < 1e-9: return 0.0
        a_norm = a_f - a_mean; b_norm = b_f - b_mean
        denom = (np.linalg.norm(a_norm) * np.linalg.norm(b_norm))
        return float(np.dot(a_norm, b_norm) / (denom + 1e-9)) if denom > 1e-9 else 0.0
    except Exception:
        return 0.0

def fft_radial_energy(img, K=config.FFT_BINS):
    """Calculates FFT radial energy in K bins."""
    if img is None or fft2 is None: return [0.0] * K
    try:
        img_float = img.astype(np.float32)
        f = fftshift(fft2(img_float)); mag = np.abs(f)
        if mag.size == 0: return [0.0] * K
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6; bins = np.linspace(0, rmax, K+1)
        feats = [float(np.mean(mag[mask]) if (mask := (r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]
        return feats
    except Exception:
        return [0.0] * K

def lbp_hist_safe_p8(img, P=config.LBP_POINTS, R=config.LBP_RADIUS):
    """Calculates a safe LBP histogram with P=8 (10 bins)."""
    if sk_lbp is None or img is None: return [0.0] * config.LBP_BINS
    try:
        min_v, max_v = np.min(img), np.max(img); rng = max_v - min_v
        if rng < 1e-9: g8 = np.zeros_like(img, dtype=np.uint8)
        else: g8 = (np.clip((img - min_v) / rng, 0, 1) * 255.0).astype(np.uint8)
        
        codes = sk_lbp(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes.ravel(), bins=config.LBP_BINS, range=(0, config.LBP_BINS), density=True)
        
        if len(hist) != config.LBP_BINS:
            hist = np.pad(hist, (0, config.LBP_BINS - len(hist)))[:config.LBP_BINS]
            
        return hist.astype(np.float32).tolist()
    except Exception:
        return [0.0] * config.LBP_BINS
    

# Main feature extraction functions

def compute_or_load_fingerprints():
    """Loads existing fingerprints or computes them if necessary."""
    print("\n--- Checking/Computing Scanner Fingerprints ---")
    
    # Use paths from config
    if os.path.exists(config.FP_OUT_PATH) and os.path.exists(config.ORDER_NPY):
        try:
            with open(config.FP_OUT_PATH, "rb") as f:
                scanner_fingerprints = pickle.load(f)
            fp_keys = np.load(config.ORDER_NPY, allow_pickle=True).tolist()
            
            # Use constants from config
            if len(fp_keys) == config.EXPECTED_FP_KEYS and len(scanner_fingerprints) == config.EXPECTED_FP_KEYS:
                print(f"✅ Loaded existing {len(fp_keys)} fingerprints and keys.")
                return scanner_fingerprints, fp_keys
            else:
                print(f"⚠️ Fingerprint/key file mismatch. Recomputing...")
        except Exception as e:
            print(f"⚠️ Error loading existing fingerprints: {e}. Recomputing...")

    # --- Computation Logic ---
    start_time_fp = time.time()
    scanner_fingerprints = {}
    if not os.path.exists(config.FLATFIELD_RESIDUALS_PATH):
        print(f"❌ Error: Flatfield residuals file not found at {config.FLATFIELD_RESIDUALS_PATH}")
        return None, None
    try:
        with open(config.FLATFIELD_RESIDUALS_PATH, "rb") as f:
            flatfield_residuals = pickle.load(f)
        print("🔄 Calculating average fingerprint for each scanner...")
        if not isinstance(flatfield_residuals, dict):
            raise ValueError("Invalid flatfield residual format")

        # ... (rest of the fingerprint computation logic remains the same) ...
        valid_scanner_count = 0
        skipped_scanners = []
        for scanner, res_data in tqdm(flatfield_residuals.items(), desc="Fingerprints"):
            if not isinstance(res_data, dict) or 'all' not in res_data:
                skipped_scanners.append(f"{scanner} (invalid data structure)"); continue
            residuals = res_data.get('all', [])
            if not residuals: skipped_scanners.append(f"{scanner} (no residuals)"); continue
            valid_residuals = [res for res in residuals if res is not None and isinstance(res, np.ndarray)]
            if not valid_residuals: skipped_scanners.append(f"{scanner} (no valid numpy residuals)"); continue
            try:
                first_shape = valid_residuals[0].shape
                consistent_residuals = [res for res in valid_residuals if res.shape == first_shape]
                if not consistent_residuals: skipped_scanners.append(f"{scanner} (no consistent shape)"); continue
                stack = np.stack(consistent_residuals, axis=0)
                fingerprint = np.mean(stack, axis=0)
                scanner_fingerprints[scanner] = fingerprint.astype(np.float32)
                valid_scanner_count += 1
            except Exception as e: skipped_scanners.append(f"{scanner} (error: {e})"); continue

        if skipped_scanners: print(f"⚠️ Skipped {len(skipped_scanners)} scanners.")
        if not scanner_fingerprints: print("❌ No valid fingerprints generated."); return None, None

        # Use paths from config
        with open(config.FP_OUT_PATH, "wb") as f:
            pickle.dump(scanner_fingerprints, f, protocol=pickle.HIGHEST_PROTOCOL)
        fp_keys = sorted(scanner_fingerprints.keys())
        np.save(config.ORDER_NPY, np.array(fp_keys))
        
        end_time_fp = time.time()
        print(f"✅ Saved {len(scanner_fingerprints)} fingerprints and {len(fp_keys)} keys.")
        print(f"   Calculation time: {end_time_fp - start_time_fp:.2f} seconds")
        
        # Use constant from config
        if len(fp_keys) != config.EXPECTED_FP_KEYS:
            print(f"❌ FATAL ERROR: Generated {len(fp_keys)} fingerprint keys, but {config.EXPECTED_FP_KEYS} are required.")
            return None, None
        return scanner_fingerprints, fp_keys
    except Exception as e:
        print(f"❌ Error during fingerprint computation: {e}")
        return None, None


def extract_27dim_features(residuals_dict, scanner_fingerprints, fp_keys):
    """Calculates the specific 27-dimension feature set."""
    if not residuals_dict or not scanner_fingerprints or not fp_keys:
        print("❌ Cannot extract 27-dim features: Missing input data.")
        return False

    if len(fp_keys) != config.EXPECTED_FP_KEYS:
        print(f"❌ Error: Expected {config.EXPECTED_FP_KEYS} fingerprint keys, found {len(fp_keys)}.")
        return False

    print("\n--- Extracting 27-Dimension Features (11 PRNU + 6 FFT + 10 LBP) ---")
    start_time_feat = time.time()
    features_27dim, labels_27dim = [], []

    print("🔄 Calculating 27-dim features...")
    if not isinstance(residuals_dict, dict):
        print(f"❌ Error: Expected residuals_dict dict, got {type(residuals_dict)}")
        return False

    for dataset_name in tqdm(residuals_dict.keys(), desc="Datasets (27-Dim Feat)"):
        dataset_data = residuals_dict.get(dataset_name)
        if not isinstance(dataset_data, dict): continue

        for scanner, dpi_dict in tqdm(dataset_data.items(), desc=f"{dataset_name} Scanners", leave=False):
            if scanner not in scanner_fingerprints: continue
            if not isinstance(dpi_dict, dict): continue

            for dpi, res_list in dpi_dict.items():
                if not isinstance(res_list, list): continue

                for res_a in res_list:
                    if res_a is None or not isinstance(res_a, np.ndarray): continue
                    try:
                        # 1. PRNU Correlations (11 features)
                        v_corr = [corr2d(res_a, scanner_fingerprints.get(k)) for k in fp_keys]

                        # 2. FFT Radial Energy (6 features)
                        v_fft  = fft_radial_energy(res_a)

                        # 3. LBP Histogram (P=8 -> 10 features)
                        v_lbp  = lbp_hist_safe_p8(res_a)

                        # Combine: 11 + 6 + 10 = 27 features
                        combined_features = v_corr + v_fft + v_lbp

                        if len(combined_features) == config.EXPECTED_FEATURE_DIM:
                            features_27dim.append(combined_features)
                            labels_27dim.append(scanner)
                        
                    except Exception as e:
                        print(f"Error calc 27-dim feat {scanner}/{dpi}: {e}")

    # --- Save 27-dim Features ---
    if features_27dim:
        try:
            # Use path from config
            with open(config.FEATURES_27DIM_OUT, "wb") as f:
                pickle.dump({"features": features_27dim, "labels": labels_27dim}, f, protocol=pickle.HIGHEST_PROTOCOL)
            end_time_feat = time.time()
            print(f"✅ Saved 27-dim features (shape: {len(features_27dim)} x {config.EXPECTED_FEATURE_DIM}) to {config.FEATURES_27DIM_OUT}")
            print(f"   27-dim feature extraction time: {end_time_feat - start_time_feat:.2f} seconds")
            return True
        except Exception as e:
            print(f"❌ Error saving 27-dim features: {e}")
            return False
    else:
        print("⚠️ No 27-dim features were extracted.")
        return False