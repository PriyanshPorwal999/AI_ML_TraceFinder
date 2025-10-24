"""
predict_cnn.py
Contains functions to load the trained hybrid CNN model (expecting 27 features)
and predict the scanner source for a single input image.
Loads artifacts from ../../../artifacts/
and model from ../../models/cnn/models/ ON DEMAND.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import pywt
import cv2
try:
    from skimage.feature import local_binary_pattern as sk_lbp
except ImportError:
    print("ERROR: scikit-image not found. LBP features disabled. Install: pip install scikit-image")
    sk_lbp = None
import time
import pandas as pd

# ---------------------------
# 1) Path Setup & Global Variables
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_LOAD_DIR = os.path.join(BASE_DIR, "src", "models", "cnn", "models")

# --- Define paths but DO NOT load globally ---
# --- CHANGE: Point to the model trained on 27 features ---
MODEL_PATH = os.path.join(MODEL_LOAD_DIR, "scanner_hybrid_best.keras") # Verify filename!
# --- END CHANGE ---
LE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl") # The 27dim scaler
FP_PATH = os.path.join(ARTIFACTS_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(ARTIFACTS_DIR, "fp_keys.npy") # Should contain 11 keys

IMG_SIZE = (256, 256)
NUM_EXPECTED_FEATURES = 27
EXPECTED_FP_KEYS = 11

# --- Global variables to hold loaded objects (initialized to None) ---
HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None
CNN_LOAD_ATTEMPTED = False
CNN_LOAD_SUCCESS = False # Changed variable name for clarity

# ---------------------------
# 2) Loader Function (Called on demand)
# ---------------------------
def _load_cnn_artifacts():
    """Loads model and artifacts only when needed."""
    global HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS, CNN_LOAD_ATTEMPTED, CNN_LOAD_SUCCESS

    # If already attempted and failed, or already succeeded, don't retry immediately
    if CNN_LOAD_ATTEMPTED and not CNN_LOAD_SUCCESS:
         print("❌ CNN Artifacts previously failed to load. Check paths and files.")
         return False
    if CNN_LOAD_SUCCESS:
         return True # Already loaded

    CNN_LOAD_ATTEMPTED = True # Mark that we are trying to load
    print("\n--- Loading CNN Model and Artifacts for Prediction (27 Features) ---")
    start_load_time = time.time()
    try:
        required_files = [MODEL_PATH, LE_PATH, SCALER_PATH, FP_PATH, ORDER_NPY]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Error: Missing required prediction artifact files:")
            for f in missing_files: print(f"   - {f}")
            CNN_LOAD_SUCCESS = False
            return False # Indicate failure

        tf.keras.backend.clear_session() # Clear session before loading
        HYBRID_MODEL = tf.keras.models.load_model(MODEL_PATH)
        with open(LE_PATH, "rb") as f: LABEL_ENCODER = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: FEATURE_SCALER = pickle.load(f)
        with open(FP_PATH, "rb") as f: SCANNER_FPS = pickle.load(f)
        FP_KEYS = np.load(ORDER_NPY, allow_pickle=True).tolist()

        # --- VALIDATION CHECKS ---
        scaler_dim = getattr(FEATURE_SCALER, 'n_features_in_', -1)
        model_feat_dim = -1
        try: # Try getting expected shape from the model input layer named during training
            feat_input_layer = HYBRID_MODEL.get_layer(name="handcrafted_features_input")
            model_feat_dim = feat_input_layer.input_shape[-1]
        except ValueError: pass # Ignore if layer name is different

        # Critical checks - if these fail, loading is unsuccessful
        if scaler_dim != NUM_EXPECTED_FEATURES:
            print(f"❌ Error: Loaded scaler expects {scaler_dim} features, but {NUM_EXPECTED_FEATURES} are required.")
            CNN_LOAD_SUCCESS = False; return False
        if model_feat_dim != -1 and model_feat_dim != NUM_EXPECTED_FEATURES:
            print(f"❌ Error: Loaded model expects {model_feat_dim} features, but {NUM_EXPECTED_FEATURES} are required.")
            CNN_LOAD_SUCCESS = False; return False
        if len(FP_KEYS) != EXPECTED_FP_KEYS:
            print(f"❌ Error: Loaded {len(FP_KEYS)} fingerprint keys, but {EXPECTED_FP_KEYS} are required for 27-dim feats.")
            CNN_LOAD_SUCCESS = False; return False
        # --- END CHECKS ---

        end_load_time = time.time()
        print(f"✅ CNN Model and artifacts loaded successfully (Time: {end_load_time - start_load_time:.2f} seconds)")
        print(f"   Expecting {NUM_EXPECTED_FEATURES} handcrafted features.")
        CNN_LOAD_SUCCESS = True # Set flag on success
        return True

    except Exception as e:
        print(f"❌ An unexpected error occurred loading CNN artifacts: {e}")
        # Reset globals on failure
        HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None
        CNN_LOAD_SUCCESS = False
        return False # Indicate failure

# ---------------------------
# 3) Helper Functions (Robust versions)
# ---------------------------
# (to_gray, resize_to, normalize_img, denoise_wavelet_img,
#  preprocess_residual_pywt, corr2d, fft_radial_energy, lbp_hist_safe_p8
#  - No changes needed in these helpers from the previous version)
def to_gray(img):
    if img is None: return None
    if img.ndim == 3:
        if img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def resize_to(img, size=IMG_SIZE):
    if img is None or img.size == 0: return None
    try: return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    except cv2.error: return None
def normalize_img(img):
    if img is None: return None
    return img.astype(np.float32) / 255.0
def denoise_wavelet_img(img):
    if img is None or pywt is None: return img
    try:
        coeffs = pywt.dwt2(img, 'haar'); cA, (cH, cV, cD) = coeffs; cH[:] = 0; cV[:] = 0; cD[:] = 0
        denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        if denoised.shape != img.shape:
            denoised_resized = resize_to(denoised, (img.shape[1], img.shape[0]))
            denoised = denoised_resized if denoised_resized is not None else img
        return denoised
    except Exception: return img
def preprocess_residual_pywt(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None: raise ValueError(f"Cannot read image: {os.path.basename(image_path)}")
        img_gray = to_gray(img);
        if img_gray is None or img_gray.size == 0: raise ValueError("Grayscale failed.")
        img_resized = resize_to(img_gray)
        if img_resized is None: raise ValueError("Resize failed.")
        img_norm = normalize_img(img_resized)
        denoised = denoise_wavelet_img(img_norm)
        if denoised is None: raise ValueError("Denoising failed.")
        residual = img_norm - denoised
        if residual.shape != IMG_SIZE:
             residual_resized = resize_to(residual)
             if residual_resized is None: raise ValueError("Final resize failed.")
             residual = residual_resized
        return residual.astype(np.float32)
    except Exception as e: raise
def corr2d(a, b):
    if a is None or b is None or not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.shape != b.shape: return 0.0
    try:
        a_f=a.astype(np.float32).ravel(); b_f=b.astype(np.float32).ravel(); a_mean, b_mean = np.mean(a_f), np.mean(b_f); a_std, b_std = np.std(a_f), np.std(b_f)
        if a_std < 1e-9 or b_std < 1e-9: return 0.0
        a_norm=a_f-a_mean; b_norm=b_f-b_mean; denom=(np.linalg.norm(a_norm)*np.linalg.norm(b_norm))
        return float(np.dot(a_norm,b_norm)/(denom+1e-9)) if denom>1e-9 else 0.0
    except Exception: return 0.0
def fft_radial_energy(img, K=6):
    if img is None: return [0.0] * K
    try:
        from scipy.fft import fft2, fftshift
        img_float = img.astype(np.float32); f=fftshift(fft2(img_float)); mag=np.abs(f)
        if mag.size == 0: return [0.0] * K
        h,w=mag.shape; cy,cx=h//2,w//2; yy,xx=np.ogrid[:h,:w]; r=np.sqrt((yy-cy)**2+(xx-cx)**2)
        rmax=r.max()+1e-6; bins=np.linspace(0,rmax,K+1)
        feats=[float(np.mean(mag[mask]) if (mask := (r>=bins[i])&(r<bins[i+1])).any() else 0.0) for i in range(K)]
        return feats
    except Exception: return [0.0] * K
def lbp_hist_safe_p8(img, P=8, R=1.0):
    expected_bins = P + 2
    if sk_lbp is None or img is None: return [0.0] * expected_bins
    try:
        min_v,max_v=np.min(img),np.max(img); rng=max_v-min_v
        if rng<1e-9: g8=np.zeros_like(img,dtype=np.uint8)
        else: g8=(np.clip((img-min_v)/rng,0,1)*255.0).astype(np.uint8)
        codes=sk_lbp(g8, P=P, R=R, method="uniform")
        hist,_=np.histogram(codes.ravel(),bins=expected_bins,range=(0,expected_bins),density=True)
        if len(hist)!=expected_bins: hist=np.pad(hist,(0,expected_bins-len(hist)))[:expected_bins]
        return hist.astype(np.float32).tolist()
    except Exception: return [0.0] * expected_bins

def make_feats_from_res(residual):
    # This function now assumes global vars SCANNER_FPS, FP_KEYS, FEATURE_SCALER are loaded
    if residual is None: raise ValueError("Invalid residual input.")
    if not all([SCANNER_FPS, FP_KEYS, FEATURE_SCALER]): raise Exception("Required artifacts not loaded.")
    if len(FP_KEYS) != EXPECTED_FP_KEYS: raise ValueError(f"Expected {EXPECTED_FP_KEYS} keys, found {len(FP_KEYS)}")
    try:
        v_corr = [corr2d(residual, SCANNER_FPS.get(k)) for k in FP_KEYS]
        v_fft  = fft_radial_energy(residual, K=6)
        v_lbp  = lbp_hist_safe_p8(residual, P=8, R=1.0)
        combined_features = v_corr + v_fft + v_lbp
        current_len = len(combined_features)
        if current_len != NUM_EXPECTED_FEATURES:
            raise ValueError(f"Feature length mismatch: expected {NUM_EXPECTED_FEATURES}, got {current_len}.")
        v = np.array(combined_features, dtype=np.float32).reshape(1, -1)
        # Check scaler compatibility before transform
        scaler_dim = getattr(FEATURE_SCALER, 'n_features_in_', -1)
        if scaler_dim != NUM_EXPECTED_FEATURES:
             raise ValueError(f"Scaler expects {scaler_dim} features, data has {NUM_EXPECTED_FEATURES}.")
        scaled_features = FEATURE_SCALER.transform(v)
        return scaled_features
    except KeyError as e: raise KeyError(f"Fingerprint key not found: {e}")
    except Exception as e: raise Exception(f"Feature calculation error (27-dim): {e}")

# ---------------------------
# 4) Main Prediction Function
# ---------------------------
def predict_scanner_cnn(image_path):
    """
    Predicts scanner for a single image using the hybrid CNN model (27 features).
    Loads artifacts ON DEMAND if not already loaded.
    Returns: (predicted_label, probability_dataframe, class_names_list) or (None, None, None) on error.
    """
    global CNN_LOAD_SUCCESS
    # --- Load artifacts if necessary ---
    if not CNN_LOAD_SUCCESS:
        if not _load_cnn_artifacts():
            print("❌ Error: CNN Model/artifacts failed to load on demand.")
            return None, None, None # Return failure if loading fails

    # --- Check if file exists ---
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None, None, None

    try:
        # 1. Preprocess image -> residual
        start_pp = time.time()
        residual = preprocess_residual_pywt(image_path)
        if residual is None or residual.shape != IMG_SIZE:
             raise ValueError("Preprocessing failed.")
        x_img = np.expand_dims(residual, axis=(0, -1)).astype(np.float32)

        # 2. Calculate 27 features -> scaled vector
        x_feat = make_feats_from_res(residual).astype(np.float32)
        pp_time = time.time() - start_pp

        # 3. Predict probabilities
        start_pred = time.time()
        # Ensure model is callable
        if not callable(getattr(HYBRID_MODEL, 'predict', None)):
             raise TypeError("Loaded HYBRID_MODEL is not a callable Keras model.")
        probabilities = HYBRID_MODEL.predict([x_img, x_feat], verbose=0)[0]
        pred_time = time.time() - start_pred

        # 4. Decode results
        predicted_idx = np.argmax(probabilities)
        output_classes = LABEL_ENCODER.classes_
        if predicted_idx >= len(output_classes):
             raise IndexError("Index out of bounds.")
        predicted_label = output_classes[predicted_idx]

        # 5. Create DataFrame
        prob_df = pd.DataFrame({'Class': output_classes, 'Probability': probabilities})
        prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)

        return predicted_label, prob_df, output_classes.tolist()

    except ValueError as ve: print(f"❌ CNN Prediction failed (Data Error): {ve}"); return None, None, None
    except IndexError as ie: print(f"❌ CNN Prediction failed (Index Error): {ie}"); return None, None, None
    except TypeError as te: print(f"❌ CNN Prediction failed (Type Error - Model issue?): {te}"); return None, None, None
    except Exception as e: print(f"❌ Unexpected CNN prediction error: {e}"); return None, None, None

# ---------------------------
# 5) Example Usage Block
# ---------------------------
if __name__ == "__main__":
    print("\n--- Running CNN Prediction Example (27 Features - Direct Execution) ---")
    test_image_dir = os.path.join(BASE_DIR, "data", "Test")
    test_image_paths = [
        os.path.join(test_image_dir, "s1_33.tif"),
        os.path.join(test_image_dir, "s11_104.tif"),
        os.path.join(test_image_dir, "non_existent_image.jpg"),
    ]
    if not os.path.isdir(test_image_dir): print(f"⚠️ Warning: Test directory not found: {test_image_dir}")

    # --- Manually trigger loading if running script directly ---
    if not _load_cnn_artifacts():
        print("❌ Exiting example: Artifacts failed to load.")
    else:
        for img_path in test_image_paths:
            print(f"\nProcessing: {os.path.relpath(img_path, BASE_DIR)}")
            start_single_pred = time.time()
            pred_label, prob_dataframe, class_names = predict_scanner_cnn(img_path)
            end_single_pred = time.time()
            if pred_label is not None and prob_dataframe is not None:
                print(f"   Predicted Scanner: {pred_label}")
                try:
                    print(f"   Top Confidence: {prob_dataframe.iloc[0]['Probability'] * 100:.2f}%")
                    print("   Top 3 Probabilities:")
                    prob_df_display = prob_dataframe.copy()
                    prob_df_display['Probability'] = (prob_df_display['Probability'] * 100).map('{:.2f}%'.format)
                    print(prob_df_display.head(3).to_string(index=False))
                    print(f"   (Time: {end_single_pred - start_single_pred:.3f}s)")
                except Exception as disp_e: print(f"   Could not display details: {disp_e}")
            else: print("   Prediction failed.")






# """
# predict_cnn.py
# Contains functions to load the trained hybrid CNN model (expecting 27 features)
# and predict the scanner source for a single input image.
# Loads artifacts from ../../../artifacts/
# and model from ../../models/cnn/models/
# """

# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# import pywt
# import cv2
# try:
#     from skimage.feature import local_binary_pattern as sk_lbp
# except ImportError:
#     print("ERROR: scikit-image not found. LBP features disabled. Install: pip install scikit-image")
#     sk_lbp = None
# import time
# import pandas as pd

# # ---------------------------
# # 1) Path Setup & Global Variables
# # ---------------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

# ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
# MODEL_LOAD_DIR = os.path.join(BASE_DIR, "src", "models", "cnn", "models")

# # Point to the model trained on 27 features
# MODEL_PATH = os.path.join(MODEL_LOAD_DIR, "scanner_hybrid_best.keras") # Or _final.keras
# LE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
# SCALER_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl") # The 27dim scaler
# FP_PATH = os.path.join(ARTIFACTS_DIR, "scanner_fingerprints.pkl")
# ORDER_NPY = os.path.join(ARTIFACTS_DIR, "fp_keys.npy") # Should contain 11 keys

# IMG_SIZE = (256, 256)
# NUM_EXPECTED_FEATURES = 27
# EXPECTED_FP_KEYS = 11

# # --- Load models and preprocessors ---
# HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None
# LOAD_SUCCESS = False
# try:
#     required_files = [MODEL_PATH, LE_PATH, SCALER_PATH, FP_PATH, ORDER_NPY]
#     missing_files = [f for f in required_files if not os.path.exists(f)]
#     if missing_files:
#         print(f"❌ Error: Missing required prediction artifact files:")
#         for f in missing_files: print(f"   - {f}")
#     else:
#         print("--- Loading CNN Model and Artifacts for Prediction (27 Features) ---")
#         start_load_time = time.time()
#         tf.keras.backend.clear_session()
#         HYBRID_MODEL = tf.keras.models.load_model(MODEL_PATH)
#         with open(LE_PATH, "rb") as f: LABEL_ENCODER = pickle.load(f)
#         with open(SCALER_PATH, "rb") as f: FEATURE_SCALER = pickle.load(f)
#         with open(FP_PATH, "rb") as f: SCANNER_FPS = pickle.load(f)
#         FP_KEYS = np.load(ORDER_NPY, allow_pickle=True).tolist()

#         # --- VALIDATION CHECKS ---
#         scaler_dim = getattr(FEATURE_SCALER, 'n_features_in_', -1)
#         model_feat_dim = -1
#         try:
#              feat_input_layer = HYBRID_MODEL.get_layer(name="handcrafted_features_input")
#              model_feat_dim = feat_input_layer.input_shape[-1]
#         except ValueError: pass # Ignore if layer name is different

#         if scaler_dim != NUM_EXPECTED_FEATURES:
#              print(f"⚠️ Warning: Loaded scaler expects {scaler_dim} features, but {NUM_EXPECTED_FEATURES} are configured.")
#         if model_feat_dim != -1 and model_feat_dim != NUM_EXPECTED_FEATURES:
#              print(f"⚠️ Warning: Loaded model expects {model_feat_dim} features, but {NUM_EXPECTED_FEATURES} are configured.")
#         if len(FP_KEYS) != EXPECTED_FP_KEYS:
#              print(f"⚠️ Warning: Loaded {len(FP_KEYS)} fingerprint keys, but {EXPECTED_FP_KEYS} were expected for 27-dim features.")
#         # --- END CHECKS ---

#         end_load_time = time.time()
#         print(f"✅ CNN Model and artifacts loaded (Time: {end_load_time - start_load_time:.2f} seconds)")
#         print(f"   Expecting {NUM_EXPECTED_FEATURES} handcrafted features.")
#         LOAD_SUCCESS = True

# except FileNotFoundError: pass
# except Exception as e:
#     print(f"❌ An unexpected error occurred loading CNN artifacts: {e}")
#     HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None

# # ---------------------------
# # 2) Helper Functions (Robust versions)
# # ---------------------------
# # (to_gray, resize_to, normalize_img, denoise_wavelet_img,
# #  preprocess_residual_pywt, corr2d, fft_radial_energy, lbp_hist_safe_p8,
# #  make_feats_from_res - These functions remain the same as the previous version)
# def to_gray(img):
#     if img is None: return None
#     if img.ndim == 3:
#         if img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#         elif img.shape[2] == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img

# def resize_to(img, size=IMG_SIZE):
#     if img is None or img.size == 0: return None
#     try: return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
#     except cv2.error: return None

# def normalize_img(img):
#     if img is None: return None
#     return img.astype(np.float32) / 255.0

# def denoise_wavelet_img(img):
#     if img is None or pywt is None: return img
#     try:
#         coeffs = pywt.dwt2(img, 'haar'); cA, (cH, cV, cD) = coeffs
#         cH[:] = 0; cV[:] = 0; cD[:] = 0
#         denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
#         if denoised.shape != img.shape:
#             denoised_resized = resize_to(denoised, (img.shape[1], img.shape[0]))
#             denoised = denoised_resized if denoised_resized is not None else img
#         return denoised
#     except Exception: return img

# def preprocess_residual_pywt(image_path):
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
#         if img is None: raise ValueError(f"Cannot read image: {os.path.basename(image_path)}")
#         img_gray = to_gray(img);
#         if img_gray is None or img_gray.size == 0: raise ValueError("Grayscale failed.")
#         img_resized = resize_to(img_gray)
#         if img_resized is None: raise ValueError("Resize failed.")
#         img_norm = normalize_img(img_resized)
#         denoised = denoise_wavelet_img(img_norm)
#         if denoised is None: raise ValueError("Denoising failed.")
#         residual = img_norm - denoised
#         if residual.shape != IMG_SIZE:
#              residual_resized = resize_to(residual)
#              if residual_resized is None: raise ValueError("Final resize failed.")
#              residual = residual_resized
#         return residual.astype(np.float32)
#     except Exception as e: raise

# def corr2d(a, b):
#     if a is None or b is None or not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.shape != b.shape: return 0.0
#     try:
#         a_f=a.astype(np.float32).ravel(); b_f=b.astype(np.float32).ravel()
#         a_mean, b_mean = np.mean(a_f), np.mean(b_f); a_std, b_std = np.std(a_f), np.std(b_f)
#         if a_std < 1e-9 or b_std < 1e-9: return 0.0
#         a_norm=a_f-a_mean; b_norm=b_f-b_mean
#         denom=(np.linalg.norm(a_norm)*np.linalg.norm(b_norm))
#         return float(np.dot(a_norm,b_norm)/(denom+1e-9)) if denom>1e-9 else 0.0
#     except Exception: return 0.0

# def fft_radial_energy(img, K=6):
#     if img is None: return [0.0] * K
#     try:
#         from scipy.fft import fft2, fftshift # Ensure imported here too
#         img_float = img.astype(np.float32)
#         f=fftshift(fft2(img_float)); mag=np.abs(f)
#         if mag.size == 0: return [0.0] * K
#         h,w=mag.shape; cy,cx=h//2,w//2
#         yy,xx=np.ogrid[:h,:w]; r=np.sqrt((yy-cy)**2+(xx-cx)**2)
#         rmax=r.max()+1e-6; bins=np.linspace(0,rmax,K+1)
#         feats=[float(np.mean(mag[mask]) if (mask := (r>=bins[i])&(r<bins[i+1])).any() else 0.0) for i in range(K)]
#         return feats
#     except Exception: return [0.0] * K

# def lbp_hist_safe_p8(img, P=8, R=1.0):
#     expected_bins = P + 2
#     if sk_lbp is None or img is None: return [0.0] * expected_bins
#     try:
#         min_v,max_v=np.min(img),np.max(img); rng=max_v-min_v
#         if rng<1e-9: g8=np.zeros_like(img,dtype=np.uint8)
#         else: g8=(np.clip((img-min_v)/rng,0,1)*255.0).astype(np.uint8)
#         codes=sk_lbp(g8, P=P, R=R, method="uniform")
#         hist,_=np.histogram(codes.ravel(),bins=expected_bins,range=(0,expected_bins),density=True)
#         if len(hist)!=expected_bins: hist=np.pad(hist,(0,expected_bins-len(hist)))[:expected_bins]
#         return hist.astype(np.float32).tolist()
#     except Exception: return [0.0] * expected_bins

# def make_feats_from_res(residual):
#     if residual is None: raise ValueError("Invalid residual input.")
#     if not all([SCANNER_FPS, FP_KEYS, FEATURE_SCALER]): raise Exception("Required artifacts not loaded.")
#     if len(FP_KEYS) != EXPECTED_FP_KEYS: raise ValueError(f"Expected {EXPECTED_FP_KEYS} keys, found {len(FP_KEYS)}")
#     try:
#         v_corr = [corr2d(residual, SCANNER_FPS.get(k)) for k in FP_KEYS]
#         v_fft  = fft_radial_energy(residual, K=6)
#         v_lbp  = lbp_hist_safe_p8(residual, P=8, R=1.0)
#         combined_features = v_corr + v_fft + v_lbp
#         current_len = len(combined_features)
#         if current_len != NUM_EXPECTED_FEATURES:
#             raise ValueError(f"Feature length mismatch: expected {NUM_EXPECTED_FEATURES}, got {current_len}.")
#         v = np.array(combined_features, dtype=np.float32).reshape(1, -1)
#         scaled_features = FEATURE_SCALER.transform(v)
#         return scaled_features
#     except KeyError as e: raise KeyError(f"Fingerprint key not found: {e}")
#     except Exception as e: raise Exception(f"Feature calculation error (27-dim): {e}")

# # ---------------------------
# # 3) Main Prediction Function
# # ---------------------------
# # --- REMOVED @tf.function decorator ---
# def predict_scanner_cnn(image_path):
#     """
#     Predicts scanner for a single image using the loaded (27-feature) hybrid CNN model.
#     Returns: (predicted_label, probability_dataframe, class_names_list) or (None, None, None) on error.
#     """
#     if not LOAD_SUCCESS or not all([HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS]):
#         print("❌ Error: CNN Model/artifacts (27-dim) not loaded correctly.")
#         return None, None, None
#     if not os.path.exists(image_path):
#         print(f"❌ Error: Image file not found: {image_path}")
#         return None, None, None

#     try:
#         # 1. Preprocess image -> residual
#         start_pp = time.time()
#         residual = preprocess_residual_pywt(image_path)
#         if residual is None or residual.shape != IMG_SIZE:
#              raise ValueError("Preprocessing failed to produce valid residual.")
#         x_img = np.expand_dims(residual, axis=(0, -1)).astype(np.float32) # (1, 256, 256, 1)

#         # 2. Calculate 27 handcrafted features -> scaled vector
#         x_feat = make_feats_from_res(residual).astype(np.float32) # (1, 27)
#         pp_time = time.time() - start_pp

#         # 3. Predict probabilities (using eager execution now)
#         start_pred = time.time()
#         # --- CHANGE: Direct model prediction ---
#         probabilities = HYBRID_MODEL.predict([x_img, x_feat], verbose=0)[0]
#         # --- END CHANGE ---
#         pred_time = time.time() - start_pred

#         # 4. Decode results
#         predicted_idx = np.argmax(probabilities)
#         output_classes = LABEL_ENCODER.classes_
#         if predicted_idx >= len(output_classes):
#              raise IndexError("Predicted index out of bounds for label encoder classes.")
#         predicted_label = output_classes[predicted_idx]

#         # 5. Create DataFrame
#         if len(probabilities) != len(output_classes):
#              min_len = min(len(probabilities), len(output_classes))
#              prob_df = pd.DataFrame({'Class': output_classes[:min_len],'Probability': probabilities[:min_len]})
#              print(f"⚠️ Warning: Model output dim ({len(probabilities)}) != LabelEncoder classes ({len(output_classes)}). Truncating.")
#         else:
#              prob_df = pd.DataFrame({'Class': output_classes, 'Probability': probabilities})
#         prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)

#         return predicted_label, prob_df, output_classes.tolist()

#     except ValueError as ve: print(f"❌ CNN Prediction failed (Data Error): {ve}"); return None, None, None
#     except IndexError as ie: print(f"❌ CNN Prediction failed (Index Error): {ie}"); return None, None, None
#     except Exception as e: print(f"❌ Unexpected CNN prediction error: {e}"); return None, None, None


# # ---------------------------
# # 4) Example Usage Block
# # ---------------------------
# if __name__ == "__main__":
#     print("\n--- Running CNN Prediction Example (27 Features) ---")
#     test_image_dir = os.path.join(BASE_DIR, "data", "Test")
#     test_image_paths = [
#         os.path.join(test_image_dir, "s1_33.tif"),
#         os.path.join(test_image_dir, "s11_104.tif"),
#         os.path.join(test_image_dir, "non_existent_image.jpg"),
#     ]
#     if not os.path.isdir(test_image_dir): print(f"⚠️ Warning: Test directory not found: {test_image_dir}")

#     for img_path in test_image_paths:
#         print(f"\nProcessing: {os.path.relpath(img_path, BASE_DIR)}")
#         start_single_pred = time.time()
#         pred_label, prob_dataframe, class_names = predict_scanner_cnn(img_path)
#         end_single_pred = time.time()

#         if pred_label is not None and prob_dataframe is not None:
#             print(f"   Predicted Scanner: {pred_label}")
#             try:
#                 print(f"   Top Confidence: {prob_dataframe.iloc[0]['Probability'] * 100:.2f}%")
#                 print("   Top 3 Probabilities:")
#                 prob_df_display = prob_dataframe.copy()
#                 prob_df_display['Probability'] = (prob_df_display['Probability'] * 100).map('{:.2f}%'.format)
#                 print(prob_df_display.head(3).to_string(index=False))
#                 print(f"   (Total prediction time: {end_single_pred - start_single_pred:.3f}s)")
#             except Exception as disp_e: print(f"   Could not retrieve probability details: {disp_e}")
#         else:
#              print("   Prediction failed.")






# """
# predict_cnn.py
# Contains functions to load the trained hybrid CNN model (expecting 27 features)
# and predict the scanner source for a single input image.
# Loads artifacts from ../../../artifacts/
# and model from ../../models/cnn/models/
# """

# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# import pywt
# import cv2
# try:
#     from skimage.feature import local_binary_pattern as sk_lbp
# except ImportError:
#     print("ERROR: scikit-image not found. Please install it: pip install scikit-image")
#     sk_lbp = None
# import time
# import pandas as pd

# # ---------------------------
# # 1) Path Setup & Global Variables
# # ---------------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

# ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
# MODEL_LOAD_DIR = os.path.join(BASE_DIR, "src", "models", "cnn", "models")

# # Input artifact paths - use the model trained on 27 features
# MODEL_PATH = os.path.join(MODEL_LOAD_DIR, "scanner_hybrid_best.keras") # Or _final.keras if that's the 27-dim one
# LE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
# SCALER_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl") # Should be the 27-dim scaler
# FP_PATH = os.path.join(ARTIFACTS_DIR, "scanner_fingerprints.pkl")
# ORDER_NPY = os.path.join(ARTIFACTS_DIR, "fp_keys.npy") # Should contain 11 keys

# IMG_SIZE = (256, 256)
# # --- CHANGE: Set expected features to 27 ---
# NUM_EXPECTED_FEATURES = 27
# EXPECTED_FP_KEYS = 11
# # --- END CHANGE ---

# # --- Load models and preprocessors ---
# HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None
# LOAD_SUCCESS = False
# try:
#     required_files = [MODEL_PATH, LE_PATH, SCALER_PATH, FP_PATH, ORDER_NPY]
#     missing_files = [f for f in required_files if not os.path.exists(f)]
#     if missing_files:
#         print(f"❌ Error: Missing required prediction artifact files:")
#         for f in missing_files: print(f"   - {f}")
#     else:
#         print("--- Loading CNN Model and Artifacts for Prediction (27 Features) ---")
#         start_load_time = time.time()
#         tf.keras.backend.clear_session()
#         HYBRID_MODEL = tf.keras.models.load_model(MODEL_PATH)
#         with open(LE_PATH, "rb") as f: LABEL_ENCODER = pickle.load(f)
#         with open(SCALER_PATH, "rb") as f: FEATURE_SCALER = pickle.load(f)
#         with open(FP_PATH, "rb") as f: SCANNER_FPS = pickle.load(f)
#         FP_KEYS = np.load(ORDER_NPY, allow_pickle=True).tolist()

#         # --- VALIDATION CHECKS ---
#         scaler_dim = getattr(FEATURE_SCALER, 'n_features_in_', -1)
#         model_feat_dim = -1
#         try: model_feat_dim = HYBRID_MODEL.get_layer("handcrafted_features_input").input_shape[-1]
#         except: pass # Ignore if layer name is different or not found

#         if scaler_dim != NUM_EXPECTED_FEATURES:
#              print(f"⚠️ Warning: Loaded scaler expects {scaler_dim} features, but {NUM_EXPECTED_FEATURES} are configured.")
#         if model_feat_dim != -1 and model_feat_dim != NUM_EXPECTED_FEATURES:
#              print(f"⚠️ Warning: Loaded model expects {model_feat_dim} features, but {NUM_EXPECTED_FEATURES} are configured.")
#         if len(FP_KEYS) != EXPECTED_FP_KEYS:
#              print(f"⚠️ Warning: Loaded {len(FP_KEYS)} fingerprint keys, but {EXPECTED_FP_KEYS} were expected for 27-dim features.")
#         # --- END CHECKS ---

#         end_load_time = time.time()
#         print(f"✅ CNN Model and artifacts loaded (Time: {end_load_time - start_load_time:.2f} seconds)")
#         print(f"   Expecting {NUM_EXPECTED_FEATURES} handcrafted features.")
#         LOAD_SUCCESS = True

# except FileNotFoundError: pass # Error already printed
# except Exception as e:
#     print(f"❌ An unexpected error occurred loading CNN artifacts: {e}")
#     HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None

# # ---------------------------
# # 2) Helper Functions (Robust versions)
# # ---------------------------
# def to_gray(img):
#     if img is None: return None
#     if img.ndim == 3:
#         if img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#         elif img.shape[2] == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img

# def resize_to(img, size=IMG_SIZE):
#     if img is None or img.size == 0: return None
#     try: return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
#     except cv2.error: return None

# def normalize_img(img):
#     if img is None: return None
#     return img.astype(np.float32) / 255.0

# def denoise_wavelet_img(img):
#     if img is None: return None
#     try:
#         coeffs = pywt.dwt2(img, 'haar'); cA, (cH, cV, cD) = coeffs
#         cH[:] = 0; cV[:] = 0; cD[:] = 0
#         denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
#         if denoised.shape != img.shape:
#             denoised_resized = resize_to(denoised, (img.shape[1], img.shape[0])) # W, H
#             denoised = denoised_resized if denoised_resized is not None else img
#         return denoised
#     except Exception: return img

# def preprocess_residual_pywt(image_path):
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
#         if img is None: raise ValueError(f"Cannot read image: {os.path.basename(image_path)}")
#         img_gray = to_gray(img);
#         if img_gray is None or img_gray.size == 0: raise ValueError("Grayscale failed.")
#         img_resized = resize_to(img_gray)
#         if img_resized is None: raise ValueError("Resize failed.")
#         img_norm = normalize_img(img_resized)
#         denoised = denoise_wavelet_img(img_norm)
#         if denoised is None: raise ValueError("Denoising failed.")
#         residual = img_norm - denoised
#         if residual.shape != IMG_SIZE:
#              residual_resized = resize_to(residual)
#              if residual_resized is None: raise ValueError("Final resize failed.")
#              residual = residual_resized
#         return residual.astype(np.float32)
#     except Exception as e: raise

# def corr2d(a, b):
#     if a is None or b is None or not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.shape != b.shape: return 0.0
#     try:
#         a_f=a.astype(np.float32).ravel(); b_f=b.astype(np.float32).ravel()
#         a_mean, b_mean = np.mean(a_f), np.mean(b_f); a_std, b_std = np.std(a_f), np.std(b_f)
#         if a_std < 1e-9 or b_std < 1e-9: return 0.0
#         a_norm=a_f-a_mean; b_norm=b_f-b_mean
#         denom=(np.linalg.norm(a_norm)*np.linalg.norm(b_norm))
#         return float(np.dot(a_norm,b_norm)/(denom+1e-9)) if denom>1e-9 else 0.0
#     except Exception: return 0.0

# def fft_radial_energy(img, K=6):
#     if img is None: return [0.0] * K
#     try:
#         img_float = img.astype(np.float32)
#         f=fftshift(fft2(img_float)); mag=np.abs(f)
#         if mag.size == 0: return [0.0] * K
#         h,w=mag.shape; cy,cx=h//2,w//2
#         yy,xx=np.ogrid[:h,:w]; r=np.sqrt((yy-cy)**2+(xx-cx)**2)
#         rmax=r.max()+1e-6; bins=np.linspace(0,rmax,K+1)
#         feats=[float(np.mean(mag[mask]) if (mask := (r>=bins[i])&(r<bins[i+1])).any() else 0.0) for i in range(K)]
#         return feats
#     except Exception: return [0.0] * K

# def lbp_hist_safe_p8(img, P=8, R=1.0): # Use P=8 for 10 bins
#     expected_bins = P + 2 # 10
#     if sk_lbp is None or img is None: return [0.0] * expected_bins
#     try:
#         min_v,max_v=np.min(img),np.max(img); rng=max_v-min_v
#         if rng<1e-9: g8=np.zeros_like(img,dtype=np.uint8)
#         else: g8=(np.clip((img-min_v)/rng,0,1)*255.0).astype(np.uint8)
#         codes=sk_lbp(g8, P=P, R=R, method="uniform")
#         hist,_=np.histogram(codes.ravel(),bins=expected_bins,range=(0,expected_bins),density=True)
#         if len(hist)!=expected_bins: hist=np.pad(hist,(0,expected_bins-len(hist)))[:expected_bins]
#         return hist.astype(np.float32).tolist()
#     except Exception: return [0.0] * expected_bins

# def make_feats_from_res(residual):
#     """Calculates the 27-dim handcrafted feature vector."""
#     if residual is None: raise ValueError("Invalid residual input.")
#     if not all([SCANNER_FPS, FP_KEYS, FEATURE_SCALER]):
#         raise Exception("Required artifacts not loaded.")
#     if NUM_EXPECTED_FEATURES != 27:
#          print(f"Warning: Expected 27 features, but NUM_EXPECTED_FEATURES={NUM_EXPECTED_FEATURES}")
#          # Proceed assuming 27 is correct, but log the discrepancy.

#     try:
#         # 1. PRNU Correlations (11 features)
#         if len(FP_KEYS) != EXPECTED_FP_KEYS:
#              raise ValueError(f"Expected {EXPECTED_FP_KEYS} fingerprint keys, found {len(FP_KEYS)}")
#         v_corr = [corr2d(residual, SCANNER_FPS.get(k)) for k in FP_KEYS]

#         # 2. FFT Radial Energy (6 features)
#         v_fft  = fft_radial_energy(residual, K=6)

#         # 3. LBP Histogram (P=8 -> 10 features)
#         v_lbp  = lbp_hist_safe_p8(residual, P=8, R=1.0)

#         # Combine: 11 + 6 + 10 = 27 features
#         combined_features = v_corr + v_fft + v_lbp

#         # Verify length before scaling
#         current_len = len(combined_features)
#         if current_len != NUM_EXPECTED_FEATURES:
#             raise ValueError(f"Final 27-dim feature length mismatch: expected {NUM_EXPECTED_FEATURES}, got {current_len}.")

#         # Reshape for scaler and scale
#         v = np.array(combined_features, dtype=np.float32).reshape(1, -1)
#         scaled_features = FEATURE_SCALER.transform(v)
#         return scaled_features

#     except KeyError as e: raise KeyError(f"Fingerprint key not found: {e}")
#     except Exception as e: raise Exception(f"Feature calculation error (27-dim): {e}")


# # ---------------------------
# # 3) Main Prediction Function
# # ---------------------------
# @tf.function(input_signature=[
#     tf.TensorSpec(shape=(1, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
#     tf.TensorSpec(shape=(1, NUM_EXPECTED_FEATURES if NUM_EXPECTED_FEATURES > 0 else None), dtype=tf.float32)
# ])
# def _predict_step(model, img_input, feat_input):
#     """Internal TensorFlow graph function for prediction."""
#     return model([img_input, feat_input], training=False)

# def predict_scanner_cnn(image_path):
#     """
#     Predicts scanner for a single image using the loaded (27-feature) hybrid CNN model.
#     """
#     if not LOAD_SUCCESS or not all([HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS]):
#         print("❌ Error: CNN Model/artifacts (27-dim) not loaded correctly.")
#         return None, None, None
#     if not os.path.exists(image_path):
#         print(f"❌ Error: Image file not found: {image_path}")
#         return None, None, None

#     try:
#         # 1. Preprocess image -> residual
#         start_pp = time.time()
#         residual = preprocess_residual_pywt(image_path)
#         if residual is None or residual.shape != IMG_SIZE:
#              raise ValueError("Preprocessing failed.")
#         x_img = np.expand_dims(residual, axis=(0, -1)).astype(np.float32) # (1, 256, 256, 1)

#         # 2. Calculate 27 handcrafted features -> scaled vector
#         x_feat = make_feats_from_res(residual).astype(np.float32) # (1, 27)
#         pp_time = time.time() - start_pp

#         # 3. Predict probabilities
#         start_pred = time.time()
#         img_tensor = tf.convert_to_tensor(x_img)
#         feat_tensor = tf.convert_to_tensor(x_feat)

#         # Explicit shape check before prediction
#         if feat_tensor.shape[-1] != NUM_EXPECTED_FEATURES:
#             raise ValueError(f"Feature tensor shape {feat_tensor.shape} incompatible with expected {NUM_EXPECTED_FEATURES} features.")

#         probabilities_tensor = _predict_step(HYBRID_MODEL, img_tensor, feat_tensor)
#         probabilities = probabilities_tensor.numpy()[0]
#         pred_time = time.time() - start_pred

#         # 4. Decode results
#         predicted_idx = np.argmax(probabilities)
#         if predicted_idx >= len(LABEL_ENCODER.classes_):
#              raise IndexError("Predicted index out of bounds.")
#         predicted_label = LABEL_ENCODER.classes_[predicted_idx]

#         # 5. Create DataFrame
#         output_classes = LABEL_ENCODER.classes_
#         if len(probabilities) != len(output_classes):
#              min_len = min(len(probabilities), len(output_classes))
#              prob_df = pd.DataFrame({'Class': output_classes[:min_len],'Probability': probabilities[:min_len]})
#              print(f"⚠️ Warning: Model output dim ({len(probabilities)}) != LabelEncoder classes ({len(output_classes)}).")
#         else:
#              prob_df = pd.DataFrame({'Class': output_classes, 'Probability': probabilities})

#         prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)

#         return predicted_label, prob_df, output_classes

#     except ValueError as ve: print(f"❌ CNN Prediction failed (Data Error): {ve}"); return None, None, None
#     except Exception as e: print(f"❌ Unexpected CNN prediction error: {e}"); return None, None, None


# # ---------------------------
# # 4) Example Usage
# # ---------------------------
# if __name__ == "__main__":
#     print("\n--- Running CNN Prediction Example (27 Features) ---")
#     test_image_dir = os.path.join(BASE_DIR, "data", "Test")
#     test_image_paths = [ # Use examples expected to exist
#         os.path.join(test_image_dir, "s1_33.tif"),
#         os.path.join(test_image_dir, "s11_104.tif"),
#     ]
#     if not os.path.isdir(test_image_dir): print(f"Warning: Test directory not found: {test_image_dir}")

#     for img_path in test_image_paths:
#         print(f"\nProcessing: {os.path.relpath(img_path, BASE_DIR)}")
#         if not os.path.exists(img_path): print("   File not found."); continue

#         pred_label, prob_dataframe, class_names = predict_scanner_cnn(img_path)

#         if pred_label is not None and prob_dataframe is not None:
#             print(f"   Predicted Scanner: {pred_label}")
#             try:
#                 print(f"   Top Confidence: {prob_dataframe.iloc[0]['Probability'] * 100:.2f}%")
#                 print("   Top 3 Probabilities:")
#                 prob_df_display = prob_dataframe.copy()
#                 prob_df_display['Probability'] = (prob_df_display['Probability'] * 100).map('{:.2f}%'.format)
#                 print(prob_df_display.head(3).to_string(index=False))
#             except: print("   Could not retrieve probability details.")






# """
# predict_cnn.py
# Contains functions to load the trained hybrid CNN model and predict
# the scanner source for a single input image. Loads artifacts from ../../../artifacts/
# and model from ../../models/cnn/models/
# """

# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# import pywt
# import cv2
# # Ensure scikit-image is installed
# try:
#     from skimage.feature import local_binary_pattern as sk_lbp
# except ImportError:
#     print("ERROR: scikit-image not found. Please install it: pip install scikit-image")
#     sk_lbp = None # Set to None if not found
# import time
# import pandas as pd

# # ---------------------------
# # 1) Path Setup & Global Variables
# # ---------------------------
# # Determine paths relative to this script's location
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

# ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
# MODEL_LOAD_DIR = os.path.join(BASE_DIR, "src", "models", "cnn", "models")

# # Input artifact paths - use the 'best' model saved by checkpoint for predictions
# MODEL_PATH = os.path.join(MODEL_LOAD_DIR, "scanner_hybrid_best.keras") # <--- USE BEST MODEL
# LE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
# SCALER_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl")
# FP_PATH = os.path.join(ARTIFACTS_DIR, "scanner_fingerprints.pkl")
# ORDER_NPY = os.path.join(ARTIFACTS_DIR, "fp_keys.npy")

# IMG_SIZE = (256, 256) # Must match training image size
# NUM_EXPECTED_FEATURES = -1 # Determined at load time

# # --- Load models and preprocessors once when the module is imported ---
# HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None
# LOAD_SUCCESS = False # Flag to indicate if loading was successful
# try:
#     # Check if necessary files exist before attempting to load
#     required_files = [MODEL_PATH, LE_PATH, SCALER_PATH, FP_PATH, ORDER_NPY]
#     missing_files = [f for f in required_files if not os.path.exists(f)]
#     if missing_files:
#         # Print a clear error message about missing files
#         print(f"❌ Error: Missing required prediction artifact files:")
#         for f in missing_files: print(f"   - {f}")
#         print("   Please ensure training completed successfully or files were copied correctly.")
#         # Don't raise an exception here, let the LOAD_SUCCESS flag handle it
#     else:
#         print("--- Loading CNN Model and Artifacts for Prediction ---")
#         start_load_time = time.time()

#         # Set Keras backend settings (optional, for potential stability)
#         tf.keras.backend.clear_session()
#         # tf.config.threading.set_inter_op_parallelism_threads(1)
#         # tf.config.threading.set_intra_op_parallelism_threads(1)

#         HYBRID_MODEL = tf.keras.models.load_model(MODEL_PATH)
#         with open(LE_PATH, "rb") as f: LABEL_ENCODER = pickle.load(f)
#         with open(SCALER_PATH, "rb") as f: FEATURE_SCALER = pickle.load(f)
#         with open(FP_PATH, "rb") as f: SCANNER_FPS = pickle.load(f)
#         FP_KEYS = np.load(ORDER_NPY, allow_pickle=True).tolist()

#         # Determine expected feature dimension robustly
#         if hasattr(FEATURE_SCALER, 'n_features_in_'):
#             NUM_EXPECTED_FEATURES = FEATURE_SCALER.n_features_in_
#         elif hasattr(FEATURE_SCALER, 'mean_') and FEATURE_SCALER.mean_ is not None:
#             NUM_EXPECTED_FEATURES = len(FEATURE_SCALER.mean_)
#         else:
#             try: # Try inferring from model
#                  feat_input_layer = HYBRID_MODEL.get_layer("handcrafted_features_input") # Use layer name
#                  NUM_EXPECTED_FEATURES = feat_input_layer.input_shape[-1]
#             except Exception:
#                  print("⚠️ Warning: Could not determine expected feature dim from scaler or model.")

#         end_load_time = time.time()
#         print(f"✅ Model and artifacts loaded (Time: {end_load_time - start_load_time:.2f} seconds)")
#         if NUM_EXPECTED_FEATURES > 0:
#             print(f"   Expecting {NUM_EXPECTED_FEATURES} handcrafted features.")
#         LOAD_SUCCESS = True # Set flag to True only if all loads succeed

# except FileNotFoundError:
#      # Error already printed above
#      pass # Keep LOAD_SUCCESS as False
# except Exception as e:
#     print(f"❌ An unexpected error occurred loading artifacts: {e}")
#     # Set all to None so predict function fails gracefully later
#     HYBRID_MODEL, LABEL_ENCODER, FEATURE_SCALER, SCANNER_FPS, FP_KEYS = None, None, None, None, None


# # ---------------------------
# # 2) Helper Functions (Copied from processing/feature extraction)
# # ---------------------------
# def to_gray(img):
#     """Converts BGR or RGBA image to grayscale."""
#     if img is None: return None
#     if img.ndim == 3:
#         if img.shape[2] == 4: # Handle RGBA
#             return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#         elif img.shape[2] == 3: # Handle BGR
#             return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img # Assume already grayscale if 2D

# def resize_to(img, size=IMG_SIZE):
#     if img is None or img.size == 0: return None
#     try: return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
#     except cv2.error: return None

# def normalize_img(img):
#     if img is None: return None
#     return img.astype(np.float32) / 255.0

# def denoise_wavelet_img(img):
#     if img is None: return None
#     try:
#         coeffs = pywt.dwt2(img, 'haar'); cA, (cH, cV, cD) = coeffs
#         cH[:] = 0; cV[:] = 0; cD[:] = 0
#         denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
#         if denoised.shape != img.shape:
#             denoised_resized = resize_to(denoised, (img.shape[1], img.shape[0]))
#             denoised = denoised_resized if denoised_resized is not None else img
#         return denoised
#     except Exception: return img # Fallback

# def preprocess_residual_pywt(image_path):
#     """Reads, preprocesses, denoises image, returns residual."""
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
#         if img is None: raise ValueError(f"Cannot read image: {os.path.basename(image_path)}")
#         img_gray = to_gray(img);
#         if img_gray is None or img_gray.size == 0: raise ValueError("Grayscale conversion failed.")
#         img_resized = resize_to(img_gray)
#         if img_resized is None: raise ValueError("Resize failed.")
#         img_norm = normalize_img(img_resized)
#         denoised = denoise_wavelet_img(img_norm)
#         if denoised is None: raise ValueError("Denoising failed.")
#         residual = img_norm - denoised
#         if residual.shape != IMG_SIZE:
#              residual_resized = resize_to(residual)
#              if residual_resized is None: raise ValueError("Final resize failed.")
#              residual = residual_resized
#         return residual.astype(np.float32)
#     except Exception as e:
#         # print(f"Error preprocessing {os.path.basename(image_path)}: {e}")
#         raise # Re-raise to be caught in predict function

# def corr2d(a, b):
#     """Normalized cross-correlation."""
#     if a is None or b is None or not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.shape != b.shape: return 0.0
#     a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
#     a_mean, b_mean = a.mean(), b.mean()
#     a_std, b_std = a.std(), b.std()
#     # Handle flat arrays (std dev = 0)
#     if a_std < 1e-9 or b_std < 1e-9: return 0.0
#     a_norm = a - a_mean
#     b_norm = b - b_mean
#     denom = (np.linalg.norm(a_norm) * np.linalg.norm(b_norm))
#     return float(np.dot(a_norm, b_norm) / (denom + 1e-9)) if denom > 1e-9 else 0.0


# def fft_radial_energy(img, K=6):
#     """Calculates radial energy distribution in FFT magnitude spectrum."""
#     if img is None: return [0.0] * K
#     try:
#         f = fftshift(fft2(img)); mag = np.abs(f)
#         h, w = mag.shape; cy, cx = h//2, w//2
#         yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
#         rmax = r.max() + 1e-6; bins = np.linspace(0, rmax, K+1)
#         # Calculate mean for each radial bin, handle empty bins
#         feats = [float(np.mean(mag[(r >= bins[i]) & (r < bins[i+1])]) if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]
#         return feats
#     except Exception: return [0.0] * K # Return zeros on failure

# def lbp_hist_safe(img, P=8, R=1.0):
#     """Calculates uniform LBP histogram safely."""
#     expected_bins = P + 2
#     if sk_lbp is None:
#         print("Error: scikit-image needed for LBP.")
#         return [0.0] * expected_bins
#     if img is None: return [0.0] * expected_bins
#     try:
#         # Normalize residual robustly before converting to uint8
#         min_v, max_v = img.min(), img.max()
#         rng = max_v - min_v
#         if rng < 1e-9: g8 = np.zeros_like(img, dtype=np.uint8)
#         else: g8 = ((img - min_v) / rng * 255.0).astype(np.uint8)

#         codes = sk_lbp(g8, P=P, R=R, method="uniform")
#         # Ensure bins cover the full range of uniform LBP codes (0 to P+1)
#         hist, _ = np.histogram(codes.ravel(), bins=np.arange(expected_bins + 1), range=(0, expected_bins), density=True)
#         # Check length just in case histogram returns unexpected size
#         if len(hist) != expected_bins: hist = np.pad(hist, (0, expected_bins - len(hist)))[:expected_bins]
#         return hist.astype(np.float32).tolist()
#     except Exception as e:
#         # print(f"Error in lbp_hist_safe: {e}")
#         return [0.0] * expected_bins


# def make_feats_from_res(residual):
#     """Calculates the full handcrafted feature vector from a residual."""
#     if residual is None: raise ValueError("Invalid residual input.")
#     if not all([SCANNER_FPS, FP_KEYS, FEATURE_SCALER]):
#         raise Exception("Required artifacts (fingerprints, keys, scaler) not loaded.")

#     try:
#         # Calculate each feature component
#         v_corr = [corr2d(residual, SCANNER_FPS.get(k)) for k in FP_KEYS]
#         v_fft  = fft_radial_energy(residual, K=6)
#         v_lbp  = lbp_hist_safe(residual, P=8, R=1.0) # Match training params P=8, R=1.0
#         combined_features = v_corr + v_fft + v_lbp

#         # Verify and adjust length BEFORE scaling
#         current_len = len(combined_features)
#         if NUM_EXPECTED_FEATURES > 0 and current_len != NUM_EXPECTED_FEATURES:
#             # print(f"Warning: Feature vector length mismatch: expected {NUM_EXPECTED_FEATURES}, got {current_len}. Adjusting.")
#             if current_len < NUM_EXPECTED_FEATURES:
#                 combined_features.extend([0.0] * (NUM_EXPECTED_FEATURES - current_len)) # Pad with zeros
#             else:
#                 combined_features = combined_features[:NUM_EXPECTED_FEATURES] # Truncate

#         # Reshape and scale
#         v = np.array(combined_features, dtype=np.float32).reshape(1, -1)
#         scaled_features = FEATURE_SCALER.transform(v)
#         return scaled_features

#     except KeyError as e: raise KeyError(f"Fingerprint key not found: {e}")
#     except Exception as e: raise Exception(f"Feature calculation error: {e}")

# # ---------------------------
# # 3) Main Prediction Function
# # ---------------------------
# # Use tf.function for potential performance improvement and graph execution
# @tf.function(input_signature=[
#     tf.TensorSpec(shape=(1, 256, 256, 1), dtype=tf.float32),
#     tf.TensorSpec(shape=(1, None), dtype=tf.float32) # Allow flexible feature dim in signature
# ])
# def _predict_step(model, img_input, feat_input):
#     """Internal prediction step wrapped in tf.function."""
#     return model([img_input, feat_input], training=False)

# def predict_scanner_cnn(image_path):
#     """
#     Predicts scanner for a single image using the loaded hybrid CNN model.
#     Returns: (predicted_label, probability_dataframe, class_names)
#              or (None, None, None) on failure.
#     """
#     if not LOAD_SUCCESS:
#         print("❌ Error: Model/artifacts not loaded. Cannot predict.")
#         return None, None, None

#     try:
#         # 1. Preprocess image -> residual
#         residual = preprocess_residual_pywt(image_path)
#         x_img = np.expand_dims(residual, axis=(0, -1)).astype(np.float32) # Ensure correct shape and type

#         # 2. Calculate handcrafted features -> scaled feature vector
#         x_feat = make_feats_from_res(residual).astype(np.float32) # Ensure correct type

#         # Verify feature shape matches expected input if known
#         if NUM_EXPECTED_FEATURES > 0 and x_feat.shape[1] != NUM_EXPECTED_FEATURES:
#              raise ValueError(f"Scaled feature shape mismatch: expected (1, {NUM_EXPECTED_FEATURES}), got {x_feat.shape}")

#         # 3. Predict probabilities using loaded model via tf.function
#         # Convert numpy arrays to Tensors for tf.function
#         img_tensor = tf.convert_to_tensor(x_img)
#         feat_tensor = tf.convert_to_tensor(x_feat)
#         probabilities_tensor = _predict_step(HYBRID_MODEL, img_tensor, feat_tensor)
#         probabilities = probabilities_tensor.numpy()[0] # Convert back to numpy array

#         # 4. Decode results
#         predicted_idx = np.argmax(probabilities)
#         predicted_label = LABEL_ENCODER.classes_[predicted_idx]

#         # 5. Create DataFrame for output
#         prob_df = pd.DataFrame({
#             'Class': LABEL_ENCODER.classes_,
#             'Probability': probabilities
#         })
#         prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)

#         return predicted_label, prob_df, LABEL_ENCODER.classes_

#     except ValueError as ve: # Catch specific preprocessing/feature errors
#          print(f"❌ Prediction failed for {os.path.basename(image_path)} (Preprocessing/Feature Error): {ve}")
#          return None, None, None
#     except Exception as e:
#         print(f"❌ An unexpected error occurred during CNN prediction for {os.path.basename(image_path)}: {e}")
#         # import traceback; traceback.print_exc() # Uncomment for debugging
#         return None, None, None

# # ---------------------------
# # 4) Example Usage (if run directly as a script)
# # ---------------------------
# if __name__ == "__main__":
#     print("\n--- Running CNN Prediction Example ---")
#     # Define test images relative to the project root
#     test_image_dir = os.path.join(BASE_DIR, "data", "Test")
#     test_image_paths = [
#         os.path.join(test_image_dir, "s1_33.tif"),     # Example Canon120-1
#         os.path.join(test_image_dir, "s11_104.tif"),   # Example HP
#         os.path.join(test_image_dir, "NonExistent.tif") # Example non-existent file
#     ]

#     if not os.path.isdir(test_image_dir):
#         print(f"Warning: Test directory not found at {test_image_dir}")

#     for img_path in test_image_paths:
#         print(f"\nProcessing: {img_path}")
#         if not os.path.exists(img_path):
#             print("   File not found.")
#             continue

#         pred_label, prob_dataframe, class_names = predict_scanner_cnn(img_path)

#         if pred_label is not None:
#             print(f"   Predicted Scanner: {pred_label}")
#             print(f"   Top Confidence: {prob_dataframe.iloc[0]['Probability'] * 100:.2f}%")
#             print("   Top 3 Probabilities:")
#             prob_dataframe['Probability'] = (prob_dataframe['Probability'] * 100).map('{:.2f}%'.format)
#             print(prob_dataframe.head(3).to_string(index=False))

