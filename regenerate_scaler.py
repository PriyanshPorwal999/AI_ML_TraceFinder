import os
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical # Keep for stratify
import time

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Project Root
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Input paths
RES_PATH = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl") # Still needed for alignment
# --- CHANGE: Point to the 27-dimension feature file ---
HANDCRAFTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features_27dim.pkl") # <-- MODIFIED
# --- END CHANGE ---

# Output paths (Will overwrite existing)
LE_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
SCALER_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl")

EXPECTED_FEATURE_DIM = 27 # Explicitly set expected dimension

# --- Reproducibility Seed (MUST match training) ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# tf.random.set_seed(SEED) # Not needed

print("\n--- Regenerating Scaler and Label Encoder (for 27 Features) ---")
print("--- Loading Precomputed Data ---")
try:
    # Load residuals (for alignment)
    print(f"Loading residuals from: {RES_PATH}")
    if not os.path.exists(RES_PATH): raise FileNotFoundError(f"Residual file not found: {RES_PATH}")
    with open(RES_PATH, "rb") as f: residuals_dict = pickle.load(f)

    # Load the 27-dimension features
    print(f"Loading handcrafted features from: {HANDCRAFTED_FEATURES_PATH}")
    if not os.path.exists(HANDCRAFTED_FEATURES_PATH): raise FileNotFoundError(f"Feature file not found: {HANDCRAFTED_FEATURES_PATH}")
    with open(HANDCRAFTED_FEATURES_PATH, "rb") as f: feature_data = pickle.load(f)

    # Validate feature_data structure
    if not isinstance(feature_data, dict) or "features" not in feature_data or "labels" not in feature_data:
        raise ValueError(f"Invalid structure in feature file {HANDCRAFTED_FEATURES_PATH}.")

    X_feat_full = np.array(feature_data["features"], dtype=np.float32)
    y_labels_full = np.array(feature_data["labels"])
    num_handcrafted_features = X_feat_full.shape[1]
    print(f"✅ Loaded handcrafted features ({X_feat_full.shape[0]} samples, {num_handcrafted_features} dims)")
    # --- ADDED CHECK ---
    if num_handcrafted_features != EXPECTED_FEATURE_DIM:
         raise ValueError(f"Loaded features have {num_handcrafted_features} dimensions, but expected {EXPECTED_FEATURE_DIM}.")
    # --- END CHECK ---


    # --- Reconstruct Alignment Indices ---
    print("🔄 Reconstructing alignment indices (using residuals)...")
    processed_indices = []
    current_feature_index = 0
    reconstruction_successful = True
    for dataset_name in ["Official", "Wikipedia"]:
        if dataset_name not in residuals_dict: continue
        for scanner in sorted(residuals_dict[dataset_name].keys()):
            if scanner not in residuals_dict[dataset_name]: continue
            for dpi in sorted(residuals_dict[dataset_name][scanner].keys()):
                res_list = residuals_dict[dataset_name][scanner][dpi]
                for i, res in enumerate(res_list):
                     if current_feature_index < len(y_labels_full):
                          if y_labels_full[current_feature_index] == scanner:
                               if res is not None and isinstance(res, np.ndarray) and res.shape == (256, 256):
                                    processed_indices.append(current_feature_index)
                               current_feature_index += 1
                          else: reconstruction_successful = False; break
                     else: reconstruction_successful = False; break
                if not reconstruction_successful: break
            if not reconstruction_successful: break
        if not reconstruction_successful: break

    if not reconstruction_successful or current_feature_index != len(y_labels_full):
        raise ValueError("Mismatch during residual reconstruction.")

    X_feat_full_aligned = X_feat_full[processed_indices]
    y_labels_full_aligned = y_labels_full[processed_indices]

    if X_feat_full_aligned.shape[0] != len(processed_indices) or X_feat_full_aligned.shape[0] == 0:
         raise ValueError("Alignment failed or resulted in empty data.")
    print(f"✅ Alignment successful ({X_feat_full_aligned.shape[0]} samples).")


    # --- Recreate Label Encoding and Split ---
    print("\n--- Recreating Data Split and Fitting Scaler ---")
    le = LabelEncoder()
    y_int_full_aligned = le.fit_transform(y_labels_full_aligned)
    num_classes = len(le.classes_)

    _, _, X_feat_tr, _, _, _ = train_test_split(
        np.zeros((X_feat_full_aligned.shape[0], 1)), # Dummy X_img
        X_feat_full_aligned,
        y_int_full_aligned, # Use int labels for stratify
        test_size=0.2, random_state=SEED, stratify=y_int_full_aligned
    )
    print(f"   Train split features shape: {X_feat_tr.shape}")

    # --- Fit and Save the Scaler ---
    scaler = StandardScaler()
    print("   Fitting StandardScaler on training features...")
    scaler.fit(X_feat_tr) # Fit ONLY on the training part

    # --- Save Scaler and Encoder ---
    def save_pickle_robust(obj, path):
        try:
            with open(path, "wb") as f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Saved artifact to {path}")
        except Exception as e: print(f"❌ Error saving artifact {path}: {e}")

    save_pickle_robust(le, LE_SAVE_PATH)
    save_pickle_robust(scaler, SCALER_SAVE_PATH) # Overwrite the old scaler
    # Check the expected features of the saved scaler
    saved_dim = getattr(scaler, 'n_features_in_', -1)
    if saved_dim == EXPECTED_FEATURE_DIM:
        print(f"✅ New Scaler (expecting {saved_dim} features) saved.")
    else:
        print(f"⚠️ Warning: Saved scaler expects {saved_dim} features, but {EXPECTED_FEATURE_DIM} were used.")


except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
except ValueError as e:
    print(f"❌ Error processing data: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")

print("\n--- Scaler Regeneration Script Finished ---")






# import os
# import pickle
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from tensorflow.keras.utils import to_categorical # Keep for stratify
# import time

# # --- Path Setup ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Project Root
# ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
# # Ensure artifacts dir exists
# os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# # Input paths
# RES_PATH = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl") # Needed for alignment
# HANDCRAFTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "enhanced_features.pkl") # Use the 33-dim features

# # Output paths (Only encoder and scaler needed)
# LE_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
# SCALER_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl") # We will overwrite this

# # --- Reproducibility Seed (MUST match training) ---
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# # tf.random.set_seed(SEED) # Not needed just for scaler

# print("\n--- Regenerating Scaler and Label Encoder ---")
# print("--- Loading Precomputed Data ---")
# try:
#     # Load residuals (needed only for aligning features correctly)
#     print(f"Loading residuals from: {RES_PATH}")
#     if not os.path.exists(RES_PATH): raise FileNotFoundError(f"Residual file not found: {RES_PATH}")
#     with open(RES_PATH, "rb") as f: residuals_dict = pickle.load(f)

#     # Load the 33-dimension features
#     print(f"Loading handcrafted features from: {HANDCRAFTED_FEATURES_PATH}")
#     if not os.path.exists(HANDCRAFTED_FEATURES_PATH): raise FileNotFoundError(f"Feature file not found: {HANDCRAFTED_FEATURES_PATH}")
#     with open(HANDCRAFTED_FEATURES_PATH, "rb") as f: feature_data = pickle.load(f)

#     # Validate feature_data structure
#     if not isinstance(feature_data, dict) or "features" not in feature_data or "labels" not in feature_data:
#         raise ValueError(f"Invalid structure in feature file {HANDCRAFTED_FEATURES_PATH}.")

#     X_feat_full = np.array(feature_data["features"], dtype=np.float32)
#     y_labels_full = np.array(feature_data["labels"])
#     num_handcrafted_features = X_feat_full.shape[1]
#     print(f"✅ Loaded handcrafted features ({X_feat_full.shape[0]} samples, {num_handcrafted_features} dims)")
#     if num_handcrafted_features != 33:
#             print(f"⚠️ Warning: Loaded features have {num_handcrafted_features} dimensions, expected 33 based on 'enhanced_features.pkl'.")


#     # --- Reconstruct Alignment Indices (same logic as training) ---
#     # This step is crucial to ensure we use the *exact same data split* as the original training run
#     print("🔄 Reconstructing alignment indices (using residuals)...")
#     processed_indices = []
#     current_feature_index = 0
#     reconstruction_successful = True
#     # Iterate residuals in the exact same sorted order
#     for dataset_name in ["Official", "Wikipedia"]:
#         if dataset_name not in residuals_dict: continue
#         for scanner in sorted(residuals_dict[dataset_name].keys()):
#             if scanner not in residuals_dict[dataset_name]: continue
#             for dpi in sorted(residuals_dict[dataset_name][scanner].keys()):
#                 res_list = residuals_dict[dataset_name][scanner][dpi]
#                 for i, res in enumerate(res_list):
#                         if current_feature_index < len(y_labels_full):
#                             if y_labels_full[current_feature_index] == scanner:
#                                 # We only care about the index alignment, not the residual itself here
#                                 if res is not None and isinstance(res, np.ndarray) and res.shape == (256, 256):
#                                     processed_indices.append(current_feature_index)
#                                 current_feature_index += 1
#                             else: reconstruction_successful = False; break
#                         else: reconstruction_successful = False; break
#                 if not reconstruction_successful: break
#             if not reconstruction_successful: break
#         if not reconstruction_successful: break

#     if not reconstruction_successful or current_feature_index != len(y_labels_full):
#         raise ValueError("Mismatch during residual reconstruction. Cannot create split.")

#     X_feat_full_aligned = X_feat_full[processed_indices]
#     y_labels_full_aligned = y_labels_full[processed_indices]

#     if X_feat_full_aligned.shape[0] != len(processed_indices) or X_feat_full_aligned.shape[0] == 0:
#             raise ValueError("Alignment failed or resulted in empty data.")
#     print(f"✅ Alignment successful ({X_feat_full_aligned.shape[0]} samples).")


#     # --- Recreate Label Encoding and Split (to get training features for scaler) ---
#     print("\n--- Recreating Data Split and Fitting Scaler ---")
#     le = LabelEncoder()
#     y_int_full_aligned = le.fit_transform(y_labels_full_aligned)
#     num_classes = len(le.classes_)
#     # We don't need the categorical labels here, just the integer ones for stratify
#     # y_cat_full_aligned = to_categorical(y_int_full_aligned, num_classes=num_classes)

#     # Recreate the exact same train/test split using the SEED
#     # We only need X_feat_tr from this split to fit the scaler
#     _, _, X_feat_tr, _, _, _ = train_test_split(
#         np.zeros((X_feat_full_aligned.shape[0], 1)), # Dummy X_img, not needed
#         X_feat_full_aligned,
#         y_int_full_aligned, # Use integer labels for stratification
#         test_size=0.2, random_state=SEED, stratify=y_int_full_aligned
#     )
#     print(f"   Train split features shape: {X_feat_tr.shape}")

#     # --- Fit and Save the Scaler ---
#     scaler = StandardScaler()
#     print("   Fitting StandardScaler on training features...")
#     scaler.fit(X_feat_tr) # Fit ONLY on the training part of the aligned features

#     # --- Save the Correct Scaler and Encoder ---
#     def save_pickle_robust(obj, path):
#         try:
#             with open(path, "wb") as f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f"✅ Saved artifact to {path}")
#         except Exception as e: print(f"❌ Error saving artifact {path}: {e}")

#     save_pickle_robust(le, LE_SAVE_PATH) # Save encoder as well for consistency
#     save_pickle_robust(scaler, SCALER_SAVE_PATH) # Overwrite the old scaler
#     print(f"✅ New Scaler (expecting {scaler.n_features_in_} features) saved.")

# except FileNotFoundError as e:
#     print(f"❌ Error loading data: {e}")
# except ValueError as e:
#     print(f"❌ Error processing data: {e}")
# except Exception as e:
#     print(f"❌ An unexpected error occurred: {e}")

# print("\n--- Scaler Regeneration Script Finished ---")


