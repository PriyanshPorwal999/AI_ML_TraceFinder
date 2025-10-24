"""
eval_hybrid_cnn.py
Evaluates the trained hybrid CNN model (expecting 27 features)
on the test split created during training.
Loads data and artifacts from ../../../artifacts/
Loads the trained model from ./models/ (expecting _27dim suffix)
Prints classification report and saves confusion matrix plot to ../../../results/
"""

import os
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split # Needed to recreate the split
from sklearn.preprocessing import LabelEncoder # Needed for label encoding
from tensorflow.keras.utils import to_categorical # Needed for label encoding
import time

# ---------------------------
# 1) Path Setup
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_LOAD_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Input artifact paths
RES_PATH = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl")
# --- CHANGE: Point to the 27-dimension feature file ---
HANDCRAFTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features_27dim.pkl") # <-- MODIFIED
# --- END CHANGE ---
LE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl") # Should now be the 27-dim scaler
# --- CHANGE: Load the model trained on 27 features ---
MODEL_PATH = os.path.join(MODEL_LOAD_DIR, "scanner_hybrid_final_27dim.keras") # <-- MODIFIED (or _best_27dim.keras)
# --- END CHANGE ---

# Output path
PLOT_SAVE_PATH = os.path.join(RESULTS_DIR, "cnn_confusion_matrix_27dim.png") # Changed plot name

EXPECTED_FEATURE_DIM = 27 # Explicitly set

# ---------------------------
# 2) Reproducibility Seed (MUST match training)
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# 3) Load Preprocessors and Model
# ---------------------------
print("\n--- Loading Preprocessors and Trained Model for Evaluation (27 Features) ---")
model, le, scaler = None, None, None # Initialize
try:
    required_files = [LE_PATH, SCALER_PATH, MODEL_PATH]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing: raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    with open(LE_PATH, "rb") as f: le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    scaler_dim = getattr(scaler, 'n_features_in_', -1)
    if scaler_dim != EXPECTED_FEATURE_DIM:
         print(f"⚠️ Warning: Loaded scaler expects {scaler_dim} features, but {EXPECTED_FEATURE_DIM} are required.")
         # Decide if this is fatal or just a warning
         # raise ValueError("Scaler dimension mismatch")

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(MODEL_PATH)
    num_classes = len(le.classes_)
    print(f"✅ Loaded model from {MODEL_PATH}")
    print(f"✅ Loaded Label Encoder (Classes: {num_classes})")
    print(f"✅ Loaded Feature Scaler (Expecting {scaler_dim if scaler_dim > 0 else 'Unknown'} features)")

    # Optional: Verify model input shape matches expected features
    model_feat_dim = -1
    try:
         feat_input_layer = model.get_layer(name="handcrafted_features_input")
         model_feat_dim = feat_input_layer.input_shape[-1]
         if model_feat_dim != EXPECTED_FEATURE_DIM:
             print(f"⚠️ Warning: Model input layer expects {model_feat_dim} features, data/scaler uses {EXPECTED_FEATURE_DIM}.")
             # raise ValueError("Model input shape mismatch")
    except Exception: pass # Ignore if layer name not found

except FileNotFoundError as e: print(f"❌ Error: {e}. Ensure training artifacts and model exist."); exit()
except Exception as e: print(f"❌ Error loading files: {e}"); exit()

# ---------------------------
# 4) Rebuild the Test Dataset EXACTLY as in Training
# ---------------------------
print("\n--- Rebuilding Test Dataset for Evaluation (27 Features) ---")
start_rebuild_time = time.time()
X_img_te, X_feat_te, y_te_cat = None, None, None
try:
    if not os.path.exists(RES_PATH): raise FileNotFoundError(f"Residual file missing: {RES_PATH}")
    if not os.path.exists(HANDCRAFTED_FEATURES_PATH): raise FileNotFoundError(f"Feature file missing: {HANDCRAFTED_FEATURES_PATH}")
    with open(RES_PATH, "rb") as f: residuals_dict = pickle.load(f)
    with open(HANDCRAFTED_FEATURES_PATH, "rb") as f: feature_data = pickle.load(f)
    if not isinstance(feature_data, dict) or "features" not in feature_data or "labels" not in feature_data:
        raise ValueError("Invalid structure in feature file.")
    X_feat_full = np.array(feature_data["features"], dtype=np.float32)
    y_labels_full = np.array(feature_data["labels"])
    if X_feat_full.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(f"Loaded features have {X_feat_full.shape[1]} dims, expected {EXPECTED_FEATURE_DIM}.")

    # Reconstruct X_img_full aligned
    X_img_list, processed_indices = [], []
    current_feature_index = 0; reconstruction_successful = True
    print("🔄 Reconstructing image residuals (aligned)...")
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
                                    X_img_list.append(np.expand_dims(res, axis=-1))
                                    processed_indices.append(current_feature_index)
                               current_feature_index += 1
                          else: reconstruction_successful = False; break
                     else: reconstruction_successful = False; break
                if not reconstruction_successful: break
            if not reconstruction_successful: break
        if not reconstruction_successful: break
    if not reconstruction_successful or current_feature_index != len(y_labels_full):
        raise ValueError("Mismatch during residual reconstruction.")
    X_img_full = np.array(X_img_list, dtype=np.float32)
    X_feat_full_aligned = X_feat_full[processed_indices]
    y_labels_full_aligned = y_labels_full[processed_indices]
    if X_img_full.shape[0] != X_feat_full_aligned.shape[0] or X_img_full.shape[0] == 0:
        raise ValueError("Aligned data shape mismatch or empty.")
    print(f"   Image residuals reconstructed ({X_img_full.shape[0]} samples).")

    # --- Re-encode Labels and Recreate Split ---
    print("   🔄 Recreating train/test split...")
    y_int_full_aligned = le.transform(y_labels_full_aligned)
    y_cat_full_aligned = to_categorical(y_int_full_aligned, num_classes=num_classes)
    _, X_img_te, _, X_feat_te, _, y_te_cat = train_test_split(
        X_img_full, X_feat_full_aligned, y_cat_full_aligned,
        test_size=0.2, random_state=SEED, stratify=y_int_full_aligned
    )

    # --- Scale Test Features ---
    print("   🔄 Scaling test features...")
    X_feat_te = scaler.transform(X_feat_te) # Use the loaded scaler

    end_rebuild_time = time.time()
    print(f"✅ Test dataset rebuilt successfully (Time: {end_rebuild_time - start_rebuild_time:.2f} seconds)")
    print(f"   Test Images shape: {X_img_te.shape}")
    print(f"   Test Features shape: {X_feat_te.shape}")
    print(f"   Test Labels shape: {y_te_cat.shape}")

except FileNotFoundError as e: print(f"❌ Error loading data for rebuild: {e}"); exit()
except ValueError as e: print(f"❌ Error rebuilding test dataset: {e}"); exit()
except Exception as e: print(f"❌ Unexpected error rebuilding test dataset: {e}"); exit()

# ---------------------------
# 5) Evaluate Model on Test Set
# ---------------------------
print("\n--- Evaluating Model (27 Features) ---")
start_eval_time = time.time()
if X_img_te is not None and X_feat_te is not None and y_te_cat is not None and X_img_te.shape[0] > 0:
    try:
        print("🔄 Predicting on test set...")
        # Verify feature shape before prediction
        if X_feat_te.shape[1] != EXPECTED_FEATURE_DIM:
            raise ValueError(f"Test features have {X_feat_te.shape[1]} dims, model expects {EXPECTED_FEATURE_DIM}")

        y_pred_prob = model.predict([X_img_te, X_feat_te], verbose=1, batch_size=32)

        y_pred_idx = np.argmax(y_pred_prob, axis=1)
        y_true_idx = np.argmax(y_te_cat, axis=1)

        test_acc = accuracy_score(y_true_idx, y_pred_idx)
        print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

        print("\n✅ Classification Report:")
        target_names = le.classes_.tolist()
        try:
            report = classification_report(y_true_idx, y_pred_idx, target_names=target_names, zero_division=0)
            print(report)
        except ValueError as report_error: print(f"Could not generate report: {report_error}")

        print("\n🔄 Generating Confusion Matrix...")
        try:
            cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(num_classes))
            plt.figure(figsize=(10, 8)) # Adjusted size
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 8})
            plt.xlabel("Predicted Label"); plt.ylabel("True Label")
            plt.title(f"Confusion Matrix (27 Features - Accuracy: {test_acc*100:.2f}%)")
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9); plt.tight_layout()
            plt.savefig(PLOT_SAVE_PATH); print(f"✅ Confusion Matrix saved to {PLOT_SAVE_PATH}")
            # plt.show()
            plt.close() # Close plot figure
        except Exception as plot_error: print(f"❌ Error generating plot: {plot_error}")

    except ValueError as pred_err: # Catch specific shape mismatch during predict
        print(f"❌ EVALUATION FAILED: {pred_err}")
        print("   Ensure model was trained on 27 features and scaler matches.")
    except Exception as e:
        print(f"❌ An error occurred during evaluation: {e}")
        import traceback; traceback.print_exc()
else:
    print("⚠️ Test dataset could not be rebuilt or is empty. Cannot evaluate.")

end_eval_time = time.time()
print(f"\n--- Evaluation Complete (Time: {end_eval_time - start_eval_time:.2f} seconds) ---")






