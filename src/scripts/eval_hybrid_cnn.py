"""
evaluate_cnn.py
Evaluates the trained hybrid CNN model (expecting 27 features)
on the test split created during training.
"""

import os
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import time
import sys
import json

# --- Path Setup ---
# This file is in src/scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR) # To import helpers if needed in future

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_LOAD_DIR = os.path.join(PROJECT_ROOT, "src", "models", "cnn") # Load models from here
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Input artifact paths
RES_PATH = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features_27dim.pkl") # Use 27-dim features
LE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl") # 27-dim scaler
MODEL_PATH = os.path.join(MODEL_LOAD_DIR, "models", "scanner_hybrid_final.keras") # 27-dim final model

# Output path
PLOT_SAVE_PATH = os.path.join(RESULTS_DIR, "cnn_confusion_matrix_27dim.png")
REPORT_SAVE_PATH = os.path.join(RESULTS_DIR, "cnn_classification_report.json") 

EXPECTED_FEATURE_DIM = 27
SEED = 42

# ---------------------------
# 2) Reproducibility Seed
# ---------------------------
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------------------
# 3) Load Preprocessors and Model
# ---------------------------
print("\n--- Loading Preprocessors and Trained Model for Evaluation (27 Features) ---")
model, le, scaler = None, None, None
try:
    required = [LE_PATH, SCALER_PATH, MODEL_PATH]
    missing = [f for f in required if not os.path.exists(f)]
    if missing: raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")
    with open(LE_PATH, "rb") as f: le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    scaler_dim = getattr(scaler, 'n_features_in_', -1)
    if scaler_dim != EXPECTED_FEATURE_DIM:
         print(f"⚠️ Warning: Loaded scaler expects {scaler_dim} features, script requires {EXPECTED_FEATURE_DIM}.")
         # Exit if scaler definitely won't match
         exit()
         
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(MODEL_PATH)
    num_classes = len(le.classes_)
    print(f"✅ Loaded model from {MODEL_PATH}"); print(f"✅ Loaded Label Encoder (Classes: {num_classes})")
    print(f"✅ Loaded Feature Scaler (Expecting {scaler_dim} features)")
except FileNotFoundError as e: print(f"❌ Error: {e}. Run training script first."); exit()
except Exception as e: print(f"❌ Error loading files: {e}"); exit()

# ---------------------------
# 4) Rebuild the Test Dataset
# ---------------------------
print("\n--- Rebuilding Test Dataset for Evaluation (27 Features) ---")
start_rebuild_time = time.time()
X_img_te, X_feat_te, y_te_cat = None, None, None
try:
    if not os.path.exists(RES_PATH): raise FileNotFoundError(f"Missing: {RES_PATH}")
    if not os.path.exists(FEATURES_PATH): raise FileNotFoundError(f"Missing: {FEATURES_PATH}")
    with open(RES_PATH, "rb") as f: residuals_dict = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feature_data = pickle.load(f)
    if not isinstance(feature_data, dict): raise ValueError("Invalid feature file structure.")
    X_feat_full = np.array(feature_data["features"], dtype=np.float32)
    y_labels_full = np.array(feature_data["labels"])
    if X_feat_full.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(f"Features have {X_feat_full.shape[1]} dims, expected {EXPECTED_FEATURE_DIM}.")
    
    print("🔄 Reconstructing image residuals (aligned)...")
    X_img_list, processed_indices = [], []
    current_feature_index = 0; reconstruction_successful = True
    # --- This loop MUST match the feature_extractor_cnn.py loop order ---
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
    
    print("   🔄 Recreating train/test split...")
    y_int_full_aligned = le.transform(y_labels_full_aligned)
    y_cat_full_aligned = to_categorical(y_int_full_aligned, num_classes=num_classes)
    _, X_img_te, _, X_feat_te, _, y_te_cat = train_test_split(
        X_img_full, X_feat_full_aligned, y_cat_full_aligned,
        test_size=0.2, random_state=SEED, stratify=y_int_full_aligned
    )
    print("   🔄 Scaling test features...")
    X_feat_te = scaler.transform(X_feat_te)
    end_rebuild_time = time.time()
    print(f"✅ Test dataset rebuilt successfully (Time: {end_rebuild_time - start_rebuild_time:.2f}s)")
    print(f"   Test Images shape: {X_img_te.shape}, Features: {X_feat_te.shape}, Labels: {y_te_cat.shape}")

except FileNotFoundError as e: print(f"❌ Error loading data: {e}"); exit()
except ValueError as e: print(f"❌ Error rebuilding test set: {e}"); exit()
except Exception as e: print(f"❌ Unexpected error rebuilding test set: {e}"); exit()

# ---------------------------
# 5) Evaluate Model
# ---------------------------
print("\n--- Evaluating Model (27 Features) ---")
start_eval_time = time.time()
if X_img_te is not None and X_feat_te is not None and y_te_cat is not None and X_img_te.shape[0] > 0:
    try:
        print("🔄 Predicting on test set...")
        model_feat_dim = -1
        try:
             # Check the input layer name from your train_cnn.py (it's "handcrafted_features")
             model_feat_dim = model.get_layer(name="handcrafted_features").input_shape[-1]
             if model_feat_dim != EXPECTED_FEATURE_DIM:
                 raise ValueError(f"Model expects {model_feat_dim} features, data has {EXPECTED_FEATURE_DIM}.")
        except Exception as shape_e:
             print(f"Warning: Could not verify model input shape ({shape_e}). Assuming compatible.")
        
        y_pred_prob = model.predict([X_img_te, X_feat_te], verbose=1, batch_size=32)
        y_pred_idx = np.argmax(y_pred_prob, axis=1)
        y_true_idx = np.argmax(y_te_cat, axis=1)
        test_acc = accuracy_score(y_true_idx, y_pred_idx)
        print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
        print("\n✅ Classification Report:")
        target_names = le.classes_.tolist()
        # try:
        #     report = classification_report(y_true_idx, y_pred_idx, target_names=target_names, zero_division=0)
        #     print(report)
        # except ValueError as report_error: print(f"Report Error: {report_error}")
        
        try:
            # Get report as a string to print
            report_str = classification_report(y_true_idx, y_pred_idx, target_names=target_names, zero_division=0)
            print(report_str)

            # Get report as a dictionary to save
            report_dict = classification_report(y_true_idx, y_pred_idx, target_names=target_names, zero_division=0, output_dict=True)

            # Save the dictionary as a JSON file
            with open(REPORT_SAVE_PATH, 'w') as f:
                json.dump(report_dict, f, indent=4)
            print(f"✅ Classification Report saved to {REPORT_SAVE_PATH}")

        except ValueError as report_error: 
            print(f"Report Error: {report_error}")
        
        print("\n🔄 Generating Confusion Matrix...")
        try:
            cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(num_classes))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 8})
            plt.xlabel("Predicted Label"); plt.ylabel("True Label")
            plt.title(f"Confusion Matrix (27 Features - Accuracy: {test_acc*100:.2f}%)")
            plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout(); plt.savefig(PLOT_SAVE_PATH)
            print(f"✅ Confusion Matrix saved to {PLOT_SAVE_PATH}")
            plt.close()
        except Exception as plot_error: print(f"❌ Plotting Error: {plot_error}")
    
    except ValueError as pred_err: 
        print(f"❌ EVALUATION FAILED: {pred_err}")
    except Exception as e:
        print(f"❌ Evaluation Error: {e}")
        import traceback; traceback.print_exc()
else:
    print("⚠️ Test dataset empty. Cannot evaluate.")
end_eval_time = time.time()
print(f"\n--- Evaluation Complete (Time: {end_eval_time - start_eval_time:.2f} seconds) ---")
