"""
train_hybrid_cnn.py
Trains the hybrid CNN model using precomputed residuals and the specific
27-dimension handcrafted feature set (11 PRNU + 6 FFT + 10 LBP).
Reads artifacts from ../../../artifacts/
Saves the trained model to ./models/ (with _27dim suffix)
Saves associated artifacts (scaler, encoder, history) to ../../../artifacts/
"""

import os
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

# ---------------------------
# 1) Path Setup
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Input artifact paths
RES_PATH = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl")
# --- CHANGE: Point to the 27-dimension feature file ---
HANDCRAFTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features_27dim.pkl") # <-- MODIFIED
# --- END CHANGE ---

# Output artifact paths (add suffix to avoid overwriting original artifacts if needed)
LE_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_label_encoder.pkl") # Can potentially reuse/overwrite
SCALER_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_feat_scaler.pkl") # Can potentially reuse/overwrite
HISTORY_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_training_history.pkl")
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "scanner_hybrid_best.keras")
FINAL_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "scanner_hybrid_final.keras")

# Configuration
EXPECTED_FEATURE_DIM = 27 # Explicitly set
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50

# ---------------------------
# 2) Reproducibility & GPU Setup
# ---------------------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'
print(f"--- Using device: {device_name} ---")
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(e)

# ---------------------------
# 3) Load Precomputed Data
# ---------------------------
print("\n--- Loading Precomputed Data for Training (27 Features) ---")
start_load_time = time.time()
X_img_full, X_feat_full_aligned, y_labels_full_aligned = None, None, None # Initialize
try:
    print(f"Loading residuals from: {RES_PATH}")
    if not os.path.exists(RES_PATH): raise FileNotFoundError(f"Residual file missing: {RES_PATH}")
    with open(RES_PATH, "rb") as f: residuals_dict = pickle.load(f)

    print(f"Loading 27-dim handcrafted features from: {HANDCRAFTED_FEATURES_PATH}")
    if not os.path.exists(HANDCRAFTED_FEATURES_PATH): raise FileNotFoundError(f"Feature file missing: {HANDCRAFTED_FEATURES_PATH}")
    with open(HANDCRAFTED_FEATURES_PATH, "rb") as f: feature_data = pickle.load(f)
    if not isinstance(feature_data, dict) or "features" not in feature_data or "labels" not in feature_data:
        raise ValueError("Invalid structure in 27-dim feature file.")
    X_feat_full = np.array(feature_data["features"], dtype=np.float32)
    y_labels_full = np.array(feature_data["labels"])
    num_loaded_features = X_feat_full.shape[1]
    print(f"✅ Loaded features ({X_feat_full.shape[0]} samples, {num_loaded_features} dims)")
    if num_loaded_features != EXPECTED_FEATURE_DIM:
        raise ValueError(f"Loaded features have {num_loaded_features} dims, expected {EXPECTED_FEATURE_DIM}.")

    # --- Reconstruct Image Data (X_img_full) Aligned ---
    print("   🔄 Reconstructing image residuals (aligned)...")
    X_img_list, processed_indices = [], []
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
    print(f"      Image residuals reconstructed ({X_img_full.shape[0]} samples).")

except FileNotFoundError as e: print(f"❌ Error loading data: {e}"); exit()
except ValueError as e: print(f"❌ Error processing data: {e}"); exit()
except Exception as e: print(f"❌ Unexpected error loading data: {e}"); exit()

# ---------------------------
# 4) Preprocess Data for Training
# ---------------------------
print("\n--- Preprocessing Data for Training ---")
try:
    # ---- Encode labels ----
    le = LabelEncoder(); y_int_full_aligned = le.fit_transform(y_labels_full_aligned)
    num_classes = len(le.classes_)
    y_cat_full_aligned = to_categorical(y_int_full_aligned, num_classes)
    print(f"   Found {num_classes} classes: {', '.join(le.classes_)}")

    # ---- Train/Test Split ----
    X_img_tr, X_img_val, X_feat_tr, X_feat_val, y_tr, y_val = train_test_split(
        X_img_full, X_feat_full_aligned, y_cat_full_aligned,
        test_size=0.2, random_state=SEED, stratify=y_int_full_aligned
    )
    print(f"   Train split: {X_img_tr.shape[0]} samples / Val split: {X_img_val.shape[0]} samples")

    # ---- Scale Handcrafted Features ----
    scaler = StandardScaler(); X_feat_tr = scaler.fit_transform(X_feat_tr)
    X_feat_val = scaler.transform(X_feat_val)

    # ---- Save Scaler and Encoder ----
    def save_pickle_robust(obj, path):
        try:
            with open(path, "wb") as f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Saved artifact to {path}")
        except Exception as e: print(f"❌ Error saving artifact {path}: {e}")

    save_pickle_robust(le, LE_SAVE_PATH) # Overwrite existing
    save_pickle_robust(scaler, SCALER_SAVE_PATH) # Overwrite existing with 27-dim scaler

    end_load_time = time.time()
    print(f"✅ Data prepared successfully (Time: {end_load_time - start_load_time:.2f} seconds)")

except Exception as e:
    print(f"❌ Error during data preparation: {e}")
    exit()

# ---------------------------
# 5) Build and Train Model
# ---------------------------
print("\n--- Building and Training Hybrid CNN Model (27 Features) ---")
start_train_time = time.time()
if X_img_tr.shape[0] == 0: print("❌ Error: Training data empty."); exit()

with tf.device(device_name):
    try:
        # --- Model Architecture ---
        img_input_shape = X_img_tr.shape[1:]
        feat_input_shape = (EXPECTED_FEATURE_DIM,) # Use constant

        img_in  = keras.Input(shape=img_input_shape, name="residual_image_input")
        feat_in = keras.Input(shape=feat_input_shape, name="handcrafted_features_input") # 27 features

        hp_kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=np.float32).reshape((3,3,1,1))
        try: hp = layers.Conv2D(1,(3,3),padding="same",use_bias=False,trainable=False,name="hp_filter")(img_in)
        except Exception: hp = img_in # Fallback if layer creation fails

        # CNN Branch (Simplified slightly for potentially faster training)
        x = layers.Conv2D(32, (3,3), padding="same")(hp); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3,3), padding="same", activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x); x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, (3,3), padding="same")(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3,3), padding="same", activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x); x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3,3), padding="same")(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3,3), padding="same", activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x); x = layers.Dropout(0.30)(x)

        x = layers.Conv2D(256, (3,3), padding="same")(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x) # (None, 256)

        # Feature Branch
        f = layers.Dense(64)(feat_in); f = layers.BatchNormalization()(f); f = layers.Activation('relu')(f)
        f = layers.Dropout(0.20)(f) # (None, 64)

        # Fusion & Classification
        z = layers.Concatenate()([x, f]) # (None, 256 + 64 = 320)
        z = layers.Dense(256, activation="relu")(z)
        z = layers.Dropout(0.40)(z)
        output = layers.Dense(num_classes, activation="softmax")(z)

        model = keras.Model(inputs=[img_in, feat_in], outputs=output, name="scanner_hybrid_27dim")
        try:
             hp_layer = model.get_layer("hp_filter", None)
             if hp_layer: hp_layer.set_weights([hp_kernel])
        except Exception: pass

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        print("\n--- Model Summary ---"); model.summary(line_length=100)

        # --- TF Data Pipelines ---
        print("\n--- Preparing Data Loaders ---")
        train_ds = tf.data.Dataset.from_tensor_slices(((X_img_tr, X_feat_tr), y_tr)) \
            .shuffle(buffer_size=min(len(y_tr), 10000), seed=SEED, reshuffle_each_iteration=True) \
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(((X_img_val, X_feat_val), y_val)) \
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        print("✅ Data loaders prepared.")

        # --- Callbacks ---
        print("\n--- Setting up Callbacks ---")
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            keras.callbacks.ModelCheckpoint(BEST_MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=0),
        ]
        print("✅ Callbacks defined.")

        # --- Train ---
        print("\n--- Starting Training ---")
        history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
        end_train_time = time.time()
        print(f"\n--- Training Complete (Time: {(end_train_time - start_train_time)/60:.2f} minutes) ---")

        # --- Save Final Model & History ---
        print("\n--- Saving Final Model and History ---")
        try: model.save(FINAL_MODEL_SAVE_PATH); print(f"✅ Final model saved to {FINAL_MODEL_SAVE_PATH}")
        except Exception as e: print(f"❌ Error saving final model: {e}")
        save_pickle_robust(history.history, HISTORY_SAVE_PATH)

    except Exception as e:
        print(f"❌ An error occurred during model building or training: {e}")
        import traceback; traceback.print_exc()

print("\n--- Training Script Finished ---")


