import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys

# --- Path Setup ---
# This file is in src/scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Input path
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "metadata_features.csv")
# Output paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "baseline")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts") # Save test splits here

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def train_models():
    print("[Start] Starting baseline model training...")
    start_time = time.time()
    
    print(f"Loading dataset from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Error: CSV file not found at {CSV_PATH}")
        print(r"   Please run the baseline feature extractor (src\features\baseline\extractor_frontend.py) first.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading CSV: {e}")
        return

    try:
        X = df.drop(columns=["file_name", "class_label", "error"], errors='ignore')
        y = df["class_label"]
        if X.empty or y.empty:
            print("[ERROR] Error: No data (X or y) to train on. Check CSV file."); return
            
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

        # --- Split Data ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        print("Saving test data for evaluation...")
        # --- PATHS UPDATED ---
        joblib.dump(X_test, os.path.join(ARTIFACTS_DIR, "baseline_X_test.pkl"))
        joblib.dump(y_test, os.path.join(ARTIFACTS_DIR, "baseline_y_test.pkl"))

        # --- Scale Data ---
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl")) # Save scaler with models
        print("   Scaler saved.")

        # --- Train Random Forest ---
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
        print("   Random Forest model saved.")

        # --- Train SVM ---
        print("Training SVM...")
        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
        print("   SVM model saved.")

        end_time = time.time()
        print("\n" + "---" * 10)
        print(f"[SUCCESS] Baseline models trained and saved successfully!")
        print(f"   Total time: {end_time - start_time:.2f} seconds")
    
    except KeyError as e:
         print(f"[ERROR] Error: Missing expected column in CSV: {e}. Check {CSV_PATH}.")
    except Exception as e:
         print(f"[ERROR] An error occurred during training: {e}")

if __name__ == "__main__":
    train_models()
