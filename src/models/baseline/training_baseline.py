import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models():
    print("🚀 Starting model training...")
    start_time = time.time()

    # --- Set up relative paths ---
    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Go up 3 levels to the project root
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
    
    # Define paths
    CSV_PATH = os.path.join(PROJECT_ROOT, "results", "metadata_features.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data") # For saving X_test, y_test

    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Load Data ---
    print(f"Loading dataset from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please run the 'build_features.py' script first.")
        return

    X = df.drop(columns=["file_name", "class_label", "error"], errors='ignore')
    y = df["class_label"]
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Save the test set for consistent evaluation in the main app
    print("Saving test data for later evaluation...")
    joblib.dump(X_test, os.path.join(DATA_DIR, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(DATA_DIR, "y_test.pkl"))

    # --- Scale Data ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print("Scaler saved.")

    # --- Train Random Forest ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    print("Random Forest model saved.")

    # --- Train SVM ---
    print("Training SVM...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train_scaled, y_train)
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
    print("SVM model saved.")

    end_time = time.time()
    print("\n---" * 10)
    print(f"✅ Models trained and saved successfully!")
    print(f"   Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    train_models()