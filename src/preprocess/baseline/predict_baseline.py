import cv2
import os
import numpy as np
import joblib  
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd

# --- Set up relative paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'models', 'baseline'))

MODEL_DIR = os.path.join(MODEL_ROOT, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RF_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "svm.pkl")

# --- Load models and scaler once ---
try:
    # --- FIXED ---
    # Load using joblib, not pickle
    SCALER = joblib.load(SCALER_PATH)
    RF_MODEL = joblib.load(RF_PATH)
    SVM_MODEL = joblib.load(SVM_PATH)
    # --- END FIX ---
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please run 'training_baseline.py' first.")
    SCALER, RF_MODEL, SVM_MODEL = None, None, None


def load_and_preprocess(img_path, size=(512, 512)):
    """Loads and preprocesses a single image from a file path."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_metadata_features(img, file_path):
    """Computes the 10 metadata features for a preprocessed image."""
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    
    hist = np.histogram(pixels, bins=256, range=(0, 1))[0]
    ent = entropy(hist + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def predict_scanner(img_path, model_choice="rf"):
    """
    Takes a file path, runs full preprocessing and feature extraction,
    and returns the prediction.
    """
    if SCALER is None or RF_MODEL is None or SVM_MODEL is None:
        raise Exception("Models are not loaded. Run training script.")

    # 1. Process the image
    img = load_and_preprocess(img_path)
    
    # 2. Extract features
    features = compute_metadata_features(img, img_path)

    # 3. Scale features
    df = pd.DataFrame([features])
    # Re-order columns to match training order
    try:
        df = df[SCALER.feature_names_in_] 
    except AttributeError:
        print("Warning: Could not get feature names from scaler. Assuming order is correct.")
        pass 
        
    X_scaled = SCALER.transform(df)

    # 4. Make prediction
    model = RF_MODEL if model_choice == "rf" else SVM_MODEL
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    return pred, prob, model.classes_

    # prob = model.predict_proba(X_scaled)[0]
    # return prob, model.classes_



