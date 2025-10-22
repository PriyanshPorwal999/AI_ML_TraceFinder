import pandas as pd
import numpy as np
import os
import cv2
from scipy.stats import skew, kurtosis, entropy
import time

def extract_features(image_path, class_label):
    """Extracts statistical and metadata features from a single image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "file_name": os.path.basename(image_path),
                "class_label": class_label,
                "error": "Unreadable file"
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Basic shape + file info
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024  # KB
        aspect_ratio = round(width / height, 3)

        # Stats
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        # Entropy
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)

        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        return {
            "file_name": os.path.basename(image_path),
            "class_label": class_label,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "entropy": round(shannon_entropy, 3),
            "edge_density": round(edge_density, 3)
        }
    except Exception as e:
        return {
            "file_name": os.path.basename(image_path),
            "class_label": class_label,
            "error": str(e)
        }

def process_dataset():
    """Scans the dataset folder, extracts features, and saves them to a CSV."""
    
    # --- Paths Updated ---
    # These paths are relative to the project's root folder,
    # going up three levels from src/features/baseline
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    dataset_root = os.path.join(base_dir, "data", "official")
    results_dir = os.path.join(base_dir, "results")
    
    # Create the results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "metadata_features.csv")
    
    if not os.path.isdir(dataset_root):
        print(f"Error: Dataset root path not found at: {dataset_root}")
        return

    print(f"🔎 Scanning dataset at: {dataset_root}")
    start_time = time.time()
    records = []

    for root, dirs, files in os.walk(dataset_root):
        rel_path = os.path.relpath(root, dataset_root)
        if rel_path == ".":
            continue  # skip the root itself

        class_label = rel_path.replace(os.sep, "/")
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

        if len(image_files) > 0:
            print(f"  -> Processing Class '{class_label}' ({len(image_files)} images)...")

        for fname in image_files:
            path = os.path.join(root, fname)
            rec = extract_features(path, class_label)
            records.append(rec)

    if not records:
        print("No images found. Aborting.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # --- Save features ---
    df.to_csv(save_path, index=False)
    
    end_time = time.time()
    print("\n---" * 10)
    print(f"✅ Feature extraction complete!")
    print(f"   Total images processed: {len(df)}")
    print(f"   Total classes found: {df['class_label'].nunique()}")
    print(f"   Total time: {end_time - start_time:.2f} seconds")
    print(f"💾 Features saved to: {save_path}")

# --- Main execution block ---
if __name__ == "__main__":
    process_dataset()