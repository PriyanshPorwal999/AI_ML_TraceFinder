import pandas as pd
import numpy as np
import os
import cv2
from scipy.stats import skew, kurtosis, entropy

def extract_features(image_path, class_label):
    """
    Extracts statistical and metadata features from a single image.
    (This is pure backend logic - no streamlit code)
    """
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

def run_extraction_process(dataset_root):
    """
    Scans the dataset folder, extracts features, and yields progress updates.
    This is a 'generator' function that the UI can loop over.
    """
    if not os.path.isdir(dataset_root):
        yield "Error: Dataset path not found", -1, None
        return

    # --- 1. Find all files first ---
    files_to_process = []
    class_labels = []
    for root, dirs, files in os.walk(dataset_root):
        rel_path = os.path.relpath(root, dataset_root)
        if rel_path == ".": continue
        
        class_label = rel_path.replace(os.sep, "/")
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                files_to_process.append(os.path.join(root, fname))
                class_labels.append(class_label)
    
    if not files_to_process:
        yield "Warning: No image files found", -1, None
        return
    
    total_files = len(files_to_process)
    yield f"Found {total_files} images in {len(set(class_labels))} classes.", 0.0, None

    # --- 2. Process files and yield progress ---
    records = []
    image_paths_for_preview = {} 
    
    for i, (fpath, label) in enumerate(zip(files_to_process, class_labels)):
        rec = extract_features(fpath, label)
        records.append(rec)
        
        if label not in image_paths_for_preview:
            image_paths_for_preview[label] = fpath
        
        percent_complete = (i + 1) / total_files
        status_text = f"Processing {i+1}/{total_files}: {os.path.basename(fpath)}"
        yield status_text, percent_complete, None # Yield progress
    
    # --- 3. All done, yield the final results ---
    df = pd.DataFrame(records)
    yield "Processing complete!", 1.0, (df, image_paths_for_preview)