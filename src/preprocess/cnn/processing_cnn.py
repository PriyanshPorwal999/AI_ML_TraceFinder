"""
processing_cnn.py
Data loading and preprocessing for CNN.
- Handles `Flatfield`, `Official`, `Wikipedia` datasets from ../../../data/
- Produces: residual images (256x256x1) and saves them to ../../../artifacts/
"""

import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Optional: wavelet denoising
import pywt
# Ensure scikit-image is installed for denoise_wavelet: pip install scikit-image
# from skimage.restoration import denoise_wavelet
from scipy.signal import wiener as scipy_wiener

# ---------------------------
# 1) Global Parameters
# ---------------------------
IMG_SIZE = (256, 256)
DENOISE_METHOD = "wavelet"  # "wavelet" or "wiener"
# Use slightly fewer workers than max CPU count to leave resources for OS
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# ---------------------------
# 2) Path Setup (Relative to this script's location)
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Project Root

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True) # Ensure artifacts folder exists

# ---------------------------
# 3) Helper Functions
# ---------------------------
def to_gray(img):
    """Converts image to grayscale if it's color."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=IMG_SIZE):
    """Resizes image using area interpolation (good for shrinking)."""
    # Ensure input is a valid image before resizing
    if img is None or img.size == 0:
        # print("Warning: Invalid input to resize_to.")
        return None
    try:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        # print(f"Warning: cv2.resize failed: {e}")
        return None


def normalize_img(img):
    """Normalizes image pixel values to the range [0.0, 1.0]."""
    if img is None: return None
    return img.astype(np.float32) / 255.0

def denoise_wavelet_img(img):
    """Applies wavelet denoising using pywt (haar wavelet)."""
    if img is None: return None
    try:
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        # Zero out detail coefficients
        cH[:] = 0; cV[:] = 0; cD[:] = 0
        denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        # Ensure output shape matches input, especially for odd dimensions
        if denoised.shape != img.shape:
            # print(f"Warning: Wavelet output shape {denoised.shape} != input {img.shape}. Resizing output.")
            denoised_resized = resize_to(denoised, (img.shape[1], img.shape[0])) # Use (W, H) for cv2.resize
            if denoised_resized is None: # Check if resize failed
                 # print("Warning: Resize after wavelet failed, returning original.")
                 return img
            denoised = denoised_resized
        return denoised
    except Exception as e:
        # print(f"Warning: Wavelet denoising failed - {e}, returning original.")
        return img # Return original image if denoising fails


def preprocess_image(fpath, method=DENOISE_METHOD):
    """
    Reads an image, preprocesses it (gray, resize, normalize),
    denoises it, and returns the noise residual.
    Returns None if the image cannot be read or processed.
    """
    try:
        # Use cv2.IMREAD_IGNORE_ORIENTATION to potentially handle EXIF issues
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            # print(f"Warning: Could not read image {os.path.basename(fpath)}, skipping.")
            return None

        img_gray = to_gray(img)
        # Check if grayscale conversion resulted in a valid image
        if img_gray is None or img_gray.size == 0:
             # print(f"Warning: Grayscale conversion failed for {os.path.basename(fpath)}, skipping.")
             return None

        img_resized = resize_to(img_gray)
        if img_resized is None: # Check if resize failed
             # print(f"Warning: Resize failed for {os.path.basename(fpath)}, skipping.")
             return None

        img_norm = normalize_img(img_resized)

        # Denoising
        if method == "wiener":
            # Wiener might require specific parameters or fail on certain images
            try:
                den = scipy_wiener(img_norm, mysize=(5,5))
            except Exception as e:
                # print(f"Warning: Wiener denoising failed for {os.path.basename(fpath)} - {e}, using original.")
                den = img_norm # Fallback if Wiener fails
        elif method == "wavelet":
            den = denoise_wavelet_img(img_norm)
        else:
            print(f"Error: Unknown denoise method: {method}")
            return None # Return None for unknown method

        # Handle case where denoising might have returned None
        if den is None:
             # print(f"Warning: Denoising returned None for {os.path.basename(fpath)}, using original.")
             den = img_norm

        residual = img_norm - den
        # Ensure residual has the correct target size after potential resize in denoise
        if residual.shape != IMG_SIZE:
             # print(f"Warning: Residual shape {residual.shape} incorrect, resizing to {IMG_SIZE}.")
             residual_resized = resize_to(residual)
             if residual_resized is None:
                  # print(f"Warning: Final resize failed for {os.path.basename(fpath)}, skipping.")
                  return None
             residual = residual_resized

        return residual.astype(np.float32)
    except Exception as e:
        # Catch-all for other unexpected errors during processing
        # print(f"Error processing {os.path.basename(fpath)}: {e}")
        return None


def process_folder(folder_path, use_dpi_subfolders=True):
    """
    Processes all valid image files within a given folder structure.
    Handles potential errors during file processing gracefully.
    Returns a dictionary containing the residuals.
    """
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder not found {folder_path}, skipping.")
        return {}

    residuals_dict = {}
    # Ensure we only process directories at the top level
    try:
        top_level_items = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    except OSError as e:
        print(f"Error listing directory {folder_path}: {e}")
        return {}


    # Overall progress bar for top-level folders (scanners or datasets)
    for item_name in tqdm(top_level_items, desc=f"Processing {os.path.basename(folder_path)}"):
        item_dir = os.path.join(folder_path, item_name)
        current_item_residuals = {} # Temporary dict for the current item

        try:
            if use_dpi_subfolders:
                # Expect structure like: Scanner/DPI/image.tif
                sub_items = sorted([d for d in os.listdir(item_dir) if os.path.isdir(os.path.join(item_dir, d))])
                if not sub_items: # Handle case where images might be directly in the scanner folder
                     files = [os.path.join(item_dir, f) for f in os.listdir(item_dir)
                             if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")) and not f.startswith('._')]
                     if files:
                        processed_residuals = parallel_process_images(files, item_name)
                        if processed_residuals:
                            current_item_residuals['all'] = processed_residuals # Use 'all' if no DPI
                else:
                    for sub_name in sub_items: # e.g., '150', '300'
                        sub_dir = os.path.join(item_dir, sub_name)
                        files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)
                                if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")) and not f.startswith('._')]
                        if files:
                            processed_residuals = parallel_process_images(files, f"{item_name}/{sub_name}")
                            if processed_residuals:
                                 current_item_residuals[sub_name] = processed_residuals
            else:
                # Case for Flatfield (no dpi subfolders): Flatfield/ScannerName/image.tif
                files = [os.path.join(item_dir, f) for f in os.listdir(item_dir)
                         if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")) and not f.startswith('._')]
                if files:
                    processed_residuals = parallel_process_images(files, item_name)
                    if processed_residuals:
                        # Flatfield residuals are stored directly under scanner name for simplicity later
                        residuals_dict[item_name] = {'all': processed_residuals} # Store under 'all' key
                        continue # Skip adding to current_item_residuals as it's handled directly

            # Add the processed residuals for this item if it's not empty (for Official/Wiki)
            if use_dpi_subfolders and current_item_residuals:
                residuals_dict[item_name] = current_item_residuals

        except OSError as e:
            print(f"Error accessing directory {item_dir}: {e}")
            continue # Skip this item if not accessible

    return residuals_dict


def parallel_process_images(file_list, desc_prefix=""):
    """Processes a list of image files in parallel using ThreadPoolExecutor."""
    residuals = []
    # Limit workers further if MAX_WORKERS is very high, avoid overloading system
    num_workers = min(MAX_WORKERS, len(file_list), 32)
    if num_workers <= 0: return []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(preprocess_image, f): f for f in file_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Images ({desc_prefix})", leave=False, mininterval=0.5):
            try:
                res = fut.result()
                if res is not None:
                    if res.shape == IMG_SIZE:
                        residuals.append(res)
                    # else: # Optional: Log shape mismatches if debugging needed
                    #    print(f"Shape mismatch: {futures[fut]} -> {res.shape}")
            except Exception as e:
                 # Error likely printed in preprocess_image, good to log the filename here too
                 # print(f"Error retrieving result for {os.path.basename(futures[fut])}: {e}")
                 pass
    return residuals


def save_pickle(obj, path):
    """Saves an object to a pickle file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol
        print(f"✅ Saved data to {path}")
    except Exception as e:
        print(f"❌ Error saving pickle file {path}: {e}")

# ---------------------------
# 4) Main Execution Block
# ---------------------------
if __name__ == "__main__":
    start_total_time = time.time()
    print(f"Starting preprocessing...")
    print(f"Using up to {MAX_WORKERS} workers for parallel processing.")
    print(f"Target Image Size: {IMG_SIZE}, Denoise Method: {DENOISE_METHOD}")
    print(f"Reading data from: {DATA_DIR}")
    print(f"Saving artifacts to: {ARTIFACTS_DIR}")


    # --- Official + Wikipedia ---
    datasets_to_process = ["Official", "Wikipedia"]
    official_wiki_residuals = {}
    print(f"\n--- Processing Official and Wikipedia Datasets ---")
    for dataset in datasets_to_process:
        dataset_dir = os.path.join(DATA_DIR, dataset)
        official_wiki_residuals[dataset] = process_folder(dataset_dir, use_dpi_subfolders=True)

    OW_RES_PATH = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl")
    save_pickle(official_wiki_residuals, OW_RES_PATH)

    # --- Flatfield ---
    print(f"\n--- Processing Flatfield Dataset ---")
    flatfield_dir = os.path.join(DATA_DIR, "Flatfield")
    # Structure: Flatfield/ScannerName/image.tif -> use_dpi_subfolders=False
    flatfield_residuals = process_folder(flatfield_dir, use_dpi_subfolders=False)

    FLAT_RES_PATH = os.path.join(ARTIFACTS_DIR, "flatfield_residuals.pkl")
    save_pickle(flatfield_residuals, FLAT_RES_PATH)

    # --- Summary ---
    total_scanners_flat = len(flatfield_residuals)
    total_images_flat = sum(len(res_data.get('all', [])) for res_data in flatfield_residuals.values())

    print(f"\n✅ Flatfield Processing Summary:")
    print(f"   Scanners processed: {total_scanners_flat}")
    print(f"   Total valid residuals generated: {total_images_flat}")

    total_images_ow = 0
    total_scanners_ow = 0
    for ds_name, scanners in official_wiki_residuals.items():
        ds_images = sum(len(dpi_list) for dpi_dict in scanners.values() for dpi_list in dpi_dict.values())
        total_images_ow += ds_images
        total_scanners_ow += len(scanners) # Count scanners per dataset
        print(f"\n✅ {ds_name} Processing Summary:")
        print(f"   Scanners processed: {len(scanners)}")
        print(f"   Total valid residuals generated: {ds_images}")


    end_total_time = time.time()
    total_minutes = (end_total_time - start_total_time) / 60
    print(f"\n--- Total Preprocessing Time: {total_minutes:.2f} minutes ---")

