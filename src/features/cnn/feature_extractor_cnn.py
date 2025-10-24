import os
import pickle
import time
# Import paths and constants from our config file
import config
# Import the main pipeline functions from our functions file
from feature_extractor_functions import compute_or_load_fingerprints, extract_27dim_features

def main():
    """Runs the full feature extraction pipeline."""
    total_start_time = time.time()
    print("--- Starting Feature Extraction Pipeline (27-Dimension Version) ---")

    # Step 1: Ensure Fingerprints Exist (should be 11)
    scanner_fingerprints, fp_keys = compute_or_load_fingerprints()

    # Proceed only if fingerprints are valid
    if scanner_fingerprints and fp_keys:
        # Step 2: Load OW Residuals
        try:
            if not os.path.exists(config.OW_RESIDUALS_PATH):
                raise FileNotFoundError(f"Official/Wikipedia residuals file not found: {config.OW_RESIDUALS_PATH}")
            
            with open(config.OW_RESIDUALS_PATH, "rb") as f:
                residuals_dict_main = pickle.load(f)
            print(f"   Loaded Official/Wikipedia residuals.")

            # Step 3: Extract 27-Dimension Features
            success_27dim = extract_27dim_features(residuals_dict_main, scanner_fingerprints, fp_keys)

            if not success_27dim:
                print("\n⚠️ 27-dim feature extraction did not complete successfully.")

        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("   Cannot extract features. Run 'src/preprocess/cnn/processing_cnn.py' first.")
        except Exception as e:
            print(f"❌ An error occurred during 27-dim feature extraction: {e}")
    else:
        print("❌ Feature extraction cannot proceed without valid 11-key fingerprints.")

    total_end_time = time.time()
    print(f"\n--- Total Pipeline Time: {(total_end_time - total_start_time)/60:.2f} minutes ---")

if __name__ == "__main__":
    main()






