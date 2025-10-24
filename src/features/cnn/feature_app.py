"""
feature_app.py
Streamlit UI for the Trace Finder project.
Provides a button to start the feature extraction pipeline
and shows live progress.
"""

import streamlit as st
import os
import pickle
import time
import sys

# Add the directory to the Python path if config/functions are in a subdir
# If all files are in the same folder, you can skip this
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(SCRIPT_DIR) 

try:
    import config
    # We rename the import to be clear
    import feature_extractor_functions as fef 
except ImportError:
    st.error("ERROR: Failed to import 'config.py' or 'feature_extractor_functions.py'.")
    st.info("Please make sure all three files (`app.py`, `config.py`, `feature_extractor_functions.py`) are in the same directory.")
    st.stop()


st.set_page_config(page_title="Trace Finder Pipeline", layout="wide")
st.title("🔎 Trace Finder - Feature Extraction Pipeline")
st.write("""
Welcome to the feature extraction pipeline for the Forensic Scanner Identification project.
This app will:
1.  **Load/Compute Scanner Fingerprints:** Generate the 11 "master" scanner noise patterns.
2.  **Load Residuals:** Load the main dataset of scanned image residuals.
3.  **Extract 27-Dim Features:** Process every residual to generate the final feature/label file.

This process can take several minutes.
""")

# We use session state to prevent re-running if the user interacts with the app
if 'pipeline_started' not in st.session_state:
    st.session_state.pipeline_started = False

if st.button("🚀 Start Feature Extraction", disabled=st.session_state.pipeline_started):
    st.session_state.pipeline_started = True
    
    st.header("Pipeline Log")
    
    # Create dedicated containers for each step
    st.subheader("Step 1: Scanner Fingerprints")
    fp_log_area = st.empty()
    fp_progress_area = st.empty()

    st.subheader("Step 2: Load Residuals")
    load_log_area = st.empty()

    st.subheader("Step 3: Extract 27-Dim Features")
    feat_log_area = st.empty()
    feat_progress_area = st.empty() # For the nested progress bars

    overall_start_time = time.time()

    try:
        # --- Run Step 1 ---
        fp_log_area.info("Starting fingerprint computation...")
        scanner_fingerprints, fp_keys = fef.compute_or_load_fingerprints(
            log_container=fp_log_area, 
            progress_container=fp_progress_area
        )
        if not (scanner_fingerprints and fp_keys):
            raise Exception("Fingerprint generation failed. Check logs above.")
        
        fp_log_area.success("✅ Fingerprints computed/loaded successfully.")

        # --- Run Step 2 ---
        load_log_area.info(f"Loading residuals from {config.OW_RESIDUALS_PATH}...")
        if not os.path.exists(config.OW_RESIDUALS_PATH):
            raise FileNotFoundError(f"Official/Wikipedia residuals file not found: {config.OW_RESIDUALS_PATH}")
        
        with open(config.OW_RESIDUALS_PATH, "rb") as f:
            residuals_dict_main = pickle.load(f)
        load_log_area.success(f"✅ Residuals loaded successfully.")

        # --- Run Step 3 ---
        feat_log_area.info("Starting 27-dim feature extraction...")
        success = fef.extract_27dim_features(
            residuals_dict_main, scanner_fingerprints, fp_keys,
            log_container=feat_log_area,
            progress_container=feat_progress_area
        )
        if not success:
            raise Exception("Feature extraction failed. Check logs above.")
        
        feat_log_area.success("✅ Feature extraction complete!")

        # --- Finish ---
        overall_end_time = time.time()
        st.success(f"--- 🎉 Pipeline Finished Successfully! ---")
        st.info(f"Total time: {(overall_end_time - overall_start_time)/60:.2f} minutes")
        st.balloons()
        st.session_state.pipeline_started = False # Allow re-run

    except Exception as e:
        st.error(f"--- ❌ Pipeline Failed ---")
        st.exception(e)
        st.session_state.pipeline_started = False # Allow re-run

# Instructions to run the app
st.sidebar.header("How to Run")
st.sidebar.info("""
1. Make sure you have all required libraries:
   `pip install streamlit numpy scikit-image scipy`
2. Place `app.py`, `config.py`, and `feature_extractor_functions.py` in the same folder.
3. Run this app from your terminal:
   `streamlit run app.py`
""")