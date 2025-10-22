import streamlit as st
import pandas as pd
import os
from PIL import Image
import time

# --- Import the backend logic ---
from extractor_backend import run_extraction_process

st.set_page_config(page_title="Feature Extractor", layout="wide")
st.title("📊 AI TraceFinder - Feature Extraction Tool")
st.write("This tool scans a dataset of images, extracts metadata features, and saves them to a CSV file.")

# --- Helper for Download Button ---
@st.cache_data
def convert_df_to_csv(df):
    """Caches the conversion of a DataFrame to CSV for fast downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- 1. UI Inputs for Paths ---
st.header("1. Set Paths")
st.write("Provide the path to your 'official' data folder and where to save the output CSV.")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
default_data_path = os.path.join(base_dir, "data", "official")
default_save_path = os.path.join(base_dir, "results", "metadata_features.csv")

dataset_root = st.text_input("Enter dataset root path:", default_data_path)
save_path = st.text_input("Enter CSV save path:", default_save_path)


# --- 2. Button to Start Extraction ---
st.header("2. Run Extraction")
if st.button("🚀 Start Feature Extraction"):
    st.info(f"🔎 Starting scan...")
    start_time = time.time()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    final_df = None
    image_paths = None

    # Clean the input path from the text box
    clean_dataset_root = dataset_root.strip(' "') 
    
    # --- THIS IS THE FIX ---
    # Pass the 'clean_dataset_root' to the backend, not the original 'dataset_root'
    for (status, percent, data) in run_extraction_process(clean_dataset_root):
    # --- END FIX ---
        
        if percent == -1: # Error case
            st.error(status)
            break
        
        if percent == 0.0: # Initial "found files" message
            st.success(status)

        status_text.text(status)
        progress_bar.progress(percent)
        
        if percent == 1.0: # Final data payload
            if data is not None:
                end_time = time.time()
                st.success(f"✅ Feature extraction complete in {end_time - start_time:.2f} seconds.")
                (final_df, image_paths) = data
            else:
                pass
    # --- 3. Save results (if successful) ---
    if final_df is not None:
        try:
            # Clean the path variable to remove extra quotes or spaces
            clean_save_path = save_path.strip(' "')

            # Check if the path is a directory. If so, append a default filename.
            if os.path.isdir(clean_save_path):
                st.warning(f"Path is a directory. Appending default filename 'metadata_features.csv'")
                clean_save_path = os.path.join(clean_save_path, "metadata_features.csv")

            os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)
            final_df.to_csv(clean_save_path, index=False)
            st.info(f"💾 File saved to: {clean_save_path}")
            
            # Store in session state to show results
            st.session_state['feature_df'] = final_df
            st.session_state['image_preview_paths'] = image_paths
        except Exception as e:
            st.error(f"Error saving file: {e}")

# --- 4. View Results (runs if 'feature_df' is in memory) ---
if 'feature_df' in st.session_state:
    st.header("3. Results")
    df = st.session_state['feature_df']
    
    # Download Button
    csv_data = convert_df_to_csv(df)
    st.download_button(
        label="💾 Download metadata_features.csv",
        data=csv_data,
        file_name="metadata_features.csv",
        mime="text/csv",
    )
    
    # Graph
    st.subheader("Class Distribution (Graph)")
    st.write("This shows the number of images processed for each scanner class.")
    st.bar_chart(df['class_label'].value_counts())
    
    # Sample Images
    st.subheader("Sample Images (One per Class)")
    image_paths = st.session_state['image_preview_paths']
    cols = st.columns(5)
    
    i = 0
    for label, path in image_paths.items():
        try:
            img = Image.open(path)
            with cols[i % 5]:
                st.image(img, caption=label, use_column_width=True)
            i += 1
        except Exception as e:
            st.warning(f"Could not load image {path}: {e}")
    
    st.subheader("Extracted Features (Preview)")
    st.dataframe(df.head(20))
