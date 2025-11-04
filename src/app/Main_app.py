import streamlit as st
import pandas as pd
import joblib # For baseline models
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time
from PIL import Image # For displaying uploaded image
import subprocess # For running scripts
import json
import cv2
import pickle


# --- Path Setup ---
# This file is in src/app/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


# Add src to path so it can find preprocess, models, scripts
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


try:
    from scripts.predict_forensics import (
        predict_scanner_hybrid_forensics, 
        infer_tamper_image, 
        infer_tamper_single_patch,
        FORENSICS_AVAILABLE, HAS_IMG, HAS_PATCH, # Use these flags
        preprocess_residual_pywt
    )
    print("✅ Forensics module imported successfully.")
    from scripts.explainability import make_gradcam_heatmap, get_superimposed_image
    print("✅ Explainability modules imported successfully.")
    from scripts.reporting import generate_pdf_report
    print("✅ Report modules imported successfully.")

except ImportError as e:
    print(f"⚠️ Warning: Could not load forensics components: {e}.")
    # Define placeholder flags to avoid breaking the app
    FORENSICS_AVAILABLE = False
    HAS_IMG = False
    HAS_PATCH = False


# --- Safe Import Baseline ---
BASELINE_AVAILABLE = False
try:
    from preprocess.baseline.predict_baseline import predict_scanner as predict_scanner_baseline
    from scripts.training_baseline import train_models as train_baseline_models # <-- Updated import path
    BASELINE_AVAILABLE = True
    print("✅ Baseline components imported successfully.")
except ImportError as e:
    print(f"⚠️ Warning: Could not load baseline components: {e}.")
    def predict_scanner_baseline(path, model_choice): return None, None, []
    def train_baseline_models(): print("Baseline training module not found.")

# --- Safe Import CNN ---
CNN_AVAILABLE = False
try:
    from preprocess.cnn.predict_cnn import predict_scanner_cnn, _load_cnn_artifacts
    if _load_cnn_artifacts(): # Try loading artifacts at start
         CNN_AVAILABLE = True
         print("✅ CNN components imported and artifacts loaded successfully.")
    else:
         print("⚠️ Warning: CNN function imported but artifacts failed to load. CNN disabled.")
except ImportError as e:
    print(f"⚠️ Warning: Could not import CNN functions: {e}. CNN disabled.")
    def predict_scanner_cnn(path): return None, None, []
except Exception as e:
    print(f"❌ ERROR during CNN component import/load: {e}")
    CNN_AVAILABLE = False
    def predict_scanner_cnn(path): return None, None, []

# --- Define Paths ---
# Baseline paths
BASELINE_SCRIPT_ROOT = os.path.join(SRC_DIR, 'scripts') # <-- Path to training_baseline.py
BASELINE_MODEL_DIR = os.path.join(SRC_DIR, "models", "baseline")
BASELINE_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts") # <-- Test splits moved here
BASELINE_FEATURES_CSV = os.path.join(PROJECT_ROOT, "results", "metadata_features.csv") 
# CNN paths
CNN_MODEL_DIR = os.path.join(SRC_DIR, "models", "cnn")
CNN_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


# === Baseline Evaluation Function ===
def evaluate_baseline_model(model_path, name):
    st.write(f"--- Evaluating {name} ---")
    try:
        # --- PATHS UPDATED ---
        X_test_path = os.path.join(BASELINE_ARTIFACTS_DIR, "X_test.pkl") # <-- Moved to artifacts
        y_test_path = os.path.join(BASELINE_ARTIFACTS_DIR, "y_test.pkl") # <-- Moved to artifacts
        scaler_path = os.path.join(BASELINE_ARTIFACTS_DIR, "scaler.pkl") # <-- Moved to artifacts
        # scaler_path = os.path.join(BASELINE_MODEL_DIR, "scaler.pkl")
        # --- END PATHS ---

        required=[X_test_path,y_test_path,scaler_path,model_path]; missing=[p for p in required if not os.path.exists(p)]
        if missing: raise FileNotFoundError(f"Missing: {', '.join([os.path.basename(p) for p in missing])}")
        
        X_test=joblib.load(X_test_path); y_test=joblib.load(y_test_path); scaler=joblib.load(scaler_path); model=joblib.load(model_path)
        X_test_scaled=scaler.transform(X_test); y_pred=model.predict(X_test_scaled)
        
        # st.subheader(f"{name} Classification Report"); target_names_baseline=model.classes_
        # try: report=classification_report(y_test,y_pred,target_names=target_names_baseline,zero_division=0); st.text(report)
        # except ValueError as report_error: st.error(f"Report Error: {report_error}")

        # === Classification Report as DataFrame ===
        st.subheader(f"{name} Classification Report")
        target_names_baseline = model.classes_

        try:
            report_dict = classification_report(
                y_test, y_pred, target_names=target_names_baseline, output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.reset_index(inplace=True)
            report_df.rename(columns={"index": "Class"}, inplace=True)
            st.dataframe(
                report_df.style.format(precision=2),
                use_container_width=True,
            )
        except ValueError as report_error:
            st.error(f"Report Error: {report_error}")    


        st.subheader(f"{name} Confusion Matrix")
        try:
            cm=confusion_matrix(y_test,y_pred,labels=target_names_baseline); fig,ax=plt.subplots(figsize=(10,7))
            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=target_names_baseline,yticklabels=target_names_baseline,ax=ax,annot_kws={"size": 8})
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.xticks(rotation=45,ha='right',fontsize=9); plt.yticks(rotation=0,fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as plot_error: st.error(f"Plotting Error: {plot_error}")
    
    except FileNotFoundError as e: st.error(f"Cannot evaluate baseline: {e}")
    except Exception as e: st.error(f"Baseline evaluation error: {e}")

# === Baseline Feature Explorer ===
def baseline_feature_explorer():
    st.subheader("📊 Baseline Feature Explorer")
    try:
        if not os.path.exists(BASELINE_FEATURES_CSV): raise FileNotFoundError(f"File not found: {BASELINE_FEATURES_CSV}")
        df=pd.read_csv(BASELINE_FEATURES_CSV); st.dataframe(df.head())
        numeric_cols=df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: st.warning("No numeric features."); return
        feature=st.selectbox("Choose baseline feature:", numeric_cols, key="baseline_feat_select")
        if feature and feature in df.columns:
            if df[feature].isnull().all(): st.warning(f"Feature '{feature}' is all null.")
            else:
                fig,ax=plt.subplots(); sns.histplot(df[feature].dropna(),bins=30,kde=True,ax=ax)
                ax.set_title(f"Distribution of {feature}"); st.pyplot(fig); plt.close(fig)
        elif feature: st.warning(f"Feature '{feature}' not found.")
    except FileNotFoundError as e: st.error(f"{e}"); st.info(r"Run baseline feature extraction (src\features\baseline\extractor_frontend.py).")
    except Exception as e: st.error(f"Feature explorer error: {e}")

# === STREAMLIT UI ===


# === STREAMLIT UI ===
st.set_page_config(page_title="AI TraceFinder", layout="wide")
# st.title("📊 AI TraceFinder - Scanner Identification")
LOGO_PATH = os.path.join(PROJECT_ROOT, "img", "logo.png")



col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(LOGO_PATH, width=1600)

# Custom CSS for styling all images
st.markdown("""
<style>
img {
    height: 450px !important;              /* Force custom height */
    opacity: 1;                          /* Make it slightly transparent */
    filter: brightness(1.1) contrast(1.05);/* Increase brightness and contrast */
    border-radius: 15px;                   /* Rounded corners */
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.4); /* Soft glow */
    transition: all 0.5s ease-in-out;      /* Smooth hover transition */
}
img:hover {
    opacity: 1;                            /* Full brightness on hover */
    transform: scale(1.03);                /* Slight zoom on hover */
    filter: brightness(1.3);
}
</style>
""", unsafe_allow_html=True)




# st.write("Logo path:", LOGO_PATH)
# st.write("File exists:", os.path.exists(LOGO_PATH))

# # Custom HTML + CSS logo block
# st.markdown(f"""
#     <div style='text-align: center; margin-top: -40px;'>
#         <img src="file:///{LOGO_PATH.replace("\\", "/")}"  
#              style="
#                 width: 500px;                /* Change width */
#                 height: 500px;               /* Change height */
#                 border-radius: 1%;          /* Makes it circular */
#                 object-fit: cover;           /* Keeps proportions */
#                 box-shadow: 0px 0px 10px rgba(0,255,255,0.4); /* Soft glow */
#              ">
#     </div>
# """, unsafe_allow_html=True)


# col1, col2, col3 = st.columns([1, 6, 1])
# with col2:
#     st.image(LOGO_PATH, use_column_width=True)


# --- Sidebar ---
st.sidebar.title("Navigation")
# --- Base menu ---
menu_options = ["🚀 Predict Scanner", "🛠️ Project Pipeline"]
# --- Conditional menu items ---
if BASELINE_AVAILABLE:
    menu_options.extend([
        "📊 Evaluate Baseline",
        "⚙️ Train Baseline",
        "🔍 Explore Baseline Features"
    ])
if CNN_AVAILABLE:
    menu_options.extend([
        "📈 Evaluate CNN",
        "🧠 Train CNN"
    ])
# --- Always available menu ---
menu_options.append("🖼️ Dataset Visualization")
# --- Default selection ---
default_index = 0
# --- Sidebar Menu ---
menu = st.sidebar.radio(
    "Choose Action", 
    menu_options, 
    index=default_index, 
    key="main_menu"
)



# st.set_page_config(page_title="AI TraceFinder", layout="wide")
# st.title("📊 AI TraceFinder - Scanner Identification")
# st.sidebar.title("Navigation")
# menu_options=["Predict Scanner"]
# if BASELINE_AVAILABLE: menu_options.extend(["Evaluate Baseline","Train Baseline","Explore Baseline Features"])
# if CNN_AVAILABLE: menu_options.extend(["Evaluate CNN", "Train CNN"])
# menu_options.append("📊 Dataset Visualization")
# default_index=0
# menu=st.sidebar.radio("Choose Action", menu_options, index=default_index, key="main_menu")





# === PREDICT SCANNER PAGE ===
if menu == "🚀 Predict Scanner":
    st.header("Upload Image to Identify Scanner Source")

    # --- Collect available models dynamically ---
    available_model_types = []
    if BASELINE_AVAILABLE:
        available_model_types.append("Baseline (RF/SVM)")
    if FORENSICS_AVAILABLE:
        available_model_types.append("Hybrid Forensics (CNN + Tamper Check)")
    elif CNN_AVAILABLE:
        available_model_types.append("CNN (Hybrid - 27 Feat) - No Tamper")

    if not available_model_types:
        st.error("❌ No models loaded.")
        st.stop()

    # --- Stylish segmented model selector ---
    model_type = st.segmented_control(
        "Select Model Type",
        available_model_types,
        key="predict_model_type"
    )

    # --- Two-column layout ---
    col_input, col_results = st.columns(2)

    # ===============================================================
    # ============ LEFT COLUMN - IMAGE INPUT SECTION ================
    # ===============================================================
    with col_input:
        st.subheader("Your Image")
        with st.container(border=True):
            if model_type == "Baseline (RF/SVM)":
                baseline_model_choice_str = st.selectbox(
                    "Algorithm", ["Random Forest", "SVM"], key="baseline_model_predict"
                )
                uploaded_file = st.file_uploader(
                    "Upload a scanned image", 
                    type=["tif", "tiff", "jpg", "png", "jpeg"], 
                    key="baseline_uploader"
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload a scanned image", 
                    type=["tif", "tiff", "jpg", "png", "jpeg"], 
                    key="cnn_uploader"
                )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            # =======================================================
            # === NEW SECTION: Show Noise Residual Map (Expander) ===
            # =======================================================
            with st.expander("Show Noise Residual Map"):
                try:
                    # Create temp file here (shared between both columns)
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    # Generate and display residual map
                    with st.spinner("Generating noise map..."):
                        residual_image = preprocess_residual_pywt(temp_path)
                        residual_display = cv2.normalize(residual_image, None, 0, 1, cv2.NORM_MINMAX)
                        # st.image(residual_display, caption="Scanner Noise Residual", use_container_width=True)
                        st.image(residual_display, caption="Scanner Noise Residual", use_container_width=True, clamp=True)

                except Exception as e:
                    st.error(f"Could not generate noise map: {e}")
        else:
            temp_path = None  # fallback if not uploaded

    # ===============================================================
    # ============ RIGHT COLUMN - ANALYSIS RESULTS ==================
    # ===============================================================
    with col_results:
        st.subheader("Analysis Results")

        if uploaded_file is not None and temp_path:
            try:
                with st.spinner("Analyzing..."):

                    # ========== BASELINE MODEL ==========
                    if model_type == "Baseline (RF/SVM)" and BASELINE_AVAILABLE:
                        model_code = "rf" if baseline_model_choice_str == "Random Forest" else "svm"
                        pred_label, prob_list, classes = predict_scanner_baseline(
                            temp_path, model_choice=model_code
                        )

                        # --- Tabs for results ---
                        tab_scanner, _ = st.tabs(["Scanner Identification", "Tamper Detection"])
                        with tab_scanner:
                            st.metric("Predicted Scanner", pred_label)
                            if len(prob_list) == len(classes):
                                prob_df = pd.DataFrame({
                                    'Class': classes,
                                    'Confidence': [p * 100 for p in prob_list]
                                })
                                st.bar_chart(
                                    prob_df.set_index('Class').sort_values('Confidence', ascending=False)
                                )
                            else:
                                st.warning("⚠️ Probability/Class mismatch.")



                        # ========== HYBRID FORENSICS ==========
                    elif model_type.startswith("Hybrid Forensics") and (FORENSICS_AVAILABLE or CNN_AVAILABLE):
                        s_label, s_conf, all_probs_dict = None, None, {}
                        t_res = None  # Initialize t_res
                    
                        # --- Scanner Prediction ---
                        if FORENSICS_AVAILABLE:
                            # --- Import hybrid model and feature extractor for explainability ---
                            from scripts.predict_forensics import hyb_model, make_scanner_feats_from_res
                    
                            s_label, s_conf, all_probs_dict = predict_scanner_hybrid_forensics(temp_path)
                    
                            # --- Tamper Detection ---
                            if HAS_IMG:
                                t_res = infer_tamper_image(temp_path)
                                tamper_source = "Image-Level (18D)"
                            elif HAS_PATCH:
                                t_res = infer_tamper_single_patch(temp_path)
                                tamper_source = "Patch Fallback (22D)"
                            else:
                                tamper_source = "Disabled"
                    
                        else:
                            pred_label_cnn, prob_df_cnn, _ = predict_scanner_cnn(temp_path)
                            s_label, s_conf, all_probs_dict = pred_label_cnn, 0.0, {}
                            st.warning("Tamper check artifacts missing. Showing only Scanner ID.")
                    
                        # --- Tabs for results ---
                        tab_scanner, tab_tamper, tab_explain = st.tabs([
                            "Scanner Identification",
                            "Tamper Detection",
                            "🔬 Explainability (Grad-CAM)"
                        ])
                    
                        # --- SCANNER IDENTIFICATION TAB ---
                        with tab_scanner:
                            st.metric("Predicted Scanner", s_label, f"{s_conf:.2f}% Confidence")
                            if all_probs_dict:
                                prob_df = pd.DataFrame(list(all_probs_dict.items()), columns=['Class', 'Probability'])
                                prob_df['Confidence'] = prob_df['Probability'] * 100
                                st.bar_chart(prob_df.set_index('Class').sort_values('Confidence', ascending=False))
                    
                        # --- TAMPER DETECTION TAB ---
                        with tab_tamper:
                            if FORENSICS_AVAILABLE and t_res:
                                st.metric(
                                    "Tamper Label",
                                    t_res["tamper_label"],
                                    delta=f"{t_res['confidence']:.1f}% Confidence",
                                    delta_color=("inverse" if t_res['tamper_label'] == 'Tampered' else 'normal')
                                )
                                st.caption(f"Method: {tamper_source}")
                                st.write(f"Tampered Probability: **{t_res['prob_tampered']:.3f}** (Threshold: {t_res['threshold']:.3f})")
                                if t_res['hits'] != -1:
                                    st.write(f"Patch Hits: **{t_res['hits']}**")
                            else:
                                st.info("Tamper detection is only available with Hybrid Forensics model.")
                    
                        # --- EXPLAINABILITY TAB ---
                        with tab_explain:
                            st.subheader("Model Decision Visualization (Grad-CAM)")
                            st.write("This shows which parts of the *noise residual* the model focused on.")
                            st.caption("Note: This refers to the **`last_conv_layer`** we named in the training script.")
                    
                            if st.button("Generate Heatmap"):
                                with st.spinner("Generating Grad-CAM..."):
                                    try:
                                        # 1. Get the preprocessed inputs
                                        residual = preprocess_residual_pywt(temp_path)
                                        x_img = np.expand_dims(residual, axis=(0, -1)).astype(np.float32)
                    
                                        # 2. Get the 27-dim features
                                        x_feat = make_scanner_feats_from_res(residual)
                                       
                                        # 'last_conv_layer' = The new name we just added
                                        # 'conv2d_13' = The old name from the error message
                                        possible_layer_names = ["last_conv_layer", "conv2d_13"]
                    
                                        # 3. Generate heatmap
                                        heatmap = make_gradcam_heatmap(
                                            x_img,
                                            x_feat,
                                            hyb_model,             # Loaded hybrid model
                                            # "last_conv_layer"      # Layer used for Grad-
                                            possible_layer_names
                                        )
                    
                                        # 4. Superimpose heatmap on the original image
                                        superimposed_img, heatmap_img = get_superimposed_image(temp_path, heatmap, alpha=0.5)
                    
                                        # 5. Display results
                                        st.write("Heatmap (Red = Important):")
                                        st.image(
                                            superimposed_img,
                                            caption="Grad-CAM Superimposed on Original Image",
                                            use_container_width=True
                                        )
                    
                                        st.write("Raw Heatmap (Normalized):")
                                        st.image(
                                            heatmap_img,
                                            caption="Raw Heatmap",
                                            use_container_width=True
                                        )
                    
                                    except Exception as e:
                                        st.error(f"Could not generate Grad-CAM: {e}")
                                        st.exception(e)

                        # ... (This is the end of your 'with tab_explain:' block) ...
                                
                    # --- ADD THIS NEW SECTION for PDF Report ---
                    st.divider()
                    st.subheader("📥 Download Report")
                    
                    try:
                        # 1. Create data for the report
                        # We need to re-generate the residual image
                        residual_image_for_pdf = preprocess_residual_pywt(temp_path)

                        report_data = {
                            "image_path": temp_path,
                            "residual_image": residual_image_for_pdf,
                            "scanner_prediction": s_label,
                            "scanner_confidence": s_conf,
                            "tamper_label": t_res["tamper_label"] if t_res else "N/A",
                            "tamper_confidence": t_res["confidence"] if t_res else 0.0,
                            "probabilities": all_probs_dict
                        }
                        
                        # 2. Generate PDF in memory
                        pdf_bytes = generate_pdf_report(report_data)
                        
                        # 3. Add Download Button
                        st.download_button(
                            label="Download Forensic Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"forensic_report_{os.path.basename(uploaded_file.name)}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Could not generate PDF report: {e}")
                        st.exception(e) # Show full error for debugging

            except Exception as e:
                st.error(f"Prediction error: {e}")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
        else:
            st.info("Upload an image to see the analysis.")



# ===============================================================
# ============= NEW: PROJECT PIPELINE WIZARD PAGE ===============
# ===============================================================
elif menu == "🛠️ Project Pipeline":
    st.header("🛠️ Project Training Pipeline")
    st.write("Follow these steps to preprocess data and train your models.")

    # --- Define file paths for our checks ---
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Official")
    RESIDUALS_PKL = os.path.join(ARTIFACTS_DIR, "official_wiki_residuals.pkl")
    
    BASELINE_FEAT_CSV = os.path.join(PROJECT_ROOT, "results", "metadata_features.csv")
    BASELINE_MODEL_PKL = os.path.join(PROJECT_ROOT, "src", "models", "baseline", "models", "random_forest.pkl")
    
    CNN_FEAT_PKL = os.path.join(ARTIFACTS_DIR, "features_27dim.pkl")
    CNN_MODEL_KERA = os.path.join(PROJECT_ROOT, "src", "models", "cnn", "models", "scanner_hybrid_best.keras")

    # Get the python executable path
    PYTHON_EXE = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")


    tab1, tab2, tab3, tab4 = st.tabs([
        "**Milestone 1: Preprocessing**", 
        "**Milestone 2: Baseline Models**", 
        "**Milestone 3: CNN Model**", 
        "**Milestone 4: Deployment**"
    ])

    # --- TAB 1: PREPROCESSING ---
    with tab1:
        st.subheader("Step 1: Check Dataset")
        
        # Check 1: Do we have data?
        if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
            st.success(f"✅ Dataset found at `{DATA_DIR}`")
        else:
            st.error(f"❌ Dataset not found at `{DATA_DIR}`.")
            st.info("Please add your 'Official' dataset folder with images.")

        st.divider()
        
        st.subheader("Step 2: Run Image Preprocessing")
        st.write("This will scan all images, extract noise residuals, and save them.")
        
        # Check 2: Are residuals already processed?
        if os.path.exists(RESIDUALS_PKL):
            st.success(f"✅ Preprocessing complete! Artifact found: `{RESIDUALS_PKL}`")
        else:
            st.warning("⚠️ Preprocessing has not been run.")

        if st.button("Run Preprocessing"):
            script_path = os.path.join(SRC_DIR, "preprocess", "cnn", "processing_cnn.py")
            with st.spinner("Processing images... This may take several minutes."):
                try:
                    result = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
                    st.text(result.stdout)
                    if result.stderr:
                        st.warning(result.stderr)
                    st.success("Preprocessing complete! Refresh to see status update.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during preprocessing: {e.stderr}")

    # --- TAB 2: BASELINE MODELS ---
    with tab2:
        # Prerequisite check
        if not os.path.exists(RESIDUALS_PKL):
            st.warning("Please complete Milestone 1 first.")
            st.stop()

        st.subheader("Step 1: Extract Baseline Features")
        
        # Check 1: Are features extracted?
        if os.path.exists(BASELINE_FEAT_CSV):
            st.success(f"✅ Baseline features extracted! Artifact found: `{BASELINE_FEAT_CSV}`")
            st.dataframe(pd.read_csv(BASELINE_FEAT_CSV).head())
        else:
            st.warning("⚠️ Baseline features have not been extracted.")

        if st.button("Run Baseline Feature Extraction"):
            script_path = os.path.join(SRC_DIR, "features", "baseline", "build_features.py")
            with st.spinner("Extracting baseline features..."):
                try:
                    result = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
                    st.text(result.stdout)
                    st.success("Baseline feature extraction complete!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during extraction: {e.stderr}")
        
        st.divider()
        
        st.subheader("Step 2: Train Baseline Models")

        # Check 2: Are models trained?
        if os.path.exists(BASELINE_MODEL_PKL):
            st.success(f"✅ Baseline models trained! Artifact found: `{BASELINE_MODEL_PKL}`")
        else:
            st.warning("⚠️ Baseline models have not been trained.")
        
        if st.button("Train Baseline Models (RF & SVM)"):
            script_path = os.path.join(SRC_DIR, "scripts", "training_baseline.py")
            with st.spinner("Training baseline models..."):
                try:
                    result = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
                    st.text(result.stdout)
                    st.success("Baseline training complete!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during training: {e.stderr}")

    # --- TAB 3: CNN MODEL ---
    with tab3:
        # Prerequisite check
        if not os.path.exists(RESIDUALS_PKL):
            st.warning("Please complete Milestone 1 first.")
            st.stop()

        st.subheader("Step 1: Extract CNN Features (27-Dim)")
        
        # Check 1: Are features extracted?
        if os.path.exists(CNN_FEAT_PKL):
            st.success(f"✅ CNN features extracted! Artifact found: `{CNN_FEAT_PKL}`")
        else:
            st.warning("⚠️ CNN features have not been extracted.")

        if st.button("Run CNN Feature Extraction (27-Dim)"):
            script_path = os.path.join(SRC_DIR, "features", "cnn", "feature_extractor_cnn.py")
            with st.spinner("Extracting CNN features... This may take a minute."):
                try:
                    result = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
                    st.text(result.stdout)
                    st.success("CNN feature extraction complete!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during extraction: {e.stderr}")
        
        st.divider()
        
        st.subheader("Step 2: Train Hybrid CNN Model")

        # Check 2: Are models trained?
        if os.path.exists(CNN_MODEL_KERA):
            st.success(f"✅ Hybrid CNN model trained! Artifact found: `{CNN_MODEL_KERA}`")
        else:
            st.warning("⚠️ Hybrid CNN model has not been trained.")
        
        if st.button("Train Hybrid CNN Model"):
            st.warning("This may take a long time and requires a GPU for best results.")
            script_path = os.path.join(SRC_DIR, "scripts", "train_hybrid_cnn.py")
            with st.spinner("Training Hybrid CNN model..."):
                try:
                    result = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
                    st.text(result.stdout)
                    st.success("Hybrid CNN training complete!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during training: {e.stderr}")
        
        st.divider()
        st.subheader("Step 3: Explainability (SHAP / Grad-CAM)")
        st.info("This feature is not yet implemented. This step would run SHAP or Grad-CAM to generate visualizations.")

    # --- TAB 4: DEPLOYMENT ---
    with tab4:
        st.header("✅ Your Application is Ready!")
        st.info("You can now go to the '🚀 Predict Scanner' page in the sidebar to use your trained models.")
        
        if os.path.exists(CNN_MODEL_KERA):
            st.success("Hybrid CNN Model is trained and ready.")
        else:
            st.warning("Hybrid CNN Model is not trained.")
            
        if os.path.exists(BASELINE_MODEL_PKL):
            st.success("Baseline Models are trained and ready.")
        else:
            st.warning("Baseline Models are not trained.")




# === DATASET VISUALIZATION PAGE ===
elif menu == "🖼️ Dataset Visualization":
    st.header("📊 Dataset Visualization Dashboard")
    st.write("View class distribution, random samples, and dataset statistics.")

    from scripts.visualize_data import get_image_data, get_dataset_summary
    import random
    from PIL import Image

    # === Cache the dataset scan ===
    @st.cache_data(show_spinner=False)
    def cached_get_image_data(base_dir):
        return get_image_data(base_dir)

    @st.cache_data(show_spinner=False)
    def cached_get_dataset_summary(base_dir):
        return get_dataset_summary(base_dir)

    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Official")

    if not os.path.exists(DATA_DIR):
        st.error(f"Dataset folder not found: {DATA_DIR}")
        st.stop()

    # Use cached function ✅
    with st.spinner("Analyzing dataset..."):
        df, class_counts, stats = cached_get_dataset_summary(DATA_DIR)

    if df is None:
        st.warning("No images found in dataset.")
        st.stop()

    # --- Summary Stats ---
    st.subheader("📦 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Classes", stats["total_classes"])
    c2.metric("Total Images", stats["total_images"])
    c3.metric("Avg. Resolution", stats["avg_resolution"])
    c4.metric("Most Common Format", stats["common_format"])

    # --- Class Distribution ---
    st.subheader("📈 Class Distribution")
    st.bar_chart(class_counts.set_index("Class")["Image Count"])

    # --- Random Samples ---
    st.subheader("🖼️ Sample Images from Each Class")
    for class_name in class_counts["Class"]:
        subset = df[df["Class"] == class_name]
        if subset.empty:
            continue

        sample_paths = random.sample(subset["Path"].tolist(), min(3, len(subset)))

        # ✅ Use expander for cleaner UI
        with st.expander(f"📁 {class_name} ({len(subset)} images)"):
            cols = st.columns(len(sample_paths))
            for i, img_path in enumerate(sample_paths):
                try:
                    img = Image.open(img_path)
                    cols[i].image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    continue

            # ===================================================
            # === NEW SECTION: Noise Map Generation Button =======
            # ===================================================
            st.divider()
            if st.button(f"Generate Noise Map for {class_name}", key=f"noise_map_{class_name}"):
                with st.spinner("Generating sample noise map..."):
                    try:
                        # Use first sample image from this class
                        residual_image = preprocess_residual_pywt(sample_paths[0])
                        residual_display = cv2.normalize(residual_image, None, 0, 1, cv2.NORM_MINMAX)
                        st.image(
                            residual_display,
                            caption=f"Noise Map for {class_name}",
                            use_container_width=True,
                            clamp=True
                        )
                    except Exception as e:
                        st.error(f"Error generating noise map: {e}")




# === EVALUATION PAGE (Unified for Baseline + CNN) ===
elif (menu == "📊 Evaluate Baseline" and BASELINE_AVAILABLE) or (menu == "📈 Evaluate CNN" and CNN_AVAILABLE):
    st.header("📊 Model Evaluation Dashboard")

    # --- Define paths (for CNN section) ---
    cnn_cm_path = os.path.join(RESULTS_DIR, "cnn_confusion_matrix_27dim.png")
    cnn_report_path = os.path.join(RESULTS_DIR, "cnn_classification_report.json")

    # --- Tabs for Baseline & CNN ---
    tab_baseline, tab_cnn = st.tabs(["🧩 Baseline (RF & SVM)", "🧠 Hybrid CNN"])

    # ===============================================================
    # ================= BASELINE EVALUATION TAB =====================
    # ===============================================================
    with tab_baseline:
        st.subheader("Baseline Model Evaluation (Test Set)")
        st.info("Ensure baseline training artifacts exist before evaluating.")

        rf_model_path = os.path.join(BASELINE_MODEL_DIR, "models", "random_forest.pkl")
        svm_model_path = os.path.join(BASELINE_MODEL_DIR, "models", "svm.pkl")

        # --- Random Forest Evaluation ---
        with st.container(border=True):
            st.markdown("### 🌲 Random Forest Evaluation")
            if os.path.exists(rf_model_path):
                try:
                    evaluate_baseline_model(rf_model_path, "Random Forest")
                except Exception as e:
                    st.error(f"RF Evaluation Error: {e}")
            else:
                st.warning("RF model not found.")

        st.divider()

        # --- SVM Evaluation ---
        with st.container(border=True):
            st.markdown("### ⚙️ SVM Evaluation")
            if os.path.exists(svm_model_path):
                try:
                    evaluate_baseline_model(svm_model_path, "SVM")
                except Exception as e:
                    st.error(f"SVM Evaluation Error: {e}")
            else:
                st.warning("SVM model not found.")

    # ===============================================================
    # ==================== CNN EVALUATION TAB =======================
    # ===============================================================
    with tab_cnn:
        st.subheader("Hybrid CNN Model Evaluation (27-Dim)")

        # --- 1. Classification Report ---
        with st.expander("📋 Classification Report", expanded=True):
            if os.path.exists(cnn_report_path):
                try:
                    with open(cnn_report_path, 'r') as f:
                        report_dict = json.load(f)

                    # Display both JSON and formatted DataFrame
                    st.json(report_dict)

                    df_report = pd.DataFrame(report_dict).transpose()
                    if 'support' in df_report.columns:
                        df_report['support'] = df_report['support'].astype(float).astype(int)

                    st.markdown("#### 📑 Detailed Metrics")
                    st.dataframe(df_report.style.highlight_max(axis=0, color='lightgreen'))
                except Exception as e:
                    st.error(f"Error loading classification report: {e}")
            else:
                st.warning("Classification report (`cnn_classification_report.json`) not found.")
                st.info("You must run the evaluation script first to generate the report:")
                st.code("python src/scripts/eval_hybrid_cnn.py", language="bash")

        # --- 2. Confusion Matrix ---
        with st.expander("📉 Confusion Matrix", expanded=True):
            if os.path.exists(cnn_cm_path):
                st.image(cnn_cm_path, caption="CNN Confusion Matrix (27 Feat)", use_container_width=True)
            else:
                st.warning("Confusion matrix (`cnn_confusion_matrix_27dim.png`) not found.")
                st.info("Run the evaluation script to generate this image.")
        
        st.divider()
        with st.expander("🎓 Training & Validation History", expanded=True):
            history_path = os.path.join(ARTIFACTS_DIR, "hybrid_training_history.pkl") 
            # Note: Your training script saves it as "hybrid_training_history_27dim.pkl"
            # Please check your 'artifacts' folder and use the correct filename.
            
            # Let's check for the correct file name based on your 'train_hybrid_cnn.py'
            history_path_27dim = os.path.join(ARTIFACTS_DIR, "hybrid_training_history.pkl")

            if os.path.exists(history_path):
                with open(history_path, "rb") as f:
                    history = pickle.load(f)
                
                # Convert history dict to DataFrame for easy plotting
                history_df = pd.DataFrame(history)
                
                st.write("Model Accuracy over Epochs:")
                st.line_chart(history_df[['accuracy', 'val_accuracy']])
                
                st.write("Model Loss over Epochs:")
                st.line_chart(history_df[['loss', 'val_loss']])
            else:
                st.info(f"Training history file not found at `{history_path}`. Run '🧠 Train CNN' to generate it.")




# elif menu=="Evaluate Baseline" and BASELINE_AVAILABLE:
#     st.header("Baseline Model Evaluation"); st.write("Results on test set.")
#     st.info("Ensure baseline training artifacts exist.")
#     rf_model_path=os.path.join(BASELINE_MODEL_DIR, "models", "random_forest.pkl")
#     if os.path.exists(rf_model_path): evaluate_baseline_model(rf_model_path,"Random Forest")
#     else: st.warning("RF model not found.")
#     st.divider()
#     svm_model_path=os.path.join(BASELINE_MODEL_DIR, "models", "svm.pkl")
#     if os.path.exists(svm_model_path): evaluate_baseline_model(svm_model_path,"SVM")
#     else: st.warning("SVM model not found.")    


# elif menu=="Train Baseline" and BASELINE_AVAILABLE:
elif menu=="⚙️ Train Baseline" and BASELINE_AVAILABLE:
    st.header("Train Baseline Models"); st.write(f"Uses: `{os.path.relpath(BASELINE_FEATURES_CSV, PROJECT_ROOT)}`")
    st.warning("Overwrites existing artifacts in `src/models/baseline/` and `artifacts/`.")
    if st.button("Start Baseline Training",key="train_baseline_button"):
        if not os.path.exists(BASELINE_FEATURES_CSV): st.error(f"Not found: {BASELINE_FEATURES_CSV}")
        else:
             with st.spinner("Training baseline..."):
                start_time_base=time.time()
                try:
                    # Run the script from its new location
                    # We pass the full python executable path to avoid venv issues
                    python_exe = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
                    script_path = os.path.join(BASELINE_SCRIPT_ROOT, "training_baseline.py")
                    # Use subprocess to run the script in its own process
                    # This is more robust than os.chdir
                    result = subprocess.run([python_exe, script_path], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
                    st.text(result.stdout) # Show output from the script
                    if result.stderr: st.warning(result.stderr)
                    end_time_base=time.time(); st.success(f"✅ Done ({end_time_base-start_time_base:.2f}s).")
                except subprocess.CalledProcessError as e:
                    st.error(f"Baseline train error (see console): {e.stderr}")
                except Exception as e: 
                    st.error(f"Baseline train error: {e}")

# elif menu=="Explore Baseline Features" and BASELINE_AVAILABLE: 
elif menu=="🔍 Explore Baseline Features" and BASELINE_AVAILABLE:
    baseline_feature_explorer()

# elif menu == "Evaluate CNN" and CNN_AVAILABLE:
#     st.header("CNN Model Evaluation")

#     # --- Define paths ---
#     cnn_cm_path = os.path.join(RESULTS_DIR, "cnn_confusion_matrix_27dim.png")
#     cnn_report_path = os.path.join(RESULTS_DIR, "cnn_classification_report.json")

#     # --- 1. Display Classification Report ---
#     st.subheader("Classification Report")
    
#     if os.path.exists(cnn_report_path):
#         try:
#             # Load the saved JSON report
#             with open(cnn_report_path, 'r') as f:
#                 report_dict = json.load(f)
            
#             # Convert it to a Pandas DataFrame
#             df_report = pd.DataFrame(report_dict).transpose()
            
#             # Format the 'support' column to be a clean integer
#             if 'support' in df_report.columns:
#                  df_report['support'] = df_report['support'].astype(float).astype(int)
            
#             # Display the DataFrame as a nice table
#             st.dataframe(df_report)
            
#         except Exception as e:
#             st.error(f"Error loading classification report: {e}")
#     else:
#         st.warning("Classification report (`cnn_classification_report.json`) not found.")
#         st.info("You must run the evaluation script first to generate the report:")
#         st.code("python src/scripts/eval_hybrid_cnn.py", language="bash")

#     # --- 2. Display Confusion Matrix ---
#     st.subheader("Confusion Matrix")
    
#     if os.path.exists(cnn_cm_path):
#         st.image(cnn_cm_path, caption="CNN Confusion Matrix (27 Feat)")
#     else:
#         st.warning("Confusion matrix (`cnn_confusion_matrix_27dim.png`) not found.")
#         st.info("Run the evaluation script to generate this image.")


# elif menu=="Train CNN" and CNN_AVAILABLE:
elif menu=="🧠 Train CNN" and CNN_AVAILABLE:
    st.header("Train CNN Model Info"); 
    st.warning("⚠️ Run from terminal (long, GPU recommended).")
    # --- FIX: Use raw string for path ---
    st.write(r"Command (from project root `D:\Project_Trace_Finder`):")
    st.code("python src/scripts/train_cnn.py", language="bash") # <-- Updated path
    # --- END FIX ---
    st.markdown("- Loads artifacts.\n- Splits/scales data.\n- Saves scaler/encoder.\n- Defines/trains/saves model & history.")

elif menu=="Test CNN" and CNN_AVAILABLE:
     st.header("Test CNN on Folder Info"); st.info("Predicts images in `data/Test`.")
     # --- FIX: Use raw string for path ---
     st.write(r"Command (from project root `D:\Project_Trace_Finder`):")
     st.code("python test_cnn_folder.py", language="bash") # <-- No path change needed
     # --- END FIX ---
     st.markdown("- Loads best CNN model.\n- Processes images, predicts.\n- Saves results to `results/cnn_hybrid_folder_results.csv`.")
     cnn_test_results_path=os.path.join(RESULTS_DIR,"cnn_hybrid_folder_results.csv")
     if os.path.exists(cnn_test_results_path):
          st.subheader("Last Test Run:")
          try:
               df_test=pd.read_csv(cnn_test_results_path); st.dataframe(df_test.head(10))
               @st.cache_data
               def get_test_csv_data(df): return df.to_csv(index=False).encode('utf-8')
               csv_test_data=get_test_csv_data(df_test)
               st.download_button("💾 Download Full CSV", csv_test_data, "cnn_hybrid_folder_results.csv", "text/csv", key="download_cnn_test_csv")
          except Exception as e: st.warning(f"Could not display test CSV: {e}")
     else: st.write("(Run script to generate results)")

else: st.sidebar.warning(f"Action '{menu}' unavailable."); st.info("Select action.")
