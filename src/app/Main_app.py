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

# --- Path Setup ---
# This file is in src/app/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
# Add src to path so it can find preprocess, models, scripts
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

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
st.set_page_config(page_title="AI TraceFinder", layout="wide")
st.title("📊 AI TraceFinder - Scanner Identification")
st.sidebar.title("Navigation")
menu_options=["Predict Scanner"]
if BASELINE_AVAILABLE: menu_options.extend(["Evaluate Baseline","Train Baseline","Explore Baseline Features"])
if CNN_AVAILABLE: menu_options.extend(["Evaluate CNN (Script Info)","Train CNN (Script Info)","Test CNN (Script Info)"])
menu_options.append("📊 Dataset Visualization")
default_index=0
menu=st.sidebar.radio("Choose Action", menu_options, index=default_index, key="main_menu")

# --- Page Content ---
if menu == "Predict Scanner":
    st.header("Upload Image to Identify Scanner Source")
    available_model_types=[]
    if BASELINE_AVAILABLE: available_model_types.append("Baseline (RF/SVM)")
    if CNN_AVAILABLE: available_model_types.append("CNN (Hybrid - 27 Feat)")
    if not available_model_types: st.error("❌ No models loaded."); st.stop()
    model_type=st.selectbox("Select Model Type", available_model_types, key="predict_model_type")

    # Baseline Prediction
    if model_type=="Baseline (RF/SVM)" and BASELINE_AVAILABLE:
        st.subheader("Baseline Prediction")
        baseline_model_choice_str=st.selectbox("Algorithm",["Random Forest","SVM"],key="baseline_model_predict")
        uploaded_file_base=st.file_uploader("Upload Image",type=["tif","tiff","jpg","png","jpeg"],key="baseline_uploader")
        if uploaded_file_base is not None:
            with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(uploaded_file_base.name)[1]) as tmp_file_base:
                tmp_file_base.write(uploaded_file_base.getvalue()); temp_path_baseline=tmp_file_base.name
            try:
                st.image(uploaded_file_base, caption="Uploaded", width=256) # FIX: Use uploaded file
                with st.spinner("Analyzing (Baseline)..."):
                    model_code="rf" if baseline_model_choice_str=="Random Forest" else "svm"
                    pred_label, prob_list, classes=predict_scanner_baseline(temp_path_baseline, model_choice=model_code)
                if pred_label is not None and prob_list is not None and classes is not None and len(classes)>0:
                    st.success(f"🖼️ Prediction: **{pred_label}**")
                    st.write("Probabilities:")
                    if len(prob_list)==len(classes):
                        prob_df=pd.DataFrame({'Class':classes,'Probability':prob_list}); prob_df['Confidence (%)']=prob_df['Probability']*100
                        prob_df_display=prob_df[['Class','Confidence (%)']].copy(); prob_df_display['Confidence (%)']=prob_df_display['Confidence (%)'].map('{:.2f}%'.format)
                        st.dataframe(prob_df_display.sort_values(by='Confidence (%)',ascending=False,key=lambda x:x.str.rstrip('%').astype(float)))
                    else: st.warning("Prob/class mismatch.")
                else: st.error("Baseline prediction failed.")
            except FileNotFoundError as e: st.error(f"Baseline artifacts missing: {e}")
            except Exception as e: st.error(f"Baseline prediction error: {e}")
            finally:
                if 'temp_path_baseline' in locals() and os.path.exists(temp_path_baseline):
                    try: os.remove(temp_path_baseline)
                    except OSError: pass

    # CNN Prediction
    elif model_type=="CNN (Hybrid - 27 Feat)" and CNN_AVAILABLE:
        st.subheader("CNN Prediction (27 Features)")
        uploaded_file_cnn=st.file_uploader("Upload Image",type=["tif","tiff","jpg","png","jpeg"],key="cnn_uploader")
        if uploaded_file_cnn is not None:
             with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(uploaded_file_cnn.name)[1]) as tmp_file_cnn:
                 tmp_file_cnn.write(uploaded_file_cnn.getvalue()); temp_path_cnn=tmp_file_cnn.name
             try:
                st.image(uploaded_file_cnn, caption="Uploaded", width=256) # FIX: Use uploaded file
                with st.spinner("Analyzing (CNN)..."):
                    pred_label_cnn, prob_df_cnn, _=predict_scanner_cnn(temp_path_cnn)
                if pred_label_cnn is not None and prob_df_cnn is not None:
                    st.success(f"🧠 Prediction: **{pred_label_cnn}**")
                    st.write("Probabilities:")
                    prob_df_cnn_display=prob_df_cnn.copy(); prob_df_cnn_display.rename(columns={'Probability':'Confidence (%)'},inplace=True)
                    prob_df_cnn_display['Confidence (%)']=(prob_df_cnn_display['Confidence (%)']*100).map('{:.2f}%'.format)
                    st.dataframe(prob_df_cnn_display) # Already sorted
                else: st.error("CNN prediction failed. Check console logs.")
             except Exception as e: st.error(f"CNN prediction error: {e}")
             finally:
                if 'temp_path_cnn' in locals() and os.path.exists(temp_path_cnn):
                      try: os.remove(temp_path_cnn)
                      except OSError: pass


# elif menu == "📊 Dataset Visualization":
#     st.header("📊 Dataset Visualization Dashboard")
#     st.write("View class distribution, random samples, and dataset statistics.")

#     from scripts.visualize_data import get_image_data, get_dataset_summary

#     @st.cache_data(show_spinner=False)
#     def cached_get_image_data(base_dir):
#         return get_image_data(base_dir)


#     @st.cache_data(show_spinner=False)
#     def cached_get_dataset_summary(base_dir):
#         return get_dataset_summary(base_dir)

#     # from scripts.visualize_data import get_dataset_summary
#     DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Official")

#     if not os.path.exists(DATA_DIR):
#         st.error(f"Dataset folder not found: {DATA_DIR}")
#         st.stop()

#     with st.spinner("Analyzing dataset..."):
#         df, class_counts, stats = get_dataset_summary(DATA_DIR)

#     if df is None:
#         st.warning("No images found in dataset.")
#         st.stop()

#     # --- Summary Stats ---
#     st.subheader("📦 Dataset Summary")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Classes", stats["total_classes"])
#     c2.metric("Total Images", stats["total_images"])
#     c3.metric("Avg. Resolution", stats["avg_resolution"])
#     c4.metric("Most Common Format", stats["common_format"])

#     # --- Class Distribution ---
#     st.subheader("📈 Class Distribution")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x="Class", y="Image Count", data=class_counts, hue="Class", dodge=False, legend=False, palette="viridis")
#     plt.xticks(rotation=90)
#     st.pyplot(fig)
#     plt.close(fig)

#     # --- Random Samples ---
#     st.subheader("🖼️ Sample Images from Each Class")
#     import random
#     from PIL import Image

#     for class_name in class_counts["Class"]:
#         subset = df[df["Class"] == class_name]
#         if subset.empty:
#             continue
#         sample_paths = random.sample(subset["Path"].tolist(), min(3, len(subset)))
#         st.markdown(f"### 📁 {class_name} ({len(subset)} images)")
#         cols = st.columns(len(sample_paths))
#         for i, img_path in enumerate(sample_paths):
#             try:
#                 img = Image.open(img_path)
#                 cols[i].image(img, use_container_width=True)
#             except:
#                 continue


elif menu == "📊 Dataset Visualization":
    st.header("📊 Dataset Visualization Dashboard")
    st.write("View class distribution, random samples, and dataset statistics.")

    from scripts.visualize_data import get_image_data, get_dataset_summary

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
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Class",
        y="Image Count",
        data=class_counts,
        hue="Class",
        dodge=False,
        legend=False,
        palette="viridis"
    )
    plt.xticks(rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    # --- Random Samples ---
    st.subheader("🖼️ Sample Images from Each Class")
    import random
    from PIL import Image

    for class_name in class_counts["Class"]:
        subset = df[df["Class"] == class_name]
        if subset.empty:
            continue
        sample_paths = random.sample(subset["Path"].tolist(), min(3, len(subset)))
        st.markdown(f"### 📁 {class_name} ({len(subset)} images)")
        cols = st.columns(len(sample_paths))
        for i, img_path in enumerate(sample_paths):
            try:
                img = Image.open(img_path)
                cols[i].image(img, use_container_width=True)
            except:
                continue



elif menu=="Evaluate Baseline" and BASELINE_AVAILABLE:
    st.header("Baseline Model Evaluation"); st.write("Results on test set.")
    st.info("Ensure baseline training artifacts exist.")
    rf_model_path=os.path.join(BASELINE_MODEL_DIR, "models", "random_forest.pkl")
    if os.path.exists(rf_model_path): evaluate_baseline_model(rf_model_path,"Random Forest")
    else: st.warning("RF model not found.")
    st.divider()
    svm_model_path=os.path.join(BASELINE_MODEL_DIR, "models", "svm.pkl")
    if os.path.exists(svm_model_path): evaluate_baseline_model(svm_model_path,"SVM")
    else: st.warning("SVM model not found.")    

elif menu=="Train Baseline" and BASELINE_AVAILABLE:
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

elif menu=="Explore Baseline Features" and BASELINE_AVAILABLE: baseline_feature_explorer()

elif menu=="Evaluate CNN (Script Info)" and CNN_AVAILABLE:
    st.header("CNN Evaluation Info"); st.info("Run via terminal for details.")
    # --- FIX: Use raw string for path ---
    st.write(r"Command (from project root `D:\Project_Trace_Finder`):")
    st.code("python src/scripts/evaluate_cnn.py", language="bash") # <-- Updated path
    # --- END FIX ---
    st.markdown("- Loads final CNN model.\n- Recreates test split.\n- Prints metrics.\n- Saves plot to `results/cnn_confusion_matrix_27dim.png`.")
    cnn_cm_path=os.path.join(RESULTS_DIR,"cnn_confusion_matrix_27dim.png")
    if os.path.exists(cnn_cm_path): st.image(cnn_cm_path,caption="CNN Confusion Matrix (27 Feat)")
    else: st.write("(Run script to generate plot)")

elif menu=="Train CNN (Script Info)" and CNN_AVAILABLE:
    st.header("Train CNN Model Info"); st.warning("⚠️ Run from terminal (long, GPU recommended).")
    # --- FIX: Use raw string for path ---
    st.write(r"Command (from project root `D:\Project_Trace_Finder`):")
    st.code("python src/scripts/train_cnn.py", language="bash") # <-- Updated path
    # --- END FIX ---
    st.markdown("- Loads artifacts.\n- Splits/scales data.\n- Saves scaler/encoder.\n- Defines/trains/saves model & history.")

elif menu=="Test CNN (Script Info)" and CNN_AVAILABLE:
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
