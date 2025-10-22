import streamlit as st
import pandas as pd
import joblib  
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import numpy as np

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(SRC_DIR)

from preprocess.baseline.predict_baseline import predict_scanner
from models.baseline.training_baseline import train_models

# --- Define Paths for App ---
MODEL_ROOT = os.path.join(SRC_DIR, 'models', 'baseline')
MODEL_DIR = os.path.join(MODEL_ROOT, "models")
DATA_DIR = os.path.join(MODEL_ROOT, "data")
RESULTS_DIR = os.path.abspath(os.path.join(SRC_DIR, '..', 'results'))
CSV_PATH = os.path.join(RESULTS_DIR, "metadata_features.csv")


# === EVALUATION FUNCTION ===
def evaluate_model(model_path, name):
    """Loads saved test data and models to show evaluation metrics."""
    try:
        # Load the pre-saved test data
        X_test = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
        y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))
        
        # --- FIXED ---
        # Load models and scaler using joblib
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        model = joblib.load(model_path)
        # --- END FIX ---

    except FileNotFoundError:
        st.error(f"Models or test data not found. Please run the 'Train Models' tab first.")
        return

    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    st.subheader(f"{name} Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader(f"{name} Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# === FEATURE EXPLORER ===
def feature_explorer():
    st.subheader("📊 Feature Explorer")
    try:
        df = pd.read_csv(CSV_PATH)
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            feature = st.selectbox("Choose feature to visualize", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[feature], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)
    except FileNotFoundError:
        st.error(f"Feature file not found: {CSV_PATH}")
        st.info("Please run the `src/features/baseline/build_features.py` script first.")


# === STREAMLIT UI ===
st.title("📊 AI TraceFinder - Baseline Model")

menu = st.sidebar.radio("Choose Action", 
    ["1. Predict Scanner", "2. Evaluate Models", "3. Train Models", "4. Feature Explorer"])

if menu == "1. Predict Scanner":
    st.header("Upload an image to identify the scanner")
    uploaded_file = st.file_uploader("Upload an Image", type=["tif", "tiff", "jpg", "png", "jpeg"])
    model_choice_str = st.selectbox("Choose Model", ["Random Forest", "SVM"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # st.image(temp_path, caption="Uploaded Image")
            st.image(uploaded_file, caption="Uploaded Image")
            
            with st.spinner("Analyzing image and extracting features..."):
                model_code = "rf" if model_choice_str == "Random Forest" else "svm"
                
                prob, classes = predict_scanner(temp_path, model_choice=model_code)
            
            st.write("🔍 Class Probabilities:")
            
            prob_df = pd.DataFrame({'Class': classes, 'Probability': prob})
            prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
            
            # Now, get our official prediction from the top row of the table
            pred = prob_df.iloc[0]['Class']
            # Display the success message and the dataframe
            st.success(f"🖼️ Predicted Scanner: **{pred}**")

            st.dataframe(prob_df)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
        finally:
            os.remove(temp_path)

elif menu == "2. Evaluate Models":
    st.header("Model Evaluation on Test Set")
    st.write("This report is generated from the test data saved during the last training run.")
    
    st.subheader("Evaluate Random Forest")
    evaluate_model(os.path.join(MODEL_DIR, "random_forest.pkl"), "Random Forest")
    
    st.subheader("Evaluate SVM")
    evaluate_model(os.path.join(MODEL_DIR, "svm.pkl"), "SVM")

elif menu == "3. Train Models":
    st.header("Re-Train Baseline Models")
    st.warning("This will re-train the models and overwrite existing ones.")
    if st.button("Train Models Now"):
        with st.spinner("Training models... This may take a moment."):
            start_time = time.time()
            
            train_models() 
            
            end_time = time.time()
            st.success(f"✅ Models trained successfully in {end_time - start_time:.2f} seconds.")

elif menu == "4. Feature Explorer":
    feature_explorer()