# 📊 AI TraceFinder: Forensic Scanner Identification

Detecting document forgery by analyzing a scanner's unique digital fingerprint.

---

## 📘 Table of Contents
- [About the Project](#-about-the-project)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Demo / Screenshots](#-demo--screenshots)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 About the Project

Scanned documents like legal agreements, official certificates, and financial records are easy to forge. It's often impossible to tell if a scanned document is legitimate or if it was created using an unauthorized, fraudulent device.

**AI TraceFinder** solves this problem by identifying the source scanner used to create a digital image.

Every scanner, due to its unique hardware, introduces microscopic and invisible *“fingerprints”* into an image. These include specific noise patterns, texture artifacts, and compression traces. This project uses machine learning to train models that recognize these unique signatures, allowing you to:

- Attribute a scanned document to a specific scanner model.  
- Detect forgeries where unauthorized scanners were used.  
- Verify the authenticity of scanned evidence in a forensic context.

> 🖼️ *(Your screenshot of the main app prediction interface would look great here!)*

---

## 🛠 Tech Stack

This project leverages a modern stack for machine learning, image processing, and web application delivery.

| Category | Technology | Purpose |
|-----------|-------------|----------|
| **Backend & ML** | **Python** | Core programming language |
| | **Scikit-learn** | Random Forest & SVM (Baseline Models) |
| | **Pandas** | Data manipulation and CSV handling |
| | **OpenCV** | Image processing (loading, color conversion, etc.) |
| | **NumPy** | numerical operations |
| | **TensorFlow / Keras** | For CNN Model |
| **Frontend & UI** | **Streamlit** | Creating the interactive web application |
| | **Matplotlib & Seaborn** | Data visualization (confusion matrix, plots) |
| | **Pillow (PIL)** | Displaying sample images in the UI |
| **Tooling** | **Git & GitHub** | Version control and source management |
| | **venv** | Python virtual environment management |

---

## ✨ Features

- 🧩 **Modular Feature Extraction:** Streamlit app to scan image directories, extract 10+ metadata features, and generate a feature CSV.  
- 📊 **Data Visualization:** View class distribution graphs, sample images from each class, and a full data preview.  
- 💾 **Downloadable Results:** Download the complete feature CSV directly from the app.  
- 🤖 **Baseline Model Pipeline:**  
  - **Train:** Build Random Forest and SVM models from the feature CSV.  
  - **Evaluate:** View detailed classification reports and confusion matrices for both models.  
  - **Predict:** Upload any image for instant scanner identification.  
- 🔀 **Dual Model Support:** Choose between Random Forest or SVM for your prediction.  
- 🧠 **Deep Learning Model:** Integration of a CNN for end-to-end image-based classification.

---

## 📸 Demo / Screenshots

Showcase your project! Add screenshots of your application in action.

1. **Main Prediction App**  
   *(src/models/baseline/Main_app.py)*  
   <!-- > Add screenshot of the prediction tab with an image uploaded. -->
   ![Main prediction app demo](./img/Main%20Prediction%20App.png)

2. **Feature Extraction App**  
   *(src/features/baseline/extractor_frontend.py)*  
   <!-- > Add screenshot of the feature extraction UI showing the graph and sample images. -->
   ![Feature extraction app demo](./img/Feature%20Extraction%20App.png)

3. **Model Evaluation Page**  
   *(Main_app.py - "Evaluate Models" Tab)*  
   <!-- > Add screenshot of the classification report and confusion matrix. -->
   ![Model evaluation demo](./img/Model%20Evaluation%20Page.png)

4. **Data Visualization Page**
   ![Data visualization demo](./img/Data%20Visualization%20Page.png)

---

## 🚀 Installation

Follow these steps to set up the project locally.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/PriyanshPorwal999/AI_ML_TraceFinder.git
cd AI_ML_TraceFinder





<!-- #  AI TraceFinder — Forensic Scanner Identification  

##  Overview  
AI TraceFinder is a forensic machine learning platform that identifies the **source scanner device** used to digitize a document or image. Each scanner (brand/model) introduces unique **noise, texture, and compression artifacts** that serve as a fingerprint. By analyzing these patterns, AI TraceFinder enables **fraud detection, authentication, and forensic validation** in scanned documents.  

---

##  Goals & Objectives  
- Collect and label scanned document datasets from multiple scanners  
- Robust preprocessing (resize, grayscale, normalize, denoise)  
- Extract scanner-specific features (noise, FFT, PRNU, texture descriptors)  
- Train classification models (ML + CNN)  
- Apply explainability tools (Grad-CAM, SHAP)  
- **Deploy an interactive app for scanner source identification**  
- Deliver **accurate, interpretable results** for forensic and legal use cases  

---

##  Methodology 
1. **Data Collection & Labeling**  
   - Gather scans from 3–5 scanner models/brands  
   - Create a structured, labeled dataset  

2. **Preprocessing**  
   - Resize, grayscale, normalize  
   - Optional: denoise to highlight artifacts  

3. **Feature Extraction**  
   - PRNU patterns, FFT, texture descriptors (LBP, edge features)  

4. **Model Training**  
   - Baseline ML: SVM, Random Forest, Logistic Regression  
   - Deep Learning: CNN with augmentation  

5. **Evaluation & Explainability**  
   - Metrics: Accuracy, F1-score, Confusion Matrix  
   - Interpretability: Grad-CAM, SHAP feature maps  

6. **Deployment**  
   - Streamlit app → upload scanned image → predict scanner model  
   - Display confidence score and key feature regions  

---

##  Actionable Insights for Forensics  
- **Source Attribution:** Identify which scanner created a scanned copy of a document.  
- **Fraud Detection:** Detect forgeries where unauthorized scanners were used.  
- **Legal Verification:** Validate whether scanned evidence originated from approved devices.  
- **Tamper Resistance:** Differentiate between authentic vs. tampered scans.  
- **Explainability:** Provide visual evidence of how classification was made.  

---

##  Architecture (Conceptual)  
Input ➜ Preprocessing ➜ Feature Extraction + Modeling ➜ Evaluation & Explainability ➜ Prediction App  

---

## ⏳ 8-Week Roadmap (Milestones)  
- **W1:** Dataset collection (min. 3–5 scanners), labeling, metadata analysis  
- **W2:** Preprocessing pipeline (resize, grayscale, normalize, optional denoise)  
- **W3:** Feature extraction (noise maps, FFT, LBP, texture descriptors)  
- **W4:** Baseline ML models (SVM, RF, Logistic Regression) + evaluation  
- **W5:** CNN model training with augmentation, hyperparameter tuning  
- **W6:** Model evaluation (accuracy, F1, confusion matrix) + Grad-CAM/SHAP analysis  
- **W7:** Streamlit app development → image upload, prediction, confidence output  
- **W8:** Final documentation, results, presentation, and demo handover  

---

##  Suggested Project Structure  
```bash
ai-tracefinder/
├─ app.py              
├─ src/
│  ├─ ingest/           
│  ├─ preprocess/        
│  ├─ features/          
│  ├─ models/            
│  ├─ explain/           
│  └─ utils/             
├─ data/                 
├─ notebooks/            
├─ reports/              
└─ README.md
``` -->


