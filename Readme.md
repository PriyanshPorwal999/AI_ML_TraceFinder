# 📊 AI TraceFinder: Forensic Scanner Identification

Detecting document forgery by analyzing a scanner's unique digital fingerprint.

---

## 📘 Table of Contents
- [About the Project](#-about-the-project)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Demo / Screenshots](#-demo--screenshots)
- [Accuracy & Performance](#-accuracy-&-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [Contact](#-contact)

---

## 🎯 About the Project

Scanned documents like legal agreements, official certificates, and financial records are easy to forge. It's often impossible to tell if a scanned document is legitimate or if it was created using an unauthorized, fraudulent device.

**AI TraceFinder** solves this problem by identifying the source scanner used to create a digital image.

Every scanner, due to its unique hardware, introduces microscopic and invisible *“fingerprints”* into an image. These include specific noise patterns, texture artifacts, and compression traces. This project uses machine learning to train models that recognize these unique signatures, allowing you to:

- Attribute a scanned document to a specific scanner model.  
- Detect forgeries where unauthorized scanners were used.  
- Verify the authenticity of scanned evidence in a forensic context.

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

1.  **Main Prediction App**
    ![Main prediction app demo](./img/Main%20Prediction%20App.png)

2.  **Feature Extraction App**
    ![Feature extraction app demo](./img/Feature%20Extraction%20App.png)

3.  **Model Evaluation Page**
    ![Model evaluation demo](./img/Model%20Evaluation%20Page.png)

4.  **Data Visualization Page**
    ![Data visualization demo](./img/Data%20Visualization%20Page.png)

---

## 📈 Accuracy & Performance
* **Hybrid CNN model test accuracy: 82.21%**
* **Overall weighted avg:** Precision 0.83, Recall 0.82, F1-score 0.82, Test sample(517 images)

---

## ⚙️ Installation

Follow these steps to set up the project locally.

### 1️. Clone the repository
```bash
git clone https://github.com/PriyanshPorwal999/AI_ML_TraceFinder.git
cd AI_ML_TraceFinder
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run src/app/Main_app.py
```
Your app should now be available at 👉 http://localhost:8501
---


### 🧪 Usage
- Launch the app using Streamlit.

- Upload a scanned image (JPG/PNG).

- The model predicts the scanner brand/model.

- View confidence score and residual noise visualization (scanner fingerprint).

### 🤝 Contributing

Contributions are welcome!

To contribute:

- Fork this repository
- Create a new branch (feature/your-feature-name)
- Commit your changes
- Push to the branch
- Open a Pull Request

### 🧑‍💻 Contact

Priyansh Porwal
📍 B.Tech CSE | JECRC Foundation
💼 AI & ML Enthusiast | Research & Development