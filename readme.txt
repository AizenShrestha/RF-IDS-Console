🔐 Random Forest Intrusion Detection System (IDS)

This project is a research-focused Intrusion Detection System built using a Random Forest classifier trained on flow-based features from the HIKARI-2021 dataset.
The system predicts whether network flows are benign or malicious and provides probability scores for transparency.

🎯 Goal

To demonstrate a reproducible IDS workflow, including:

Flow-based feature preprocessing

Training-time class imbalance handling (SMOTE + class weights)

Threshold tuning for sensitivity control

A lightweight, functional Streamlit detection console

Model explainability in notebooks (SHAP)

Note: Class imbalance handling happens during training only.
The deployed Streamlit app loads the trained model and does not re-balance data.

🏗️ Project Structure (matches this repository)
RF-IDS-Console/
│
├── appFinal.py                     # Streamlit IDS console (inference only)
│
├── models/
│   ├── rf_smote70_classweight.joblib
│   └── rf_threshold.json
│
├── dataset/
│   └── processed/
│       ├── preprocessor.joblib     # Imputer + scaler used during training
│       ├── training_features.txt
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_eda_HIKARI2021.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│
├── assets/
│
└── README.md


Raw datasets and large intermediates are intentionally not included.

📦 Installation
git clone https://github.com/AizenShrestha/RF-IDS-Console.git
cd RF-IDS-Console
pip install -r requirements.txt

▶️ Running the IDS Console

Start the Streamlit detection interface:

streamlit run appFinal.py


You can upload:

🔹 Raw flow CSV

Must contain the exact flow feature names used during training.

🔹 Preprocessed numeric data

Directly compatible with the saved scaler (preprocessor.joblib).

The app will:

Validate feature structure

Apply the training-time preprocessing pipeline

Predict using the trained Random Forest

Return attack/benign labels with probabilities

Apply your selected detection threshold

The app does NOT perform SMOTE or balancing.
All balancing occurs offline during training inside the notebooks.

🔬 Reproducing the Training Workflow (offline only)

Open notebooks in this order:

1️⃣ 01_eda_HIKARI2021.ipynb
2️⃣ 02_preprocessing.ipynb — scaling, SMOTE, feature prep
3️⃣ 03_model_development.ipynb — train RF, evaluate, tune threshold, SHAP results


The exported model and preprocessing artifacts are used by the Streamlit app.

⚠️ Dataset Notice

The full HIKARI-2021 dataset is not included due to licensing and size.
A small example subset is provided for testing the prototype.

🧾 Reproducibility Statement

This repository includes:

Full training notebooks

The exact preprocessing pipeline used during inference

The trained Random Forest model + tuned threshold

Feature schema used during training & deployment

A deterministic Streamlit inference console

Offline SHAP explainability (not part of the app UI)
