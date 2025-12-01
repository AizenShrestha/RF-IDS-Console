🔐 Random Forest Intrusion Detection System (IDS)

This project is a research-focused Intrusion Detection System using a Random Forest classifier trained on flow-based features from the HIKARI-2021 dataset.
The system predicts whether network flows are benign or malicious and provides probability scores for each prediction.

🎯 Goal

To demonstrate a fully reproducible IDS pipeline with:

Real-world flow-based features

Class imbalance handling (SMOTE + class weights)

Feature preprocessing (imputation + scaling)

Threshold tuning

A working Streamlit detection console

Model explainability in notebooks (SHAP)

🏗️ Project Structure (matches this repository)
RF-IDS-Console/
│
├── appFinal.py                         # Streamlit IDS console (no SHAP)
│
├── models/
│   ├── rf_smote70_classweight.joblib
│   └── rf_threshold.json
│
├── dataset/
│   └── processed/
│       ├── preprocessor.joblib
│       ├── training_features.txt
│       └── y_test.csv / y_train_bal_70_30.csv
│
├── notebooks/
│   ├── 01_eda_HIKARI2021.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_SHAP_Explainability.ipynb   # Explainability is in notebooks only
│
├── assets/
│
└── README.md

📦 Installation
git clone https://github.com/AizenShrestha/RF-IDS-Console.git
cd RF-IDS-Console
pip install -r requirements.txt

▶️ Running the IDS Console

Start the Streamlit detection app:

streamlit run appFinal.py


You can upload either:

🔹 Raw flow CSV

Must contain all required flow features used during training.

🔹 Preprocessed numeric CSV/XLS

Directly compatible with the saved scaler (preprocessor.joblib).

The app will automatically:

Validate feature names

Apply the same preprocessing pipeline (imputer + scaler)

Run Random Forest predictions

Return class + probability

SHAP is not part of the app.
Explainability is provided exclusively in the training notebooks.

🔬 Reproducing the Training Pipeline

Open the notebooks in this order:

1️⃣ 01_eda_HIKARI2021.ipynb – Dataset inspection and structure
2️⃣ 02_preprocessing.ipynb – Scaling, SMOTE, feature prep
3️⃣ 03_model_development.ipynb – Train RF + tune threshold + SHAP results

SHAP results help understand feature importance but are not used in the deployment app.

⚠️ Dataset Notice

The full HIKARI-2021 dataset is not included due to licensing and size constraints.
A small sample is included only for demonstration.

🧾 Reproducibility Statement

This repository includes:

The complete training workflow

The exact preprocessing pipeline

Trained model + tuned threshold

Feature schema used during deployment

A reliable Streamlit detection console

Notebook-level explainability (SHAP)

All experiments and predictions can be reproduced by running the provided notebooks and application.
