🔐 Random Forest Intrusion Detection System (IDS)

This project is a small, research-focused Intrusion Detection System built using a Random Forest classifier trained on flow-based network traffic (HIKARI-2021).
It can predict whether network flows are benign or malicious and gives a probability score for each prediction.

🎯 Goal:
To demonstrate a reproducible IDS pipeline with real-world flow statistics, class imbalance handling, and model explainability.

🚀 What This IDS Can Do

✔ Detects attacks using only network flow statistics
✔ Works on raw flow data (if features match the training format)
✔ Accepts preprocessed numeric test data
✔ Comes with a working Streamlit prototype

🏗️ Project Structure
RF_IDS_Console/
│
├── app/                  # Streamlit web prototype
│   └── app.py
│
├── models/               # Trained model + tuned threshold
│   └── rf_smote70_classweight.joblib
│   └── rf_threshold.json
│
├── dataset/
│   ├── raw/              # Example flow CSV (small sample only)
│   └── processed/
│       └── preprocessor.joblib
│
├── notebooks/            # Reproducible training workflow
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_SHAP_Explainability.ipynb
│
├── results/              # Performance reports + graphs
│
├── requirements.txt
└── README.md

📦 Installation

Make sure Python 3.x is installed, then run:

git clone <your-repository-link>
cd RF_IDS_Project
pip install -r requirements.txt

▶️ Running the IDS Prototype

Start the Streamlit app:
streamlit run app/app.py


Then upload either:

🔹 Raw flow CSV (must follow the same feature format used during training)
🔹 Preprocessed numeric CSV/XLS from the training notebooks

The app will automatically:

Validate features

Apply the same scaler used in training

Predict and show attack probabilities

🔬 Reproducing the Training Process

Open the notebooks in this order:

1️⃣ 01_EDA.ipynb – Explore / inspect data
2️⃣ 02_Preprocessing.ipynb – SMOTE, scaling, feature selection
3️⃣ 03_Model_Training.ipynb – Train Random Forest + tune threshold
4️⃣ 04_SHAP_Explainability.ipynb – Understand feature importance

The saved model and scaler are automatically exported to /models and /dataset/processed.

⚠️ Dataset Notice

The full HIKARI-2021 dataset is not included due to licensing and size limits.
A small sample is provided for testing.
You can download the full dataset from its official source if needed for retraining.

🧾 Reproducibility Statement

This repository contains:

The complete training pipeline

Exported preprocessing scaler

Saved model + tuned threshold

Same feature schema used during deployment

A working prototype

All experiments and predictions can be reproduced exactly by running the notebooks and app included in this project.
