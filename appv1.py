import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ================================================================
# 1) MODEL + PREPROCESSOR FILE LOCATIONS
# ================================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback when running in environments without __file__ (e.g. some notebooks)
    BASE_DIR = Path.cwd()

MODELS_DIR = BASE_DIR / "models"
PROC_DIR = BASE_DIR / "dataset" / "processed"

MODEL_FILE = MODELS_DIR / "rf_smote70_classweight.joblib"
THRESH_FILE = MODELS_DIR / "rf_threshold.json"
PREPROC_FILE = PROC_DIR / "preprocessor.joblib"


# ================================================================
# 2) LOAD ARTEFACTS (MODEL + PREPROCESSOR + THRESHOLD)
# ================================================================
@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_preprocessor():
    if not PREPROC_FILE.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROC_FILE}")
    return joblib.load(PREPROC_FILE)


def load_threshold(default: float = 0.5) -> float:
    """
    Load tuned decision threshold if available, else fall back to default.
    The threshold is stored separately so tuning does not change model weights.
    """
    if THRESH_FILE.exists():
        try:
            with open(THRESH_FILE, "r") as f:
                data = json.load(f)
            return float(data.get("best_threshold", default))
        except Exception:
            # If JSON is corrupted or missing key, fall back safely
            return default
    return default


# ================================================================
# 3) PREDICTION PIPELINE (RAW + PREPROCESSED SUPPORT)
# ================================================================
def make_predictions(df_raw: pd.DataFrame, model, preproc, threshold: float):
    """
    df_raw:
        DataFrame uploaded by the user (may contain raw flow features,
        identifiers, or already-preprocessed numeric vectors).

    Returns:
        result_df: DataFrame with attack_probability, predicted_label, predicted_class.
        dropped_cols: list of non-numeric columns that were ignored.
        mode: string describing how the data was interpreted.
    """

    # Keep original for later merging with predictions
    df_original = df_raw.copy()

    # 1) Keep only numeric columns (drop IPs, hostnames, timestamps, etc.)
    df_numeric = df_raw.select_dtypes(include=[np.number])
    dropped_cols = [c for c in df_original.columns if c not in df_numeric.columns]

    if df_numeric.shape[1] == 0:
        raise ValueError("❌ Error: No numeric flow features detected. Upload numeric flow data only.")

    # 2) Retrieve the exact feature schema used during training from the preprocessor
    if not hasattr(preproc, "feature_names_in_"):
        raise ValueError("❌ Preprocessor does not expose 'feature_names_in_'. Cannot verify schema.")

    expected_cols = list(preproc.feature_names_in_)
    required = set(expected_cols)
    present = set(df_numeric.columns)

    missing_cols = list(required - present)
    overlap = present & required

    # ------------------------------------------------------------
    # RAW MODE:
    # All expected training feature names are present in the upload
    # ------------------------------------------------------------
    if len(missing_cols) == 0:
        # Reorder numeric columns to match training order exactly
        df_numeric = df_numeric[expected_cols]

        # Reject missing values in required features
        if df_numeric.isnull().any().any():
            bad_cols = df_numeric.columns[df_numeric.isnull().any()].tolist()
            raise ValueError(
                "❌ Missing values detected in required flow features.\n\n"
                "⚠ Affected columns:\n" + "\n".join(f"- {c}" for c in bad_cols) +
                "\n\nFlow statistics must be complete to ensure reliable intrusion detection. "
                "Please clean or re-extract the flows using a feature pipeline compatible "
                "with the training data."
            )

        # Apply the same preprocessing as during training
        X_proc = preproc.transform(df_numeric)
        mode = "RAW FLOW FEATURES"

    # ------------------------------------------------------------
    # PARTIAL RAW:
    # Some expected features found, but not all → reject as unsafe
    # ------------------------------------------------------------
    elif len(overlap) > 0:
        raise ValueError(
            "❌ Raw flow data detected but rejected because some required features are missing.\n\n"
            "⚠ Missing feature columns:\n" + "\n".join(f"- {c}" for c in missing_cols) +
            "\n\nTo classify raw traffic safely, the feature extraction pipeline must produce "
            "ALL required flow metrics used during training. Otherwise, predictions would be unreliable."
        )

    # ------------------------------------------------------------
    # PREPROCESSED NUMERIC MODE:
    # No overlap with training feature names → assume already transformed vectors
    # ------------------------------------------------------------
    else:
        # Reject NaNs in numeric vectors
        if df_numeric.isnull().any().any():
            bad_cols = df_numeric.columns[df_numeric.isnull().any()].tolist()
            raise ValueError(
                "❌ Missing values detected in numeric input.\n\n"
                "⚠ Affected columns:\n" + "\n".join(f"- {c}" for c in bad_cols) +
                "\n\nPreprocessed numeric vectors must not contain NaN values. "
                "Please ensure the data is exported correctly from the training pipeline."
            )

        # Check dimensionality matches model expectations
        n_features_model = getattr(model, "n_features_in_", None)
        if n_features_model is not None and df_numeric.shape[1] != n_features_model:
            raise ValueError(
                f"❌ Incorrect numeric input shape.\n\n"
                f"Expected **{n_features_model}** features but received **{df_numeric.shape[1]}**.\n\n"
                "Upload either:\n"
                " • A complete raw flow CSV with the same feature schema as training, or\n"
                " • Preprocessed numeric test vectors exported from the training pipeline."
            )

        X_proc = df_numeric.to_numpy()
        mode = "PREPROCESSED NUMERIC INPUT"

    # 3) Predict with the Random Forest model
    proba_attack = model.predict_proba(X_proc)[:, 1]
    y_pred = (proba_attack >= threshold).astype(int)
    pred_class = ["Attack" if v == 1 else "Benign" for v in y_pred]

    # 4) Build result DataFrame
    result = pd.DataFrame(
        {
            "attack_probability": proba_attack,
            "predicted_label": y_pred,
            "predicted_class": pred_class,
        },
        index=df_original.index,
    )

    return result, dropped_cols, mode


# ================================================================
# 4) STREAMLIT USER INTERFACE
# ================================================================
def main():
    st.set_page_config(page_title="Random Forest IDS", layout="wide")
    st.title("🔐 Random Forest Intrusion Detection Prototype")

    # Load artefacts
    try:
        model = load_model()
        preproc = load_preprocessor()
        threshold = load_threshold(0.5)
    except Exception as e:
        st.error(e)
        st.stop()

    # Sidebar info
    st.sidebar.header("📌 Model Information")
    st.sidebar.write(f"**Classifier:** `{MODEL_FILE.name}`")
    st.sidebar.write(f"**Preprocessor:** `{PREPROC_FILE.name}`")
    st.sidebar.write(f"**Decision Threshold:** `{threshold:.2f}`")

    # Upload instructions
    st.markdown("### 📤 Upload Flow Data (CSV / XLS / XLSX)")
    st.write(
        "Upload either:\n"
        "• **Raw network flow data** (flow features + optional identifiers such as IPs or timestamps; no label column), or\n"
        "• **Preprocessed numeric test data** generated from the training pipeline.\n\n"
        "Label columns such as `Label` or `traffic_category` should be removed before upload. "
        "During deployment, the IDS operates on unlabeled traffic."
    )

    uploaded = st.file_uploader("Select a file", type=["csv", "xls", "xlsx"])

    if uploaded is not None:
        # Read CSV or Excel safely with fallback
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                try:
                    df = pd.read_excel(uploaded)
                except Exception:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ File read error: {e}")
            return

        st.write("🔎 **Preview of uploaded data:**")
        st.dataframe(df.head())

        if st.button("🚀 Run IDS Prediction"):
            try:
                result_df, dropped_cols, mode = make_predictions(df, model, preproc, threshold)
            except Exception as e:
                st.error(e)
                return

            # Mode banner
            if mode == "RAW FLOW FEATURES":
                st.success("🔧 Mode Detected: **RAW FLOW FEATURES**")
            else:
                st.info("🔧 Mode Detected: **PREPROCESSED NUMERIC INPUT**")

            st.caption(
                "Raw mode: automatic preprocessing is applied (feature alignment, ordering, scaling).  \n"
                "Preprocessed mode: input is assumed to be already transformed using the training pipeline."
            )

            # Show predictions
            st.markdown("### 🧠 Prediction Results")
            merged = df.copy()
            merged["attack_probability"] = result_df["attack_probability"]
            merged["predicted_label"] = result_df["predicted_label"]
            merged["predicted_class"] = result_df["predicted_class"]
            st.dataframe(merged.head(50))

            # Inform about ignored non-numeric columns
            if dropped_cols:
                st.info(
                    "ℹ The following non-numeric columns were ignored during prediction:\n"
                    + ", ".join(dropped_cols)
                )

            # Summary statistics
            st.markdown("### 📊 Summary of Predictions")
            counts = (
                result_df["predicted_class"]
                .value_counts()
                .rename_axis("Class")
                .reset_index(name="Count")
            )
            counts["Percent"] = (counts["Count"] / len(result_df) * 100).round(2)
            st.table(counts)
            st.bar_chart(counts.set_index("Class")["Count"])

            attacks = counts.loc[counts["Class"] == "Attack", "Count"].sum()
            st.success(
                f"Processed **{len(result_df)}** flows — 🚨 **{int(attacks)}** predicted as attacks."
            )

    else:
        st.info("📎 Upload a CSV or Excel file to start using the IDS prototype.")


if __name__ == "__main__":
    main()
