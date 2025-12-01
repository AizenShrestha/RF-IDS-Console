import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

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
X_TEST_FILE = PROC_DIR / "X_test_proc.npy"
Y_TEST_FILE = PROC_DIR / "y_test.csv"


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


def load_threshold(default: float = 0.4) -> float:
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
        raise ValueError(
            "❌ No numeric flow features detected. "
            "Please upload data with numeric flow statistics (e.g. counts, rates, durations)."
        )

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
# 4) VALIDATION ON HELD-OUT HIKARI TEST SET
# ================================================================
def run_validation_page(model, threshold: float):
    st.header("📊 Model Validation – HIKARI-2021 Test Split")

    st.write(
        "This view evaluates the trained Random Forest IDS on the held-out "
        "HIKARI-2021 test split used in the dissertation. It is designed to "
        "reproduce the offline metrics reported in the evaluation chapter."
    )

    if not X_TEST_FILE.exists() or not Y_TEST_FILE.exists():
        st.error(
            "Test split files not found. Expected:\n"
            f"- {X_TEST_FILE}\n"
            f"- {Y_TEST_FILE}\n"
            "Please ensure the processed test data is available in the project directory."
        )
        return

    if st.button("▶ Run validation on HIKARI test data"):
        try:
            X_test = np.load(X_TEST_FILE)
            y_test = pd.read_csv(Y_TEST_FILE).squeeze()
        except Exception as e:
            st.error(f"Failed to load test data: {e}")
            return

        with st.spinner("Evaluating model on HIKARI-2021 test split..."):
            proba_attack = model.predict_proba(X_test)[:, 1]
            y_pred = (proba_attack >= threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, labels=[0, 1], zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        st.subheader("Overall performance")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Test samples", len(y_test))

        st.write("Per-class precision, recall and F1-score:")
        metrics_df = pd.DataFrame(
            {
                "Class": ["Benign (0)", "Attack (1)"],
                "Precision": precision.round(4),
                "Recall": recall.round(4),
                "F1-score": f1.round(4),
                "Support": support,
            }
        )
        st.table(metrics_df)

        st.write("Confusion matrix (rows = true class, columns = predicted class):")
        cm_df = pd.DataFrame(
            cm,
            index=["True Benign (0)", "True Attack (1)"],
            columns=["Pred Benign (0)", "Pred Attack (1)"],
        )
        st.table(cm_df)

        st.text("Classification report (for reference):")
        st.code(classification_report(y_test, y_pred, digits=4), language="text")


# ================================================================
# 5) STREAMLIT USER INTERFACE – DETECTION CONSOLE
# ================================================================
def run_detection_page(model, preproc, threshold: float):
    st.header("🔍 Detection Console – Flow-Based IDS")

    st.markdown(
        "This console applies the trained Random Forest model to new network flows. "
        "The IDS operates on flow-level statistics (counts, rates, timings) and does "
        "not inspect packet payloads."
    )

    st.markdown("#### 1. Upload flow data (CSV / XLS / XLSX)")
    st.write(
        "You can upload either:\n"
        "• **Raw flow statistics** using the same feature schema as during training, or\n"
        "• **Preprocessed numeric test data** exported from the training pipeline.\n\n"
        "Label columns (e.g. `Label`, `traffic_category`) should be removed before upload. "
    )

    uploaded = st.file_uploader("Choose a flow data file", type=["csv", "xls", "xlsx"])

    if uploaded is None:
        st.info("Upload a CSV or Excel file to analyse network flows.")
        return

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

    st.markdown("#### 2. Preview of uploaded data")
    st.dataframe(df.head())

    if st.button("🚀 Run intrusion detection"):
        with st.spinner("Scoring flows with the Random Forest IDS model..."):
            try:
                result_df, dropped_cols, mode = make_predictions(
                    df, model, preproc, threshold
                )
            except Exception as e:
                st.error(e)
                return

        # Mode banner
        if mode == "RAW FLOW FEATURES":
            st.success(
                "Mode: **RAW FLOW FEATURES** — columns were aligned, ordered and scaled "
                "using the stored preprocessing pipeline."
            )
        else:
            st.info(
                "Mode: **PREPROCESSED NUMERIC INPUT** — the file already matches the "
                "training feature layout, so only the classifier was applied."
            )

        st.caption(
            "Raw mode expects the same flow feature names as in training. "
            "Preprocessed mode is intended for exported test vectors from the project."
        )

        # --------------------------------------------------------
        # Per-flow predictions
        # --------------------------------------------------------
        st.markdown("### 3. Per-flow intrusion decisions")
        merged = df.copy()
        merged["attack_probability"] = result_df["attack_probability"]
        merged["predicted_label"] = result_df["predicted_label"]
        merged["predicted_class"] = result_df["predicted_class"]
        st.dataframe(merged.head(50))

        # --------------------------------------------------------
        # Summary dashboard
        # --------------------------------------------------------
        st.markdown("### 4. Traffic summary and attack distribution")

        counts = (
            result_df["predicted_class"]
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Count")
        )
        counts["Percent"] = (counts["Count"] / len(result_df) * 100).round(2)

        total_flows = len(result_df)
        attack_count = int(counts.loc[counts["Class"] == "Attack", "Count"].sum())
        benign_count = int(counts.loc[counts["Class"] == "Benign", "Count"].sum())
        attack_rate = (attack_count / total_flows * 100) if total_flows > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total flows processed", total_flows)
        c2.metric("Predicted attacks", attack_count)
        c3.metric("Attack rate (%)", f"{attack_rate:.2f}")

        # Simple qualitative risk indicator based on attack rate
        if attack_rate == 0:
            st.success("Risk level: Low – no attacks detected in this batch of flows.")
        elif attack_rate < 5:
            st.info("Risk level: Moderate – a small fraction of flows are flagged as attacks.")
        else:
            st.warning(
                "Risk level: Elevated – a noticeable proportion of flows are flagged as attacks. "
                "Further investigation would be recommended in a live environment."
            )

        st.table(counts)
        st.bar_chart(counts.set_index("Class")["Count"])

        # Inform about ignored non-numeric columns
        if dropped_cols:
            st.info(
                "The following non-numeric columns were ignored during prediction:\n"
                + ", ".join(dropped_cols)
            )

        st.success(
            f"Finished scoring **{total_flows}** flows — 🚨 **{attack_count}** predicted as attacks."
        )

    # ------------------------------------------------------------
    # About section
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### About this prototype")
    st.write(
        "This interface is a research prototype for a flow-based Intrusion Detection System (IDS). "
        "It uses a Random Forest model trained on the HIKARI-2021 dataset and applies the same "
        "preprocessing pipeline (feature alignment and scaling) used during training. "
        "The goal is to demonstrate a reproducible IDS workflow rather than a production UI."
    )


# ================================================================
# 6) MAIN APP ENTRY POINT
# ================================================================
def main():
    st.set_page_config(
        page_title="Random Forest IDS Console",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Random Forest Intrusion Detection")
    st.caption(
        "Flow-based Intrusion Detection System (IDS) trained on the HIKARI-2021 dataset. "
        "The console supports both offline validation and interactive analysis of network flows."
    )

    # Load artefacts once
    try:
        model = load_model()
        preproc = load_preprocessor()
        threshold = load_threshold(0.5)
    except Exception as e:
        st.error(e)
        st.stop()

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select view",
        ("🔍 Detection (Upload Data)", "📊 Validation (HIKARI Test Split)"),
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Model Information")
    st.sidebar.write(f"**Classifier:** `{MODEL_FILE.name}`")
    st.sidebar.write(f"**Preprocessor:** `{PREPROC_FILE.name}`")
    st.sidebar.write(f"**Decision threshold:** `{threshold:.2f}`")

    if page.startswith("🔍"):
        run_detection_page(model, preproc, threshold)
    else:
        run_validation_page(model, threshold)


if __name__ == "__main__":
    main()
