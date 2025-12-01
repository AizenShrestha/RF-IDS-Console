import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ================================================================
# 1) PROJECT PATHS AND ARTEFACT LOCATIONS
# ================================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback when running from interactive / notebook environments
    BASE_DIR = Path.cwd()

MODELS_DIR = BASE_DIR / "models"
PROC_DIR = BASE_DIR / "dataset" / "processed"

MODEL_FILE = MODELS_DIR / "rf_smote70_classweight.joblib"
PREPROC_FILE = PROC_DIR / "preprocessor.joblib"

# Processed HIKARI test split (used only in validation view)
X_TEST_FILE = PROC_DIR / "X_test_proc.npy"
Y_TEST_FILE = PROC_DIR / "y_test.csv"

# Fixed operating threshold for the detection console
# Chosen as a "balanced" sensitivity mode from validation experiments
DETECTION_THRESHOLD = 0.60


# ================================================================
# 2) LOADING MODEL + PREPROCESSOR
# ================================================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest classifier from disk."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_preprocessor():
    """Load the fitted preprocessing pipeline used during training."""
    if not PREPROC_FILE.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROC_FILE}")
    return joblib.load(PREPROC_FILE)


# ================================================================
# 3) PREDICTION PIPELINE (RAW + PREPROCESSED SUPPORT)
# ================================================================
def make_predictions(df_raw: pd.DataFrame, model, preproc, threshold: float):
    """
    Core prediction helper used by the detection console.

    df_raw:
        DataFrame uploaded by the user. It may contain:
        - raw flow features (+ identifiers like IPs, timestamps), or
        - already-preprocessed numeric vectors.

    Returns:
        result_df     : DataFrame with attack_probability, predicted_label, predicted_class.
        dropped_cols  : list of non-numeric columns that were ignored.
        mode          : "RAW FLOW FEATURES" or "PREPROCESSED NUMERIC INPUT".
    """

    # Keep original for merging with predictions later
    df_original = df_raw.copy()

    # 1) Remove non-numeric columns (IPs, hostnames, timestamps, etc.)
    df_numeric = df_raw.select_dtypes(include=[np.number])
    dropped_cols = [c for c in df_original.columns if c not in df_numeric.columns]

    if df_numeric.shape[1] == 0:
        raise ValueError(
            "❌ Error: No numeric flow features detected.\n"
            "Please upload a file containing numeric flow statistics."
        )

    # 2) Retrieve the feature schema used during training
    if not hasattr(preproc, "feature_names_in_"):
        raise ValueError(
            "❌ Preprocessor does not expose 'feature_names_in_'.\n"
            "Cannot verify feature schema against the training pipeline."
        )

    expected_cols = list(preproc.feature_names_in_)
    required = set(expected_cols)
    present = set(df_numeric.columns)

    missing_cols = list(required - present)
    overlap = present & required

    # ------------------------------------------------------------
    # RAW FLOW MODE:
    #   All training feature names are present → treat as raw HIKARI-style flows
    # ------------------------------------------------------------
    if len(missing_cols) == 0:
        # Reorder numeric columns to match training order exactly
        df_numeric = df_numeric[expected_cols]

        # Reject missing values in required features
        if df_numeric.isnull().any().any():
            bad_cols = df_numeric.columns[df_numeric.isnull().any()].tolist()
            raise ValueError(
                "❌ Missing values detected in required flow features.\n\n"
                "⚠ Affected columns:\n"
                + "\n".join(f"- {c}" for c in bad_cols)
                + "\n\nFlows must be complete to ensure reliable detection. "
                  "Please clean or re-extract the flow statistics."
            )

        # Apply the same preprocessing as during training
        X_proc = preproc.transform(df_numeric)
        mode = "RAW FLOW FEATURES"

    # ------------------------------------------------------------
    # PARTIAL RAW MODE:
    #   Some required features present but not all → reject as unsafe
    # ------------------------------------------------------------
    elif len(overlap) > 0:
        raise ValueError(
            "❌ Raw flow data detected but rejected because some required "
            "features from the training pipeline are missing.\n\n"
            "⚠ Missing feature columns:\n"
            + "\n".join(f"- {c}" for c in missing_cols)
            + "\n\nTo classify raw traffic safely, the feature extraction step must "
              "produce the full set of flow metrics used during training."
        )

    # ------------------------------------------------------------
    # PREPROCESSED NUMERIC MODE:
    #   No overlap with training feature names → assume numeric vectors
    # ------------------------------------------------------------
    else:
        # Reject NaNs in numeric vectors
        if df_numeric.isnull().any().any():
            bad_cols = df_numeric.columns[df_numeric.isnull().any()].tolist()
            raise ValueError(
                "❌ Missing values detected in numeric input.\n\n"
                "⚠ Affected columns:\n"
                + "\n".join(f"- {c}" for c in bad_cols)
                + "\n\nPreprocessed numeric vectors must be exported cleanly from the "
                  "training pipeline (no NaNs)."
            )

        # Check dimensionality against model expectation
        n_features_model = getattr(model, "n_features_in_", None)
        if n_features_model is not None and df_numeric.shape[1] != n_features_model:
            raise ValueError(
                "❌ Incorrect numeric input shape.\n\n"
                f"Expected **{n_features_model}** features but received "
                f"**{df_numeric.shape[1]}**.\n\n"
                "Upload either:\n"
                " • A raw HIKARI-style flow CSV with the full feature set, or\n"
                " • Preprocessed numeric vectors exported from the training notebook."
            )

        X_proc = df_numeric.to_numpy()
        mode = "PREPROCESSED NUMERIC INPUT"

    # 3) Predict probabilities and apply decision threshold
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
# 4) VALIDATION HELPER – HIKARI-2021 TEST SPLIT
# ================================================================
@st.cache_data
def load_hikari_test():
    """Load preprocessed HIKARI-2021 test split for validation."""
    if not X_TEST_FILE.exists():
        raise FileNotFoundError(f"X_test file not found at {X_TEST_FILE}")
    if not Y_TEST_FILE.exists():
        raise FileNotFoundError(f"y_test file not found at {Y_TEST_FILE}")

    X_test = np.load(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).squeeze().to_numpy()
    return X_test, y_test


def evaluate_on_hikari(model, threshold: float):
    """Evaluate model on the stored HIKARI-2021 test split."""
    X_test, y_test = load_hikari_test()
    proba_attack = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_attack >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_text = classification_report(
        y_test, y_pred, target_names=["Benign (0)", "Attack (1)"], digits=4, zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "supports": supports,
        "confusion_matrix": cm,
        "report_text": report_text,
        "n_samples": len(y_test),
    }
    return metrics


# ================================================================
# 5) DETECTION VIEW (UPLOAD + PREDICT, FIXED 0.60 THRESHOLD)
# ================================================================
def detection_view(model, preproc):
    st.subheader("📡 Detection Console – Flow-Based IDS")

    st.markdown(
        "This console applies the trained Random Forest model to **new network flows**. "
        "The IDS operates on flow-level statistics (counts, rates, timings) and does "
        "not inspect packet payloads."
    )

    st.markdown("### IDS sensitivity for live detection")

    st.write(
        f"**Decision threshold used:** `{DETECTION_THRESHOLD:.2f}`  "
        "(flows with attack probability above this value are classified as attacks)."
    )

    # Short explanation of what this threshold means
    if DETECTION_THRESHOLD < 0.45:
        st.info(
            "⚠️ High Sensitivity Mode – maximises recall (very few attacks missed) "
            "but increases false positives."
        )
    elif 0.45 <= DETECTION_THRESHOLD < 0.55:
        st.info(
            "ℹ️ Default Mode – standard decision point based on the training distribution."
        )
    else:
        st.success(
            "✔️ Balanced Mode (recommended) – reduces false alerts while maintaining "
            "high recall for attacks."
        )

    st.markdown("### 1. Upload flow data (CSV / XLS / XLSX)")
    st.write(
        "You can upload either:\n"
        "• **Raw flow statistics** using the same feature schema as during training, or\n"
        "• **Preprocessed numeric test data** exported from the training pipeline.\n\n"
        "Label columns such as `Label` or `traffic_category` should be removed before upload."
    )

    uploaded = st.file_uploader("Choose a flow data file", type=["csv", "xls", "xlsx"])

    if uploaded is None:
        st.info("📎 Upload a CSV or Excel file to analyse network flows.")
        return

    # Read CSV or Excel safely
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
            result_df, dropped_cols, mode = make_predictions(
                df, model, preproc, threshold=DETECTION_THRESHOLD
            )
        except Exception as e:
            st.error(e)
            return

        # Mode banner
        if mode == "RAW FLOW FEATURES":
            st.success(
                "🔧 Mode detected: **RAW FLOW FEATURES** – feature alignment and scaling "
                "applied using the stored preprocessing pipeline."
            )
        else:
            st.info(
                "🔧 Mode detected: **PREPROCESSED NUMERIC INPUT** – input assumed to already "
                "match the training feature layout."
            )

        st.caption(
            "Raw mode: automatic preprocessing is applied (feature alignment, ordering, scaling).  \n"
            "Preprocessed mode: input is assumed to already follow the training pipeline."
        )

        # Per-flow predictions
        st.markdown("### 🧠 Prediction results")
        merged = df.copy()
        merged["attack_probability"] = result_df["attack_probability"]
        merged["predicted_label"] = result_df["predicted_label"]
        merged["predicted_class"] = result_df["predicted_class"]
        st.dataframe(merged.head(50))

        if dropped_cols:
            st.info(
                "ℹ The following non-numeric columns were ignored during prediction:\n"
                + ", ".join(dropped_cols)
            )

        # Summary statistics
        st.markdown("### 📊 Summary of predictions")
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


# ================================================================
# 6) VALIDATION VIEW (HIKARI-2021 TEST SPLIT WITH THRESHOLD SLIDER)
# ================================================================
def validation_view(model):
    st.subheader("🧪 Model Validation – HIKARI-2021 Test Split")

    st.write(
        "This view evaluates the trained Random Forest IDS on the held-out "
        "**HIKARI-2021** test split used in the dissertation. It reproduces the "
        "offline metrics reported in the evaluation chapter and lets you explore "
        "how the decision threshold affects recall and precision."
    )

    st.markdown("### Threshold selection")

    threshold = st.slider(
        "Decision threshold (higher = fewer alerts, lower = more alerts)",
        min_value=0.10,
        max_value=0.90,
        value=DETECTION_THRESHOLD,
        step=0.01,
    )

    # Mode description based on threshold
    if threshold < 0.45:
        st.info(
            "⚠️ High Sensitivity Mode – maximises recall for the attack class "
            "(very few intrusions missed) but increases false positives."
        )
    elif 0.45 <= threshold < 0.55:
        st.info(
            "ℹ️ Default Mode – reflects the natural decision point of the model given "
            "the training distribution."
        )
    else:
        st.success(
            "✔️ Balanced Mode – reduces false alerts while still maintaining high "
            "recall for attacks. This is a realistic IDS operating point."
        )

    if st.button("▶ Run validation on HIKARI test data"):
        try:
            metrics = evaluate_on_hikari(model, threshold)
        except Exception as e:
            st.error(e)
            return

        st.markdown("### Overall performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Test samples", f"{metrics['n_samples']}")
        with col3:
            st.metric("Threshold used", f"{threshold:.2f}")

        # Per-class precision / recall / F1
        st.markdown("### Per-class precision, recall and F1-score")
        df_prf = pd.DataFrame(
            {
                "Class": ["Benign (0)", "Attack (1)"],
                "Precision": metrics["precisions"],
                "Recall": metrics["recalls"],
                "F1-score": metrics["f1s"],
                "Support": metrics["supports"],
            }
        )
        df_prf["Precision"] = df_prf["Precision"].round(4)
        df_prf["Recall"] = df_prf["Recall"].round(4)
        df_prf["F1-score"] = df_prf["F1-score"].round(4)
        st.table(df_prf)

        st.info(
            "For an IDS, **high recall for the attack class** is usually prioritised "
            "over precision, because missing an intrusion is more harmful than "
            "raising extra alerts. The threshold slider allows this trade-off to be "
            "explored explicitly."
        )

        # Confusion matrix
        st.markdown("### Confusion matrix (rows = true class, columns = predicted class)")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["True Benign (0)", "True Attack (1)"],
            columns=["Pred Benign (0)", "Pred Attack (1)"],
        )
        st.table(cm_df)

        # Full classification report (text form)
        st.markdown("### Classification report (text)")
        st.code(metrics["report_text"])


# ================================================================
# 7) MAIN APP ENTRYPOINT
# ================================================================
def main():
    st.set_page_config(
        page_title="Random Forest Intrusion Detection Console",
        layout="wide",
    )

    st.title("🛡️ Random Forest Intrusion Detection Console")
    st.caption(
        "Flow-based Intrusion Detection System (IDS) trained on the HIKARI-2021 dataset. "
        "The console supports **live detection** on uploaded flow data and **offline "
        "validation** on the held-out test split."
    )

    # Load artefacts once
    try:
        model = load_model()
        preproc = load_preprocessor()
    except Exception as e:
        st.error(e)
        st.stop()

    # Sidebar – navigation and model info
    st.sidebar.title("Navigation")
    view = st.sidebar.radio(
        "Select view",
        options=[
            "Detection (Upload Data)",
            "Validation (HIKARI Test Split)",
        ],
    )

    st.sidebar.title("Model Information")
    st.sidebar.write(f"**Classifier:** `{MODEL_FILE.name}`")
    st.sidebar.write(f"**Preprocessor:** `{PREPROC_FILE.name}`")
    st.sidebar.write(f"**Detection threshold:** `{DETECTION_THRESHOLD:.2f}`")

    if DETECTION_THRESHOLD < 0.45:
        mode_text = "High sensitivity (recall-focused)"
    elif 0.45 <= DETECTION_THRESHOLD < 0.55:
        mode_text = "Default (dataset-based)"
    else:
        mode_text = "Balanced sensitivity (recommended)"

    st.sidebar.write(f"**Operating mode:** {mode_text}")

    # Route to selected view
    if view == "Detection (Upload Data)":
        detection_view(model, preproc)
    else:
        validation_view(model)


if __name__ == "__main__":
    main()
