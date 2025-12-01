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

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)

    # Per-class precision, recall, f1-score, support
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Return ONLY the values used in the UI
    metrics = {
        "accuracy": acc,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "supports": supports,
        "confusion_matrix": cm,
        "n_samples": len(y_test),
    }

    return metrics

# ================================================================
# 0) SIMPLE LOGIN / AUTH GATE
# ================================================================
def login_view():
    with st.form("login_form"):
        st.markdown("### 🔐 Login")
        st.markdown(
            '<p class="login-subtitle">Sign in to access the IDS console.</p>',
            unsafe_allow_html=True,
        )

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    # Authentication logic
    if submitted:
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")



# ================================================================
# 5) DETECTION VIEW (UPLOAD + PREDICT, FIXED 0.60 THRESHOLD)
# ================================================================
def detection_view(model, preproc):
    st.subheader("📡 Detection Console – Flow-Based IDS")

    # Hero description
    st.write(
        "Apply the trained Random Forest classifier to new network flow records. "
        "The IDS operates on flow-level statistical features (counts, timings, "
        "rates) and does not inspect packet payloads."
    )

    # Quick-glance metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset", "HIKARI-2021")
    with col2:
        st.metric("Operating mode", "Moderate-Sensitivity")
  
    # Threshold card
    st.markdown(
        f"""
        <div class="rf-card">
            <span class="rf-pill">Threshold {DETECTION_THRESHOLD:.2f}</span>
            <p style="margin-top:0.6rem;">
                The IDS uses a fixed decision threshold of <strong>{DETECTION_THRESHOLD:.2f}</strong>,
                chosen during validation as a practical compromise between limiting false alerts
                and maintaining strong attack detection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload instructions card
    st.markdown(
        """
        <div class="rf-card">
            <span class="rf-pill">Input data</span>
            <p style="margin-top:0.6rem; margin-bottom:0.4rem;">
                You can analyse either raw flow statistics or preprocessed feature vectors:
            </p>
            <ul style="margin-top:0;">
                <li><strong>Raw flow statistics</strong> using the same feature schema as during training.</li>
                <li><strong>Preprocessed numeric test data</strong> exported from the training pipeline.</li>
            </ul>
            <p style="margin-top:0.6rem; margin-bottom:0.4rem;">
                All raw uploads must contain the full set of flow features used during training; missing or mismatched columns cannot be processed by the IDS.
            </p>
            <p style="margin-top:0.4rem;">
                Columns such as <code>Label</code> or <code>traffic_category</code> should be removed before upload.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File uploader (unchanged logic)
    uploaded = st.file_uploader(
        "Choose a flow data file", type=["csv", "xls", "xlsx"]
    )

    if uploaded is None:
        st.info("📎 Upload a CSV or Excel file to analyse network flows.")
        return

    # ... keep the rest of your detection_view exactly as you have it now


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

        # ------------------------------------------------------------
        # Per-flow predictions (detailed view, 81 features preserved)
        # ------------------------------------------------------------
        st.markdown("### 🧠 Prediction results")

        # Small badge to make it clear why the table is wide
        st.markdown(
            "<span class='rf-pill'>81 flow features per record + model outputs</span>",
            unsafe_allow_html=True,
        )

        merged = df.copy()
        merged["attack_probability"] = result_df["attack_probability"]
        merged["predicted_label"] = result_df["predicted_label"]
        merged["predicted_class"] = result_df["predicted_class"]

        # Put the big table inside an expander so the page stays clean
        with st.expander("Show per-flow prediction table (first 50 flows)", expanded=False):
            st.caption(
                "Each row corresponds to a single network flow with all 81 features "
                "plus the model's prediction outputs."
            )
            st.dataframe(merged.head(50), height=400)

        if dropped_cols:
            st.info(
                "The following non-numeric columns were ignored during prediction:\n"
                + ", ".join(dropped_cols)
            )

        # ------------------------------------------------------------
        # Summary statistics
        # ------------------------------------------------------------
        st.markdown("### 📊 Summary of predictions")

        counts = (
            result_df["predicted_class"]
            .value_counts()
            .reindex(["Benign", "Attack"])   # enforce consistent order
            .dropna()
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
            st.success(
                "Low risk: No flows in this dataset were classified as attacks."
            )
        elif attack_rate < 5:
            st.info(
                "Moderate risk: Only a small proportion of flows were classified as attacks."
            )
        else:
            st.warning(
                "Elevated risk: A substantial proportion of flows were classified as attacks. "
                "In an operational IDS this would trigger further investigation."
            )

        st.table(counts)
        st.bar_chart(counts.set_index("Class")["Count"])

        st.success(
            f"Finished scoring **{total_flows}** flows — 🎯 **{attack_count}** predicted as attacks."
        )



# ================================================================
# 6) VALIDATION VIEW (HIKARI-2021 TEST SPLIT WITH THRESHOLD SLIDER)
# ================================================================
def validation_view(model):
    st.subheader("🧪 Model Validation – Test Split")

    st.write(
        "This view analyses the Random Forest IDS on a reserved test split drawn "
        "from the HIKARI-2021 dataset. It reflects the evaluation carried out in "
        "the project notebooks and lets you explore how different decision "
        "thresholds affect accuracy, recall, and precision."
    )

    st.markdown("### Threshold selection")

    threshold = st.slider(
        "Decision threshold (higher = fewer alerts, lower = more alerts)",
        min_value=0.10,
        max_value=0.90,
        value=DETECTION_THRESHOLD,
        step=0.01,
    )
    st.markdown("")
    
      # Mode description based on threshold
    if threshold <= 0.40:
        st.warning(
            "⚠️ High-Sensitivity Mode – prioritises maximum recall to minimise missed attacks. "
            "The IDS flags nearly all malicious flows, but at the cost of many false positives. "
            "Best suited for environments where any missed intrusion is unacceptable."
        )

    elif 0.40 < threshold <= 0.70:
        st.info(
            "🟡 Moderate-Sensitivity Mode – reduces false positives while still retaining strong "
            "attack detection. Recall remains high (though no longer maximal), making this a "
            "practical compromise between alert volume and detection reliability."
        )

    else:
        st.error(
            "🔴 Low-Sensitivity Mode – significantly reduces false positives, but recall drops "
            "substantially. Only high-confidence attacks are detected, and a noticeable number "
            "of intrusions may be missed. Use with caution."
        )

        
    if st.button("▶ Run validation"):
        try:
            metrics = evaluate_on_hikari(model, threshold)
        except Exception as e:
            st.error(e)
            return

        st.markdown("### 📈 Overall performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Test samples", f"{metrics['n_samples']}")
        with col3:
            st.metric("Threshold used", f"{threshold:.2f}")
            
        st.markdown("")
        
        # Per-class precision / recall / F1
        st.markdown("### 📊 Per-class precision, recall and F1-score")
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
        st.markdown("")
        
        # Confusion matrix
        st.markdown("### 🧩 Confusion matrix (rows = true class, columns = predicted class)")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["True Benign (0)", "True Attack (1)"],
            columns=["Pred Benign (0)", "Pred Attack (1)"],
        )
        st.table(cm_df)
        
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Light theming helpers for cards / pills -------------------
def inject_css():
    bg_image = get_base64_image("assets/login.jpg")  # make sure this path is correct
    is_logged_in = st.session_state.get("logged_in", False)

    if not is_logged_in:
        # Use background image on login page
        bg_style = f'''
            background: url("data:image/jpeg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        '''
    else:
        # Plain white background after login
        bg_style = '''
            background: #ffffff;
        '''

    st.markdown(
        f"""
        <style>
        .rf-card {{
            padding: 1.1rem 1.25rem;
            border-radius: 0.8rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background-color: rgba(250, 250, 252, 0.9);
            margin-bottom: 1rem;
        }}
        .rf-pill {{
            display: inline-block;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            background-color: rgba(86, 156, 214, 0.12);
            color: rgb(38, 76, 119);
        }}
        /* Improve sidebar typography */
        [data-testid="stSidebar"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        /* Navigation title ("Navigation") */
        [data-testid="stSidebar"] h2 {{
            margin-top: 1rem !important;
            margin-bottom: 0.7rem !important;
        }}

        /* "Select view" subtitle */
        [data-testid="stSidebar"] p {{
            font-size: 14px !important;
            color: #555 !important;
            margin-bottom: 0.3rem !important;
        }}

        /* Radio button container rows */
        [data-testid="stSidebar"] div[role="radiogroup"] label {{
            padding: 6px 6px !important;
            border-radius: 6px !important;
            transition: background-color 0.2s ease-in-out;
            cursor: pointer;
        }}

        /* Hover effect for options */
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background-color: rgba(0, 0, 0, 0.05) !important;
        }}

        /* Selected option highlight */
        [data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {{
            background-color: rgba(60, 120, 230, 0.12) !important;
            border-left: 3px solid rgb(60, 120, 230) !important;
        }}

        /* Radio option text */
        [data-testid="stSidebar"] div[role="radiogroup"] span {{
            font-size: 15px !important;
        }}

        h1 {{
            margin-top: -3rem !important;
        }}
        
        /* ---------- LOGIN FORM AS CENTERED CARD ---------- */
        div[data-testid="stForm"] {{
            width: 380px;
            max-width: 90%;
            margin: 6rem auto 0 auto;
            padding: 2.2rem 2rem 2rem 2rem;
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background-color: #ffffff;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
        }}

        div[data-testid="stForm"] h3 {{
            margin-bottom: 0.4rem;
        }}

        div[data-testid="stForm"] p.login-subtitle {{
            font-size: 0.95rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }}

        div[data-testid="stForm"] button[kind="primary"] {{
            width: 100% !important;
            margin-top: 0.8rem;
        }}

        /* ---------- FULL PAGE BACKGROUND IMAGE / COLOR ----------- */
        html, body, [data-testid="stAppViewContainer"] {{
{bg_style}
        }}

        /* ---------- USER MENU (Top-right) ---------- */
        .user-menu select {{
            padding: 4px 10px !important;
            border-radius: 8px !important;
            background-color: #fff !important;
            font-size: 14px !important;
            height: 34px !important;
            border: 1px solid #CCC !important;
        }}

        .user-menu .stSelectbox {{
            margin-top: -10px !important;
            width: 150px !important;
            float: right !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



# ================================================================
# 7) MAIN APP ENTRYPOINT
# ================================================================
def main():

    # must be the first Streamlit call
    st.set_page_config(
        page_title="Random Forest Intrusion Detection",
        layout="wide",
    )
    
    inject_css()

    # ------------------ LOGIN GATE ------------------
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if not st.session_state.logged_in:
        login_view()
        return

    # ------------------ HEADER: TITLE + USER MENU ------------------
    header_left, header_right = st.columns([7, 1])

    with header_left:
        st.title("🛡️ Random Forest Intrusion Detection")
        st.caption(
            "The console supports **live detection** on uploaded flow data and **offline "
            "validation** on the held-out test split."
        )

    with header_right:
        st.markdown('<div class="user-menu">', unsafe_allow_html=True)

        user_choice = st.selectbox(
            "",
            [f"👤 {st.session_state.username or 'admin'}", "Logout"],
            index=0,
            label_visibility="collapsed",
        )

        if user_choice == "Logout":
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ LOAD ARTEFACTS ------------------
    try:
        model = load_model()
        preproc = load_preprocessor()
    except Exception as e:
        st.error(e)
        st.stop()

    # ------------------ SIDEBAR NAVIGATION ------------------
    st.sidebar.title("📚 Navigation")
    view = st.sidebar.radio(
        "Select view",
        options=[
            "📡 Detection (Upload Data)",
            "🧪 Validation (Test Split)",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.title("🧬 Model Information")
    st.sidebar.write(f"**Classifier:** `{MODEL_FILE.name}`")
    st.sidebar.write(f"**Preprocessor:** `{PREPROC_FILE.name}`")

    # ------------------ ROUTE VIEWS ------------------
    if view == "📡 Detection (Upload Data)":
        detection_view(model, preproc)
    else:
        validation_view(model)


if __name__ == "__main__":
    main()
