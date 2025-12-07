import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from google.cloud import firestore
import joblib


# -----------------------------
# Basic config
# -----------------------------
st.set_page_config(
    page_title="SleepScope | Insomnia Analysis",
    page_icon="😴",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 🔧 CHANGE THIS to match your actual Firestore collection name
FIRESTORE_COLLECTION = "sleepscope_sessions"  # TODO: update if different


# -----------------------------
# Helper: Load models safely
# -----------------------------
@st.cache_resource
def load_subtype_pipeline():
    """Load subtype model, scaler, feature list, and label map."""
    try:
        model = joblib.load(MODELS_DIR / "subtype_model.pkl")
        scaler = joblib.load(MODELS_DIR / "subtype_scaler.pkl")

        with open(MODELS_DIR / "subtype_features.json", "r") as f:
            feature_names = json.load(f)

        with open(MODELS_DIR / "subtype_label_map.json", "r") as f:
            label_map = json.load(f)

        # label_map is likely { "0": "Subtype A", "1": "Subtype B", ... }
        # ensure keys are int
        label_map = {int(k): v for k, v in label_map.items()}

        return model, scaler, feature_names, label_map
    except Exception as e:
        st.error(f"Error loading subtype model or config: {e}")
        return None, None, None, None


@st.cache_resource
def get_firestore_client():
    """
    Returns a Firestore client.

    Assumes GOOGLE_APPLICATION_CREDENTIALS or equivalent is already
    configured in your Render environment (same as your FastAPI backend).
    """
    try:
        client = firestore.Client()
        return client
    except Exception as e:
        st.warning(
            "Could not connect to Firestore. "
            "Check credentials / environment variables."
        )
        st.info(str(e))
        return None


# -----------------------------
# Helper: ISI severity based on total score
# -----------------------------
def get_isi_severity_label(isi_total: int) -> str:
    """
    Standard ISI interpretation:
      0–7   : No clinically significant insomnia
      8–14  : Subthreshold insomnia
      15–21 : Moderate clinical insomnia
      22–28 : Severe clinical insomnia
    """
    if isi_total <= 7:
        return "No clinically significant insomnia"
    elif isi_total <= 14:
        return "Subthreshold insomnia"
    elif isi_total <= 21:
        return "Moderate clinical insomnia"
    else:
        return "Severe clinical insomnia"


# -----------------------------
# Helper: Predict subtype
# -----------------------------
def predict_subtype(feature_values: dict):
    """
    feature_values: dict { feature_name: float }
    Uses feature_names from subtype_features.json to build ordered vector.
    """
    model, scaler, feature_names, label_map = load_subtype_pipeline()
    if model is None or feature_names is None:
        st.error("Subtype model not loaded. Please check model files.")
        return None, None

    # Build vector in the exact feature order
    x = []
    for name in feature_names:
        val = feature_values.get(name, 0.0)  # default 0.0 if missing
        x.append(float(val))

    x = np.array(x).reshape(1, -1)

    # Scale -> predict
    try:
        x_scaled = scaler.transform(x)
    except Exception:
        # If scaler is not available or fails, try without scaling
        x_scaled = x

    raw_pred = model.predict(x_scaled)[0]

    # If model output is numeric class index, map to label
    if isinstance(raw_pred, (int, np.integer)):
        subtype_label = label_map.get(int(raw_pred), f"Class {raw_pred}")
    else:
        subtype_label = str(raw_pred)

    return raw_pred, subtype_label


# -----------------------------
# Helper: Fetch ISI–PHQ9 correlation from Firestore
# -----------------------------
def fetch_isi_phq9_data():
    client = get_firestore_client()
    if client is None:
        return pd.DataFrame()

    try:
        docs = client.collection(FIRESTORE_COLLECTION).stream()
        rows = []
        for d in docs:
            data = d.to_dict()
            # 🔧 CHANGE KEYS if your Firestore uses different field names
            isi_val = data.get("isi_total") or data.get("isi_score")
            phq_val = data.get("phq9_total") or data.get("phq9_score")
            session_id = data.get("session_id", d.id)

            if isi_val is not None and phq_val is not None:
                rows.append(
                    {
                        "session_id": session_id,
                        "ISI": float(isi_val),
                        "PHQ9": float(phq_val),
                    }
                )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Error reading from Firestore: {e}")
        return pd.DataFrame()


# -----------------------------
# Layout: Main title
# -----------------------------
st.markdown(
    """
    <div class="big-title">SleepScope 💤</div>
    <div class="subtitle">
        Insomnia Severity Prediction · Subtype Classification · Depression Correlation
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🏠 Overview",
        "👤 User Dashboard",
        "🩺 Clinician (PSG Upload)",
        "📊 Correlation Explorer",
        "ℹ️ About / How it Works",
    ]
)

# =====================================================
#  TAB 1: OVERVIEW
# =====================================================
with tabs[0]:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Project Summary")
        st.write(
            """
            **SleepScope** is an explainable ML framework designed to:

            - Estimate **insomnia severity**, based on ISI scores.
            - Perform **insomnia subtype classification** using a trained ML model.
            - Explore the **correlation between insomnia and depression**, via ISI and PHQ-9 scores stored in Firestore.
            - Provide a **clinician-facing PSG upload section** for extending the analysis to polysomnography data.

            This Streamlit app combines all components into a single, interactive dashboard
            for both users and clinicians.
            """
        )

    with col2:
        st.markdown("### Demo Flow")
        st.markdown(
            """
            1. Go to **User Dashboard**  
               → Enter ISI & PHQ-9 totals  
               → Get severity and subtype prediction.
               
            2. Go to **Correlation Explorer**  
               → View real-time ISI–PHQ9 correlation from Firestore.
               
            3. Go to **Clinician (PSG)**  
               → Upload PSG/EDF file (concept demo).  
            """
        )


# =====================================================
#  TAB 2: USER DASHBOARD
# =====================================================
with tabs[1]:
    st.subheader("User Dashboard – Questionnaire-based Analysis")

    st.write(
        """
        This section is meant for **end users** who have already answered
        insomnia (ISI) and depression (PHQ-9) questionnaires.  
        We use the **total scores** as model inputs.
        """
    )

    with st.form("user_input_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            isi_total = st.number_input(
                "ISI Total Score (0–28)",
                min_value=0,
                max_value=28,
                value=10,
                step=1,
                help="Sum of all 7 ISI items.",
            )
        with c2:
            phq9_total = st.number_input(
                "PHQ-9 Total Score (0–27)",
                min_value=0,
                max_value=27,
                value=8,
                step=1,
                help="Sum of all 9 PHQ-9 items.",
            )
        with c3:
            sleep_duration = st.number_input(
                "Sleep Duration (hours)",
                min_value=0.0,
                max_value=15.0,
                value=5.5,
                step=0.5,
            )

        st.markdown("#### Optional additional features")

        # ✅ Try to build feature set dynamically from subtype_features.json
        _, _, subtype_feature_names, _ = load_subtype_pipeline()
        extra_features = {}

        if subtype_feature_names:
            st.caption(
                "Below, you can provide feature values used by the subtype model. "
                "Defaults are 0.0 if left unchanged."
            )
            # Create 2 columns for nicer layout
            col_a, col_b = st.columns(2)
            for i, fname in enumerate(subtype_feature_names):
                # Some features might be isi_total, phq9_total, sleep_duration.
                # We'll auto-fill them if names match exactly.
                default = 0.0
                fname_lower = fname.lower()
                if "isi" in fname_lower and "total" in fname_lower:
                    default = float(isi_total)
                elif "phq" in fname_lower:
                    default = float(phq9_total)
                elif "sleep" in fname_lower and "duration" in fname_lower:
                    default = float(sleep_duration)

                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    extra_features[fname] = st.number_input(
                        f"{fname}",
                        value=default,
                        format="%.3f",
                    )
        else:
            st.info(
                "Subtype feature configuration not found. "
                "Subtype prediction will be disabled."
            )

        submitted = st.form_submit_button("Run Analysis")

    if submitted:
        # 1) ISI severity (rule-based)
        severity_label = get_isi_severity_label(int(isi_total))

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Insomnia Severity (based on ISI)")
            st.metric(
                label="Severity Category",
                value=severity_label,
                delta=f"ISI Total: {isi_total}",
            )
            st.write(
                """
                *Interpretation is based on standard ISI cutoffs.*
                """
            )

        # 2) Subtype prediction (ML model)
        with col_right:
            st.markdown("### Insomnia Subtype (ML Model)")
            if subtype_feature_names:
                raw_label, pretty_label = predict_subtype(extra_features)
                if pretty_label is not None:
                    st.success(f"Predicted Subtype: **{pretty_label}**")
                    st.caption(f"Raw model output: `{raw_label}`")
                else:
                    st.error("Subtype prediction failed. Check logs / console.")
            else:
                st.warning(
                    "Subtype model not configured. "
                    "Ensure subtype_features.json and subtype_model.pkl exist."
                )


# =====================================================
#  TAB 3: CLINICIAN (PSG UPLOAD)
# =====================================================
with tabs[2]:
    st.subheader("Clinician View – PSG / EDF Upload")

    st.write(
        """
        This section is meant for **clinicians** to upload polysomnography (PSG)
        files (e.g., EDF). In the current prototype, we focus on:

        - Demonstrating **file upload and handling**
        - Explaining how EEG/PSG features can be combined with questionnaire data
        - Showing how this connects to the trained `psg_model.pkl` in the backend

        > In a deployment-limited environment like Render, full EDF parsing may
        > require additional libraries (e.g., `mne`) and environment configuration.
        """
    )

    uploaded_psg = st.file_uploader(
        "Upload PSG / EDF file",
        type=["edf", "EDF"],
        accept_multiple_files=False,
    )

    if uploaded_psg is not None:
        st.success(f"File `{uploaded_psg.name}` uploaded successfully.")
        st.info(
            """
            In your full backend, this file would be:

            1. Passed to the **PSG preprocessing pipeline** (`app.utils.preprocess`).
            2. Converted into **feature vectors**.
            3. Fed into `psg_model.pkl` for prediction.
            4. Optionally combined with ISI / PHQ-9 scores for richer analysis.

            For the current demo, you can describe this pipeline to the panel
            while showing that the upload mechanism is already in place.
            """
        )

        # If you want, you can save the file temporarily (if allowed in your environment):
        # temp_path = BASE_DIR / "tmp_psg.edf"
        # with open(temp_path, "wb") as f:
        #     f.write(uploaded_psg.getbuffer())


# =====================================================
#  TAB 4: CORRELATION EXPLORER
# =====================================================
with tabs[3]:
    st.subheader("Correlation Explorer – ISI vs PHQ-9")

    st.write(
        """
        This section computes and visualizes the **correlation between insomnia severity**
        and **depression symptoms** using ISI and PHQ-9 scores stored in Firestore.
        """
    )

    df_corr = fetch_isi_phq9_data()

    if df_corr.empty:
        st.warning(
            "No data found in Firestore or unable to connect. "
            "Ensure the collection name and credentials are correct."
        )
    else:
        st.markdown("#### Sample Data")
        st.dataframe(df_corr.head())

        corr_val = df_corr[["ISI", "PHQ9"]].corr().iloc[0, 1]
        st.metric(
            label="Pearson Correlation (ISI vs PHQ-9)",
            value=f"{corr_val:.3f}",
        )

        st.markdown("#### Scatter Plot")
        st.write(
            "Each point represents a **session** with both ISI and PHQ-9 scores."
        )
        st.scatter_chart(df_corr, x="ISI", y="PHQ9")

        st.caption(
            """
            A higher positive correlation suggests that higher insomnia severity
            is associated with higher depression scores in the observed population.
            """
        )


# =====================================================
#  TAB 5: ABOUT / HOW IT WORKS
# =====================================================
with tabs[4]:
    st.subheader("About SleepScope & Technical Workflow")

    st.markdown(
        """
        ### 1. Data Sources

        - **Questionnaire data**:  
          - Insomnia Severity Index (ISI) – severity of insomnia  
          - PHQ-9 – depression symptoms  
        - **Optional PSG data** (EDF):
          - EEG/EOG/EMG channels extracted as features for advanced modelling.

        ### 2. ML Components

        1. **ISI-based Severity Estimation (Rule-based)**  
           - ISI total score is categorized into severity levels:
             - 0–7: No clinically significant insomnia  
             - 8–14: Subthreshold insomnia  
             - 15–21: Moderate clinical insomnia  
             - 22–28: Severe clinical insomnia  

        2. **Subtype Classification (ML Model)**  
           - Uses `subtype_model.pkl`, `subtype_scaler.pkl`, and
             `subtype_features.json`.  
           - Features are collected from the user and scaled before prediction.  
           - Output label is mapped via `subtype_label_map.json`.

        3. **Depression Correlation (Explainable Insight)**  
           - ISI and PHQ-9 scores are stored in Firestore along with a session ID.  
           - Correlation between ISI and PHQ-9 is computed and visualized to
             study how insomnia might co-occur with depression.

        ### 3. Architecture

        - **Backend** (already deployed on Render):  
          - Handles model training / storage, scoring logic, and Firestore integration.
        - **This Streamlit app** (same repo):
          - Acts as a lightweight, Python-based frontend.
          - Directly loads `.pkl` models and Firestore data.
          - Provides separate views for **users** and **clinicians**.

        ### 4. Why Streamlit?

        - Rapid prototype for demo under strict time constraints.
        - Eliminates complex JS–backend integration issues.
        - Still demonstrates:
          - End-to-end ML workflow  
          - Data pipeline  
          - Real-time analytics & explainability  
        """
    )
