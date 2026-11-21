"""
Customer Churn Risk Predictor
-----------------------------
Streamlit app that serves a pre-trained RandomForestClassifier.
No training happens here: the model is loaded from Randomforestclassifier.joblib.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "Randomforestclassifier.joblib"

# Risk levels, same cut-offs as "Churn Risk Predictor v1" in the notebook
HIGH_THRESHOLD = 0.5
MEDIUM_THRESHOLD = 0.3

# Features, in the same order used during training
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORY_OPTIONS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "PaperlessBilling": ["No", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
}
CATEGORICAL_FEATURES = list(CATEGORY_OPTIONS)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="wide")


# --------------------------------------------------------------------------
# Model + helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def expected_columns(model) -> list:
    """Columns the model was trained on (one-hot encoded), in training order."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    cols = list(NUMERIC_FEATURES)
    for feat, options in CATEGORY_OPTIONS.items():
        cols += [f"{feat}_{o}" for o in options]
    return cols


def encode(df: pd.DataFrame, model) -> pd.DataFrame:
    """Apply the same one-hot encoding used in training and align the columns."""
    df = df[FEATURES].copy()
    for col, options in CATEGORY_OPTIONS.items():
        # Fixed categories => every dummy column exists even for a single row
        df[col] = pd.Categorical(df[col], categories=options)
    encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)
    return encoded.reindex(columns=expected_columns(model), fill_value=0)


def risk_level(probability: float) -> str:
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def predict_proba(df: pd.DataFrame, model):
    return model.predict_proba(encode(df, model))[:, 1]


# --------------------------------------------------------------------------
# Load model (fail clearly if the file is missing)
# --------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        f"Model file `{MODEL_PATH.name}` not found. "
        "Place it in the same folder as `app.py` (the repository root)."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:  # version mismatch, corrupted file, etc.
    st.error(
        "The model could not be loaded. This is usually a scikit-learn version "
        "mismatch: pin the same version in `requirements.txt` that you used in Colab."
    )
    st.exception(exc)
    st.stop()


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📉 Customer Churn Risk Predictor")
st.caption(
    "Estimate how likely a telecom customer is to leave, using a pre-trained "
    "Random Forest model."
)

tab_single, tab_batch, tab_about = st.tabs(
    ["🔎 Single customer", "📂 Batch scoring", "ℹ️ About the model"]
)

# ------------------------------ Single customer ----------------------------
with tab_single:
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.slider("Monthly charges ($)", 15.0, 125.0, 70.0, 0.5)
        contract = st.selectbox("Contract", CATEGORY_OPTIONS["Contract"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])

    with c2:
        st.markdown("**Billing**")
        payment = st.selectbox("Payment method", CATEGORY_OPTIONS["PaymentMethod"])
        paperless = st.selectbox("Paperless billing", CATEGORY_OPTIONS["PaperlessBilling"])
        override_total = st.checkbox("Enter total charges manually")
        if override_total:
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=10.0
            )
        else:
            total_charges = tenure * monthly_charges
            st.caption(f"Total charges estimated as tenure × monthly: **${total_charges:,.2f}**")

    with c3:
        st.markdown("**Services**")
        internet = st.selectbox("Internet service", CATEGORY_OPTIONS["InternetService"])
        if internet == "No":
            # Without internet, these two services can only be "No internet service"
            tech_support = "No internet service"
            online_security = "No internet service"
            st.selectbox("Tech support", [tech_support], disabled=True)
            st.selectbox("Online security", [online_security], disabled=True)
        else:
            yes_no = ["No", "Yes"]
            tech_support = st.selectbox("Tech support", yes_no)
            online_security = st.selectbox("Online security", yes_no)

    if st.button("Predict churn risk", type="primary", use_container_width=True):
        customer = pd.DataFrame(
            [
                {
                    "tenure": tenure,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "SeniorCitizen": 1 if senior == "Yes" else 0,
                    "Contract": contract,
                    "InternetService": internet,
                    "PaymentMethod": payment,
                    "PaperlessBilling": paperless,
                    "TechSupport": tech_support,
                    "OnlineSecurity": online_security,
                }
            ]
        )
        probability = float(predict_proba(customer, model)[0])
        level = risk_level(probability)

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{probability * 100:.1f}%")
        m2.metric("Risk level", level)
        st.progress(min(max(probability, 0.0), 1.0))

        if level == "HIGH":
            st.error("High risk: prioritise this customer for a retention offer.")
        elif level == "MEDIUM":
            st.warning("Medium risk: keep an eye on this customer and consider a proactive check-in.")
        else:
            st.success("Low risk: this customer is unlikely to leave in the near term.")

        st.caption(
            f"Risk levels: LOW < {MEDIUM_THRESHOLD:.0%}, "
            f"MEDIUM {MEDIUM_THRESHOLD:.0%} to {HIGH_THRESHOLD:.0%}, HIGH ≥ {HIGH_THRESHOLD:.0%}."
        )

# ------------------------------ Batch scoring ------------------------------
with tab_batch:
    st.subheader("Score many customers at once")
    st.write(
        "Upload a CSV with these columns (one row per customer): "
        f"`{', '.join(FEATURES)}`."
    )

    template = pd.DataFrame(
        [
            [2, 95.0, 190.0, 0, "Month-to-month", "Fiber optic", "Electronic check", "Yes", "No", "No"],
            [60, 60.0, 3600.0, 0, "Two year", "DSL", "Credit card (automatic)", "No", "Yes", "Yes"],
        ],
        columns=FEATURES,
    )
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="churn_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customers CSV", type="csv")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()

        missing_cols = [c for c in FEATURES if c not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            data = data.copy()
            for col in NUMERIC_FEATURES:
                data[col] = pd.to_numeric(data[col], errors="coerce")

            valid = data[NUMERIC_FEATURES].notna().all(axis=1)
            for col, options in CATEGORY_OPTIONS.items():
                valid &= data[col].isin(options)

            n_bad = int((~valid).sum())
            if n_bad:
                st.warning(
                    f"{n_bad} row(s) skipped because of missing numbers or unrecognised "
                    "category values. Check spelling against the template."
                )

            scored = data[valid].copy()
            if scored.empty:
                st.error("No valid rows to score.")
            else:
                probs = predict_proba(scored, model)
                scored["ChurnRisk%"] = (probs * 100).round(1)
                scored["RiskLevel"] = [risk_level(p) for p in probs]
                scored = scored.sort_values("ChurnRisk%", ascending=False)

                k1, k2, k3 = st.columns(3)
                k1.metric("Customers scored", len(scored))
                k2.metric("HIGH risk", int((scored["RiskLevel"] == "HIGH").sum()))
                k3.metric("MEDIUM risk", int((scored["RiskLevel"] == "MEDIUM").sum()))

                st.dataframe(scored, use_container_width=True)
                st.download_button(
                    "Download scored customers",
                    scored.to_csv(index=False).encode("utf-8"),
                    file_name="churn_scored_customers.csv",
                    mime="text/csv",
                )

# ------------------------------ About --------------------------------------
with tab_about:
    st.subheader("About the model")
    st.markdown(
        """
- **Model:** `RandomForestClassifier` (100 trees), trained offline on the Telco Customer Churn dataset.
- **Target:** whether the customer churned (1) or stayed (0).
- **Output:** a churn probability, converted into a risk level (LOW / MEDIUM / HIGH).
- **Features (10):** tenure, monthly charges, total charges, senior citizen, contract,
  internet service, payment method, paperless billing, tech support, online security.
- **No training happens in this app.** The model is only loaded and used for prediction.
        """
    )

    if hasattr(model, "feature_importances_"):
        st.markdown("**Top 10 most important features**")
        importances = (
            pd.Series(model.feature_importances_, index=expected_columns(model))
            .nlargest(10)
            .sort_values(ascending=False)
        )
        st.bar_chart(importances)

    st.info(
        "Predictions are probabilities based on historical patterns, not certainties. "
        "Use them to prioritise retention efforts alongside business judgement."
    )
