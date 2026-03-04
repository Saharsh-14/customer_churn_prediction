# app.py
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide", page_icon="📊")

# --- CUSTOM HEADER ---
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #0072ff, #00c6ff);
            padding: 1rem 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.2);
            color: white;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .main-header h3 {
            font-size: 1rem;
            font-weight: 400;
            letter-spacing: 0.5px;
        }
    </style>

    <div class="main-header">
        <h1>📈 Customer Churn Prediction System</h1>
        <h3>by <b>Saharsh Jaiswal</b></h3>
    </div>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide", page_icon="📊")

# -------------------------
# Helpers & caching
# -------------------------
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # basic cleaning consistent with training
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    # replace common categories used earlier
    replace_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    # normalize strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

@st.cache_resource
def load_model():
    # model is in ../src/best_churn_model.pkl (relative to app folder)
    model_path = os.path.join(os.path.dirname(__file__), "..", "src", "best_churn_model.pkl")
    model_path = os.path.normpath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    return model

def predict_df(model, df_in):
    # ensures same cleaning as used before prediction
    df = df_in.copy()
    for c in df.columns:
        if c.lower() in ["tenure", "monthlycharges", "totalcharges", "seniorcitizen"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip().str.lower()
    df = df.fillna(0)
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    return preds, probs

def get_feature_importance(model):
    # Try to extract feature names and importances/coefficients
    try:
        pre = model.named_steps["preprocess"]
        # sklearn >=1.0 provides get_feature_names_out
        try:
            cat_features = pre.named_transformers_["cat"].get_feature_names_out()
        except Exception:
            # fallback: if OneHotEncoder inside ColumnTransformer doesn't expose get_feature_names_out
            cat_features = pre.transformers_[0][1].named_steps["encoder"].get_feature_names_out(pre.transformers_[0][2])
        num_features = pre.transformers_[1][2]
        feature_names = list(cat_features) + list(num_features)
    except Exception:
        # fallback: use columns of training dataset (if included)
        feature_names = None

    importance = None
    try:
        model_obj = model.named_steps["model"]
        if hasattr(model_obj, "feature_importances_"):
            importance = model_obj.feature_importances_
        elif hasattr(model_obj, "coef_"):
            # logistic regression
            coef = model_obj.coef_.ravel()
            importance = np.abs(coef)
    except Exception:
        importance = None

    if feature_names is not None and importance is not None and len(feature_names) == len(importance):
        fi = pd.DataFrame({"feature": feature_names, "importance": importance})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        return fi
    else:
        return None

# -------------------------
# Layout: sidebar (tabs)
# -------------------------
st.sidebar.title("Customer Churn Dashboard")
page = st.sidebar.radio("Choose view", ["Overview", "Predict", "Insights", "Dataset"])

# locate data file (assumes data in ../data/)
data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "telco_customer_churn.csv"))

# Load model & data (cached)
try:
    model = load_model()
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")
    model = None

try:
    data = load_data(data_path)
except Exception as e:
    st.sidebar.error(f"Data load error: {e}")
    data = None

# -------------------------
# Overview Tab
# -------------------------
if page == "Overview":
    st.title("📈 Overview & EDA")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Rows", data.shape[0] if data is not None else "—")
        st.metric("Columns", data.shape[1] if data is not None else "—")
        if data is not None and "Churn" in data.columns:
            churn_counts = data["Churn"].value_counts(normalize=False) if set(data["Churn"].unique()) <= {0,1} else data["Churn"].map({"yes":1,"no":0}).value_counts()
            st.metric("Churned", int(churn_counts.get(1, 0)))
            st.metric("Retention", int(churn_counts.get(0, 0)))
    with col2:
        st.subheader("Churn Distribution")
        if data is not None:
            # ensure churn numeric
            df_plot = data.copy()
            if df_plot["Churn"].dtype == "object":
                df_plot["Churn"] = df_plot["Churn"].map({"yes":1,"no":0})
            fig, ax = plt.subplots(figsize=(5,3))
            ax = df_plot["Churn"].value_counts().plot(kind="bar", rot=0)
            ax.set_xticklabels(["No", "Yes"])
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("Data not loaded")

    st.markdown("---")
    st.subheader("Numeric distributions")
    if data is not None:
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        cols = st.columns(len(num_cols))
        for i, nc in enumerate(num_cols):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(4,3))
                ax.hist(data[nc].dropna(), bins=30)
                ax.set_title(nc)
                st.pyplot(fig)
    st.markdown("---")
    st.subheader("Correlation (numeric)")
    if data is not None:
        corr = data.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

# -------------------------
# Predict Tab
# -------------------------
elif page == "Predict":
    st.title("🔮 Predict churn for a customer")
    st.markdown("Enter customer details on the left and click **Predict**. Or upload a CSV with the same schema for batch predictions.")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Customer input")
        # Build inputs - default values chosen sensibly
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen (0/1)", [0,1], index=0)
        Partner = st.selectbox("Partner", ["Yes","No"])
        Dependents = st.selectbox("Dependents", ["Yes","No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=200, value=12)
        PhoneService = st.selectbox("Phone Service", ["Yes","No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
        InternetService = st.selectbox("Internet Service", ["dsl","fiber optic","no"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
        Contract = st.selectbox("Contract", ["month-to-month","one year","two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
        PaymentMethod = st.selectbox("Payment Method", ["electronic check","mailed check","bank transfer (automatic)","credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=float(tenure*MonthlyCharges))

        if st.button("🔮 Predict single customer"):
            if model is None:
                st.error("Model not loaded.")
            else:
                inp = {
                    'gender': gender,
                    'SeniorCitizen': SeniorCitizen,
                    'Partner': Partner,
                    'Dependents': Dependents,
                    'tenure': tenure,
                    'PhoneService': PhoneService,
                    'MultipleLines': MultipleLines,
                    'InternetService': InternetService,
                    'OnlineSecurity': OnlineSecurity,
                    'OnlineBackup': OnlineBackup,
                    'DeviceProtection': DeviceProtection,
                    'TechSupport': TechSupport,
                    'StreamingTV': StreamingTV,
                    'StreamingMovies': StreamingMovies,
                    'Contract': Contract,
                    'PaperlessBilling': PaperlessBilling,
                    'PaymentMethod': PaymentMethod,
                    'MonthlyCharges': MonthlyCharges,
                    'TotalCharges': TotalCharges
                }
                pred, prob = predict_df(model, pd.DataFrame([inp]))
                if pred is not None:
                    if pred[0] == 1:
                        st.error(f"⚠️ Predicted: CHURN (probability {prob[0]:.2f})")
                    else:
                        st.success(f"✅ Predicted: No churn (probability {prob[0]:.2f})")

    with right:
        st.subheader("Batch prediction (CSV)")
        uploaded = st.file_uploader("Upload CSV with same columns (no target required)", type=["csv"])
        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            st.write("Preview of upload:")
            st.dataframe(df_upload.head())
            if st.button("Run batch prediction"):
                if model is None:
                    st.error("Model not loaded.")
                else:
                    preds, probs = predict_df(model, df_upload)
                    df_upload["pred_churn"] = preds
                    df_upload["churn_prob"] = probs
                    st.success("Batch prediction completed.")
                    st.download_button("Download results CSV", df_upload.to_csv(index=False), file_name="predictions.csv")
                    st.dataframe(df_upload.head(10))

# -------------------------
# Insights Tab
# -------------------------
elif page == "Insights":
    st.title("🔎 Model Insights")
    st.markdown("Feature importance / coefficients from the trained model (if available).")

    fi = None
    if model is not None:
        fi = get_feature_importance(model)

    if fi is not None:
        st.subheader("Top important features")
        st.dataframe(fi.head(20))
        fig, ax = plt.subplots(figsize=(6,5))
        ax.barh(fi["feature"].head(15)[::-1], fi["importance"].head(15)[::-1])
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    else:
        st.info("Feature importances not available for this model type or could not be extracted from pipeline.")

    st.markdown("---")
    st.subheader("Model & Data summary")
    st.write("Model file:", os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src", "best_churn_model.pkl")))
    if data is not None:
        st.write("Data rows:", data.shape[0])
        st.write("Churn rate:", (data["Churn"].map({"yes":1,"no":0}).mean() if data["Churn"].dtype == "object" else data["Churn"].mean()))
    else:
        st.write("Data not loaded")

# -------------------------
# Dataset Tab
# -------------------------
elif page == "Dataset":
    st.title("📂 Dataset")
    if data is not None:
        st.dataframe(data.head(200))
        if st.button("Download sample (200 rows)"):
            st.download_button("Download CSV", data.head(200).to_csv(index=False), file_name="telco_sample_200.csv")
    else:
        st.info("Data not loaded")
