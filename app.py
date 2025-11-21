
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

st.set_page_config(page_title='RF + Clustering Dashboard', layout='wide')
st.title('🧠 RF Predictions + 🎯 Customer Clustering')

# --- load saved assets ---
rf_models = {
    "MntWines": joblib.load("rf_MntWines.pkl"),
    "MntFruits": joblib.load("rf_MntFruits.pkl"),
    "MntMeatProducts": joblib.load("rf_MntMeat.pkl"),
    "MntFishProducts": joblib.load("rf_MntFish.pkl"),
    "MntSweetProducts": joblib.load("rf_MntSweet.pkl"),
    "MntGoldProds": joblib.load("rf_MntGold.pkl")
}
cluster_model = joblib.load("cluster_model.pkl")
scaler = joblib.load("cluster_scaler.pkl")
cluster_profile = pd.read_csv("cluster_profile.csv")
feat_obj = json.load(open("feature_columns.json"))
feature_cols = feat_obj["cluster_features"]
impute = json.load(open("impute_values.json"))

# --- helper to extract unique options for Marital_Status and Education if present in impute keys ---
def extract_options(prefix):
    opts = []
    for f in feature_cols:
        if f.startswith(prefix + "_"):
            opts.append(f.replace(prefix + "_", ""))
    return opts

marital_options = extract_options("Marital_Status")
education_options = extract_options("Education")

# default options if none found
if not marital_options:
    marital_options = ["Married","Together","Single","Divorced","Widow","Alone","YOLO"]
if not education_options:
    education_options = ["Basic","Graduation","Master","PhD"]

# --- Input UI: only these fields requested from user ---
st.header("Enter Customer Information (only these fields required)")
col1, col2 = st.columns(2)
with col1:
    Income = st.number_input("Income", value=impute.get("Income_imputed", 50000.0), format="%.2f")
    Age = st.number_input("Age", min_value=16, max_value=100, value=int(impute.get("Age", 35)))
    Kidhome = st.number_input("Kidhome", min_value=0, max_value=10, value=int(impute.get("Kidhome", 0)))
with col2:
    Teenhome = st.number_input("Teenhome", min_value=0, max_value=10, value=int(impute.get("Teenhome", 0)))
    Marital = st.selectbox("Marital Status", marital_options, index=0)
    Education = st.selectbox("Education", education_options, index=0)

# --- build full input row using impute defaults, then overwrite with user inputs ---
row = {c: float(impute.get(c, 0.0)) for c in feature_cols}

# map basic numeric fields to their expected feature names
# Income -> Income_imputed if present
if "Income_imputed" in row:
    row["Income_imputed"] = float(Income)
elif "Income" in row:
    row["Income"] = float(Income)

row["Age"] = float(Age) if "Age" in row else float(Age)

row["Kidhome"] = int(Kidhome) if "Kidhome" in row else int(Kidhome)
row["Teenhome"] = int(Teenhome) if "Teenhome" in row else int(Teenhome)

# set marital one-hot columns
for f in feature_cols:
    if f.startswith("Marital_Status_"):
        # set 1 where matches selected Marital, else 0
        key = f.replace("Marital_Status_", "")
        row[f] = 1.0 if key == Marital else 0.0

# set education one-hot columns
for f in feature_cols:
    if f.startswith("Education_"):
        key = f.replace("Education_", "")
        row[f] = 1.0 if key == Education else 0.0

input_df = pd.DataFrame([row], columns=feature_cols)

st.subheader("Input Preview (features sent to models)")
st.dataframe(input_df)

# --- Predict with RF models (use full feature vector for each model) ---
if st.button("🔍 Predict + Cluster"):
    rf_results = {}
    for label, mdl in rf_models.items():
        try:
            pred = mdl.predict(input_df)[0]
            # if model supports predict_proba and is classifier, get probability of positive class
            prob = None
            if hasattr(mdl, "predict_proba"):
                try:
                    probs = mdl.predict_proba(input_df)
                    # if binary classification, probability of class 1
                    if probs.shape[1] > 1:
                        prob = float(probs[0,1])
                except Exception:
                    prob = None
            rf_results[label] = {"prediction": float(pred) if (isinstance(pred, (int,float,np.number))) else str(pred),
                                 "prob": (round(prob,3) if prob is not None else None)}
        except Exception as e:
            rf_results[label] = {"error": str(e)}

    st.subheader("🔹 RF Model Outputs")
    st.table(pd.DataFrame(rf_results).T)

    # --- clustering ---
    try:
        scaled = scaler.transform(input_df)
        cluster_label = int(cluster_model.predict(scaled)[0])
        st.subheader(f"🎯 Cluster Assigned: {cluster_label}")
        # attempt to show cluster profile row from cluster_profile.csv
        if "Cluster" in cluster_profile.columns:
            prof_row = cluster_profile[cluster_profile["Cluster"]==cluster_label]
            if prof_row.empty:
                prof_row = cluster_profile.iloc[[cluster_label % len(cluster_profile)]]
        else:
            prof_row = cluster_profile.iloc[[cluster_label % len(cluster_profile)]]
        st.dataframe(prof_row.T)

        # radar chart for spending categories
        categories = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
        present = [c for c in categories if c in prof_row.columns]
        if present:
            vals = prof_row.iloc[0][present].astype(float).values
            vals = np.append(vals, vals[0])
            angles = np.linspace(0, 2*np.pi, len(present), endpoint=False)
            angles = np.append(angles, angles[0])
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6,6))
            ax.plot(angles, vals, 'o-', linewidth=2)
            ax.fill(angles, vals, alpha=0.25)
            ax.set_thetagrids(angles[:-1]*180/np.pi, present)
            ax.set_title(f"Cluster {cluster_label} Spending Pattern")
            st.pyplot(fig)
    except Exception as e:
        st.error("Clustering failed: " + str(e))
