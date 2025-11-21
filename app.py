
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

st.set_page_config(page_title='RF + Clustering Dashboard', layout='wide')
st.title('🧠 RF Predictions + 🎯 Customer Clustering')

# Load assets
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
feature_cols = json.load(open("feature_columns.json"))["cluster_features"]
impute = json.load(open("impute_values.json"))

def extract_options(prefix):
    return [f.replace(prefix + "_","") for f in feature_cols if f.startswith(prefix + "_")]

marital_options = extract_options("Marital_Status") or ["Married","Together","Single","Divorced","Widow","Alone","YOLO"]
education_options = extract_options("Education") or ["Basic","Graduation","Master","PhD"]

# Input
st.header("Enter Customer Information")
col1, col2 = st.columns(2)
with col1:
    Income = st.number_input("Income", value=impute.get("Income",50000.0))
    Age = st.number_input("Age", min_value=16, max_value=100, value=int(impute.get("Age",35)))
    Kidhome = st.number_input("Kidhome", min_value=0, max_value=10, value=int(impute.get("Kidhome",0)))
with col2:
    Teenhome = st.number_input("Teenhome", min_value=0, max_value=10, value=int(impute.get("Teenhome",0)))
    Marital = st.selectbox("Marital Status", marital_options)
    Education = st.selectbox("Education", education_options)

# Build input row
row = {c: float(impute.get(c,0.0)) for c in feature_cols}
row["Income"] = float(Income)
row["Age"] = float(Age)
row["Kidhome"] = int(Kidhome)
row["Teenhome"] = int(Teenhome)

for f in feature_cols:
    if f.startswith("Marital_Status_"):
        row[f] = 1.0 if f == f"Marital_Status_{Marital}" else 0.0
    if f.startswith("Education_"):
        row[f] = 1.0 if f == f"Education_{Education}" else 0.0

input_df = pd.DataFrame([row], columns=feature_cols)
st.subheader("Input Preview")
st.dataframe(input_df)

# Prediction
if st.button("🔍 Predict + Cluster"):
    rf_results = {}
    for label, mdl in rf_models.items():
        try:
            pred = mdl.predict(input_df)[0]
            prob = None
            if hasattr(mdl,"predict_proba"):
                try:
                    probs = mdl.predict_proba(input_df)
                    if probs.shape[1]>1:
                        prob = float(probs[0,1])
                except: pass
            rf_results[label] = {"prediction": float(pred),"prob": round(prob,3) if prob else None}
        except Exception as e:
            rf_results[label] = {"error": str(e)}
    st.subheader("🔹 RF Model Outputs")
    st.table(pd.DataFrame(rf_results).T)

    # Clustering
    try:
        scaled = scaler.transform(input_df)
        cluster_label = int(cluster_model.predict(scaled)[0])
        st.subheader(f"🎯 Cluster Assigned: {cluster_label}")
        if "Cluster" in cluster_profile.columns:
            prof_row = cluster_profile[cluster_profile["Cluster"]==cluster_label]
            if prof_row.empty:
                prof_row = cluster_profile.iloc[[cluster_label % len(cluster_profile)]]
        else:
            prof_row = cluster_profile.iloc[[cluster_label % len(cluster_profile)]]
        st.dataframe(prof_row.T)

        # Radar
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
