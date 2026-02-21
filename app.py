import streamlit as st
import joblib
import numpy as np

# Load model files
model = joblib.load("fraud_xgb_model.pkl")
threshold = joblib.load("threshold.pkl")

st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction features:")

features = []

# PCA Features V1–V28
for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    features.append(value)

# Amount
amount = st.number_input("Amount", value=0.0)
features.append(amount)

if st.button("Predict"):
    features = np.array(features).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]

    if prob > threshold:
        st.error(f"🚨 Fraud Detected! Probability: {prob:.4f}")
    else:
        st.success(f"✅ Normal Transaction. Probability: {prob:.4f}")