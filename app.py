import streamlit as st
import pickle
import numpy as np


@st.cache_resource
def load_model():
    with open("fraud_detection_system.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("💳 Fraud Detection System")
st.write("Credit Card Transaction Fraud Prediction")

st.divider()


time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

st.subheader("PCA Features (V1 to V28)")
v_features = []

for i in range(1, 29):
    v = st.number_input(f"V{i}")
    v_features.append(v)


if st.button("🔍 Predict"):
    input_data = np.array([[time, amount] + v_features])

    prediction = model.predict(input_data)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
        st.write(f"Fraud Probability: {prob:.2f}")

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

