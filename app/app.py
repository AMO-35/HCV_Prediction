import streamlit as st
import pandas as pd
import joblib

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Hepatitis C Prediction App", page_icon="🩺", layout="centered")

# ---- HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #FF5733;'>Hepatitis C Prediction App</h1>
    <p style='text-align: center;'>Predict the category of Hepatitis C based on medical parameters.</p>
    <hr style='border: 1px solid #FF5733;'>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.header("User Input Parameters")

Age = st.sidebar.slider("Age", min_value=0, max_value=100, value=32)
Sex = st.sidebar.selectbox("Sex", ["m", "f"], index=0)
ALB = st.sidebar.number_input("Albumin (ALB)", min_value=0.0, max_value=100.0, value=38.5)
ALP = st.sidebar.number_input("Alkaline Phosphatase (ALP)", min_value=0.0, max_value=500.0, value=52.5)
ALT = st.sidebar.number_input("Alanine Transaminase (ALT)", min_value=0.0, max_value=350.0, value=7.7)
AST = st.sidebar.number_input("Aspartate Transaminase (AST)", min_value=0.0, max_value=300.0, value=22.1)
BIL = st.sidebar.number_input("Bilirubin (BIL)", min_value=0.0, max_value=300.0, value=7.5)
CHE = st.sidebar.number_input("Cholinesterase (CHE)", min_value=0.0, max_value=50.0, value=6.93)
CHOL = st.sidebar.number_input("Cholesterol (CHOL)", min_value=0.0, max_value=50.0, value=3.23)
CREA = st.sidebar.number_input("Creatinine (CREA)", min_value=0.0, max_value=1050.0, value=106.0)
GGT = st.sidebar.number_input("Gamma-Glutamyl Transferase (GGT)", min_value=0.0, max_value=800.0, value=12.1)
PROT = st.sidebar.number_input("Total Protein (PROT)", min_value=0.0, max_value=100.0, value=69.0)

# ---- INPUT DATA ----
input_data = {
    "Age": Age,
    "Sex": Sex,
    "ALB": ALB,
    "ALP": ALP,
    "ALT": ALT,
    "AST": AST,
    "BIL": BIL,
    "CHE": CHE,
    "CHOL": CHOL,
    "CREA": CREA,
    "GGT": GGT,
    "PROT": PROT
}
input_data_df = pd.DataFrame([input_data])

# ---- LOAD MODEL & PREDICT ----
model = joblib.load('C:\\HCV Prediction\\models\\best_model_with_pipeline.pkl') # Ensure the model is trained and saved

# Use the model's pipeline to make predictions (it will handle the preprocessing automatically)
result = model.predict(input_data_df)[0]

# ---- DISPLAY RESULTS ----
st.markdown("""
    <h3 style='text-align: center; color: #3498db;'>User Input Data</h3>
""", unsafe_allow_html=True)
st.table(input_data_df)

# ---- RESULT DESIGN ----
prediction_text = f"Predicted Category: {result}"
prediction_color = "#e74c3c" if "Hepatitis" in result else "#2ecc71"

st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: {prediction_color}; color: white; border-radius: 10px;'>
        <h2>{prediction_text}</h2>
    </div>
""", unsafe_allow_html=True)
