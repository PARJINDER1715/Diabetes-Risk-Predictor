import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# === App Config ===
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# === Title and Description ===
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Use this app to predict whether a person is likely to have diabetes based on key health indicators.")

st.markdown("<hr>", unsafe_allow_html=True)

# === Input Section ===
st.header("📋 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, value=1,
                                  help="Number of times the patient has been pregnant")
    glucose = st.number_input("🍬 Glucose Level", min_value=0, value=120,
                              help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
    blood_pressure = st.number_input("🩸 Blood Pressure (mm Hg)", min_value=0, value=70,
                                     help="Diastolic blood pressure")
    skin_thickness = st.number_input("🧪 Skin Thickness (mm)", min_value=0, value=20,
                                     help="Triceps skinfold thickness")

with col2:
    insulin = st.number_input("💉 Insulin Level (mu U/ml)", min_value=0, value=80,
                              help="2-Hour serum insulin")
    bmi = st.number_input("⚖️ BMI", min_value=0.0, value=25.0,
                          help="Body mass index (weight in kg/(height in m)^2)")
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, value=0.5,
                          help="Function that scores likelihood of diabetes based on family history")
    age = st.number_input("🎂 Age", min_value=0, value=30,
                          help="Age of the patient")

# === Prediction ===
st.markdown("###")
if st.button("🔍 Predict"):
    st.markdown("⏳ Running prediction...")

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ **Result: At Risk** — The model predicts that the patient is likely to have **diabetes**.")
    else:
        st.success("✅ **Result: Healthy** — The model predicts that the patient is **not likely** to have diabetes.")

# === Footer with LinkedIn and Spacing ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 16px;'>
        👨‍💻 Developed by <strong>PARJINDER SINGH</strong><br><br>
        <a href='https://www.linkedin.com/in/parjinder-singh' target='_blank' style='
            background-color:#0e76a8;
            color:white;
            padding: 10px 20px;
            text-decoration:none;
            border-radius:8px;
            display:inline-block;
            font-weight:bold;
            font-size:16px;'>
            🔗 Connect on LinkedIn
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
