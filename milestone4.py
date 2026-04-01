import streamlit as st
import pandas as pd
import joblib

# Load trained model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# Page settings
st.set_page_config(page_title="Visa Predictor", layout="centered")

# Title
st.title("Visa Processing Time Predictor")


st.markdown("---")

# Input fields
st.subheader("Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=60, value=25)

country = st.selectbox(
    "Applicant Country",
    ["India", "USA", "UK"]
)

visa_type = st.selectbox(
    "Visa Type",
    ["Student", "Work", "Tourist"]
)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelors", "Masters"]
)

employment = st.selectbox(
    "Employment Status",
    ["Employed", "Unemployed"]
)

sponsor = st.selectbox(
    "Sponsor Company",
    ["Yes", "No"]
)

st.markdown("---")

# Prediction button
if st.button("🚀 Predict Processing Time"):

    # Create input data
    input_data = {
        "age": age,
        "applicant_country": country,
        "visa_type": visa_type,
        "education_level": education,
        "employment_status": employment,
        "sponsor_company": sponsor
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert categorical variables
    input_df = pd.get_dummies(input_df)

    # Match training features
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Estimated Processing Time: {round(prediction, 2)} days")