import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os

# -----------------------------------------
# Inject a professional blue‑gray theme
# -----------------------------------------
custom_css = """
<style>
.reportview-container, .main .block-container {
    background-color: #f5f7fa;
}
.sidebar .sidebar-content {
    background-color: #2c3e50;
    color: #ecf0f1;
}
.stApp h1 {
    color: #2c3e50;
}
.stApp h2, .stApp h3, .stApp h4 {
    color: #34495e;
}
.stButton > button {
    background-color: #2980b9;
    color: white;
    border-radius: 5px;
    padding: 0.5em 1em;
}
.stLabel {
    color: #2c3e50;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------
# Load trained XGBoost model and encoders
# -----------------------------------------
MODEL_PATH = os.path.join("data", "bcsc_xgb_model.pkl")
ENCODERS_PATH = os.path.join("data", "bcsc_feature_encoders.pkl")
TARGET_ENCODER_PATH = os.path.join("data", "bcsc_target_encoder.pkl")

@st.cache_resource
def load_model_and_encoders():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    feature_encoders = joblib.load(ENCODERS_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH) if os.path.exists(TARGET_ENCODER_PATH) else None
    return model, feature_encoders, target_encoder

model, feature_encoders, target_encoder = load_model_and_encoders()

if model is None or feature_encoders is None:
    st.error("Model or encoders not found. Please run the training script first.")
    st.stop()

# -----------------------------------------
# Dropdown options for each covariate
# -----------------------------------------
age_group_options = [
    "1 = Age 18-29", "2 = Age 30-34", "3 = Age 35-39", "4 = Age 40-44", 
    "5 = Age 45-49", "6 = Age 50-54", "7 = Age 55-59", "8 = Age 60-64", 
    "9 = Age 65-69", "10 = Age 70-74", "11 = Age 75-79", "12 = Age 80-84", "13 = Age ≥85"
]
race_eth_options = [
    "1 = Non-Hispanic white", "2 = Non-Hispanic black", "3 = Asian/Pacific Islander", 
    "4 = Native American", "5 = Hispanic", "6 = Other/mixed", "9 = Unknown"
]
first_degree_hx_options = ["0 = No", "1 = Yes", "9 = Unknown"]
age_menarche_options = ["0 = Age ≥14", "1 = Age 12-13", "2 = Age <12", "9 = Unknown"]
age_first_birth_options = [
    "0 = Age <20", "1 = Age 20-24", "2 = Age 25-29", "3 = Age ≥30", "4 = Nulliparous", "9 = Unknown"
]
birads_density_options = [
    "1 = Almost entirely fat", "2 = Scattered fibroglandular densities", 
    "3 = Heterogeneously dense", "4 = Extremely dense", "9 = Unknown/different system"
]
current_hrt_options = ["0 = No", "1 = Yes", "9 = Unknown"]
menopaus_options = ["1 = Pre- or peri‑menopausal", "2 = Post‑menopausal", "3 = Surgical menopause", "9 = Unknown"]
bmi_group_options = ["1 = 10-24.99", "2 = 25-29.99", "3 = 30-34.99", "4 = 35 or more", "9 = Unknown"]
biophx_options = ["0 = No", "1 = Yes", "9 = Unknown"]

# -----------------------------------------
# Build the Streamlit interface
# -----------------------------------------
st.set_page_config(page_title="Breast Cancer Risk Predictor", layout="centered")
st.title("Breast Cancer Risk Predictor")
st.write("Enter your clinical/historical data and click 'Predict Risk'.")

col1, col2 = st.columns(2)
with col1:
    age_group_sel = st.selectbox("Age Group (5-year)", age_group_options)
    race_sel = st.selectbox("Race/Ethnicity", race_eth_options)
    fhx_sel = st.selectbox("First‑degree Family History", first_degree_hx_options)
    menarche_sel = st.selectbox("Age at Menarche", age_menarche_options)
    first_birth_sel = st.selectbox("Age at First Birth", age_first_birth_options)

with col2:
    density_sel = st.selectbox("BI‑RADS Density", birads_density_options)
    hrt_sel = st.selectbox("Hormone Replacement Therapy", current_hrt_options)
    menopaus_sel = st.selectbox("Menopausal Status", menopaus_options)
    bmi_sel = st.selectbox("BMI Group (kg/m²)", bmi_group_options)
    biophx_sel = st.selectbox("Previous Breast Biopsy/Aspiration", biophx_options)

# Helper to extract the integer code from "X = Label" text
def parse_code(selection):
    try:
        return int(selection.split("=")[0].strip())
    except:
        return np.nan

input_dict = {
    'age_group_5_years': parse_code(age_group_sel),
    'race_eth': parse_code(race_sel),
    'first_degree_hx': parse_code(fhx_sel),
    'age_menarche': parse_code(menarche_sel),
    'age_first_birth': parse_code(first_birth_sel),
    'BIRADS_breast_density': parse_code(density_sel),
    'current_hrt': parse_code(hrt_sel),
    'menopaus': parse_code(menopaus_sel),
    'bmi_group': parse_code(bmi_sel),
    'biophx': parse_code(biophx_sel)
}

input_df = pd.DataFrame([input_dict])

# Ensure column order matches training
feature_cols = model.get_booster().feature_names
input_df = input_df[feature_cols].fillna(0)

# Prediction function
def predict_risk():
    proba = model.predict_proba(input_df)[0]
    classes = target_encoder.classes_ if target_encoder else ["0", "1"]
    pred_idx = np.argmax(proba)
    pred_label = classes[pred_idx]
    return pred_label, proba

if st.button("Predict Risk"):
    with st.spinner("Computing risk..."):
        label, probabilities = predict_risk()
    st.subheader("Predicted Breast Cancer History")
    st.write(f"  • Label: **{label}**")
    # Show bar chart of probabilities
    prob_df = pd.DataFrame({
        'Outcome': target_encoder.classes_ if target_encoder else ["0", "1"],
        'Probability': probabilities
    }).set_index('Outcome')
    st.bar_chart(prob_df)

    # Simple recommendations
    st.subheader("Recommendations")
    if label == "0":
        st.write("Lower risk—continue routine screening and maintain a healthy lifestyle.")
    else:
        st.write("Elevated risk—consult your healthcare provider for enhanced screening and genetic counseling.")

    # Supportive resources
    st.subheader("Resources")
    st.markdown("- [Breast Cancer Foundation Grants](https://www.bcfoundation.org/grants)")
    st.markdown("- [Local Support Groups](https://www.breastcancer.org/support/local-groups)")
    st.markdown("- [Educational Video on Breast Health](https://www.youtube.com/watch?v=example)")
