!pip install streamlit
import streamlit as st
import pandas as pd
import joblib

@st.cache(allow_output_mutation=True)
def load_artifacts():
    return joblib.load("data/bcsc_xgb_model.pkl")

model = load_artifacts()

st.title("BCSC Breast Cancer Risk Predictor")
st.sidebar.header("Patient Profile")

# --- your mappings ---
age_map = {
    "18–29":1, "30–34":2, "35–39":3, "40–44":4, "45–49":5,
    "50–54":6, "55–59":7, "60–64":8, "65–69":9, "70–74":10,
    "75–79":11, "80–84":12, ">85":13
}
race_map = {
    "Non-Hispanic white":1, "Non-Hispanic black":2,
    "Asian/Pacific Islander":3, "Native American":4,
    "Hispanic":5, "Other/mixed":6, "Unknown":9
}
men_map = {"<12":2, "12–13":1, ">14":0, "Unknown":9}
afb_map = {"<20":0, "20–24":1, "25–29":2, ">30":3, "Nulliparous":4, "Unknown":9}
fh_map = {"No":0, "Yes":1, "Unknown":9}
den_map = {
    "Almost entirely fat":1, "Scattered fibroglandular":2,
    "Heterogeneously dense":3, "Extremely dense":4, "Unknown":9
}
hrt_map = {"No":0, "Yes":1, "Unknown":9}
meno_map = {"Pre-/peri-menopausal":1, "Post-menopausal":2, "Surgical menopause":3, "Unknown":9}
bmi_map = {"10–24.99":1, "25–29.99":2, "30–34.99":3, "≥35":4, "Unknown":9}
bx_map = {"No":0, "Yes":1, "Unknown":9}

# --- sidebar selects ---
age       = st.sidebar.selectbox("Age group", list(age_map.keys()))
race      = st.sidebar.selectbox("Race/ethnicity", list(race_map.keys()))
menarche  = st.sidebar.selectbox("Age at menarche", list(men_map.keys()))
afb       = st.sidebar.selectbox("Age at first birth", list(afb_map.keys()))
fam_hx    = st.sidebar.selectbox("First-degree family history", list(fh_map.keys()))
density   = st.sidebar.selectbox("BI-RADS density", list(den_map.keys()))
hrt       = st.sidebar.selectbox("Current hormone therapy use", list(hrt_map.keys()))
meno      = st.sidebar.selectbox("Menopausal status", list(meno_map.keys()))
bmi       = st.sidebar.selectbox("BMI group", list(bmi_map.keys()))
biopsy    = st.sidebar.selectbox("Biopsy history", list(bx_map.keys()))

# --- build the DataFrame correctly ---
input_dict = {
    "age_group_5_years":       age_map[age],
    "race_eth":                race_map[race],
    "age_menarche":            men_map[menarche],
    "age_first_birth":         afb_map[afb],
    "first_degree_hx":         fh_map[fam_hx],                  # <- fix here
    "BIRADS_breast_density":   den_map[density],
    "current_hrt":             hrt_map[hrt],
    "menopaus":                meno_map[meno],
    "bmi_group":               bmi_map[bmi],
    "biophx":                  bx_map[biopsy],
}

X_input = pd.DataFrame([input_dict])

if st.button("Predict risk"):
    prob = model.predict_proba(X_input)[0, 1]
    st.write(f"🔍 **Predicted probability of prior breast cancer:** {prob:.1%}")
