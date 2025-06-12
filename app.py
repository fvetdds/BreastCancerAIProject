import streamlit as st
import pandas as pd
import joblib

# --- Page config & title ---
st.set_page_config(page_title="Breast Cancer Risk", layout="centered")
st.title("🎗️ Breast Cancer Risk Predictor")

# --- Load artifacts ---
model = joblib.load("models/bcsc_xgb_model.pkl")
threshold = joblib.load("models/threshold.pkl")

# --- Sidebar inputs ---
st.sidebar.header("Patient Profile")
def sel(label, opts): return st.sidebar.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])

age_groups = {1:"18–29",2:"30–34",3:"35–39",4:"40–44",5:"45–49",6:"50–54",7:"55–59",8:"60–64",9:"65–69",10:"70–74",11:"75–79",12:"80–84",13:">85"}
race_eth   = {1:"NH white",2:"NH black",3:"Asian/PI",4:"Native Am",5:"Hispanic",6:"Other"}
menarche    = {0:">14",1:"12–13",2:"<12"}
birth_age   = {0:"<20",1:"20–24",2:"25–29",3:">30",4:"Nulliparous"}
fam_hist    = {0:"No",1:"Yes"}
biopsy      = {0:"No",1:"Yes"}
density     = {1:"Almost fat",2:"Scattered",3:"Hetero-dense",4:"Extremely"}
hormone_use = {0:"No",1:"Yes"}
menopause   = {1:"Pre/peri",2:"Post",3:"Surgical"}
bmi_group   = {1:"10–24.9",2:"25–29.9",3:"30–34.9",4:"35+"}

inputs = {
    "age_group":       sel("Age group", age_groups),
    "race_ethnicity":  sel("Race/Ethnicity", race_eth),
    "age_menarche":    sel("Age at menarche", menarche),
    "age_first_birth": sel("Age at first birth", birth_age),
    "family_history":  sel("Family history", fam_hist),
    "personal_biopsy": sel("Personal biopsy history", biopsy),
    "density":         sel("BI-RADS density", density),
    "hormone_use":     sel("Hormone use", hormone_use),
    "menopausal_status": sel("Menopausal status", menopause),
    "bmi_group":       sel("BMI group", bmi_group),
}

# --- Prediction ---
df_new = pd.DataFrame([inputs])
prob = model.predict_proba(df_new)[0,1]
label = "⚠️ High risk" if prob >= threshold else "✅ Low risk"

# --- Risk bucket text ---
if prob < 0.20:
    bucket = "Low risk (<20%)"
elif prob < 0.50:
    bucket = "Moderate risk (20–50%)"
else:
    bucket = "High risk (>50%)"

# --- Display ---
st.subheader("Results")
st.metric("Predicted probability", f"{prob:.1%}", delta=None)
st.write(f"**Risk bucket:** {bucket}")
st.write(f"**Binary call (thr={threshold:.2f}):** {label}")

st.markdown("---")
st.write("**Inputs:**")
st.json(inputs)
