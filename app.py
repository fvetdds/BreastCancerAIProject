import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Breast Cancer Risk", layout="centered")
st.title("🎗️ Breast Cancer Risk Predictor")

# ─── Load model & threshold ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# ─── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Your information")
def sel(label, opts):
    return st.sidebar.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])

# Define your feature options
age_groups  = {1:"18–29", 2:"30–34", 3:"35–39", 4:"40–44", 5:"45–49", 6:"50–54", 7:"55–59", 8:"60–64", 9:"65–69", 10:"70–74", 11:"75–79", 12:"80–84", 13:">85"}
race_eth    = {1:"White", 2:"Black", 3:"Asian or Pacific Island", 4:"Native American", 5:"Hispanic", 6:"Other"}
menarche    = {0:">14", 1:"12–13", 2:"<12"}
birth_age   = {0:"<20", 1:"20–24", 2:"25–29", 3:">30", 4:"Nulliparous"}
fam_hist    = {0:"No", 1:"Yes"}
biopsy      = {0:"No", 1:"Yes"}
density     = {1:"Almost fat", 2:"Scattered", 3:"Hetero-dense", 4:"Extremely"}
hormone_use = {0:"No", 1:"Yes"}
menopause   = {1:"Pre/peri", 2:"Post", 3:"Surgical"}
bmi_group   = {1:"10–24.9", 2:"25–29.9", 3:"30–34.9", 4:"35+"}

# Collect inputs
inputs = {
    "age_group":         sel("Age group", age_groups),
    "race_eth":          sel("Race/Ethnicity", race_eth),
    "age_menarche":      sel("Age at 1st period", menarche),
    "age_first_birth":   sel("Age at first birth", birth_age),
    "family_history":    sel("Family history of cancer", fam_hist),
    "personal_biopsy":   sel("Personal biopsy history", biopsy),
    "density":           sel("BI-RADS density", density),
    "hormone_use":       sel("Hormone use", hormone_use),
    "menopausal_status": sel("Menopausal status", menopause),
    "bmi_group":         sel("BMI group", bmi_group),
}

# Build DataFrame and align features
raw_df = pd.DataFrame(inputs, index=[0])
expected = model.get_booster().feature_names
# Add missing features, drop extras, and reorder
df_new = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)

# Predict probability
prob = model.predict_proba(df_new)[0, 1]

# Determine risk label
risk_str = "High risk" if prob >= threshold else "Low risk"
icon = "⚠️" if risk_str == "High risk" else "✅"

# Display results
st.subheader("Results")
st.write(f"Predicted probability: {prob:.1%}")
if risk_str == "High risk":
    st.error(f"{icon} {risk_str} (threshold = {threshold:.2f})")
else:
    st.success(f"{icon} {risk_str} (threshold = {threshold:.2f})")
