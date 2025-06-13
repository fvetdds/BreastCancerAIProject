import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎗️ EmpowerHER", layout="wide")

# ─── CACHED MODEL & FEATURES ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parent / "models"
    model = joblib.load(base / "bcsc_xgb_model.pkl")
    threshold = joblib.load(base / "threshold.pkl")
    feature_names = model.get_booster().feature_names
    return model, threshold, feature_names

# ─── CACHED PREDICTION ─────────────────────────────────────────────────────────
@st.cache_data
def compute_prediction(input_dict):
    model, threshold, feature_names = load_model()
    X = np.array([input_dict[name] for name in feature_names], dtype=np.float32).reshape(1, -1)
    prob = model.predict_proba(X)[0, 1]
    return float(prob), bool(prob >= threshold), threshold

# ─── CHOICE DICTIONARIES ───────────────────────────────────────────────────────
age_groups  = {1:"18–29", 2:"30–34", 3:"35–39", 4:"40–44", 5:"45–49", 6:"50–54",
               7:"55–59", 8:"60–64", 9:"65–69", 10:"70–74", 11:"75–79", 12:"80–84", 13:">85"}
race_eth    = {1:"White", 2:"Black", 3:"Asian or Pacific Island", 4:"Native American",
               5:"Hispanic", 6:"Other"}
menarche    = {0:">14", 1:"12–13", 2:"<12"}
birth_age   = {0:"<20", 1:"20–24", 2:"25–29", 3:">30", 4:"Nulliparous"}
fam_hist    = {0:"No", 1:"Yes"}
biopsy      = {0:"No", 1:"Yes"}
density     = {1:"Almost fat", 2:"Scattered", 3:"Hetero-dense", 4:"Extremely"}
hormone_use = {0:"No", 1:"Yes"}
menopause   = {1:"Pre/peri", 2:"Post", 3:"Surgical"}
bmi_group   = {1:"10–24.9", 2:"25–29.9", 3:"30–34.9", 4:"35+"}

# ─── HEADER IMAGE or TITLE ────────────────────────────────────────────────────
base_dir   = Path(__file__).resolve().parent
title_path = base_dir / "assets" / "title.png"
if title_path.exists():
    st.image(str(title_path), use_column_width=True)
else:
    st.title("🎗️ EmpowerHER")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Risk Insight", "Mind & Move"])

# ─── TAB 1: RISK INSIGHT ───────────────────────────────────────────────────────
with tab1:
    st.sidebar.header("Breast Cancer Risk")
    with st.sidebar.form(key="risk_form"):
        age          = st.selectbox("Age group", list(age_groups),   format_func=lambda k: age_groups[k])
        race         = st.selectbox("Race/Ethnicity", list(race_eth),format_func=lambda k: race_eth[k])
        menarche_age = st.selectbox("Age at 1st period", list(menarche),
                                    format_func=lambda k: menarche[k])
        first_birth  = st.selectbox("Age at first birth", list(birth_age),
                                    format_func=lambda k: birth_age[k])
        family_hist  = st.selectbox("Family history", list(fam_hist),
                                    format_func=lambda k: fam_hist[k])
        biopsy_hist  = st.selectbox("Biopsy history", list(biopsy),
                                    format_func=lambda k: biopsy[k])
        dens         = st.selectbox("BI-RADS density", list(density),
                                    format_func=lambda k: density[k])
        horm_use     = st.selectbox("Hormone use", list(hormone_use),
                                    format_func=lambda k: hormone_use[k])
        meno_status  = st.selectbox("Menopausal status", list(menopause),
                                    format_func=lambda k: menopause[k])
        bmi          = st.selectbox("BMI group", list(bmi_group),
                                    format_func=lambda k: bmi_group[k])
        submit       = st.form_submit_button("Predict Risk")

    if submit:
        with st.spinner("Calculating…"):
            inputs = {
                "age_group":         age,
                "race_eth":          race,
                "age_menarche":      menarche_age,
                "age_first_birth":   first_birth,
                "family_history":    family_hist,
                "personal_biopsy":   biopsy_hist,
                "density":           dens,
                "hormone_use":       horm_use,
                "menopausal_status": meno_status,
                "bmi_group":         bmi,
            }
            prob, is_high, threshold = compute_prediction(inputs)
            icon = "⚠️" if is_high else "✅"
            st.write(f"Predicted probability: **{prob:.1%}**")
            if is_high:
                st.error(f"{icon} High risk (threshold {threshold:.2f})")
            else:
                st.success(f"{icon} Low risk (threshold {threshold:.2f})")

# ─── TAB 2: MIND & MOVE ───────────────────────────────────────────────────────
with tab2:
    st.header("Mind & Move")
    st.subheader("Daily Self-Care Tips")
    tips = [
        "🧘 Practice 10 mins of mindfulness",
        "🥗 Eat ≥5 servings of fruits/veggies",
        "🚶‍♀️ 30 mins of light exercise",
        "💧 Drink ≥8 glasses of water",
        "😴 Aim for 7–8 hours of sleep",
    ]
    for tip in tips:
        st.markdown(f"- {tip}")

    st.subheader("Glow & Grow Log")
    c1, c2, c3 = st.columns(3)
    med_mins = c1.number_input("Meditation minutes", 0, 60, 0)
    ex_mins  = c2.number_input("Exercise minutes", 0, 180, 0)
    water    = c3.number_input("Water glasses", 0, 20, 0)

    diet_log = st.text_area("Diet log (meals/snacks)")
    if st.button("Save Entry"):
        st.json({
            "date":       datetime.now().strftime("%Y-%m-%d"),
            "meditation": med_mins,
            "exercise":   ex_mins,
            "water":      water,
            "diet_log":   diet_log
        })
