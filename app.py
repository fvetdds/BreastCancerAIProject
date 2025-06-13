import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page config 
st.set_page_config(page_title="🎗️ EmpowerHER", layout="wide")

# Cached loaders 
@st.cache_resource
def load_model_and_threshold():
    base = Path(__file__).resolve().parent / "models"
    model = joblib.load(base / "bcsc_xgb_model.pkl")
    threshold = joblib.load(base / "threshold.pkl")
    return model, threshold

@st.cache_data
def load_survival_data():
    data_path = Path(__file__).resolve().parent / "data" / "METABRIC.csv"
    if not data_path.exists():
        data_path = Path("../data") / "METABRIC.csv"
    return pd.read_csv(data_path)

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

def sel(label, opts, key):
    return st.sidebar.selectbox(label,
                                list(opts.keys()),
                                format_func=lambda k: opts[k],
                                key=key)
# Title or image 
BASE_DIR = Path(__file__).resolve().parent
title_img = BASE_DIR / "assets" / "title.png"
if title_img.exists():
    st.image(str(title_img), use_column_width=True)
else:
    st.title("🎗️ EmpowerHER")

# Tabs 
tab1, tab2, tab3 = st.tabs(["Risk Insight", "Thrive Forecast", "Mind & Move"])

# --- Tab 1: Risk Insight 
with tab1:
    st.sidebar.header("Your information for risk prediction")

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

    st.subheader("Breast Cancer Risk Prediction")
    if st.sidebar.button("Predict Risk", key="predict_risk"):
        with st.spinner("Running model..."):
            model, threshold = load_model_and_threshold()
            raw_df = pd.DataFrame(inputs, index=[0])
            expected = model.get_booster().feature_names
            df_new = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)

            prob = model.predict_proba(df_new)[0, 1]
            high = prob >= threshold
            icon = "⚠️" if high else "✅"
            st.write(f"Predicted probability: {prob:.1%}")
            if high:
                st.error(f"{icon} High risk (thr={threshold:.2f})")
            else:
                st.success(f"{icon} Low risk (thr={threshold:.2f})")

# --- Tab 2: Thrive Forecast 
with tab2:
    st.header("Thrive Forecast")
    if st.button("Load Survival Data", key="load_survival"):
        with st.spinner("Loading survival data..."):
            surv_df = load_survival_data()
            st.dataframe(surv_df)  

# --- Tab 3: Mind & Move 
with tab3:
    st.header("Mind & Move")
    st.subheader("Daily Selfcare Tips")
    tips = ["🧘 Practice 10 mins of mindfulness","🥗 Eat ≥5 servings of fruits/veggies","🚶‍♀️ 30 mins of light exercise","💧 Drink ≥8 glasses of water","😴 Aim for 7-8 hours of sleep"]
    for tip in tips: st.markdown(f"- {tip}")
    st.subheader("GLow & Grow Log")
    c1, c2, c3 = st.columns(3)
    with c1: med_mins = st.number_input("Meditation minutes", 0, 60, 0)
    with c2: ex_mins = st.number_input("Exercise minutes", 0, 180, 0)
    with c3: water = st.number_input("Water glasses", 0, 20, 0)
    diet_log = st.text_area("Diet log (meals/snacks)")
    if st.button("Save Entry"): st.json({"date": pd.Timestamp.now().strftime("%Y-%m-%d"),"meditation": med_mins,"exercise": ex_mins,"water": water,"diet": diet_log})
    st.subheader("Additional Resources")
    st.markdown("**YouTube Videos:**")
    vids = {"Mindfulness Meditation": "https://www.youtube.com/watch?v=inpok4MKVLM","Gentle Yoga": "https://www.youtube.com/watch?v=v7AYKMP6rOE","Healthy Eating": "https://www.youtube.com/watch?v=5kV8XjvM8k8"}
    for t,u in vids.items(): st.markdown(f"- [{t}]({u})")
    st.markdown("**Local Support (Nashville, TN):**")
    groups = [{"name":"Susan G. Komen Nashville","phone":"(615)673-6633","url":"https://komen.org/nashville"},{"name":"Vanderbilt Support","phone":"(615)322-3900","url":"https://www.vicc.org/support-groups"},{"name":"Alive Hospice","phone":"(615)327-1085","url":"https://alivehospice.org"}]
    for grp in groups: st.markdown(f"- **{grp['name']}**: {grp['phone']} | [Website]({grp['url']})")
