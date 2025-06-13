import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎗️ EmpowerHER", layout="wide")

# ─── Optional CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-baseweb="select"] > div { min-width:200px; max-width:220px; }
    [role="tab"] { font-size:18px; color:#FF8C00; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Cached loaders (won’t run until called) ─────────────────────────────────
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
        data_path = Path("/mnt/data") / "METABRIC.csv"
    return pd.read_csv(data_path)

# ─── Title or image ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
title_img = BASE_DIR / "assets" / "title.png"
if title_img.exists():
    st.image(str(title_img), use_column_width=True)
else:
    st.title("🎗️ EmpowerHER")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Risk Insight", "Thrive Forecast", "Mind & Move"])

# --- Tab 1: Risk Insight ---
with tab1:
    st.sidebar.header("Your information for risk prediction")

    def sel(label, opts):
        return st.sidebar.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])

    age_groups = {1: "18–29", 2: "30–34", 3: "35–39", 4: "40–44", 5: "45–49",
                  6: "50–54", 7: "55–59", 8: "60–64", 9: "65–69", 10: "70–74",
                  11: "75–79", 12: "80–84", 13: ">85"}
    # ... (other mapping dicts unchanged) ...
    bmi_group = {1: "10–24.9", 2: "25–29.9", 3: "30–34.9", 4: "35+"}

    inputs = {
        "age_group": sel("Age group", age_groups),
        # … all other inputs …
        "bmi_group": sel("BMI group", bmi_group),
    }

    st.subheader("Breast Cancer Risk Prediction")
    if st.sidebar.button("Predict Risk"):
        # now the model only loads when you click
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

# --- Tab 2: Thrive Forecast ---
with tab2:
    st.header("Thrive Forecast")
    if st.button("Load Survival Data"):
        surv_df = load_survival_data()
        st.dataframe(surv_df)  # or whatever processing you need

# --- Tab 3: Mind & Move ---
with tab3:
    st.header("Mind & Move")
    st.subheader("Daily Wellness Tips")
    tips = ["🧘 Practice 10 mins of mindfulness","🥗 Eat ≥5 servings of fruits/veggies","🚶‍♀️ 30 mins of light exercise","💧 Drink ≥8 glasses of water","😴 Aim for 7-8 hours of sleep"]
    for tip in tips: st.markdown(f"- {tip}")
    st.subheader("Your Daily Tracker")
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
