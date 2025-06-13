import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ─── Page config (first Streamlit call) ───────────────────────────────────────
st.set_page_config(
    page_title="🎗️ EmpowerHER",
    layout="wide"
)

# ─── Optional CSS for styling ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar dropdown width */
    [data-baseweb="select"] > div { min-width: 200px !important; max-width: 220px !important; }
    /* Tab labels font and color */
    [role="tab"] { font-size: 18px !important; color: #FF8C00 !important; }
    </style>
    """, unsafe_allow_html=True
)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("🎗️ Breast Cancer Risk factors and 5-Year Survival prediction")

# ─── Load models and data ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
# Classification model
model = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")
# Survival data (METABRIC.csv)
data_path = BASE_DIR / "data" / "METABRIC.csv"
if not data_path.exists():
    # Fallback if file is in /mnt/data
    data_path = Path('/mnt/data') / 'METABRIC.csv'
surv_df = pd.read_csv(data_path)

# ─── Tabs setup ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Risk Predictor", "5-Year Survival", "Wellness & Tracker"])

# --- Tab 1: Risk Predictor ---
with tab1:
    st.sidebar.header("Your information for risk prediction")
    def sel(label, opts):
        return st.sidebar.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])

    # Dropdown options
    age_groups  = {1:"18–29",2:"30–34",3:"35–39",4:"40–44",5:"45–49",6:"50–54",7:"55–59",8:"60–64",9:"65–69",10:"70–74",11:"75–79",12:"80–84",13:">85"}
    race_eth    = {1:"White",2:"Black",3:"Asian/Pacific",4:"Native American",5:"Hispanic",6:"Other"}
    menarche    = {0:">14",1:"12–13",2:"<12"}
    birth_age   = {0:"<20",1:"20–24",2:"25–29",3:">30",4:"Nulliparous"}
    fam_hist    = {0:"No",1:"Yes"}
    biopsy      = {0:"No",1:"Yes"}
    density     = {1:"Almost fat",2:"Scattered",3:"Hetero-dense",4:"Extremely"}
    hormone_use = {0:"No",1:"Yes"}
    menopause   = {1:"Pre/peri",2:"Post",3:"Surgical"}
    bmi_group   = {1:"10–24.9",2:"25–29.9",3:"30–34.9",4:"35+"}

    # Collect inputs
    inputs = {
        "age_group":        sel("Age group", age_groups),
        "race_eth":         sel("Race/Ethnicity", race_eth),
        "age_menarche":     sel("Age at 1st period", menarche),
        "age_first_birth":  sel("Age at first birth", birth_age),
        "family_history":   sel("Family history of cancer", fam_hist),
        "personal_biopsy":  sel("Personal biopsy history", biopsy),
        "density":          sel("BI-RADS density", density),
        "hormone_use":      sel("Hormone use", hormone_use),
        "menopausal_status":sel("Menopausal status", menopause),
        "bmi_group":        sel("BMI group", bmi_group),
    }

    # Prepare DataFrame
    raw_df = pd.DataFrame(inputs, index=[0])
    expected = model.get_booster().feature_names
    df_new = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)

    # Predict
    prob = model.predict_proba(df_new)[0, 1]
    risk_str = "High risk" if prob >= threshold else "Low risk"
    icon = "⚠️" if risk_str == "High risk" else "✅"

    # Display
    st.subheader("Breast Cancer Risk Prediction")
    st.write(f"Predicted probability: {prob:.1%}")
    if risk_str == "High risk":
        st.error(f"{icon} {risk_str} (threshold = {threshold:.2f})")
    else:
        st.success(f"{icon} {risk_str} (threshold = {threshold:.2f})")

# --- Tab 2: 5-Year Survival ---
with tab2:
    st.header("5-Year Survival by Gene Mutation Markers")
    gene_cols = [c for c in surv_df.columns if c.endswith("_mut")]
    selected = st.multiselect("Select gene markers", gene_cols)
    if selected:
        med = surv_df[selected].median()
        mask = np.logical_and.reduce([surv_df[g] >= med[g] for g in selected])
        filtered = surv_df[mask]
        if filtered.empty:
            st.warning("No matching patients.")
        else:
            base = (surv_df.overall_survival_months >= 60).mean()
            filt = (filtered.overall_survival_months >= 60).mean()
            st.write(f"Baseline 5-yr survival: {base:.1%} (n={len(surv_df)})")
            st.write(f"Filtered 5-yr survival: {filt:.1%} (n={len(filtered)})")
            st.dataframe(filtered[["patient_id","overall_survival_months"]+selected].head(10))
    else:
        st.info("Select markers to view survival probability.")

# --- Tab 3: Wellness & Tracker ---
with tab3:
    st.header("Wellness & Life Coaching & Resources")
    st.subheader("Daily Wellness Tips")
    for tip in [
        "🧘 Practice 10 mins of mindfulness",
        "🥗 Eat ≥5 servings of fruits/veggies",
        "🚶‍♀️ 30 mins of light exercise",
        "💧 Drink ≥8 glasses of water",
        "😴 Aim for 7-8 hours of sleep"
    ]:
        st.markdown(f"- {tip}")
    st.subheader("Your Daily Tracker")
    c1, c2, c3 = st.columns(3)
    with c1:
        med_mins = st.number_input("Meditation minutes", 0, 60, 0)
    with c2:
        ex_mins = st.number_input("Exercise minutes", 0, 180, 0)
    with c3:
        water = st.number_input("Glasses of water", 0, 20, 0)
    diet = st.text_area("Diet log (meals/snacks)")
    if st.button("Save Entry"):
        st.success("Entry recorded!")
        st.json({"date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                 "meditation": med_mins, "exercise": ex_mins,
                 "water": water, "diet": diet})
    st.subheader("Additional Resources")
    st.markdown("**YouTube Videos:**")
    vids = {
        "Mindfulness Meditation": "https://www.youtube.com/watch?v=inpok4MKVLM",
        "Gentle Yoga": "https://www.youtube.com/watch?v=v7AYKMP6rOE",
        "Healthy Eating": "https://www.youtube.com/watch?v=5kV8XjvM8k8"
    }
    for t,u in vids.items(): st.markdown(f"- [{t}]({u})")
    st.markdown("**Local Support Groups (Nashville, TN):**")
    for g in [
        {"name":"Susan G. Komen Nashville","phone":"(615)673-6633","url":"https://komen.org/nashville"},
        {"name":"Vanderbilt Support","phone":"(615)322-3900","url":"https://www.vicc.org/support-groups"},
        {"name":"Alive Hospice","phone":"(615)327-1085","url":"https://alivehospice.org"}
    ]:
        st.markdown(f"- **{g['name']}**: {g['phone']} | [Website]({g['url']})")
