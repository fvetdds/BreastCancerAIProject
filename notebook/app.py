!pip install joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, classification_report, confusion_matrix,accuracy_score 
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("xgb_model.pkl")

# --- Sidebar Info ---
st.sidebar.title("🧬 About this App")
st.sidebar.markdown("""
This application predicts **breast cancer prognosis risk** (Low, Medium, High) using patient features.

**Project Purpose**:  
Assist clinicians and patients in early understanding of cancer prognosis risk to support care planning.

**Model**: XGBoost Classifier  
**Inputs**: Age, tumor size, gene mutation status, treatment type  
**Risk Levels**:
- **Low**: Favorable prognosis  
- **Medium**: Moderate risk, requires monitoring  
- **High**: Elevated risk, requires intervention
""")

# --- Main Interface ---
st.title("🔍 Breast Cancer Prognosis Risk Predictor")

st.markdown("""
Please enter patient details to estimate prognosis risk.
""")

# --- User Inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=45)
tumor_size = st.slider("Tumor Size (mm)", 0, 100, 20)

gene_mutation = st.selectbox("Gene Mutation Present?", ['No', 'Yes'])
gene_mutation_bin = 1 if gene_mutation == 'Yes' else 0

treatment_type = st.selectbox("Treatment Type", ['None', 'Chemotherapy', 'Radiation', 'Hormonal'])
# One-hot encode treatment (simplified)
treatment_options = ['None', 'Chemotherapy', 'Radiation', 'Hormonal']
treatment_encoded = [1 if treatment_type == opt else 0 for opt in treatment_options]

# --- Predict Button ---
if st.button("Predict Risk"):
    # Construct input array
    input_features = [age, tumor_size, gene_mutation_bin] + treatment_encoded
    input_array = np.array(input_features).reshape(1, -1)
    
    # Model prediction
    prob = model.predict_proba(input_array)[0][1]  # Adjust index for multiclass if needed
    
    # Risk categorization
    def categorize_risk(p):
        if p < 0.33:
            return "Low Risk", "✅ Continue routine care"
        elif p < 0.66:
            return "Medium Risk", "⚠️ Monitor closely and discuss follow-up plans"
        else:
            return "High Risk", "🚨 Immediate specialist consultation recommended"
    
    risk, advice = categorize_risk(prob)
    
    # Display results
    st.markdown(f"### 🩺 Predicted Risk: **{risk}**")
    st.markdown(f"**Probability Score**: {prob:.2f}")
    st.markdown(f"**Next Steps**: {advice}")

