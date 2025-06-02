import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import os

# 1. Load the cleaned BCSC data
DATA_DIR = "data"
input_csv = os.path.join(DATA_DIR, "bcsc_concatenated_no_hist9.csv")
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Cannot find {input_csv}. Make sure you’ve concatenated and cleaned your BCSC files into this path.")
bcsc_df = pd.read_csv(input_csv)
print("Loaded BCSC data shape:", bcsc_df.shape)

# 2. Drop unwanted columns if present
for col in ["count", "year"]:
    if col in bcsc_df.columns:
        bcsc_df = bcsc_df.drop(columns=[col])
        print(f"Dropped column: {col}")

# 3. Separate features (X) and target (y)
if "breast_cancer_history" not in bcsc_df.columns:
    raise KeyError("Target column 'breast_cancer_history' not found.")
X = bcsc_df.drop(columns=["breast_cancer_history"])
y = bcsc_df["breast_cancer_history"]

# 4. Fill missing values
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype in [np.float64, np.int64]:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
        else:
            mode_vals = X[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            X[col] = X[col].fillna(fill_val)

# 5. Encode categorical features
feature_encoders = {}
for col in X.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le
    print(f"Encoded '{col}' with classes: {le.classes_}")

# 6. Encode target if it’s not numeric
if y.dtype == "object" or str(y.dtype).startswith("category"):
    target_le = LabelEncoder()
    y = target_le.fit_transform(y.astype(str))
    print("Encoded target 'breast_cancer_history' with classes:", target_le.classes_)
else:
    target_le = None

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 8. Fit XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
print("Model training complete.")

# 9. Evaluate on the test set
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
try:
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC: {auc:.4f}")
except:
    print("Could not compute ROC AUC (maybe only one class in test set).")

# 10. Save artifacts into data/
model_path = os.path.join(DATA_DIR, "bcsc_xgb_model.pkl")
encoders_path = os.path.join(DATA_DIR, "bcsc_feature_encoders.pkl")
joblib.dump(model, model_path)
print(f"Saved XGBoost model to {model_path}")
joblib.dump(feature_encoders, encoders_path)
print(f"Saved feature encoders to {encoders_path}")
if target_le is not None:
    target_path = os.path.join(DATA_DIR, "bcsc_target_encoder.pkl")
    joblib.dump(target_le, target_path)
    print(f"Saved target encoder to {target_path}")
