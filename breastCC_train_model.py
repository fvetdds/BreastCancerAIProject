import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import joblib

# 1) Load and split
df = pd.read_csv("data/bcsc_concatenated_no_9.csv")
X = df.drop(columns="breast_cancer_history")
y = df["breast_cancer_history"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train/test: {X_train.shape}/{X_test.shape}")

# 2) Handle imbalance
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# 3) Hyperparam search
base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    random_state=42
)
param_dist = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma":            [0, 1, 5],
}
search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=20,
    scoring="recall",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
best = search.best_estimator_

# 4) Retrain full train set
best.set_params(scale_pos_weight=scale_pos_weight)
best.fit(X_train, y_train)

# 5) Find G-mean threshold on test set
y_prob = best.predict_proba(X_test)[:, 1]
fpr, tpr, thresh = roc_curve(y_test, y_prob)
gmeans = np.sqrt(tpr * (1 - fpr))
idx = np.argmax(gmeans)
best_thresh = float(thresh[idx])
print(f"Optimal G-mean threshold: {best_thresh:.3f}")

# 6) Evaluate
y_pred = (y_prob >= best_thresh).astype(int)
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# 7) Save model + threshold
os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/bcsc_xgb_model.pkl")
joblib.dump(best_thresh, "models/threshold.pkl")
print("Model and threshold saved.")
