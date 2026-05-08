import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

# Load selected features dataset
df = pd.read_csv(r"c:\Cairo university\Second Year\Machine Learning\preprocessed\features_selected.csv")

meta_cols = ["case_id", "label", "meta_candidate_id", "meta_method"]
feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols]
y = df["label"]

# Train RF with same parameters as before
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running cross-validation predictions to compute confusion matrix...")
y_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

print("\n--- Confusion Matrix ---")
print(f"True Positives (Actual Aneurysm, Predicted Aneurysm) : {tp}")
print(f"False Positives (Actual Non-Aneurysm, Predicted Aneurysm): {fp}")
print(f"True Negatives (Actual Non-Aneurysm, Predicted Non-Aneurysm): {tn}")
print(f"False Negatives (Actual Aneurysm, Predicted Non-Aneurysm): {fn}")

print("\n--- Rates ---")
print(f"False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
print(f"False Negative Rate (FNR): {fnr:.4f} ({fnr*100:.2f}%)")
print(f"True Positive Rate (Recall): {tpr:.4f} ({tpr*100:.2f}%)")
print(f"True Negative Rate (Specificity): {tnr:.4f} ({tnr*100:.2f}%)")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
