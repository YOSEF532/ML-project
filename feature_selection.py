"""
Feature Selection Pipeline for Aneurysm Detection Dataset
=========================================================
Applies multiple feature selection methods and produces a final
reduced feature set saved to preprocessed/features_selected.csv.

Methods used:
1. Cleaning — remove constant/quasi-constant & highly correlated features
2. Univariate — Mutual Information + ANOVA F-test ranking
3. Model-based — Random Forest feature importance
4. Wrapper — Recursive Feature Elimination (RFE) with Logistic Regression
5. Consensus — keep features selected by ≥2 methods
"""

import warnings, os, sys, io
# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    f_classif,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# -- paths -------------------------------------------------------------
BASE   = Path(r"c:\Cairo university\Second Year\Machine Learning")
INPUT  = BASE / "preprocessed" / "features.csv"
OUTPUT = BASE / "preprocessed" / "features_selected.csv"
FIG_DIR = BASE / "preprocessed" / "feature_selection_plots"
FIG_DIR.mkdir(exist_ok=True)

# -- 0. load data ------------------------------------------------------
print("=" * 70)
print("FEATURE SELECTION PIPELINE")
print("=" * 70)

df = pd.read_csv(INPUT)
print(f"\n[0] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Identify meta / id columns vs. numeric feature columns
meta_cols = ["case_id", "label", "meta_candidate_id", "meta_method"]
feature_cols = [c for c in df.columns if c not in meta_cols]

# Separate features and label
X = df[feature_cols].copy()
y = df["label"].copy()

print(f"    Target distribution:\n{y.value_counts().to_string()}")
print(f"    Positive rate: {y.mean()*100:.2f}%")
print(f"    Starting features: {len(feature_cols)}")

# -- 1. Handle problematic values -------------------------------------
print("\n" + "-" * 70)
print("[1] Cleaning: handling inf / NaN values")

# Replace inf with NaN, then fill NaN with column median
X.replace([np.inf, -np.inf], np.nan, inplace=True)
nan_counts = X.isna().sum()
cols_with_nan = nan_counts[nan_counts > 0]
if len(cols_with_nan):
    print(f"    Columns with inf/NaN: {len(cols_with_nan)}")
    for col in cols_with_nan.index:
        print(f"      - {col}: {cols_with_nan[col]} missing")
    X.fillna(X.median(), inplace=True)
else:
    print("    No inf/NaN values found.")

# -- 2. Remove constant / quasi-constant features ---------------------
print("\n" + "-" * 70)
print("[2] Removing constant / quasi-constant features (variance < 0.01)")

vt = VarianceThreshold(threshold=0.01)
vt.fit(X)
constant_mask = ~vt.get_support()
constant_cols = X.columns[constant_mask].tolist()
if constant_cols:
    print(f"    Removed {len(constant_cols)} quasi-constant features:")
    for c in constant_cols:
        print(f"      - {c}  (var={X[c].var():.6f})")
    X.drop(columns=constant_cols, inplace=True)
else:
    print("    No quasi-constant features found.")

print(f"    Remaining features: {X.shape[1]}")

# -- 3. Remove highly correlated features -----------------------------
print("\n" + "-" * 70)
print("[3] Removing highly correlated features (|r| > 0.95)")

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
if high_corr_cols:
    print(f"    Removed {len(high_corr_cols)} highly-correlated features:")
    for c in high_corr_cols:
        partners = upper.index[upper[c] > 0.95].tolist()
        print(f"      - {c}  (correlated with {partners[:3]}{'...' if len(partners) > 3 else ''})")
    X.drop(columns=high_corr_cols, inplace=True)
else:
    print("    No highly-correlated features found.")

print(f"    Remaining features: {X.shape[1]}")

# -- Save correlation heatmap after cleaning ---------------------------
print("    Saving correlation heatmap...")
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(X.corr(), annot=False, cmap="coolwarm", center=0,
            linewidths=0.3, ax=ax, square=True)
ax.set_title("Feature Correlation Matrix (after cleaning)", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)

# -- 4. Scale features for selection methods ---------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

remaining_features = list(X.columns)
n_features = len(remaining_features)
# We will select top-K from each method; K = roughly half of remaining
K = max(10, n_features // 2)
print(f"\n    Each method will rank features; top-K = {K}")

# -- 5. Mutual Information --------------------------------------------
print("\n" + "-" * 70)
print("[4] Univariate: Mutual Information")

mi_scores = mutual_info_classif(X_scaled, y, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
mi_top = set(mi_series.head(K).index)

print(f"    Top-{K} MI features:")
for i, (feat, score) in enumerate(mi_series.head(K).items()):
    print(f"      {i+1:2d}. {feat:35s}  MI={score:.4f}")

# -- 6. ANOVA F-test --------------------------------------------------
print("\n" + "-" * 70)
print("[5] Univariate: ANOVA F-test")

f_scores, f_pvalues = f_classif(X_scaled, y)
f_series = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
f_top = set(f_series.head(K).index)

print(f"    Top-{K} ANOVA features:")
for i, (feat, score) in enumerate(f_series.head(K).items()):
    pval = f_pvalues[X.columns.get_loc(feat)]
    print(f"      {i+1:2d}. {feat:35s}  F={score:.2f}  p={pval:.2e}")

# -- 7. Random Forest Importance --------------------------------------
print("\n" + "-" * 70)
print("[6] Model-based: Random Forest Importance")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1,
)
rf.fit(X_scaled, y)
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_top = set(rf_imp.head(K).index)

print(f"    Top-{K} RF features:")
for i, (feat, score) in enumerate(rf_imp.head(K).items()):
    print(f"      {i+1:2d}. {feat:35s}  Imp={score:.4f}")

# -- 8. RFE with Logistic Regression ----------------------------------
print("\n" + "-" * 70)
print("[7] Wrapper: RFE with Logistic Regression")

lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs")
rfe = RFE(estimator=lr, n_features_to_select=K, step=3)
rfe.fit(X_scaled, y)
rfe_top = set(X.columns[rfe.support_])

print(f"    RFE selected {len(rfe_top)} features:")
rfe_ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
for i, (feat, rank) in enumerate(rfe_ranking.head(K).items()):
    print(f"      {i+1:2d}. {feat:35s}  Rank={rank}")

# -- 9. Consensus: features selected by ≥ 2 methods ------------------
print("\n" + "-" * 70)
print("[8] Consensus: keeping features selected by ≥ 2 / 4 methods")

methods = {"MI": mi_top, "ANOVA": f_top, "RF": rf_top, "RFE": rfe_top}
vote_counts = {}
for feat in remaining_features:
    votes = sum(1 for s in methods.values() if feat in s)
    vote_counts[feat] = votes

votes_df = pd.DataFrame([
    {"feature": feat, "votes": votes,
     "MI": "✓" if feat in mi_top else "",
     "ANOVA": "✓" if feat in f_top else "",
     "RF": "✓" if feat in rf_top else "",
     "RFE": "✓" if feat in rfe_top else ""}
    for feat, votes in vote_counts.items()
]).sort_values("votes", ascending=False)

consensus_features = votes_df[votes_df["votes"] >= 2]["feature"].tolist()
print(f"\n    Consensus features ({len(consensus_features)}):")
print(votes_df.to_string(index=False))

# -- 10. Evaluate before vs. after ------------------------------------
print("\n" + "-" * 70)
print("[9] Evaluation: RF cross-val with ALL vs. SELECTED features")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_eval = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1,
)

scores_all = cross_val_score(rf_eval, X_scaled, y, cv=cv, scoring="f1")
scores_sel = cross_val_score(
    rf_eval, X_scaled[consensus_features], y, cv=cv, scoring="f1"
)

print(f"    ALL features ({X_scaled.shape[1]}):      F1 = {scores_all.mean():.4f} ± {scores_all.std():.4f}")
print(f"    SELECTED features ({len(consensus_features)}): F1 = {scores_sel.mean():.4f} ± {scores_sel.std():.4f}")

# Also evaluate AUC
from sklearn.metrics import make_scorer, roc_auc_score
scores_auc_all = cross_val_score(rf_eval, X_scaled, y, cv=cv, scoring="roc_auc")
scores_auc_sel = cross_val_score(
    rf_eval, X_scaled[consensus_features], y, cv=cv, scoring="roc_auc"
)
print(f"    ALL features ({X_scaled.shape[1]}):      AUC = {scores_auc_all.mean():.4f} ± {scores_auc_all.std():.4f}")
print(f"    SELECTED features ({len(consensus_features)}): AUC = {scores_auc_sel.mean():.4f} ± {scores_auc_sel.std():.4f}")

# -- 11. Save plots ---------------------------------------------------
print("\n" + "-" * 70)
print("[10] Saving visualizations...")

# --- Feature importance bar chart (RF) ---
fig, ax = plt.subplots(figsize=(10, max(8, len(consensus_features) * 0.35)))
rf_imp_sel = rf_imp[rf_imp.index.isin(consensus_features)].sort_values()
colors = ["#2ecc71" if f in consensus_features else "#bdc3c7" for f in rf_imp_sel.index]
rf_imp_sel.plot.barh(ax=ax, color=colors, edgecolor="white")
ax.set_xlabel("Random Forest Importance")
ax.set_title("Selected Features — RF Importance", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "rf_importance_selected.png", dpi=150)
plt.close(fig)

# --- Votes bar chart ---
fig, ax = plt.subplots(figsize=(10, max(8, n_features * 0.3)))
vote_series = pd.Series(vote_counts).sort_values()
colors = ["#2ecc71" if v >= 2 else "#e74c3c" for v in vote_series.values]
vote_series.plot.barh(ax=ax, color=colors, edgecolor="white")
ax.axvline(x=2, color="black", linestyle="--", linewidth=1, label="Threshold (≥2)")
ax.set_xlabel("Number of Methods Selecting Feature")
ax.set_title("Feature Selection Consensus Votes", fontsize=14)
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "consensus_votes.png", dpi=150)
plt.close(fig)

# --- MI vs RF importance scatter ---
fig, ax = plt.subplots(figsize=(10, 8))
for feat in remaining_features:
    c = "#2ecc71" if feat in consensus_features else "#bdc3c7"
    ax.scatter(mi_series[feat], rf_imp[feat], c=c, s=50, edgecolors="k", linewidth=0.3)
    if feat in consensus_features:
        ax.annotate(feat, (mi_series[feat], rf_imp[feat]), fontsize=6, alpha=0.8)
ax.set_xlabel("Mutual Information Score")
ax.set_ylabel("Random Forest Importance")
ax.set_title("MI vs RF Importance (green = selected)", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "mi_vs_rf_scatter.png", dpi=150)
plt.close(fig)

# -- 12. Save output dataset ------------------------------------------
print("\n" + "-" * 70)
print("[11] Saving selected features dataset")

output_cols = meta_cols + consensus_features
df_out = df[output_cols].copy()

# Re-clean inf/NaN in output
for col in consensus_features:
    df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan)
df_out.fillna(df_out.median(numeric_only=True), inplace=True)

df_out.to_csv(OUTPUT, index=False)
print(f"    Saved → {OUTPUT}")
print(f"    Shape: {df_out.shape}")

# -- Summary -----------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Original features:   {len(feature_cols)}")
print(f"  After cleaning:      {n_features}")
print(f"  After selection:     {len(consensus_features)}")
print(f"  Reduction:           {len(feature_cols)} → {len(consensus_features)} "
      f"({(1 - len(consensus_features)/len(feature_cols))*100:.1f}% reduction)")
print(f"\n  Selected features:")
for i, f in enumerate(sorted(consensus_features), 1):
    print(f"    {i:2d}. {f}")
print(f"\n  Output file:  {OUTPUT}")
print(f"  Plots:        {FIG_DIR}")
print("=" * 70)
