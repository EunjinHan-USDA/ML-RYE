# ---------- MUST BE FIRST: single-thread math & hashing for full determinism ----------
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -------------------------------------------------------------------------------------

# Reproducible XGBoost (exact) + SHAP + Feature Importance
# ----------------------------------------------------------------
import re, random, warnings, hashlib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

# =========================
# GLOBAL REPRO SETTINGS
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Clean logs
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded", category=FutureWarning)

# Matplotlib defaults for bold/large text
plt.rcParams.update({
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.labelsize": 16, "axes.labelweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14
})

# =========================
# PATHS & INPUTS
# =========================
csv_path = "/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged_7.csv"
output_dir = "Aug30_GS40_50_XGBoost_outputs"
os.makedirs(output_dir, exist_ok=True)

input_features = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha", "zone",
    "GS0_20avgSrad", "GS0_20cRain",
    "GS20_30avgSrad", "GS20_30cRain","GS30_40avgSrad", "GS30_40cRain","GS40_50avgSrad", "GS40_50cRain",
    "FallcumGDD", "SpringcumGDD",
    "OM (%/100)", "Sand",  "Clay",
    "legume_preceding", "planting_method"
]
cat_features = ["zone", "legume_preceding", "planting_method"]
target_col = "biomass_mean"

# =========================
# LOAD & CLEAN
# =========================
df = pd.read_csv(csv_path)
df = df[input_features + [target_col]].dropna()
print(f"✅ Valid samples with complete data and biomass : {len(df)}")

X = df[input_features].copy()
y = df[target_col].copy()
X[cat_features] = X[cat_features].fillna("missing").astype(str)

# =========================
# 70/30 SPLIT (deterministic)
# =========================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, shuffle=True
)

print(f"Total samples: {len(X)}")
print(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

# =========================
# ONE-HOT ENCODING (stable)
# =========================
X_all = pd.concat([X_train_raw, X_test_raw], axis=0)
X_all_encoded = pd.get_dummies(X_all, columns=cat_features)
X_train = X_all_encoded.iloc[:len(X_train_raw)].copy()
X_test  = X_all_encoded.iloc[len(X_train_raw):].copy()

X_train = X_train.reindex(sorted(X_train.columns), axis=1)
X_test  = X_test.reindex(X_train.columns, axis=1)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Optional: HASH sanity check (proves identical inputs across OS/runs)
def md5(a: np.ndarray) -> str:
    return hashlib.md5(a.astype(np.float64).tobytes()).hexdigest()

print("hash X_train:", md5(X_train.values))
print("hash y_train:", md5(y_train.values))
print("hash X_test :", md5(X_test.values))
print("hash y_test :", md5(y_test.values))

# =========================
# XGBOOST (balanced parameters)
# =========================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=2,
    min_child_weight=20,
    subsample=0.90,
    colsample_bytree=0.80,
    reg_alpha=0.5,
    reg_lambda=10.0,
    gamma=0.10,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=1,
    tree_method="exact",
    verbosity=0,
)
model.fit(X_train, y_train)

# =========================
# EVALUATION + SAVE
# =========================
y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))
pct_rmse = 100.0 * rmse / float(y_test.mean())

print(f"\n📊 [TEST] RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}, %RMSE: {pct_rmse:.2f}%")

# Save test predictions
pd.DataFrame({"actual_biomass": y_test.values, "predicted_biomass": y_pred})\
  .to_csv(os.path.join(output_dir, "predictions_test.csv"), index=False)

# ===== TRAIN predictions & metrics =====
y_pred_train = model.predict(X_train)
rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
mae_tr  = float(mean_absolute_error(y_train, y_pred_train))
r2_tr   = float(r2_score(y_train, y_pred_train))
pct_rmse_tr = 100.0 * rmse_tr / float(y_train.mean())

print(f"📊 [TRAIN] RMSE: {rmse_tr:.2f}, MAE: {mae_tr:.2f}, R²: {r2_tr:.3f}, %RMSE: {pct_rmse_tr:.2f}%")

# Save train predictions
pd.DataFrame({"actual_biomass": y_train.values, "predicted_biomass": y_pred_train})\
  .to_csv(os.path.join(output_dir, "predictions_train.csv"), index=False)

# Save metrics
pd.DataFrame([
    {"split": "train", "rmse": rmse_tr, "mae": mae_tr, "r2": r2_tr, "pct_rmse": pct_rmse_tr},
    {"split": "test",  "rmse": rmse,    "mae": mae,    "r2": r2,    "pct_rmse": pct_rmse}
]).to_csv(os.path.join(output_dir, "metrics_train_test.csv"), index=False)

# =========================
# 1:1 PLOTS
# =========================
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.8, edgecolor="black", s=60, label="Samples")
vmin = min(y_test.min(), y_pred.min()); vmax = max(y_test.max(), y_pred.max())
plt.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2, label="1:1 Line")
plt.xlabel("Observed Biomass (kg/ha)")
plt.ylabel("Predicted Biomass (kg/ha)")
plt.title("XGBoost Observed vs Predicted Biomass")
plt.legend(fontsize=12, frameon=True)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "observed_vs_predicted_test.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
plt.scatter(y_train, y_pred_train, alpha=0.8, edgecolor="black", s=60, label="Samples")
vmin_tr = min(y_train.min(), y_pred_train.min()); vmax_tr = max(y_train.max(), y_pred_train.max())
plt.plot([vmin_tr, vmax_tr], [vmin_tr, vmax_tr], "r--", linewidth=2, label="1:1 Line")
plt.xlabel("Observed Biomass (kg/ha)")
plt.ylabel("Predicted Biomass (kg/ha)")
plt.title("XGBoost Observed vs Predicted Biomass")
plt.legend(fontsize=12, frameon=True)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "observed_vs_predicted_train.png"), dpi=300)
plt.close()

# =========================
# FEATURE IMPORTANCE (aggregated OHE)
# =========================
importances = model.feature_importances_
fi_df = pd.DataFrame({"feature": X_train.columns, "importance": importances})

def base_name(col: str) -> str:
    for pref in ["zone_", "legume_preceding_", "planting_method_"]:
        if col.startswith(pref):
            return pref[:-1]
    return col

fi_df["base_feature"] = fi_df["feature"].apply(base_name)
grouped = fi_df.groupby("base_feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
grouped.to_csv(os.path.join(output_dir, "feature_importance_grouped.csv"), index=False)

group_top = grouped.head(30).iloc[::-1]
plt.figure(figsize=(10, 8))
plt.barh(group_top["base_feature"], group_top["importance"])
plt.xlabel("Aggregated Importance")
plt.ylabel("Feature")
plt.title("Top 30 Aggregated Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_top30_grouped.png"), dpi=300)
plt.close()

# =========================
# SHAP ANALYSIS (collapse OHE)
# =========================
print("\n✅ Generating SHAP plots")

explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)

# Collapse one-hot encoded categoricals
ohe_groups = {c: [col for col in X_test.columns if col.startswith(f"{c}_")] for c in cat_features}
ohe_cols_all = [c for cols in ohe_groups.values() for c in cols]
non_cat_cols = [c for c in X_test.columns if c not in ohe_cols_all]
col_index = {c: i for i, c in enumerate(X_test.columns)}

n = X_test.shape[0]
agg_names = list(cat_features) + non_cat_cols
agg_shap = np.zeros((n, len(agg_names)), dtype=float)
agg_features = pd.DataFrame(index=X_test.index)

cursor = 0
for cat in cat_features:
    cols = ohe_groups[cat]
    if cols:
        idxs = [col_index[c] for c in cols]
        agg_shap[:, cursor] = shap_values_raw[:, idxs].sum(axis=1)
        labels = X_test[cols].idxmax(axis=1).str.replace(f"^{re.escape(cat)}_", "", regex=True)
        agg_features[cat] = labels
    else:
        agg_shap[:, cursor] = shap_values_raw[:, col_index[cat]]
        agg_features[cat] = X_test[cat]
    cursor += 1

for c in non_cat_cols:
    agg_shap[:, cursor] = shap_values_raw[:, col_index[c]]
    agg_features[c] = X_test[c]
    cursor += 1

# SHAP bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    agg_shap,
    features=agg_features,
    feature_names=agg_names,
    plot_type="bar",
    max_display=len(agg_names),
    show=False
)
plt.title("SHAP Feature Importance (Bar)", fontsize=18, fontweight="bold")
plt.xlabel("Mean |SHAP value| (impact on model output)", fontsize=16, fontweight="bold")
plt.ylabel("Feature", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=300)
plt.close()

# SHAP dot plot
dot_features = agg_features.copy()
for c in cat_features:
    if c in dot_features.columns:
        dot_features[c] = pd.factorize(dot_features[c])[0].astype(float)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    agg_shap,
    features=dot_features,
    feature_names=agg_names,
    max_display=len(agg_names),
    show=False
)
plt.title("SHAP Summary Plot", fontsize=18, fontweight="bold")
plt.xlabel("SHAP value (impact on model output)", fontsize=16, fontweight="bold")
plt.ylabel("Feature", fontsize=16, fontweight="bold")
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_dot.png"), dpi=300)
plt.close()

# =========================
# Final hash (for reproducibility)
# =========================
pred_hash = hashlib.md5(y_pred.astype(np.float64).tobytes()).hexdigest()
print("prediction_hash:", pred_hash)

print(f"✅ All outputs saved in: {output_dir}")
