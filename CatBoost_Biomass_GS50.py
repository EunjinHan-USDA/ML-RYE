# ---------- MUST BE FIRST: single-thread math & stable hashing for cross-OS determinism ----------
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -------------------------------------------------------------------------------------------------

# Reproducible CatBoost Quantiles + SHAP (show ALL features, bold plots, terminal-friendly prints)
import re, random, warnings, hashlib, sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
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
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded", category=FutureWarning)

# Make matplotlib text large & bold by default
plt.rcParams.update({
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.labelsize": 16, "axes.labelweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14
})

# =========================
# PATHS & INPUTS
# =========================
CSV_PATH = "/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged_7.csv"
OUTPUT_DIR = "AUG25_Biomass_yesNMAYBE_GS50_CatBOOST_outputss"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FEATURES = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha", "zone",
    "GS0_20avgSrad", "GS0_20cRain",
    "GS20_30avgSrad", "GS20_30cRain",
    "GS30_40avgSrad", "GS30_40cRain",
    "GS40_50avgSrad", "GS40_50cRain",
    "GS50avgSrad", "GS50cRain",
    "FallcumGDD", "SpringcumGDD",
    "OM (%/100)", "Sand", "Silt", "Clay",
    "legume_preceding", "planting_method"
]
CAT_FEATURES = ["zone", "legume_preceding", "planting_method"]
TARGET_COL = "biomass_mean"

# =========================
# LOAD & CLEAN
# =========================
df = pd.read_csv(CSV_PATH)
df = df[INPUT_FEATURES + [TARGET_COL]].dropna()
print(f"✅ Valid samples with complete data and biomass : {len(df)}")

X = df[INPUT_FEATURES].copy()
y = df[TARGET_COL].copy()

# Keep categoricals as strings; CatBoost handles them natively
X[CAT_FEATURES] = X[CAT_FEATURES].fillna("missing").astype(str)

# =========================
# SPLIT (PIN EXACT ROWS ACROSS RUNS/OS)
# =========================
split_path = os.path.join(OUTPUT_DIR, "fixed_split_idx.npz")
if os.path.exists(split_path):
    npz = np.load(split_path, allow_pickle=False)
    train_idx, test_idx = npz["train_idx"], npz["test_idx"]
else:
    all_idx = np.arange(len(X))
    _, _, _, _, train_idx, test_idx = train_test_split(
        X, y, all_idx, test_size=0.20, random_state=SEED, shuffle=True
    )
    np.savez_compressed(split_path, train_idx=train_idx, test_idx=test_idx)

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

print(f"Total samples after filtering: {len(df)}")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# CatBoost needs categorical column indices (0-based)
cat_idx = [X_train.columns.get_loc(c) for c in CAT_FEATURES]

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

# Optional hashes (prove identical inputs across OS/runs)
def md5(a: np.ndarray) -> str:
    return hashlib.md5(a.astype(np.float64).tobytes()).hexdigest()


# =========================
# QUANTILE REGRESSION (Deterministic via seed + single thread)
# =========================
quantiles = [0.1, 0.5, 0.9]
models, preds = {}, {}

for q in quantiles:
    print(f"\n🔁 Training quantile {q}")
    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={q}",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=SEED,     # fixed seed
        thread_count=1,       # single-thread = deterministic
        verbose=False
    )
    model.fit(train_pool, eval_set=test_pool, verbose=False)
    models[q] = model
    preds[q] = model.predict(test_pool)

# =========================
# EVALUATION (based on median model, q=0.5)
# =========================
results_df = pd.DataFrame({
    "actual_biomass": y_test.values,
    "pred_10th": preds[0.1],
    "pred_50th": preds[0.5],
    "pred_90th": preds[0.9]
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

rmse = float(np.sqrt(mean_squared_error(results_df["actual_biomass"], results_df["pred_50th"])))
mae  = float(mean_absolute_error(results_df["actual_biomass"], results_df["pred_50th"]))
r2   = float(r2_score(results_df["actual_biomass"], results_df["pred_50th"]))
pct_rmse = 100.0 * rmse / float(results_df["actual_biomass"].mean())

print(f"\n📊 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.3f}, %RMSE: {pct_rmse:.2f}%")

def pinball_loss(y_true, y_pred, alpha):
    d = y_true - y_pred
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))

for q in quantiles:
    loss = pinball_loss(results_df["actual_biomass"], results_df[f"pred_{int(q*100)}th"], q)
    print(f" Pinball loss q={q}: {loss:.4f}")

pd.DataFrame([{
    "model": "catboost_quantile_deterministic",
    "rmse": rmse, "mae": mae, "r2": r2, "pct_rmse": pct_rmse,
    "pinball_q10": pinball_loss(results_df["actual_biomass"], results_df["pred_10th"], 0.1),
    "pinball_q50": pinball_loss(results_df["actual_biomass"], results_df["pred_50th"], 0.5),
    "pinball_q90": pinball_loss(results_df["actual_biomass"], results_df["pred_90th"], 0.9)
}]).to_csv(os.path.join(OUTPUT_DIR, "metrics_reproducible.csv"), index=False)

# =========================
# UNCERTAINTY PLOT (bold styling)
# =========================
sorted_idx = results_df["actual_biomass"].argsort()
plt.figure(figsize=(10, 6))
plt.plot(results_df["actual_biomass"].values[sorted_idx], label="Actual", color="black", linewidth=2)
plt.plot(results_df["pred_50th"].values[sorted_idx], label="Predicted Median", linewidth=2)
plt.fill_between(range(len(results_df)),
                 results_df["pred_10th"].values[sorted_idx],
                 results_df["pred_90th"].values[sorted_idx],
                 alpha=0.35, label="10–90% interval")
plt.xlabel("Sample Index")
plt.ylabel("Biomass (kg/ha)")
plt.title("Predicted Biomass with Uncertainty (CatBoost Quantiles)")
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "uncertainty_plot.png"), dpi=300)
plt.close()

# =========================
# FEATURE IMPORTANCE (CatBoost native)
# =========================
importances = models[0.5].get_feature_importance(type="PredictionValuesChange", data=train_pool)
fi_df = pd.DataFrame({"feature": X_train.columns, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

top = fi_df.head(30).iloc[::-1]
plt.figure(figsize=(10, 8))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 30 Feature Importances (CatBoost)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_top30.png"), dpi=300)
plt.close()

# =========================
# SHAP (show ALL features; bold/large fonts)
# =========================
print("\n✅ Generating SHAP plots")

# TreeExplainer works with CatBoost
explainer = shap.TreeExplainer(models[0.5])
shap_values = explainer.shap_values(X_test)

# --- BAR summary (ALL features) ---
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    max_display=X_test.shape[1],   # show ALL features
    show=False
)
plt.title("SHAP Feature Importance (Bar)", fontsize=18, fontweight="bold")
plt.xlabel("Mean |SHAP value| (impact on model output)", fontsize=16, fontweight="bold")
plt.ylabel("Feature", fontsize=16, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold"); plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar.png"), dpi=300)
plt.close()

# --- DOT summary (factorize categoricals ONLY for color scale) ---
X_test_for_color = X_test.copy()
for c in CAT_FEATURES:
    X_test_for_color[c] = pd.factorize(X_test_for_color[c])[0].astype(float)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test_for_color,
    max_display=X_test_for_color.shape[1],  # ALL features
    show=False
)
plt.title("SHAP Summary Plot", fontsize=18, fontweight="bold")
plt.xlabel("SHAP value (impact on model output)", fontsize=16, fontweight="bold")
plt.ylabel("Feature", fontsize=16, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold"); plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_dot.png"), dpi=300)
plt.close()

# Optional: prediction hash proves cross-OS identity
pred_hash = hashlib.md5(results_df["pred_50th"].values.astype(np.float64).tobytes()).hexdigest()
print("prediction_hash (q=0.5):", pred_hash)

print(f"\n✅ All outputs saved in: {OUTPUT_DIR}")
