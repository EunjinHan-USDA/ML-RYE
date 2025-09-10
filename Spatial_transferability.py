# ---------- MUST BE FIRST: single-thread math & stable hashing for cross-OS determinism ----------
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -------------------------------------------------------------------------------------------------

# Reproducible CatBoost Quantiles + SHAP + Spatial Transferability (North↔South)
import re, random, warnings, hashlib
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

plt.rcParams.update({
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.labelsize": 16, "axes.labelweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14
})

# =========================
# PATHS & INPUTS
# =========================
CSV_PATH   = "/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged_7.csv"
OUTPUT_DIR = "AUG30_Primary_CatBOOST_with_SpatialTransfer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Primary-model feature set (GS20–30 primary style; adjust if needed)
INPUT_FEATURES = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha", "zone",
    "GS0_20avgSrad", "GS0_20cRain",
    "GS20_30avgSrad", "GS20_30cRain",
    "FallcumGDD", "SpringcumGDD",
    "OM (%/100)", "Sand", "Clay",
    "legume_preceding", "planting_method"
]
CAT_FEATURES = ["zone", "legume_preceding", "planting_method"]
TARGET_COL   = "biomass_mean"

# Regions for spatial transferability
SOUTH_STATES = ['FL', 'SC', 'NC', 'AL', 'AR', 'LA', 'TX']
NORTH_STATES = ['MN', 'WI', 'MI', 'NY', 'VT', 'MA', 'IA', 'IL', 'IN', 'OH', 'MO', 'NE',
                'MD', 'DE', 'PA', 'VA', 'KY']

# =========================
# LOAD & CLEAN
# =========================
df_full = pd.read_csv(CSV_PATH)

# Detect a state column robustly (not used as a feature, only for splitting)
state_col = next((c for c in ["state", "State", "STATE", "state_abbrev", "state_code"] if c in df_full.columns), None)
if state_col is None:
    raise ValueError("⚠️ No state column found (looked for: state/State/STATE/state_abbrev/state_code). "
                     "Please add a state column for spatial transferability.")

# Keep only rows with all required features + target present
df = df_full[INPUT_FEATURES + [TARGET_COL, state_col]].dropna().copy()
print(f"✅ Valid samples with complete data and biomass: {len(df)}")

X = df[INPUT_FEATURES].copy()
y = df[TARGET_COL].copy()
states = df[state_col].astype(str)

# Keep categoricals as strings; CatBoost handles them natively
X[CAT_FEATURES] = X[CAT_FEATURES].fillna("missing").astype(str)

# =========================
# 70/30 SPLIT for PRIMARY MODEL (deterministic)
# =========================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, shuffle=True
)
print(f"Total samples: {len(X)}")
print(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

# CatBoost needs categorical column indices (0-based)
cat_idx = [X_train_raw.columns.get_loc(c) for c in CAT_FEATURES]

train_pool = Pool(X_train_raw, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test_raw,  y_test,  cat_features=cat_idx)

# =========================
# QUANTILE REGRESSION (q=0.1, 0.5, 0.9) — PRIMARY MODEL
# =========================
def pinball_loss(y_true, y_pred, alpha):
    d = y_true - y_pred
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))

quantiles = [0.1, 0.5, 0.9]
models, preds = {}, {}

for q in quantiles:
    print(f"\n🔁 Training quantile {q}")
    qb = CatBoostRegressor(
        loss_function=f"Quantile:alpha={q}",
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10,
        random_seed=SEED,
        thread_count=1,        # single-thread deterministic
        od_type="Iter",        # early stopping based on eval_set
        od_wait=50,
        use_best_model=True,
        verbose=False
    )
    qb.fit(train_pool, eval_set=test_pool, verbose=False)
    models[q] = qb
    preds[q] = qb.predict(test_pool)

# Evaluate median model (q=0.5) on test & train
results_df = pd.DataFrame({
    "actual_biomass": y_test.values,
    "pred_10th": preds[0.1],
    "pred_50th": preds[0.5],
    "pred_90th": preds[0.9]
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_primary_quantiles_test.csv"), index=False)

rmse = float(np.sqrt(((results_df["actual_biomass"] - results_df["pred_50th"])**2).mean()))
mae  = float(np.abs(results_df["actual_biomass"] - results_df["pred_50th"]).mean())
r2   = float(r2_score(results_df["actual_biomass"], results_df["pred_50th"]))
pct_rmse = 100.0 * rmse / float(results_df["actual_biomass"].mean())

print(f"\n📊 [PRIMARY TEST] RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}, %RMSE: {pct_rmse:.2f}%")

train_pred_50 = models[0.5].predict(train_pool)
rmse_tr = float(np.sqrt(((y_train - train_pred_50)**2).mean()))
mae_tr  = float(np.abs(y_train - train_pred_50).mean())
r2_tr   = float(r2_score(y_train, train_pred_50))
pct_rmse_tr = 100.0 * rmse_tr / float(y_train.mean())

print(f"📊 [PRIMARY TRAIN] RMSE: {rmse_tr:.2f}, MAE: {mae_tr:.2f}, R²: {r2_tr:.3f}, %RMSE: {pct_rmse_tr:.2f}%")

# Pinball losses (test)
pb10 = pinball_loss(results_df["actual_biomass"], results_df["pred_10th"], 0.1)
pb50 = pinball_loss(results_df["actual_biomass"], results_df["pred_50th"], 0.5)
pb90 = pinball_loss(results_df["actual_biomass"], results_df["pred_90th"], 0.9)
print(f"   Pinball loss q=0.1: {pb10:.4f}")
print(f"   Pinball loss q=0.5: {pb50:.4f}")
print(f"   Pinball loss q=0.9: {pb90:.4f}")

pd.DataFrame([{
    "model": "catboost_quantile_primary",
    "test_rmse": rmse, "test_mae": mae, "test_r2": r2, "test_pct_rmse": pct_rmse,
    "train_rmse": rmse_tr, "train_mae": mae_tr, "train_r2": r2_tr, "train_pct_rmse": pct_rmse_tr,
    "pinball_q10": pb10, "pinball_q50": pb50, "pinball_q90": pb90
}]).to_csv(os.path.join(OUTPUT_DIR, "metrics_primary_quantile.csv"), index=False)

# =========================
# PLOTS — PRIMARY MODEL
# =========================
# 1:1 (TEST)
plt.figure(figsize=(8, 8))
plt.scatter(results_df["actual_biomass"], results_df["pred_50th"], edgecolor="black", alpha=0.7, s=60)
vmin = min(results_df["actual_biomass"].min(), results_df["pred_50th"].min())
vmax = max(results_df["actual_biomass"].max(), results_df["pred_50th"].max())
plt.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2, label="1:1 Line")
plt.xlabel("Observed Biomass (kg/ha)"); plt.ylabel("Predicted Biomass (kg/ha)")
plt.title("CatBoost (q=0.5) Observed vs Predicted — Test")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "obs_vs_pred_test.png"), dpi=300); plt.close()

# 1:1 (TRAIN)
plt.figure(figsize=(8, 8))
plt.scatter(y_train, train_pred_50, edgecolor="black", alpha=0.7, s=60)
vmin_tr = min(y_train.min(), train_pred_50.min()); vmax_tr = max(y_train.max(), train_pred_50.max())
plt.plot([vmin_tr, vmax_tr], [vmin_tr, vmax_tr], "r--", linewidth=2, label="1:1 Line")
plt.xlabel("Observed Biomass (kg/ha)"); plt.ylabel("Predicted Biomass (kg/ha)")
plt.title("CatBoost (q=0.5) Observed vs Predicted — Train")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "obs_vs_pred_train.png"), dpi=300); plt.close()

# Uncertainty plot (sorted by actual)
sort_idx = results_df["actual_biomass"].argsort()
plt.figure(figsize=(10, 6))
plt.plot(results_df["actual_biomass"].values[sort_idx], label="Actual", color="black", linewidth=2)
plt.plot(results_df["pred_50th"].values[sort_idx], label="Predicted Median (q=0.5)", linewidth=2)
plt.fill_between(
    range(len(results_df)),
    results_df["pred_10th"].values[sort_idx],
    results_df["pred_90th"].values[sort_idx],
    alpha=0.35, label="10–90% interval"
)
plt.xlabel("Sample Index"); plt.ylabel("Biomass (kg/ha)")
plt.title("Predicted Biomass with Uncertainty (CatBoost Quantiles)")
plt.legend(fontsize=12); plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "uncertainty_plot.png"), dpi=300); plt.close()

# SHAP (primary, q=0.5)
print("\n✅ Generating SHAP plots for primary model (q=0.5)")
explainer = shap.TreeExplainer(models[0.5])
shap_values = explainer.shap_values(X_test_raw)

# SHAP bar (all features)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_raw, plot_type="bar", max_display=X_test_raw.shape[1], show=False)
plt.title("SHAP Feature Importance (Bar)", fontsize=18, fontweight="bold")
plt.xlabel("Mean |SHAP value| (impact on model output)", fontsize=16, fontweight="bold")
plt.ylabel("Feature", fontsize=16, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar.png"), dpi=300); plt.close()

# SHAP dot (factorize categoricals for color scale only)
X_test_for_color = X_test_raw.copy()
for c in CAT_FEATURES:
    X_test_for_color[c] = pd.factorize(X_test_for_color[c])[0].astype(float)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_for_color, max_display=X_test_for_color.shape[1], show=False)
plt.title("SHAP Summary Plot", fontsize=18, fontweight="bold")
plt.xlabel("SHAP value (impact on model output)", fontsize=16, fontweight="bold")
plt.ylabel("Feature", fontsize=16, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_dot.png"), dpi=300); plt.close()

# =========================
#  SPATIAL TRANSFERABILITY (North↔South)
# =========================
print("\n🗺️  Spatial transferability evaluation (North↔South)")

# Build masks on the *filtered* df
north_mask = states.isin(NORTH_STATES)
south_mask = states.isin(SOUTH_STATES)

def summarize_transfer(train_mask, test_mask, train_label, test_label, cat_features_idx):
    X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
    X_te, y_te = X.loc[test_mask],  y.loc[test_mask]

    # Pools
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
    te_pool = Pool(X_te, y_te, cat_features=cat_features_idx)

    # Train point model (RMSE) for transferability
    m = CatBoostRegressor(
        loss_function="RMSE",
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10,
        random_seed=SEED,
        thread_count=1,
        od_type="Iter",
        od_wait=50,
        use_best_model=True,
        verbose=False
    )
    m.fit(tr_pool, eval_set=te_pool, verbose=False)

    # Predictions
    y_pred_tr = m.predict(tr_pool)
    y_pred_te = m.predict(te_pool)

    # Metrics (Train)
    rmse_tr = float(np.sqrt(((y_tr - y_pred_tr) ** 2).mean()))
    r2_tr   = float(r2_score(y_tr, y_pred_tr))
    pct_tr  = 100.0 * rmse_tr / float(y_tr.mean())

    # Metrics (Test)
    rmse_te = float(np.sqrt(((y_te - y_pred_te) ** 2).mean()))
    r2_te   = float(r2_score(y_te, y_pred_te))
    pct_te  = 100.0 * rmse_te / float(y_te.mean())

    print(f"  [{train_label}→{test_label}] "
          f"Train n={len(X_tr)}, Test n={len(X_te)} | "
          f"Train R²={r2_tr:.3f}, %RMSE={pct_tr:.2f}% | "
          f"Test R²={r2_te:.3f}, %RMSE={pct_te:.2f}%")

    row = {
        "Train Region": train_label,
        "Test Region":  test_label,
        "Train Samples": len(X_tr),
        "Test Samples":  len(X_te),
        "Train R²": r2_tr,
        "Train RMSE": rmse_tr
        "Train %RMSE": pct_tr,
        "Test R²": r2_te,
        "Test RMSE": rmse_te
        "Test %RMSE": pct_te
    }
    return row

# Build cat index once (same as primary features ordering)
cat_features_idx = [X.columns.get_loc(c) for c in CAT_FEATURES]

rows = []
# North→South (if both have data)
if north_mask.any() and south_mask.any():
    rows.append(summarize_transfer(north_mask, south_mask, "North", "South", cat_features_idx))
    rows.append(summarize_transfer(south_mask, north_mask, "South", "North", cat_features_idx))
else:
    missing = "North" if not north_mask.any() else "South"
    print(f"⚠️ Not enough data to compute transferability for: {missing}")

spatial_df = pd.DataFrame(rows)
spatial_path = os.path.join(OUTPUT_DIR, "SpatialTransfer_Summary.csv")
spatial_df.to_csv(spatial_path, index=False)

print("\n📄 Saved spatial transferability summary table →", spatial_path)

# =========================
# Final reproducibility hash for predictions
# =========================
pred_hash = hashlib.md5(results_df["pred_50th"].values.astype(np.float64).tobytes()).hexdigest()
print("prediction_hash (primary q=0.5):", pred_hash)
print(f"✅ All outputs saved in: {OUTPUT_DIR}")
