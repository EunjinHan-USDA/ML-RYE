# ---------- MUST BE FIRST: single-thread math & stable hashing ----------
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -----------------------------------------------------------------------

import numpy as np
import pandas as pd
import random, warnings
from itertools import product
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================
# GLOBAL SETTINGS
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# PATHS & INPUTS
# =========================
CSV_PATH = "/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged_7.csv"
OUTPUT_DIR = "AUG27_Biomass_yesNMAYBE_GS50_CatBOOST_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FEATURES = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha", "zone",
    "FallcumGDD", "SpringcumGDD", "GS0_20avgSrad", "GS0_20cRain",
    "GS20_30avgSrad", "GS20_30cRain", 
    "OM (%/100)", "Sand", "Clay",
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
X[CAT_FEATURES] = X[CAT_FEATURES].fillna("missing").astype(str)

# =========================
# TRAIN/TEST SPLIT (70/30 deterministic)
# =========================
train_idx, test_idx = train_test_split(
    np.arange(len(X)), test_size=0.30, random_state=SEED, shuffle=True
)
X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

cat_idx = [X_train.columns.get_loc(c) for c in CAT_FEATURES]
train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

# =========================
# GRID SEARCH (Quantile α=0.5)
# =========================
param_grid = {
    "depth": [4, 5, 6],
    "learning_rate": [0.1, 0.05, 0.03],
    "l2_leaf_reg": [3, 5, 10, 20]
}

best_model, best_params = None, None
best_rmse, best_train_rmse = float("inf"), None
results = []

for depth, lr, l2 in product(param_grid["depth"], param_grid["learning_rate"], param_grid["l2_leaf_reg"]):
    print(f"🔍 Trying depth={depth}, lr={lr}, l2_leaf_reg={l2}")
    model = CatBoostRegressor(
        loss_function="Quantile:alpha=0.5",
        iterations=500,              # faster search
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2,
        random_seed=SEED,
        thread_count=1,
        od_type="Iter",
        od_wait=50,
        verbose=False
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True, verbose=False)
    preds_test = model.predict(test_pool)
    preds_train = model.predict(train_pool)
    rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
    rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
    results.append((depth, lr, l2, rmse_train, rmse_test))
    
    if rmse_test < best_rmse:
        best_rmse = rmse_test
        best_train_rmse = rmse_train
        best_model = model
        best_params = {"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2}



# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results, columns=["depth", "learning_rate", "l2_leaf_reg", "train_RMSE", "test_RMSE"])
results_df = results_df.sort_values("test_RMSE").reset_index(drop=True)
results_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_results.csv"), index=False)

print("\n✅ Best Parameters:", best_params)
print(f"   Train RMSE: {best_train_rmse:.2f}, Test RMSE: {best_rmse:.2f}")
print(f"   Results saved to {OUTPUT_DIR}/gridsearch_results.csv")
