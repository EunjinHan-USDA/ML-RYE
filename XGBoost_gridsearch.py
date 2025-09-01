# ---------- MUST BE FIRST: single-thread math & hashing for determinism ----------
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ---------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# =========================
# GLOBAL SETTINGS
# =========================
SEED = 42
np.random.seed(SEED)

# =========================
# PATH & FEATURES
# =========================
csv_path = "/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged_7.csv"

input_features = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha", "zone",
    "GS0_20avgSrad", "GS0_20cRain",
    "GS20_30avgSrad", "GS20_30cRain",
    "FallcumGDD", "SpringcumGDD",
    "OM (%/100)", "Sand", "Clay",
    "legume_preceding", "planting_method"
]
cat_features = ["zone", "legume_preceding", "planting_method"]
target_col = "biomass_mean"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(csv_path)
df = df[input_features + [target_col]].dropna()
print(f"✅ Valid samples with complete data: {len(df)}")

X = df[input_features].copy()
y = df[target_col].copy()

# one-hot encode categoricals
X = pd.get_dummies(X, columns=cat_features, drop_first=True)

# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, shuffle=True
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# =========================
# GRID SEARCH (balanced tuning)
# =========================
param_grid = {
    "max_depth": [2, 3, 4],              # shallower trees
    "learning_rate": [0.01, 0.02, 0.03], # smaller learning rates
    "reg_lambda": [10, 20, 50],          # stronger regularization
    "min_child_weight": [5, 10, 20]      # minimum leaf weight
}

results = []

for depth, lr, reg, mcw in product(param_grid["max_depth"],
                                   param_grid["learning_rate"],
                                   param_grid["reg_lambda"],
                                   param_grid["min_child_weight"]):
    print(f"🔍 depth={depth}, lr={lr}, reg_lambda={reg}, min_child_weight={mcw}")
    
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,   # fixed trees
        learning_rate=lr,
        max_depth=depth,
        reg_lambda=reg,
        min_child_weight=mcw,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    # train / test predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train  = mean_absolute_error(y_train, y_pred_train)
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test, y_pred_test)
    pct_rmse_train = 100 * rmse_train / y_train.mean()
    pct_rmse_test  = 100 * rmse_test / y_test.mean()
    gap = rmse_test - rmse_train
    
    results.append([depth, lr, reg, mcw,
                    rmse_train, mae_train, r2_train, pct_rmse_train,
                    rmse_test, mae_test, r2_test, pct_rmse_test, gap])

# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results, columns=[
    "max_depth", "learning_rate", "reg_lambda", "min_child_weight",
    "train_RMSE", "train_MAE", "train_R2", "train_%RMSE",
    "test_RMSE", "test_MAE", "test_R2", "test_%RMSE", "gap"
])

# Sort by test RMSE (best generalization first)
results_df = results_df.sort_values("test_RMSE").reset_index(drop=True)
results_df.to_csv("xgboost_gridsearch_balanced_results.csv", index=False)

print("\n✅ Grid search complete. Results saved to xgboost_gridsearch_balanced_results.csv")
print("Top 5 rows:\n", results_df.head())
