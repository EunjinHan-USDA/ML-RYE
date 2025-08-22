import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import os

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged_7.csv")

output_dir = "CNratio_GS40_50_XGBoost_outputs"
os.makedirs(output_dir, exist_ok=True)

input_features = [
    "state", "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha", "zone",
    "GS0_20avgTavg", "GS0_20avgSrad", "GS0_20cRain", "GS0_20cGDD",
    "GS20_30avgTavg", "GS20_30avgSrad", "GS20_30cRain", "GS20_30cGDD", "GS30_40avgTavg", "GS30_40avgSrad", "GS30_40cRain", "GS30_40cGDD", "GS40_50avgTavg", "GS40_50avgSrad", "GS40_50cRain", "GS40_50cGDD",
    "FallcumGDD", "SpringcumGDD", "TotalcumGDD",
    "OM (%/100)", "Sand", "Silt", "Clay", "awc",
    "legume_preceding", "planting_method"
]

cn_ratio_col = "cn_ratio_mean"
df = df[input_features + [cn_ratio_col]].dropna()
print(f"✅ Valid samples with complete data and CN ratio: {len(df)}")

X = df[input_features].copy()
y = df[cn_ratio_col]

cat_features = ["state", "zone", "legume_preceding", "planting_method"]
X[cat_features] = X[cat_features].fillna("missing").astype(str)

# --------------------------------------------------
# 2. TRAIN-TEST SPLIT & ENCODING
# --------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_all = pd.concat([X_train_raw, X_test_raw])
X_all_encoded = pd.get_dummies(X_all, columns=cat_features)
X_train = X_all_encoded.iloc[:len(X_train_raw)].copy()
X_test = X_all_encoded.iloc[len(X_train_raw):].copy()

print(f"Total samples after filtering: {len(df)}")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --------------------------------------------------
# 3. XGBoost Regressor Training
# --------------------------------------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --------------------------------------------------
# 4. Evaluation
# --------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
percent_rmse = (rmse / y_test.mean()) * 100

print(f"\n📊 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.3f}, %RMSE: {percent_rmse:.2f}%")

results_df = pd.DataFrame({
    "actual_cn_ratio": y_test.values,
    "predicted_cn_ratio": y_pred
})
results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# --------------------------------------------------
# 5. Uncertainty Plot
# --------------------------------------------------
sorted_idx = results_df["actual_cn_ratio"].argsort()
plt.figure(figsize=(10, 6))
plt.plot(results_df["actual_cn_ratio"].values[sorted_idx], label="Actual", color="black")
plt.plot(results_df["predicted_cn_ratio"].values[sorted_idx], label="Predicted", color="blue")
plt.xlabel("Sample Index")
plt.ylabel("C:N Ratio")
plt.title("Predicted C:N Ratio (XGBoost)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "uncertainty_plot.png"))
plt.close()

# --------------------------------------------------
# 6. Feature Importance
# --------------------------------------------------
importances = model.feature_importances_
fi_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False).reset_index(drop=True)
fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

# Group one-hot encoded categories into original features
def simplify_feature_name(col):
    for prefix in ["state_", "zone_", "legume_preceding_", "planting_method_"]:
        if col.startswith(prefix):
            return prefix[:-1]  # remove trailing underscore
    return col

fi_df["base_feature"] = fi_df["feature"].apply(simplify_feature_name)
grouped_importance = fi_df.groupby("base_feature")["importance"].sum().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(10, 8))
plt.barh(grouped_importance["base_feature"][:30][::-1], grouped_importance["importance"][:30][::-1])
plt.xlabel("Aggregated Importance")
plt.title("Top 30 Aggregated Feature Importances (XGBoost - CN Ratio)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_top30_grouped.png"))
plt.close()

# --------------------------------------------------
# 7. SHAP Analysis
# --------------------------------------------------
print("\n✅ Generating SHAP plots")
X_test = X_test.astype(float)  # Ensure SHAP compatibility

explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"))
plt.close()

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_dot.png"))
plt.close()

print("✅ All outputs saved in:", output_dir)
