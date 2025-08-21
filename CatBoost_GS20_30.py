import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import os

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
df = pd.read_csv("/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged_7.csv")

output_dir = "Biomass_yesNMAYBE_GS20_30_aug7_outputs"
os.makedirs(output_dir, exist_ok=True)

input_features = [
     "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha", "zone", "GS0_20avgTavg", "GS0_20avgSrad", "GS0_20cRain", "GS0_20cGDD",
    "GS20_30avgTavg", "GS20_30avgSrad", "GS20_30cRain", "GS20_30cGDD", 
    "FallcumGDD", "SpringcumGDD",
    "OM (%/100)", "Sand", "Silt", "Clay", 
    "legume_preceding", "planting_method"
]

biomass_col = "biomass_mean"
df = df[input_features + [biomass_col]].dropna()


print(f"✅ Valid samples with complete data and biomass : {len(df)}")

X = df[input_features].copy()
y = df[biomass_col]

cat_features = ["zone", "legume_preceding", "planting_method"]
for col in cat_features:
    X[col] = X[col].fillna("missing").astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total samples after filtering: {len(df)}")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --------------------------------------------------
# 3. QUANTILE REGRESSION MODEL
# --------------------------------------------------
quantiles = [0.1, 0.5, 0.9]
models, preds = {}, {}

for q in quantiles:
    print(f"\n🔁 Training quantile {q}")
    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={q}",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        verbose=100
    )
    model.fit(X_train, y_train)
    models[q] = model
    preds[q] = model.predict(X_test)

# --------------------------------------------------
# 4. EVALUATION AND PINBALL LOSS
# --------------------------------------------------
results_df = pd.DataFrame({
    "actual_biomass": y_test.values,
    "pred_10th": preds[0.1],
    "pred_50th": preds[0.5],
    "pred_90th": preds[0.9]
})
results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

rmse = np.sqrt(mean_squared_error(results_df["actual_biomass"], results_df["pred_50th"]))
mae = mean_absolute_error(results_df["actual_biomass"], results_df["pred_50th"])
r2 = r2_score(results_df["actual_biomass"], results_df["pred_50th"])

print(f"\n📊 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.3f}")
percent_rmse = (rmse / results_df["actual_biomass"].mean()) * 100
print(f"📊 %RMSE : {percent_rmse:.2f}%")

def pinball_loss(y_true, y_pred, alpha):
    delta = y_true - y_pred
    return np.mean(np.maximum(alpha * delta, (alpha - 1) * delta))

for q in quantiles:
    loss = pinball_loss(results_df["actual_biomass"], results_df[f"pred_{int(q*100)}th"], q)
    print(f" Pinball loss q={q}: {loss:.4f}")

# --------------------------------------------------
# 5. UNCERTAINTY PLOT
# --------------------------------------------------
# --------------------------------------------------
# 5. UNCERTAINTY PLOT (Bold Styling)
# --------------------------------------------------
sorted_idx = results_df["actual_biomass"].argsort()
plt.figure(figsize=(10, 6))
plt.plot(results_df["actual_biomass"].values[sorted_idx], label="Actual", color="black", linewidth=2)
plt.plot(results_df["pred_50th"].values[sorted_idx], label="Predicted Median", color="blue", linewidth=2)
plt.fill_between(range(len(results_df)),
                 results_df["pred_10th"].values[sorted_idx],
                 results_df["pred_90th"].values[sorted_idx],
                 color="lightblue", alpha=0.5, label="10th–90th Percentile")

plt.xlabel("Sample Index", fontsize=14, fontweight='bold')
plt.ylabel("Biomass (kg/ha)", fontsize=14, fontweight='bold')
plt.title("Predicted Biomass with Uncertainty", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "uncertainty_plot.png"), dpi=300)
plt.close()


# --------------------------------------------------
# 6. FEATURE IMPORTANCE
# --------------------------------------------------
importances = models[0.5].get_feature_importance(type="PredictionValuesChange")
fi_df = pd.DataFrame({"feature": X.columns, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

plt.figure(figsize=(10, 8))
plt.barh(fi_df["feature"][:30][::-1], fi_df["importance"][:30][::-1])
plt.xlabel("Importance")
plt.title("Top 30 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_top30.png"))
plt.close()

# --------------------------------------------------
# 7. SHAP VALUES (Bold Styling)
# --------------------------------------------------
print("\n✅ Generating SHAP plots")
explainer = shap.TreeExplainer(models[0.5])
shap_values = explainer.shap_values(X_test)

# SHAP bar summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.xlabel("Mean |SHAP value| (impact on model output)", fontsize=14, fontweight='bold')
plt.ylabel("Feature", fontsize=14, fontweight='bold')
plt.title("SHAP Feature Importance (Bar)", fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=300)
plt.close()

# SHAP dot summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.xlabel("SHAP value (impact on model output)", fontsize=14, fontweight='bold')
plt.ylabel("Feature", fontsize=18, fontweight='bold')
plt.title("SHAP Summary Plot", fontsize=18, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_dot.png"), dpi=300)
plt.close()

# --------------------------------------------------
# 8. 1:1 PLOT FOR MEDIAN PREDICTION (BOLD STYLING)
# --------------------------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(results_df["actual_biomass"], results_df["pred_50th"], 
            edgecolor='black', facecolor='dodgerblue', alpha=0.7, s=60)

# Plot 1:1 line
min_val = min(results_df["actual_biomass"].min(), results_df["pred_50th"].min())
max_val = max(results_df["actual_biomass"].max(), results_df["pred_50th"].max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="1:1 Line")

plt.xlabel("Observed Biomass (kg/ha)", fontsize=14, fontweight='bold')
plt.ylabel("Predicted Biomass (kg/ha)", fontsize=14, fontweight='bold')
plt.title("1:1 Plot for Median Prediction", fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "median_prediction_1to1.png"), dpi=300)
plt.close()
