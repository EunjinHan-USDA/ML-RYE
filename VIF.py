import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ───────────────────────────────────────────────
# 1. Load Data
# ───────────────────────────────────────────────
df = pd.read_csv("/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_and_Maybe_646_Rows_with_Biomass_and_CN_Ratio_Averaged.csv")

# ───────────────────────────────────────────────
# 2. Define Feature Groups
# ───────────────────────────────────────────────
groups = {
    "temperature": [
        "GS0_20avgTavg", "GS20_30avgTavg", "GS30_40avgTavg",
        "GS40_50avgTavg", "GS50avgTavg"
    ],
    "radiation": [
        "GS0_20avgSrad", "GS20_30avgSrad", "GS30_40avgSrad",
        "GS40_50avgSrad", "GS50avgSrad"
    ],
    "rainfall": [
        "GS0_20cRain", "GS20_30cRain", "GS30_40cRain",
        "GS40_50cRain", "GS50cRain"
    ],
    "soil": [
        "OM (%/100)", "Sand", "Silt", "Clay", "awc"
    ],
    "fertilizer": [
        "N_rate_fall.kg_ha", "N_rate_spring.kg_ha"
    ],

     "GDD": [
        "GS0_20cGDD", "GS20_30cGDD", "GS30_40cGDD", "GS50cGDD", "FallcumGDD", "SpringcumGDD", 
    ]
}

# ───────────────────────────────────────────────
# 3. Create Output Directory
# ───────────────────────────────────────────────
output_dir = "Grouped_VIF"
os.makedirs(output_dir, exist_ok=True)

# ───────────────────────────────────────────────
# 4. Function to Compute and Plot VIF
# ───────────────────────────────────────────────
def calculate_vif_and_plot(df, features, group_name):
    df_clean = df[features].dropna()
    X = df_clean.values

    vif_df = pd.DataFrame()
    vif_df["Feature"] = features
    vif_df["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

    # Save CSV
    vif_df.to_csv(os.path.join(output_dir, f"VIF_{group_name}.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x="VIF", y="Feature", data=vif_df.sort_values("VIF", ascending=True), palette="crest")
    plt.title(f"VIF for {group_name.capitalize()} Variables", fontsize=14)
    plt.axvline(10, color='red', linestyle='--', label="VIF = 10 Threshold")
    plt.xlabel("VIF")
    plt.ylabel("Feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"VIF_{group_name}.png"), dpi=300)
    plt.close()

    print(f"✅ Saved VIF for group: {group_name}")
    return vif_df

# ───────────────────────────────────────────────
# 5. Loop Through Groups and Generate VIFs
# ───────────────────────────────────────────────
for group, feature_list in groups.items():
    calculate_vif_and_plot(df, feature_list, group)
