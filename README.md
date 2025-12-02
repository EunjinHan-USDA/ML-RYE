# ML-RYE

Machine learning model development for predicting cereal rye biomass and C:N ratio using weather, soil, and management features across multiple sites and growth stages.

This repository contains the analysis code, model training scripts, diagnostics, spatial-transferability evaluation, and soil data extraction tools used in the manuscript:

> **"Machine learning-based prediction of cereal rye cover crop biomass across diverse agroecosystem"**  
>   
> (Year, Journal — to be updated upon acceptance)

The workflow includes XGBoost and CatBoost modeling, VIF-based multicollinearity tests, spatial transferability analysis, and SSURGO-based soil property extraction.

---

## Repository Structure

### Modeling Scripts — Biomass
- `XGBoost_Biomass_GS40_50.py`
- `XGBoost_Biomass_GS50.py`
- `CatBoost_Biomass_GS40_50.py`
- `CatBoost_Biomass_GS50.py`

### Modeling Scripts — C:N Ratio
- `XGBoost_CNRatio_GS20_30.py`
- `XGBoost_CNRatio_GS30_40.py`
- `XGBoost_CNRatio_GS40_50.py`
- `CatBoost_CNRatio_GS20_30.py`
- `CatBoost_CNRatio_GS30_40.py`
- `CatBoost_CNRatio_GS40_50.py`

### General Stage-Specific Models
- `XGBoost_GS20_30.py`
- `XGBoost_GS30_40.py`
- `CatBoost_GS20_30.py`
- `CatBoost_GS30_40.py`

### Hyperparameter Tuning
- `XGBoost_gridsearch.py`
- `AUG29_CatBoost_Gridsearch.py`

### Diagnostics
- `VIF.py` — Variance Inflation Factor analysis for multicollinearity

### Transferability
- `Spatial_transferability.py` — Tests generalization across held-out states/sites

### Soil Data Extraction (SSURGO)
Folder: `extract_SSURGO/`

Includes examples for:
- Querying SSURGO soil properties  
- Extracting texture, organic matter, AWC, etc.  
- Preparing soil predictors compatible with the modeling dataset  

> **Note:** Some SSURGO scripts require external GIS tools or APIs. See comments within scripts.

### Other Files
- `requirements.txt`  
- `LICENSE`  
- `README.md` (this file)

---

## Installation

### Clone this repository
```bash
git clone https://github.com/EunjinHan-USDA/ML-RYE.git
cd ML-RYE

