import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

temp = pd.read_csv(os.path.join("data", "transformed", "Oats_Temp_Merge.csv"))
drought = pd.read_csv(os.path.join("data", "transformed", "Oats_Drought_Merge.csv"))

temp['yield_bu_acre'] = pd.to_numeric(temp['yield_bu_acre'], errors='coerce')
temp['temp_anomaly'] = pd.to_numeric(temp['temp_anomaly'], errors='coerce')
drought['yield_bu_acre'] = pd.to_numeric(drought['yield_bu_acre'], errors='coerce')
drought['drought_severity_index'] = pd.to_numeric(drought['drought_severity_index'], errors='coerce')
temp = temp.dropna(subset=['yield_bu_acre', 'temp_anomaly'])
drought = drought.dropna(subset=['yield_bu_acre', 'drought_severity_index'])


# correlation statistics
# temp & yield
pearson_temp = stats.pearsonr(temp['temp_anomaly'], temp['yield_bu_acre'])
spearman_temp = stats.spearmanr(temp['temp_anomaly'], temp['yield_bu_acre'])
# drought & yield
pearson_drought = stats.pearsonr(drought['drought_severity_index'], drought['yield_bu_acre'])
spearman_drought = stats.spearmanr(drought['drought_severity_index'], drought['yield_bu_acre'])

print("Correlations")
print("Temperature vs Yield:")
print(f" Pearson r = {pearson_temp.statistic:.3f}, p = {pearson_temp.pvalue:.4f}")
print(f" Spearman ρ = {spearman_temp.statistic:.3f}, p = {spearman_temp.pvalue:.4f}")

print("\nDrought vs Yield:")
print(f" Pearson r = {pearson_drought.statistic:.3f}, p = {pearson_drought.pvalue:.4f}")
print(f" Spearman ρ = {spearman_drought.statistic:.3f}, p = {spearman_drought.pvalue:.4f}")


# linear regressions
# temp regression
temp_model = smf.ols("yield_bu_acre ~ temp_anomaly", data=temp).fit()
# drought regression
drought_model = smf.ols("yield_bu_acre ~ drought_severity_index", data=drought).fit()

print("\nRegression: Temperature on Yield")
print(temp_model.summary())

print("\nRegression: Drought on Yield")
print(drought_model.summary())
