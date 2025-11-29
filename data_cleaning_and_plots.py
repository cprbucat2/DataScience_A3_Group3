import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")


def norm_colname(col):
    # remove html tags
    col = re.sub(r"<[^>]+>", "", str(col))       
    col = col.replace("\n", " ").replace("\r", " ")
    # remove many punctuation
    col = re.sub(r"[\(\)\[\]\{\}\,\/\\\:\;]", " ", col)  
    col = re.sub(r"[-]+", " ", col)
    col = re.sub(r"\s+", " ", col).strip()
    col = col.lower()
    col = col.replace(" ", "_")
    # collapse multiple underscores
    col = re.sub(r"_+", "_", col)
    return col

def find_column_by_keywords(cols, must_have=None, any_of=None):
    # search normalized column list for a column that contains all 'must_have' 
    # tokens and at least one token from 'any_of' 
    # return first match or None.
    must_have = must_have or []
    any_of = any_of or []
    for c in cols:
        ok = True
        for token in must_have:
            if token not in c:
                ok = False; break
        if not ok: 
            continue
        if any_of:
            if not any(token in c for token in any_of):
                continue
        return c
    return None

raw_dir = os.path.join("data", "raw")
trans_dir = os.path.join("data", "transformed")
os.makedirs(trans_dir, exist_ok=True)

oats_path = os.path.join(raw_dir, "NASS_Oat_Production.csv")
temp_path = os.path.join(raw_dir, "NOAA_Annual_Temperature_Anomalies.csv")
drought_path = os.path.join(raw_dir, "US_Drought_Severity_Index.csv")

# load oats data and clean column names
oats_raw = pd.read_csv(oats_path, dtype=str)
orig_cols = oats_raw.columns.tolist()
norm_map = {c: norm_colname(c) for c in orig_cols}
oats_raw = oats_raw.rename(columns=norm_map)

'''print("normalized columns:")
for orig, norm in list(norm_map.items())[:12]:
    print(f"{orig} -> {norm}")'''

cols = oats_raw.columns.tolist()
year_col = find_column_by_keywords(cols, any_of=["year", "date"])
if year_col is None:
    # common variants
    candidates = [c for c in cols if re.search(r"\b20\d{2}\b", c) or "yr" in c]
    year_col = candidates[0] if candidates else None
if year_col is None:
    raise KeyError("Could not find a year column in oats data; columns: " + ", ".join(cols))
#print("Detected oats year column:", year_col)

# production (bushels)
prod_col = find_column_by_keywords(cols, must_have=["production"], any_of=["bu", "bushel", "measured_in_bu", "value"])
# fallback: any column containing 'production' and 'bu' may be missing; try 'production' then choose numeric later
if prod_col is None:
    prod_col = find_column_by_keywords(cols, any_of=["production"])
if prod_col is None:
    raise KeyError("Could not find production column in oats data. Columns: " + ", ".join(cols))
#print("Detected oats production column:", prod_col)

# acres harvested
harvest_col = find_column_by_keywords(cols, any_of=["acres_harvested","acres","harvest"])
harvest_col_priority = [c for c in cols if "harvest" in c and "acres" in c]
if harvest_col_priority:
    harvest_col = harvest_col_priority[0]
elif harvest_col is None:
    harvest_col = find_column_by_keywords(cols, any_of=["acres"])
if harvest_col is None:
    raise KeyError("Could not find acres/harvested column in oats data. Columns: " + ", ".join(cols))
#print("Detected oats acres column:", harvest_col)

planted_col = find_column_by_keywords(cols, any_of=["planted", "acres_planted"])

# select and cast
oats = oats_raw[[year_col, prod_col, harvest_col] + ([planted_col] if planted_col else [])].copy()
oats = oats.rename(columns={year_col:"year", prod_col:"production_raw", harvest_col:"acres_harvested"})
if planted_col:
    oats = oats.rename(columns={planted_col:"acres_planted"})

# remove commas and non numeric characters in numeric columns, cast
oats["production_bu"] = oats["production_raw"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
oats["production_bu"] = pd.to_numeric(oats["production_bu"], errors="coerce")
oats["acres_harvested"] = oats["acres_harvested"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
oats["acres_harvested"] = pd.to_numeric(oats["acres_harvested"], errors="coerce")
if "acres_planted" in oats.columns:
    oats["acres_planted"] = oats["acres_planted"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    oats["acres_planted"] = pd.to_numeric(oats["acres_planted"], errors="coerce")

# year numeric
oats["year"] = pd.to_numeric(oats["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce").astype("Int64")
oats = oats.sort_values("year").reset_index(drop=True)

# compute yield using acres_harvested only
oats["yield_bu_acre"] = oats["production_bu"] / oats["acres_harvested"]
# keep minimal columns
keep = ["year", "production_bu", "acres_harvested", "yield_bu_acre"]
if "acres_planted" in oats.columns:
    keep.append("acres_planted")
oats = oats[keep]

# save cleaned oats
oats.to_csv(os.path.join(trans_dir, "Oats_Cleaned.csv"), index=False)
print("Cleaned oats rows:", len(oats))
print(oats.head())

# load NOAA temp anomalies, skip first title row
temp = pd.read_csv(temp_path, skiprows=1)
# normalize columns
temp.columns = [norm_colname(c) for c in temp.columns]
rename_map = {}
rename_map["date"] = "year"
rename_map["climdiv"] = "temp_anomaly"
temp = temp.rename(columns=rename_map)
# cast
temp["year"] = pd.to_numeric(temp["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce").astype("Int64")
temp["temp_anomaly"] = pd.to_numeric(temp["temp_anomaly"], errors="coerce").replace([-99.99, -999, -9999], pd.NA)
temp = temp.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
temp.to_csv(os.path.join(trans_dir, "Temp_Anomaly_Cleaned.csv"), index=False)
print("Temp Anom rows:", len(temp))
print(temp.head())

# load US Drought Monitor, aggregate weekly to annual
d = pd.read_csv(drought_path, dtype=str)
d.columns = [norm_colname(c) for c in d.columns]
# mapdate col -> year
mapcols = [c for c in d.columns if c.startswith("map")]
if not mapcols:
    mapdate_col = d.columns[0]
else:
    mapdate_col = mapcols[0]
d["year"] = pd.to_numeric(d[mapdate_col].astype(str).str[:4], errors="coerce").astype("Int64")

# find NONE..D4 columns
required = {}
for key in ["none","d0","d1","d2","d3","d4"]:
    candidates = [c for c in d.columns if c==key or key in c]
    required[key.upper()] = candidates[0] if candidates else None

# convert percentages to numeric and fill missing with 0
for k,v in required.items():
    if v is not None:
        d[v] = pd.to_numeric(d[v].astype(str).str.replace("%","").str.strip(), errors="coerce").fillna(0.0)
    else:
        # if a level is entirely missing, create zeros
        d[k] = 0.0

# compute weekly weighted sum (% * weight)
weights = {"NONE":0,"D0":1,"D1":2,"D2":3,"D3":4,"D4":5}
# ensure we reference column names present in dataframe
weekly_weighted = 0
for key, w in weights.items():
    colname = required.get(key)
    if colname is not None:
        weekly_weighted = weekly_weighted + d[colname] * w
    else:
        weekly_weighted = weekly_weighted + 0

# weekly score normalized to 0-5 by dividing percent
d["weekly_score"] = weekly_weighted / 100.0

# annual average of weekly_score
drought_annual = d.groupby("year", as_index=False)["weekly_score"].mean().rename(columns={"weekly_score":"drought_severity_index"})
drought_annual = drought_annual.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
drought_annual.to_csv(os.path.join(trans_dir, "Drought_Annual_Index.csv"), index=False)
print("Drought annual rows:", len(drought_annual))
print(drought_annual.head())

# merge by year, oats & temp and oats & drought
def merge_on_year(base, other):
    if "state" in base.columns and "state" in other.columns:
        return base.merge(other, on=["year","state"], how="left")
    else:
        return base.merge(other, on=["year"], how="left")

oats_temp = merge_on_year(oats, temp)
oats_drought = merge_on_year(oats, drought_annual)

oats_temp.to_csv(os.path.join(trans_dir, "Oats_Temp_Merge.csv"), index=False)
oats_drought.to_csv(os.path.join(trans_dir, "Oats_Drought_Merge.csv"), index=False)
print("Saved merges. oats_temp rows:", len(oats_temp), " oats_drought rows:", len(oats_drought))
print(oats_temp.head())
print(oats_drought.head())



# oats & temp anom analysis
df = oats_temp.dropna(subset=["yield_bu_acre", "temp_anomaly"]).copy()
print("Temp analysis rows:", len(df))
if len(df) == 0:
    print("No overlapping rows with non-null yield & temp_anomaly. Check temporal overlap.")
else:
    # time series aggregated (national average if multiple entries per year)
    ts = df.groupby("year").agg({"yield_bu_acre":"mean","temp_anomaly":"mean"}).reset_index()
    # plot dual-axis time series
    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(ts["year"], ts["yield_bu_acre"], label="Yield (bu/acre)")
    ax.set_xlabel("Year"); ax.set_ylabel("Yield (bu/acre)")
    ax2 = ax.twinx()
    ax2.plot(ts["year"], ts["temp_anomaly"], color="tab:orange", label="Temp anomaly (°F)")
    ax2.set_ylabel("Temp anomaly (°F)")
    ax.set_title("Yield and Temperature Anomaly Over Time")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout(); plt.show()

    # moving averages 5-year
    ts["yield_ma5"] = ts["yield_bu_acre"].rolling(5, min_periods=1).mean()
    ts["temp_ma5"] = ts["temp_anomaly"].rolling(5, min_periods=1).mean()
    plt.figure(figsize=(11,5))
    plt.plot(ts["year"], ts["yield_ma5"], label="Yield 5-yr MA")
    plt.plot(ts["year"], ts["temp_ma5"], label="Temp anomaly 5-yr MA")
    plt.title("5-year Moving Averages"); plt.xlabel("Year"); plt.legend(); plt.show()

    # correlations
    corr_results = {}
    for x in ["yield_bu_acre", "acres_harvested"]:
        if x in df.columns:
            a = df[x].dropna(); b = df["temp_anomaly"].dropna()
            if len(a)>1 and len(b)>1:
                pear = stats.pearsonr(a,b)
                spear = stats.spearmanr(a,b)
                corr_results[x] = {"pearson_r":pear[0], "pearson_p":pear[1], "spearman_r":spear.correlation, "spearman_p":spear.pvalue}
    print("Correlation results (temp):", corr_results)

    # scatter + regression line
    plt.figure(figsize=(8,6))
    sns.regplot(x="temp_anomaly", y="yield_bu_acre", data=df, ci=95, scatter_kws={"s":50,"alpha":0.7})
    plt.title("Yield vs Temp Anomaly"); plt.xlabel("Temp anomaly (°F)"); plt.ylabel("Yield (bu/acre)"); plt.show()

    # OLS test yield - temp_anomaly
    if len(df) >= 3:
        X = sm.add_constant(df["temp_anomaly"])
        y = df["yield_bu_acre"]
        model = sm.OLS(y, X, missing='drop').fit()
        print(model.summary())
        print("Test H0: beta_temp = 0 -> p-value for temp_anomaly:", model.pvalues.get("temp_anomaly"))
    else:
        print("Not enough rows to estimate OLS for temp model.")




# oats & drought analysis
df2 = oats_drought.dropna(subset=["yield_bu_acre", "drought_severity_index"]).copy()
print("Drought analysis rows:", len(df2))
if len(df2) == 0:
    print("No overlapping rows with non-null yield & drought_severity_index. Check temporal overlap.")
else:
    ts2 = df2.groupby("year").agg({"yield_bu_acre":"mean","drought_severity_index":"mean"}).reset_index()
    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(ts2["year"], ts2["yield_bu_acre"], label="Yield (bu/acre)")
    ax.set_xlabel("Year"); ax.set_ylabel("Yield (bu/acre)")
    ax2 = ax.twinx()
    ax2.plot(ts2["year"], ts2["drought_severity_index"], color="tab:red", label="Drought severity index")
    ax2.set_ylabel("Drought severity index")
    ax.set_title("Yield and Drought Severity Over Time")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout(); plt.show()

    # highlight extreme drought years (top 10%)
    threshold = ts2["drought_severity_index"].quantile(0.90)
    extreme_years = ts2.loc[ts2["drought_severity_index"] >= threshold, "year"].tolist()
    plt.figure(figsize=(11,5))
    plt.plot(ts2["year"], ts2["yield_bu_acre"], label="Yield (bu/acre)")
    for yv in extreme_years:
        plt.axvline(x=yv, color="red", linestyle="--", alpha=0.7)
    plt.title("Yield with Extreme Drought Years Highlighted"); plt.xlabel("Year"); plt.ylabel("Yield (bu/acre)"); plt.legend(); plt.show()

    # correlations
    if len(df2) > 1:
        pear = stats.pearsonr(df2["yield_bu_acre"].dropna(), df2["drought_severity_index"].dropna())
        spear = stats.spearmanr(df2["yield_bu_acre"].dropna(), df2["drought_severity_index"].dropna())
        print("Yield vs Drought correlations - Pearson:", pear, " Spearman:", spear)

    # scatter + regression
    plt.figure(figsize=(8,6))
    sns.regplot(x="drought_severity_index", y="yield_bu_acre", data=df2, ci=95, scatter_kws={"s":50,"alpha":0.7})
    plt.title("Yield vs Drought Severity"); plt.xlabel("Drought severity index"); plt.ylabel("Yield (bu/acre)"); plt.show()

    if len(df2) >= 3:
        X2 = sm.add_constant(df2["drought_severity_index"])
        y2 = df2["yield_bu_acre"]
        model2 = sm.OLS(y2, X2, missing='drop').fit()
        print(model2.summary())
        print("Test H0: beta_drought = 0 -> p-value:", model2.pvalues.get("drought_severity_index"))
    else:
        print("Not enough rows to estimate OLS for drought model.")


