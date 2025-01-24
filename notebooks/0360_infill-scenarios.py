# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Infill - WMO
#
# Here we infill the variables we can using WMO's scenarios.

# %%
import pandas as pd
import pandas_indexing as pix
import seaborn as sns

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    INFILLING_WMO_ID,
)
from emissions_harmonization_historical.io import load_csv

# %%
SCENARIO_TIME_ID = "20250122-140031"

# %%
harmonised_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "harmonised"
    / f"harmonised-scenarios_{SCENARIO_TIME_ID}_{HARMONISATION_ID}.csv"
)
harmonised_file

# %%
out_file = (
    DATA_ROOT / "climate-assessment-workflow" / "interim" / f"infilled-wmo_{SCENARIO_TIME_ID}_{INFILLING_WMO_ID}.csv"
)
out_file

# %%
harmonised = load_csv(harmonised_file)

# %%
# Variables we expect to get in a future WMO scenario
wmo_variables = [
    "Emissions|Montreal Gases|CFC|CFC11",
    "Emissions|Montreal Gases|CFC|CFC12",
    "Emissions|Montreal Gases|CFC|CFC113",
    "Emissions|Montreal Gases|CFC|CFC114",
    "Emissions|Montreal Gases|CFC|CFC115",
    "Emissions|Montreal Gases|HCFC22",
    "Emissions|Montreal Gases|HCFC141b",
    "Emissions|Montreal Gases|HCFC142b",
    "Emissions|Montreal Gases|CH3Br",
    "Emissions|Montreal Gases|CH3Cl",
    "Emissions|Montreal Gases|CH3CCl3",
    "Emissions|Montreal Gases|CCl4",
    "Emissions|Montreal Gases|Halon1202",
    "Emissions|Montreal Gases|Halon1211",
    "Emissions|Montreal Gases|Halon1301",
    "Emissions|Montreal Gases|Halon2402",
]
len(wmo_variables)

# %%
# TODO: switch to WMO scenario once we have one

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_PATH


# %%
def transform_rcmip_to_iamc_variable(v):
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (
        ("F-Gases|", ""),
        ("PFC|", ""),
        ("HFC4310mee", "HFC43-10"),
        ("MAGICC AFOLU", "AFOLU"),
        ("MAGICC Fossil and Industrial", "Energy and Industrial Processes"),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
ar6_wmo_scenarios = rcmip_clean.loc[
    pix.ismatch(mip_era="CMIP6")
    & pix.ismatch(scenario="ssp245")
    & pix.ismatch(region="World")
    & pix.ismatch(variable=wmo_variables)
].reset_index(["mip_era", "activity_id"], drop=True)
ar6_wmo_scenarios = ar6_wmo_scenarios.T.interpolate(method="index").T.loc[:, harmonised.columns]
ar6_wmo_scenarios


# %%
def get_sns_df(indf):
    """
    Get data frame to use with seaborn's plotting
    """
    out = indf.copy()
    out.columns.name = "year"
    out = out.stack().to_frame("value").reset_index()

    return out


# %%
sns.relplot(
    data=get_sns_df(ar6_wmo_scenarios),
    x="year",
    y="value",
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
    kind="line",
)

# %%
wmo_infilled_l = []
for model, scenario in harmonised.pix.unique(["model", "scenario"]):
    tmp = ar6_wmo_scenarios.pix.assign(model=model, scenario=scenario)
    wmo_infilled_l.append(tmp)

wmo_infilled = pix.concat(wmo_infilled_l)
wmo_infilled

# %%
out_file.parent.mkdir(exist_ok=True, parents=True)
wmo_infilled.to_csv(out_file)
out_file
