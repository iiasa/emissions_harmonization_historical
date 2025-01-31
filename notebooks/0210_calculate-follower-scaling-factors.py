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
# # Calculate

# %%
import json
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
from gcages.io import load_timeseries_csv
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    FOLLOWER_SCALING_FACTORS_ID,
    HARMONISATION_YEAR,
    VELDERS_ET_AL_2022_PROCESSING_ID,
    WMO_2022_PROCESSING_ID,
)
from emissions_harmonization_historical.infilling_followers import FOLLOW_LEADERS

# %%
FOLLOW_LEADERS.keys()

# %%
out_file = DATA_ROOT / "global-composite" / f"follower-scaling-factors_{FOLLOWER_SCALING_FACTORS_ID}.json"

# %%
velders_et_al_2022_processed_output_file = DATA_ROOT / Path(
    "global",
    "velders-et-al-2022",
    "processed",
    f"velders-et-al-2022_cmip7_global_{VELDERS_ET_AL_2022_PROCESSING_ID}.csv",
)
velders = load_timeseries_csv(
    velders_et_al_2022_processed_output_file,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)

velders

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"
HISTORICAL_GLOBAL_COMPOSITE_PATH

# %%
history = strip_pint_incompatible_characters_from_units(
    load_timeseries_csv(
        HISTORICAL_GLOBAL_COMPOSITE_PATH,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
)
history

# %%
wmo_2022_processed_output_file = DATA_ROOT / Path(
    "global",
    "wmo-2022",
    "processed",
    f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv",
)
wmo_2022_processed_output_file

# %%
wmo_processed = load_timeseries_csv(
    wmo_2022_processed_output_file,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
wmo_processed

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
ar6_history = rcmip_clean.loc[pix.isin(mip_era=["CMIP6"], scenario=["ssp245"], region=["World"])]
ar6_history = (
    ar6_history.loc[
        ~pix.ismatch(
            variable=[
                f"Emissions|{stub}|**" for stub in ["BC", "CH4", "CO", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC"]
            ]
        )
        & ~pix.ismatch(variable=["Emissions|CO2|*|**"])
        & ~pix.isin(variable=["Emissions|CO2"])
    ]
    .T.interpolate("index")
    .T
)
full_var_set = ar6_history.pix.unique("variable")
n_variables_in_full_scenario = 52
if len(full_var_set) != n_variables_in_full_scenario:
    raise AssertionError

# %% [markdown]
# ## Calculate scaling factors and intercepts for variables that don't have another source
#
# These allow us to later create pathways based on a lead gas,
# whether future or historical.

# %% [markdown]
# Have to be a bit clever with scaling to consider background/natural emissions.
#
# Instead of
#
# $$
# f = a * l
# $$
# where $f$ is the follow variable, $a$ is the scaling factor and $l$ is the lead.
#
# We want
# $$
# f - f_0 = a * (l - l_0)
# $$
# where $f_0$ is pre-industrial emissions of the follow variable and $l_0$ is pre-industrial emissions of the lead.

# %%
FOLLOW_LEADERS

# %%
ar6_history.loc[pix.ismatch(variable="**3F8*")].loc[:, 2000:2022]

# %%
PI_YEAR = 1750

leaders_scaling_factors = {}
for follower, leader in FOLLOW_LEADERS.items():
    # Need f_harmonisation_year, l_harmonisation_year,
    # f_0, l_0

    # TODO: use inversions for these components
    # (requires harmonising WMO pathways then.)
    if leader in velders.pix.unique("variable").tolist():
        lead_df = velders.loc[pix.isin(variable=[leader])]
        l_0 = 0.0  # fine assumption
    elif follower.endswith("Halon1202"):
        lead_df = wmo_processed.loc[pix.isin(variable=[leader])]
        # I don't know why we bother with this gas,
        # was all zero in CMIP6 too.
        follow_df = 0.0 * lead_df
    else:
        # l_0 = float(
        #     ar6_history.loc[pix.isin(variable=[leader])][PI_YEAR].values.squeeze()
        # )
        continue

    # if leader in history.pix.unique("variable").tolist():
    #     lead_df = history.loc[pix.isin(variable=[leader])]
    # elif leader in wmo_processed.pix.unique("variable").tolist():
    #     lead_df = wmo_processed.loc[pix.isin(variable=[leader])]
    # else:
    #     raise NotImplementedError(leader)

    l_harmonisation_year = float(lead_df[HARMONISATION_YEAR].values.squeeze())
    follow_df = ar6_history.loc[pix.isin(variable=[follower])]
    f_0 = float(follow_df[PI_YEAR].values.squeeze())
    f_harmonisation_year = float(follow_df[HARMONISATION_YEAR].values.squeeze())

    if (f_harmonisation_year - f_0) == 0.0:
        scaling_factor = 0.0
    else:
        scaling_factor = (l_harmonisation_year - l_0) / (f_harmonisation_year - f_0)

    l_unit = lead_df.pix.unique("unit")
    if len(l_unit) != 1:
        raise AssertionError
    l_unit = l_unit[0]

    f_unit = follow_df.pix.unique("unit")
    if len(f_unit) != 1:
        raise AssertionError
    f_unit = f_unit[0]

    leaders_scaling_factors[follower] = {
        "leader": leader,
        "scaling_factor": scaling_factor,
        "intercept": f_0,
        "calculation_year": HARMONISATION_YEAR,
        "l_calculation_year": l_harmonisation_year,
        "f_calculation_year": f_harmonisation_year,
        "l_unit": l_unit,
        "f_unit": f_unit,
    }

leaders_scaling_factors

# %%
out_file.parent.mkdir(exist_ok=True, parents=True)
with open(out_file, "w") as fh:
    json.dump(leaders_scaling_factors, fh, indent=2)

out_file
