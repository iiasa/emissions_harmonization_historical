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

import numpy as np
import openscm_units
import pandas_indexing as pix
from gcages.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    CMIP_CONCENTRATION_INVERSION_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    FOLLOWER_SCALING_FACTORS_ID,
    HARMONISATION_YEAR,
)
from emissions_harmonization_historical.infilling_followers import FOLLOW_LEADERS

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %%
out_file = DATA_ROOT / "global-composite" / f"follower-scaling-factors_{FOLLOWER_SCALING_FACTORS_ID}.json"

# %%
input_file = DATA_ROOT / Path("global-composite", f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv")

# %%
input_file_pi_leaders = (
    DATA_ROOT / "global" / "esgf" / "CR-CMIP-0-4-0" / f"pre-industrial_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.json"
)

# %%
with open(input_file_pi_leaders) as fh:
    pi_leaders_raw = json.load(fh)

pi_values = {k: Q(v[0], v[1]) for k, v in pi_leaders_raw.items()}
pi_values

# %%
history = load_timeseries_csv(
    input_file,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)

history

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
set(FOLLOW_LEADERS.keys())

# %%
PI_YEAR = 1750

leaders_scaling_factors = {}
for follower, leader in FOLLOW_LEADERS.items():
    # Need f_harmonisation_year, l_harmonisation_year,
    # f_0, l_0

    lead_df = history.loc[pix.isin(variable=[leader])]
    follow_df = history.loc[pix.isin(variable=[follower])]

    f_unit = follow_df.pix.unique("unit")
    if len(f_unit) != 1:
        raise AssertionError
    f_unit = f_unit[0].replace("-", "")

    l_unit = lead_df.pix.unique("unit")
    if len(l_unit) != 1:
        raise AssertionError
    l_unit = l_unit[0].replace("-", "")

    f_harmonisation_year = float(follow_df[HARMONISATION_YEAR].values.squeeze())
    l_harmonisation_year = float(lead_df[HARMONISATION_YEAR].values.squeeze())
    if not follow_df[PI_YEAR].isnull().any():
        f_0 = float(follow_df[PI_YEAR].values.squeeze())
    else:
        f_0 = pi_values[follower].to(f_unit).m

    l_0 = pi_values[leader].to(l_unit).m

    if (f_harmonisation_year - f_0) == 0.0:
        scaling_factor = 0.0
    else:
        scaling_factor = (l_harmonisation_year - l_0) / (f_harmonisation_year - f_0)

    if np.isnan(scaling_factor):
        msg = f"{f_harmonisation_year=} {l_harmonisation_year=} {f_0=} {l_0=}"
        raise AssertionError(msg)

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
