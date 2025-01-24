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
# # Infill - combination
#
# Combine all the infilling components
# and harmonised scenarios into complete scenarios.

# %%

import pandas_indexing as pix

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    INFILLING_LEFTOVERS_ID,
    INFILLING_SILICONE_ID,
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
infilled_silicone_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "interim"
    / f"infilled-silicone_{SCENARIO_TIME_ID}_{INFILLING_SILICONE_ID}.csv"
)
infilled_silicone_file

# %%
infilled_wmo_file = (
    DATA_ROOT / "climate-assessment-workflow" / "interim" / f"infilled-wmo_{SCENARIO_TIME_ID}_{INFILLING_WMO_ID}.csv"
)
infilled_wmo_file

# %%
infilled_leftovers_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "interim"
    / f"infilled-leftovers_{SCENARIO_TIME_ID}_{INFILLING_LEFTOVERS_ID}.csv"
)
infilled_leftovers_file

# %%
out_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "infilled"
    / f"infilled_{SCENARIO_TIME_ID}_{HARMONISATION_ID}_{INFILLING_SILICONE_ID}_{INFILLING_WMO_ID}_{INFILLING_LEFTOVERS_ID}.csv"  # noqa: E501
)
out_file

# %%
full_scenarios = pix.concat(
    [
        load_csv(harmonised_file),
        load_csv(infilled_silicone_file),
        load_csv(infilled_wmo_file),
        load_csv(infilled_leftovers_file),
    ]
)
full_scenarios.sort_index()

# %%
for (model, scenario), msdf in full_scenarios.groupby(["model", "scenario"]):
    expected_n_variables_per_scenario = 52
    if len(msdf.pix.unique("variable")) != expected_n_variables_per_scenario:
        msg = f"{model=} {scenario=}"
        raise AssertionError(msg)

# %%
out_file.parent.mkdir(exist_ok=True, parents=True)
full_scenarios.to_csv(out_file)
out_file
