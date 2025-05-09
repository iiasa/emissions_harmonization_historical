# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Process Adam et al., 2022 data
#
# Process data from [Adam et al., 2024](https://doi.org/10.1038/s43247-024-01946-y).

# %% [markdown]
# ## Imports

# %%
from functools import partial

import pandas as pd
import pandas_indexing as pix
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    ADAM_ET_AL_2024_PROCESSED_DB,
    ADAM_ET_AL_2024_RAW_PATH,
    HISTORY_SCENARIO_NAME,
)

# %% [markdown]
# ## Setup

# %%
pix.set_openscm_registry_as_default()


# %%
raw_file = ADAM_ET_AL_2024_RAW_PATH / "HFC-23_Global_annual_emissions.csv"

# %% [markdown]
# ## Load data

# %%
with open(raw_file) as fh:
    raw = tuple(fh.readlines())

# %% [markdown]
# ## Reformat

# %%
for line in raw:
    if "Units:" in line:
        units = line.split("Units: ")[-1].strip()
        break

else:
    msg = "units not found"
    raise AssertionError(msg)

units

# %%
raw_df = pd.read_csv(raw_file, comment="#")
# raw_df

# %%
out = raw_df[["Year", "Global_annual_emissions"]].rename(
    {"Year": "year", "Global_annual_emissions": "value"}, axis="columns"
)
out["variable"] = "Emissions|HFC23"
out["unit"] = units.replace("/", " HFC23/")
out["model"] = "Adam et al., 2024"
out["scenario"] = HISTORY_SCENARIO_NAME
out["region"] = "World"
out = out.set_index(list(set(out.columns) - {"value"}))["value"].unstack("year")
out = out.pix.convert_unit("kt HFC23/yr")
# out

# %% [markdown]
# ## Save

# %% [markdown]
# Make sure we can convert the variable names

# %%
update_index_levels_func(
    out,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
ADAM_ET_AL_2024_PROCESSED_DB.save(out, allow_overwrite=True)
