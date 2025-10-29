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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Create infilling database
#
# Here we create the infilling database.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import pandas.io.excel
import pandas_indexing as pix

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    INFILLED_OUT_DIR,
    INFILLED_OUT_DIR_ID,
    INFILLING_DB,
    PRE_PROCESSED_SCENARIO_DB,
    WMO_2022_PROCESSED_DB,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
    harmonise,
)
from emissions_harmonization_historical.zenodo import upload_to_zenodo

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""}
pd.set_option("display.max_colwidth", None)

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
# No parameters

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
# Only World level data for specific variables
# is used for infilling.
scenarios_for_infilling_db = PRE_PROCESSED_SCENARIO_DB.load(
    pix.isin(region="World", stage="global_workflow_emissions_raw_names"), progress=True
).reset_index("stage", drop=True)
if scenarios_for_infilling_db.empty:
    raise AssertionError

# scenarios_for_infilling_db

# %% [markdown]
# #### Temporary hack: interpolate scenario data to annual to allow harmonisation
#
# Make sure this stays consistent with 5094

# %%
if (
    HARMONISATION_YEAR not in scenarios_for_infilling_db
    or scenarios_for_infilling_db[HARMONISATION_YEAR].isnull().any()
):
    scenarios_for_infilling_db = scenarios_for_infilling_db.T.interpolate(method="index").T

# Drop out any timeseries that contain NaN too
scenarios_for_infilling_db = scenarios_for_infilling_db.dropna()
scenarios_for_infilling_db

# %% [markdown]
# #### Temporary hack: remove carbon removal from raw scenarios

# %%
scenarios_for_infilling_db = scenarios_for_infilling_db.loc[~pix.ismatch(variable="Carbon Removal**")]
scenarios_for_infilling_db

# %% [markdown]
# ### WMO 2022

# %%
wmo_2022_scenarios = WMO_2022_PROCESSED_DB.load(pix.ismatch(model="*projections*"))
if wmo_2022_scenarios.empty:
    raise AssertionError

# wmo_2022_scenarios

# %% [markdown]
# ### Velders et al., 2022

# %%
assert False, "Add these to infilling DB"

# %% [markdown]
# ### History

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)
# Drop out years with NaNs as they break aneris
history = history.dropna(axis="columns")
# history

# %% [markdown]
# ## Harmonise

# %%
infilling_db_raw = pix.concat([scenarios_for_infilling_db, wmo_2022_scenarios]).sort_index(axis="columns")
# infilling_db_raw

# %% [markdown]
# ### Temporary hack: linearly interpolate to ensure harmonisation works

# %%
infilling_db_raw = infilling_db_raw.T.interpolate(method="index").T
# infilling_db_raw

# %%
# Could load in user overrides from elsewhere here.
# They need to be a series with name "method".
# TODO: make sure these are consistent with what we do in 5094
user_overrides = None

# %%
harmonise_res = harmonise(
    scenarios=infilling_db_raw,
    history=history,
    harmonisation_year=HARMONISATION_YEAR,
    user_overrides=user_overrides,
)
# harmonise_res

# %% [markdown]
# ## Save

# %%
# Linearly interpolate the output so we get sensible infilling

# %%
out = harmonise_res.timeseries
for y in range(out.columns.min(), out.columns.max() + 1):
    if y not in out:
        out[y] = np.nan

out = out.sort_index(axis="columns")
out = out.T.interpolate(method="index").T
# out

# %%
INFILLING_DB.save(out, allow_overwrite=True)

# %% [markdown]
# ## Upload to Zenodo

# %%
# Rewrite as single file
INFILLED_OUT_DIR.mkdir(exist_ok=True, parents=True)
out_file_infilling_db = INFILLED_OUT_DIR / f"infilling-db_{INFILLED_OUT_DIR_ID}.csv"
out = INFILLING_DB.load()
out.to_csv(out_file_infilling_db)
out_file_infilling_db

# %%
upload_to_zenodo([out_file_infilling_db], remove_existing=False, update_metadata=True)
