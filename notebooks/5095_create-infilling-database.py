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
from functools import partial

import numpy as np
import pandas as pd
import pandas.io.excel
import pandas_indexing as pix
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.comparison import compare_close
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_OUT_DIR,
    INFILLED_OUT_DIR_ID,
    INFILLING_DB,
    VELDERS_ET_AL_2022_PROCESSED_DB,
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
scenarios_for_infilling_db = HARMONISED_SCENARIO_DB.load(
    pix.isin(region="World", workflow="for_scms"), progress=True
).reset_index("workflow", drop=True)
if scenarios_for_infilling_db.empty:
    raise AssertionError

# scenarios_for_infilling_db

# %%
if scenarios_for_infilling_db.isnull().any().any():
    raise AssertionError

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
velders_2022_scenarios = VELDERS_ET_AL_2022_PROCESSED_DB.load(~pix.ismatch(scenario="history"))
if velders_2022_scenarios.empty:
    raise AssertionError

velders_2022_scenarios = update_index_levels_func(
    velders_2022_scenarios,
    updates={
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)
velders_2022_scenarios = update_index_levels_func(
    velders_2022_scenarios, {"unit": lambda x: x.replace("HFC4310mee", "HFC4310")}
)

# velders_2022_scenarios

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
infilling_db_raw = pix.concat([scenarios_for_infilling_db, wmo_2022_scenarios, velders_2022_scenarios]).sort_index(
    axis="columns"
)
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
if harmonise_res.timeseries.isnull().any().any():
    raise AssertionError

# %%
compare_infilling_harmonisation = compare_close(
    harmonise_res.timeseries,
    infilling_db_raw.loc[:, harmonise_res.timeseries.columns],
    "harmonised",
    "input",
    isclose=partial(np.isclose, rtol=1e-4),
)
models_reharmonised = compare_infilling_harmonisation.index.get_level_values("model").unique().tolist()
if models_reharmonised != ["Velders et al., 2022"]:
    msg = f"Data from {models_reharmonised} was re-harmonised. This shouldn't happen?"
    raise AssertionError(msg)

# %%
tmp = compare_infilling_harmonisation
tmp.columns.name = "stage"
tmp = tmp.stack().to_frame("value").reset_index()
sns.relplot(
    data=tmp,
    x="year",
    y="value",
    col="variable",
    col_wrap=3,
    hue="scenario",
    style="stage",
    facet_kws=dict(sharey=False),
    kind="line",
)

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
