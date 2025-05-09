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
# # Compile historical emissions
#
# Here we compile the historical emissions
# needed for gridding and the global workflow.

# %% [markdown]
# ## Imports

# %%
from functools import partial

import pandas_indexing as pix
import pandas_openscm
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
HARMONISATION_GRIDDING_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"gridding-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
# HARMONISATION_GRIDDING_EMISSIONS_FILE

# %%
HARMONISATION_GLOBAL_WORKFLOW_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"global-workflow-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
# HARMONISATION_GLOBAL_WORKFLOW_EMISSIONS_FILE

# %%
OUT_FILE_REPORTING_CONVENTIONS = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonisation-emissions_reporting-conventions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
OUT_FILE_REPORTING_CONVENTIONS.parent.mkdir(exist_ok=True, parents=True)

# %%
OUT_FILE_GCAGES_CONVENTIONS = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonisation-emissions_gcages-conventions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
OUT_FILE_GCAGES_CONVENTIONS.parent.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load data

# %%
gridding_harmonisation_emissions = load_timeseries_csv(
    HARMONISATION_GRIDDING_EMISSIONS_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if gridding_harmonisation_emissions.empty:
    raise AssertionError

gridding_harmonisation_emissions.columns.name = "year"
# gridding_harmonisation_emissions

# %%
global_workflow_harmonisation_emissions_reporting_conventions = load_timeseries_csv(
    HARMONISATION_GLOBAL_WORKFLOW_EMISSIONS_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if global_workflow_harmonisation_emissions_reporting_conventions.empty:
    raise AssertionError

global_workflow_harmonisation_emissions_reporting_conventions.columns.name = "year"
# global_workflow_harmonisation_emissions_reporting_conventions

# %%
assert_units_match_wishes(global_workflow_harmonisation_emissions_reporting_conventions)

# %%
global_workflow_harmonisation_emissions_gcages_names = update_index_levels_func(
    global_workflow_harmonisation_emissions_reporting_conventions,
    {
        "unit": strip_pint_incompatible_characters_from_unit_string,
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        ),
    },
)
global_workflow_harmonisation_emissions_gcages_names

# %% [markdown]
# ## Combine and save

# %%
harmonisation_emissions_reporting_conventions = pix.concat(
    [gridding_harmonisation_emissions, global_workflow_harmonisation_emissions_reporting_conventions]
).pix.assign(model="history", scenario="history")
# harmonisation_emissions_reporting_conventions
harmonisation_emissions_reporting_conventions.to_csv(OUT_FILE_REPORTING_CONVENTIONS)
OUT_FILE_REPORTING_CONVENTIONS

# %%
harmonisation_emissions_gcages_conventions = pix.concat(
    [gridding_harmonisation_emissions, global_workflow_harmonisation_emissions_gcages_names]
).pix.assign(model="history", scenario="history")
# harmonisation_emissions_reporting_conventions
harmonisation_emissions_gcages_conventions.to_csv(OUT_FILE_GCAGES_CONVENTIONS)
OUT_FILE_GCAGES_CONVENTIONS
