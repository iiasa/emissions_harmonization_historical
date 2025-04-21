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
# # Create infilling database
#
# Create the infilling database
# for our climate assessment workflow.
# This needs to be consistent with the workflow,
# but isn't actually directly part of it
# (to allow for running single scenarios in future).

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing

import pandas_indexing as pix
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.harmonisation import AR7FTHarmoniser
from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.pre_processing import AR7FTPreProcessor

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
OUTPUT_PATH = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "input"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_{HARMONISATION_ID}_infilling-database.csv"
)
OUTPUT_PATH

# %%
n_processes = multiprocessing.cpu_count()

# %% [markdown]
# ## Load scenario data

# %%
scenarios_raw_global = load_global_scenario_data(
    scenario_path=DATA_ROOT / "scenarios" / "data_raw",
    scenario_time_id=SCENARIO_TIME_ID,
    progress=True,
).loc[:, :2100]  # TODO: drop 2100 end once we have usable scenario data post-2100

# %% [markdown]
# ## Create infilling database
#
# Harmonise all the scenarios
# to create our infilling database.

# %%
pre_processor = AR7FTPreProcessor.from_default_config()
pre_processor.run_co2_reclassification

# %%
harmoniser = AR7FTHarmoniser.from_default_config(
    data_root=DATA_ROOT,
    n_processes=n_processes,
)
harmoniser.historical_emissions

# %%
pre_processed = pre_processor(scenarios_raw_global)
pre_processed

# %%
# ?? WITCH HFC23 data becomes NaN from 2090 onwards
# Use this line to see the issue: pre_processed.loc[pix.isin(model=["WITCH 6.0"]) & pix.ismatch(variable="**HFC*")].dropna(how="all", axis="columns")
#    - temporary solution: just drop it
pre_processed = pre_processed.loc[~(pix.isin(model=["WITCH 6.0"]) & pix.ismatch(variable="**HFC23"))]
pre_processed

# %%
harmonised = harmoniser(pre_processed)
harmonised

# %% [markdown]
# ## Save

# %%
OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
harmonised.to_csv(OUTPUT_PATH)
OUTPUT_PATH
