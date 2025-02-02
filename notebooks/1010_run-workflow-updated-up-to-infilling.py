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
# # Run workflow - updated up to infilling
#
# Run the climate assessment workflow with our updated processing
# up to the end of infilling.

# %% [markdown]
# ## Imports

# %%
import json
import logging
import multiprocessing

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pyam
import silicone.database_crunchers
import tqdm.autonotebook as tqdman
from gcages.infilling import Infiller
from gcages.io import load_timeseries_csv
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    FOLLOWER_SCALING_FACTORS_ID,
    SCENARIO_TIME_ID,
    WMO_2022_PROCESSING_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.harmonisation import AR7FTHarmoniser
from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.workflow import run_workflow_up_to_infilling

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
OUTPUT_PATH = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"
)
OUTPUT_PATH

# %%
n_processes = multiprocessing.cpu_count()
run_checks = False  # TODO: turn on

# %% [markdown]
# ## Load scenario data

# %%
scenarios_raw_global = load_global_scenario_data(
    scenario_path=DATA_ROOT / "scenarios" / "data_raw",
    scenario_time_id=SCENARIO_TIME_ID,
    progress=True,
).loc[:, :2100]  # TODO: drop 2100 end once we have usable scenario data post-2100

# %% [markdown]
# ## Run workflow

# %%
workflow_res = run_workflow_up_to_infilling(
    scenarios_raw_global,
    n_processes=n_processes,
    run_checks=False,  # TODO: implement
    data_root=DATA_ROOT,
)
pre_processed = workflow_res.pre_processed_emissions
harmonised = workflow_res.harmonised_emissions

# %%
from emissions_harmonization_historical.infilling import AR7FTInfiller

# %%
infilled = AR7FTInfiller.from_default_config(data_root=DATA_ROOT, harmonised=harmonised)(harmonised)
full_scenarios = pix.concat([harmonised, infilled])

# %%
regress_file = DATA_ROOT / "climate-assessment-workflow/output/0010_20250127-140714_updated-workflow" / "infilled.csv"
regress = load_timeseries_csv(
    regress_file,
    index_columns=infilled.index.names,
    out_column_type=int,
)

# %%
from gcages.testing import assert_frame_equal
assert_frame_equal(
    full_scenarios,
    regress,
)

# %%
assert False, "stop"

# %% [markdown]
# ## Save

# %%
for full_path, df in (
    (OUTPUT_PATH / "pre-processed.csv", pre_processed),
    (OUTPUT_PATH / "harmonised.csv", harmonised),
    (OUTPUT_PATH / "infilled.csv", infilled),
):
    print(f"Writing {full_path}")
    full_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(full_path)
    print(f"Wrote {full_path}")
    print()
