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
import logging
import multiprocessing

from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
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
    data_root=DATA_ROOT,
    run_checks=False,  # TODO: implement
)
pre_processed = workflow_res.pre_processed_emissions
harmonised = workflow_res.harmonised_emissions
infilled = workflow_res.infilled_emissions
complete_scenarios = workflow_res.complete_scenarios

# %% [markdown]
# ## Save

# %%
for full_path, df in (
    (OUTPUT_PATH / "pre-processed.csv", pre_processed),
    (OUTPUT_PATH / "harmonised.csv", harmonised),
    (OUTPUT_PATH / "infilled.csv", infilled),
    (OUTPUT_PATH / "complete_scenarios.csv", complete_scenarios),
):
    print(f"Writing {full_path}")
    full_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(full_path)
    print(f"Wrote {full_path}")
    print()
