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
# # Upload history files to Zenodo
#
# Upload the history files to Zenodo.
# Forcing the rest of the workflow
# to retrieve the files from Zenodo ensures
# that we use a consistent set of history files,
# regardless of who/where/how we have run
# the rest of the workflow.

# %% [markdown]
# ## Imports

# %%
import pandas as pd

from emissions_harmonization_historical.constants_5000 import (
    CREATE_HISTORY_FOR_GLOBAL_WORKFLOW_ID,
    CREATE_HISTORY_FOR_GRIDDING_ID,
    HISTORY_HARMONISATION_INTERIM_DIR,
)

# %% [markdown]
# ## Setup

# %%

# %% [markdown]
# ## Load data

# %%
history_gridded_workflow = pd.read_feather(
    HISTORY_HARMONISATION_INTERIM_DIR / f"gridding-history_{CREATE_HISTORY_FOR_GRIDDING_ID}.feather"
)
# history_gridded_workflow

# %%
history_global_workflow = pd.read_feather(
    HISTORY_HARMONISATION_INTERIM_DIR / f"global-workflow-history_{CREATE_HISTORY_FOR_GLOBAL_WORKFLOW_ID}.feather"
)
# history_global_workflow

# %% [markdown]
# ## Upload to Zenodo

# %%
history_gridded_workflow.to_csv("tmp.csv")

# %%
# !du -sh tmp.csv

# %%
history_gridded_workflow.to_feather("tmp.feather")

# %%
# !du -sh tmp.feather

# %%
history_gridded_workflow.to_parquet("tmp.parquet.gzip", compression="gzip")

# %%
# !du -sh tmp.parquet.gzip

# %%
# pd.read_parquet("tmp.parquet.gzip")

# %%
logger.configure(handlers=[dict(sink=sys.stderr, level="INFO")])
logger.enable("openscm_zenodo")
