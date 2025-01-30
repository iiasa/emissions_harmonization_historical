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
# # Compare runs

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import pandas_indexing as pix

from gcages.io import load_timeseries_csv

from emissions_harmonization_historical.constants import DATA_ROOT, SCENARIO_TIME_ID

# %%
ar6_workflow_out_dir = (
    DATA_ROOT 
    / "climate-assessment-workflow" 
    / "output" 
    / "ar6-workflow-magicc"
    / SCENARIO_TIME_ID
)
ar6_workflow_out_dir

# %%
updated_workflow_magicc_v753_dir = (
    DATA_ROOT 
    / "climate-assessment-workflow" 
    / "output" 
    / "0001_magicc-v7-5-3_600-member"
    / SCENARIO_TIME_ID
)
updated_workflow_magicc_v753_dir

# %%
metadata_l = []
for label, out_dir in (
    ("ar6-workflow", ar6_workflow_out_dir),
    ("update-workflow_magiccv7.5.3", updated_workflow_magicc_v753_dir),
):
    tmp = pd.read_csv(out_dir / "metadata.csv").set_index(["model", "scenario"]).pix.assign(workflow=label)
    metadata_l.append(tmp)

metadata = pix.concat(metadata_l)
metadata

# %%
(
    metadata
    .groupby(["workflow"])["category"]
    .value_counts()
    .unstack("workflow")
    # .reorder_levels(["category", "workflow"])
    # .sort_index()
)

# %%
infilled_l = []
for label, out_dir in (
    ("ar6-workflow", ar6_workflow_out_dir),
    ("update-workflow_magiccv7.5.3", updated_workflow_magicc_v753_dir),
):
    tmp = load_timeseries_csv(out_dir / "infilled.csv",
                             index_columns=["model", "scenario", "region", "variable", "unit"],
                              out_column_type=int,
                             ).pix.assign(workflow=label)
    infilled_l.append(tmp)

infilled = pix.concat(infilled_l)
# Interpolate to make later life easier
infilled = infilled.T.interpolate("index").T
infilled

# %%
