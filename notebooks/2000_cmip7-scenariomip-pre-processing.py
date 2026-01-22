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
# # Pre-processing
#
# Here we post-process each model's data.
#
# Your life will be much simpler if you run the
# `040*` notebooks first as they have better diagnostics.

# %% [markdown]
# ## Imports

# %%
import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
import numpy as np
import pandas as pd
import pandas_indexing as pix
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.cmip7_scenariomip.pre_processing.reaggregation import ReaggregatorBasic
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import get_required_timeseries_index
from gcages.completeness import get_missing_levels
from gcages.index_manipulation import create_levels_based_on_existing, set_new_single_value_levels
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
    DATA_ROOT,
    SCENARIO_TIME_ID,
)

# %% [markdown]
# ## Set up

# %%
model: str = "REMIND-MAgPIE 3.5-4.10"

# %%
out_dir = DATA_ROOT / "cmip7-scenariomip-workflow" / "pre-processing" / CMIP7_SCENARIOMIP_PRE_PROCESSING_ID
out_dir.mkdir(exist_ok=True, parents=True)

# %%
out_db = OpenSCMDB(
    db_dir=out_dir,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
SCENARIO_DB = OpenSCMDB(
    db_dir=SCENARIO_PATH / SCENARIO_TIME_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

SCENARIO_DB.load_metadata().shape

# %%
SCENARIO_DB.load_metadata().get_level_values("model").unique()

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
model_raw = SCENARIO_DB.load(pix.isin(model=model), progress=True)
if model_raw.empty:
    raise AssertionError

model_raw

# %%
model_raw.pix.unique(["variable", "region"]).to_frame(index=False)

# %% [markdown]
# Extract the model data, keeping:
#
# - only reported timesteps
# - only data from 2015 onwards (we don't care about data before this)
# - only data up to 2100 (while reporting is still weird for data post 2100)

# %%
model_df = model_raw.loc[:, 2015:2100].dropna(how="all", axis="columns")
if model_df.empty:
    raise AssertionError

model_df.columns.name = "year"
# model_df

# %% [markdown]
# ## Pre-process

# %% [markdown]
# ### Figure out the model-specific regions

# %%
model_regions = [r for r in model_df.pix.unique("region") if r.startswith(model.split(" ")[0])]
if not model_regions:
    raise AssertionError
# model_regions

# %% [markdown]
# ### Temporary hack: assume zero for missing reporting

# %%
reaggregator = ReaggregatorBasic(model_regions=model_regions)
pre_processor = CMIP7ScenarioMIPPreProcessor(
    reaggregator=reaggregator,
    n_processes=None,  # run serially
)

required_index = get_required_timeseries_index(
    model_regions=model_regions,
    world_region=reaggregator.world_region,
    region_level=reaggregator.region_level,
    variable_level=reaggregator.variable_level,
)

variable_unit_map = {
    "|".join(v.split("|")[:2]): u
    for v, u in model_df.index.droplevel(model_df.index.names.difference(["variable", "unit"]))
    .drop_duplicates()
    .to_list()
}


def guess_unit(v_in: str) -> str:
    """Guess the unit of a given variable"""
    for k, v in variable_unit_map.items():
        if v_in.startswith(f"{k}|") or v_in == k:
            return v


tmp_l = []
for scenario, sdf in model_df.groupby("scenario"):
    mls = get_missing_levels(
        sdf.index,
        required_index,
        unit_col=reaggregator.unit_level,
    )

    zeros_hack = pd.DataFrame(
        np.zeros((mls.shape[0], sdf.shape[1])),
        columns=sdf.columns,
        index=create_levels_based_on_existing(mls, {"unit": ("variable", guess_unit)}),
    )
    zeros_hack = set_new_single_value_levels(
        zeros_hack,
        {"model": model, "scenario": scenario},
    ).reorder_levels(sdf.index.names)
    sdf_full = pix.concat([sdf, zeros_hack])

    tmp_l.append(sdf_full)

tmp = pix.concat(tmp_l)
tmp

# %%
# Assuming that all worked, update the model_df
model_df = tmp

# %% [markdown]
# ### Do the pre-processing
#
# The hard part here could be deciding on your re-aggregator.
# For now, we assume all models use the same basic reporting
# hence we can use the same re-aggregation strategy for all.

# %%
reaggregator = gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.ReaggregatorBasic(
    model_regions=model_regions
)
pre_processor = CMIP7ScenarioMIPPreProcessor(
    reaggregator=reaggregator,
    n_processes=None,  # run serially
)

# %%
pre_processing_res = pre_processor(model_df)

# %%
model_safe = model.replace(" ", "_").replace(".", "-")
model_safe

# %%
pre_processing_res.global_workflow_emissions

# %%
for stage, df in (
    ("gridding_emissions", pre_processing_res.gridding_workflow_emissions),
    ("global_workflow_emissions", pre_processing_res.global_workflow_emissions),
    ("global_workflow_emissions_raw_names", pre_processing_res.global_workflow_emissions_raw_names),
    ("assumed_zero", pre_processing_res.assumed_zero_emissions),
):
    out_db.save(df.pix.assign(stage=stage))
    print(f"Saved {stage}")
