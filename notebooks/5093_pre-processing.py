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

# %% [markdown]
# ## Imports

# %%
import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.completeness import get_missing_levels

from emissions_harmonization_historical.constants_5000 import (
    PRE_PROCESSED_SCENARIO_DB,
    RAW_SCENARIO_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "AIM"

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
model_raw = RAW_SCENARIO_DB.load(pix.ismatch(model=f"*{model}*"), progress=True)
if model_raw.empty:
    raise AssertionError

# model_raw

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
import numpy as np  # noqa: E402
from gcages.cmip7_scenariomip.pre_processing.reaggregation import ReaggregatorBasic  # noqa: E402
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import get_required_timeseries_index  # noqa: E402
from gcages.index_manipulation import create_levels_based_on_existing, set_new_single_value_levels  # noqa: E402

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
for (model_l, scenario), sdf in model_df.groupby(["model", "scenario"]):
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
        {"model": model_l, "scenario": scenario},
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

# %% [markdown]
# #### Temporary hack: turn off checks
#
# So even if the model's reporting is inconsistent,
# it should still pass pre-processing.

# %%
pre_processor.run_checks = False

# %%
# Urgh, GCAM has some unexpected reporting for stuff we've never heard of
model_df = model_df.loc[~pix.ismatch(variable="Emissions|C2F6|**")]

# %%
pre_processing_res = pre_processor(model_df)

# %%
pdf = pre_processing_res.global_workflow_emissions.loc[
    pix.ismatch(variable=["**CO2**", "**CH4", "**N2O", "**SOx"])
].openscm.to_long_data()
fg = sns.relplot(
    data=pdf,
    x="time",
    y="value",
    col="variable",
    col_order=sorted(pdf["variable"].unique()),
    col_wrap=2,
    hue="scenario",
    hue_order=sorted(pdf["scenario"].unique()),
    kind="line",
    linewidth=2.0,
    alpha=0.7,
    facet_kws=dict(sharey=False),
)
for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="gray")
    else:
        ax.set_ylim(ymin=0.0)

    ax.set_xticks(np.arange(2020, 2101, 10))
    ax.grid()

# %%
# sorted(pre_processing_res.assumed_zero_emissions.pix.unique("variable"))

# %%
pre_processing_res.global_workflow_emissions

# %%
for stage, df in (
    ("gridding_emissions", pre_processing_res.gridding_workflow_emissions),
    ("global_workflow_emissions", pre_processing_res.global_workflow_emissions),
    ("global_workflow_emissions_raw_names", pre_processing_res.global_workflow_emissions_raw_names),
    ("assumed_zero", pre_processing_res.assumed_zero_emissions),
):
    PRE_PROCESSED_SCENARIO_DB.save(df.pix.assign(stage=stage), allow_overwrite=True)
    print(f"Saved {stage}")
