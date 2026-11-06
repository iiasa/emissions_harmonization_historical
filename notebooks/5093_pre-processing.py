# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Pre-processing
#
# Here we pre-process each model's data.

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
model: str = "REMIND"
model: str = "CGEM"

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
model_raw = RAW_SCENARIO_DB.load(pix.ismatch(model=f"*{model}*"), progress=True)
if model_raw.empty:
    raise AssertionError

# %%
model_raw.loc[pix.ismatch(variable="**CO2|AFOLU", region="World")].sort_index(axis="columns")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Extract the model data, keeping:
#
# - only reported timesteps
# - only data from 2015 onwards (we don't care about data before this)
# <!-- - only data up to 2100 (while reporting is still weird for data post 2100) --> TODO handle this better

# %%
model_df = model_raw.loc[:, 2015:].dropna(how="all", axis="columns")
if model_df.empty:
    raise AssertionError

model_df.columns.name = "year"
# model_df

# %%
# sorted(model_df.pix.unique("variable"))

# %%
# AIM currently (2025.07.10) accidentally reports zeroes between model years for Carbon Removal,
# so let's fix that by delete those and interpolating
if model.startswith("AIM"):
    model_df = pd.concat(
        [
            model_df.loc[pix.ismatch(variable=["Emissions**"])].loc[:, 2015:2100:1].dropna(how="all", axis="columns"),
            model_df.loc[pix.ismatch(variable=["Carbon Removal**"])]
            .loc[:, 2015:2100:5]
            .dropna(how="all", axis="columns"),
        ]
    )

# %%
# GCAM currently (2025.07.10) reports
# Australia and New Zealand for some scenarios
# and Australia_and New Zealand for others.
# Fix this.
if model.startswith("GCAM"):
    model_df = model_df.openscm.update_index_levels({"region": lambda x: x.replace("Australia_and", "Australia and")})

# %%
# Interpolate
# (needs to be done to ensure that the CDR-Emissions correction
# works even if Carbon Removal data is reported at different time resolutions)
model_df = model_df.T.interpolate(method="index").T
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
# # CGEM goes below pre-industrial
# # when it shouldn't.
# # Fix this.
# if model.startswith("CGEM"):
#     import matplotlib.pyplot as plt

#     from emissions_harmonization_historical.constants_5000 import (
#         HISTORY_HARMONISATION_DB,
#     )

#     history_for_gridding_harmonisation = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="gridding_emissions"))
#     history_for_gridding_harmonisation = history_for_gridding_harmonisation.loc[pix.ismatch(region=["CGEM|*", "World"])]

#     res_grid_tmp = pre_processing_res.gridding_workflow_emissions.sort_index().copy()

#     issues_for = []
#     for idx, row in res_grid_tmp.iterrows():
#         idx_d = {k: v for k, v in zip(model_df.index.names, idx)}
#         if not any(v in idx_d["variable"] for v in ("|CO|", "|OC|")):
#             # Don't touch this
#             continue

#         locator = {k: v for k, v in idx_d.items() if k not in ["model", "scenario"]}

#         history_compare = history_for_gridding_harmonisation.loc[pix.isin(**locator)]
#         if history_compare.empty:
#             raise AssertionError(idx_d)

#         row_min = row.min()
#         history_min = history_compare.min(axis=1).iloc[0]

#         if row_min < history_min:
#             issues_for.append(idx_d)

#             raw = res_grid_tmp.loc[[idx]]
#             scaled = raw.copy()
#             scaled[scaled < history_min] = history_min

#             # if "CO" not in idx_d["variable"]:
#             #     continue

#             # row_min_year = row.idxmin()
#             # first_year_at_min = row.index[np.isclose(row, row_min, rtol=1e-3)].min()
#             # first_year_below_history = row.index[row < history_min].min()

#             # pre_ind_onwards_year = first_year_below_history
#             # pre_ind_onwards_year_scale_factor = history_min / raw.loc[:, pre_ind_onwards_year].iloc[0]

#             # scaled.loc[:, pre_ind_onwards_year:] = history_min

#             # if np.isinf(pre_ind_onwards_year_scale_factor):
#             #     scaled[raw == 0.0] = history_min

#             # else:
#             #     scale_factor_start_year = max(start_year, pre_ind_onwards_year - scale_years)
#             #     scaled.loc[:, scale_factor_start_year:pre_ind_onwards_year] *= np.linspace(
#             #         1.0,
#             #         pre_ind_onwards_year_scale_factor,
#             #         pre_ind_onwards_year - scale_factor_start_year + 1,
#             #     )

#             # scaled.loc[:, (raw > history_min).values.squeeze()] = raw.loc[:, (raw > history_min).values.squeeze()]

#             # # scale_factor_max = history_min / row_min
#             # # scale_factor_start_year = max(start_year, first_year_at_min - scale_years)
#             # # scale_factor = pd.Series(np.ones_like(row), index=row.index)
#             # # scale_factor.loc[first_year_at_min:] = scale_factor_max
#             # # scale_factor.loc[scale_factor_start_year:first_year_at_min] = np.linspace(
#             # #     1.0, scale_factor_max, first_year_at_min - scale_factor_start_year + 1
#             # # )
#             # # scale_factor = np.max(np.vstack([history_min / row, scale_factor]), axis=0)

#             # # scaled = raw * scale_factor
#             # # scaled.loc[np.where(np.isinf(scaled))] = history_min

#             if "CHN" in idx_d["region"] or "|CO|" in idx_d["variable"]:
#                 pdf = pix.concat(
#                     [
#                         raw.pix.assign(source="raw"),
#                         scaled.pix.assign(source="scaled"),
#                         history_compare.reset_index("purpose", drop=True).pix.assign(source="history"),
#                     ]
#                 ).sort_index(axis="columns")
#                 ax = pdf.pix.project(["variable", "region", "source"]).T.plot()
#                 ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

#                 plt.show()
#             # raise AssertionError
#             # break

#             res_grid_tmp.loc[[idx]] = scaled

#     pre_processing_res.gridding_workflow_emissions = res_grid_tmp

# %%
# Hard override the global workflow emissions for CO2 AFOLU
# to use globally reported numbers,
# even if they're not consistent with region-sector reporting.
pre_processing_res.global_workflow_emissions = pix.concat(
    [
        pre_processing_res.global_workflow_emissions.loc[~pix.isin(variable="Emissions|CO2|Biosphere")],
        model_df.loc[pix.isin(variable="Emissions|CO2|AFOLU", region="World")].pix.assign(
            variable="Emissions|CO2|Biosphere"
        ),
    ]
)

pre_processing_res.global_workflow_emissions_raw_names = pix.concat(
    [
        pre_processing_res.global_workflow_emissions_raw_names.loc[~pix.isin(variable="Emissions|CO2|AFOLU")],
        model_df.loc[pix.isin(variable="Emissions|CO2|AFOLU", region="World")],
    ]
)

# %%
pdf = pre_processing_res.global_workflow_emissions.loc[
    pix.ismatch(
        variable=[
            "**CO2**",
            # "**CH4", "**NOx", "**N2O", "**SOx",
        ]
    )
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
# pre_processing_res.global_workflow_emissions.loc[:, 2023:2025]

# %%
# sorted(pre_processing_res.assumed_zero_emissions.pix.unique("variable"))

# %%
# pre_processing_res.global_workflow_emissions.loc[
# pix.ismatch(variable="**CO2**")
# ]

# %% editable=true slideshow={"slide_type": ""}
for stage, df in (
    ("gridding_emissions", pre_processing_res.gridding_workflow_emissions),
    ("global_workflow_emissions", pre_processing_res.global_workflow_emissions),
    ("global_workflow_emissions_raw_names", pre_processing_res.global_workflow_emissions_raw_names),
    ("assumed_zero", pre_processing_res.assumed_zero_emissions),
):
    PRE_PROCESSED_SCENARIO_DB.save(df.pix.assign(stage=stage), allow_overwrite=True)
    print(f"Saved {stage}")

# %%
