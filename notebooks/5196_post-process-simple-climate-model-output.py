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
# # Post process
#
# Here we post process the results.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
from gcages.ar6.post_processing import (
    categorise_scenarios,
    get_exceedance_probabilities,
    get_exceedance_probabilities_over_time,
    get_temperatures_in_line_with_assessment,
)
from gcages.index_manipulation import set_new_single_value_levels
from gcages.post_processing import PostProcessingResult
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_EXCEEDANCE_PROBABILITIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSED_METADATA_RUN_ID_DB,
    POST_PROCESSED_TIMESERIES_EXCEEDANCE_PROBABILITIES_DB,
    POST_PROCESSED_TIMESERIES_QUANTILE_DB,
    POST_PROCESSED_TIMESERIES_RUN_ID_DB,
    SCM_OUTPUT_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pix.set_openscm_registry_as_default()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "AIM"
scm: str = "MAGICCv7.6.0a3"
output_to_pdf: bool = False

# %%
assessed_gsat_variable = "Surface Temperature (GSAT)"
gsat_assessment_median = 0.85
gsat_assessment_time_period = range(1995, 2014 + 1)
gsat_assessment_pre_industrial_period = range(1850, 1900 + 1)
quantiles_of_interest = (
    0.05,
    0.10,
    1.0 / 6.0,
    0.33,
    0.50,
    0.67,
    5.0 / 6.0,
    0.90,
    0.95,
)
exceedance_thresholds_of_interest = np.arange(1.0, 4.01, 0.5)

# %% [markdown]
# ## Load data

# %% [markdown]
# ### SCM output

# %%
raw_gsat_variable_in = "Surface Air Temperature Change"

# %%
in_df = SCM_OUTPUT_DB.load(pix.ismatch(variable=raw_gsat_variable_in, model=f"*{model}*", climate_model=f"*{scm}*"))
# in_df

# %% [markdown]
# ## Emissions

# %% [markdown]
# ## Post process

# %% [markdown]
# ### Temperatures in line with the assessment

# %%
temperatures_in_line_with_assessment = update_index_levels_func(
    get_temperatures_in_line_with_assessment(
        in_df.loc[in_df.index.get_level_values("variable").isin([raw_gsat_variable_in])],
        assessment_median=gsat_assessment_median,
        assessment_time_period=gsat_assessment_time_period,
        assessment_pre_industrial_period=gsat_assessment_pre_industrial_period,
        group_cols=["climate_model", "model", "scenario"],
    ),
    {"variable": lambda x: assessed_gsat_variable},
)
# temperatures_in_line_with_assessment

# %%
ax = temperatures_in_line_with_assessment.loc[:, 2010:2100].openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id", hue_var="scenario", style_var="climate_model"
)
ax.grid()

# %%
temperatures_in_line_with_assessment.loc[:, 2010:2030].openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id", hue_var="scenario", style_var="climate_model"
)

# %% [markdown]
# #### Quantiles

# %%
temperatures_in_line_with_assessment_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(
        temperatures_in_line_with_assessment,
        "run_id",
    ).quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
    new_name="quantile",
)
# temperatures_in_line_with_assessment_quantiles

# %% [markdown]
# ### Exceedance probabilities, peak warming and categorisation

# %%
exceedance_probabilities_over_time = get_exceedance_probabilities_over_time(
    temperatures_in_line_with_assessment,
    exceedance_thresholds_of_interest=exceedance_thresholds_of_interest,
    group_cols=["model", "scenario", "climate_model"],
    unit_col="unit",
    groupby_except_levels="run_id",
)

# %%
peak_warming = set_new_single_value_levels(temperatures_in_line_with_assessment.max(axis="columns"), {"metric": "max"})
peak_warming_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(peak_warming, "run_id").quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
    new_name="quantile",
)

eoc_warming = set_new_single_value_levels(temperatures_in_line_with_assessment[2100], {"metric": 2100})
eoc_warming_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(eoc_warming, "run_id").quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
    new_name="quantile",
)
peak_warming_year = set_new_single_value_levels(
    update_index_levels_func(
        temperatures_in_line_with_assessment.idxmax(axis="columns"),  # type: ignore # error in pandas-openscm
        {"unit": lambda x: "yr"},
    ),
    {"metric": "max_year"},
)
peak_warming_year_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(peak_warming_year, "run_id").quantile(
        quantiles_of_interest  # type: ignore # pandas-stubs out of date
    ),
    new_name="quantile",
)

exceedance_probabilities = get_exceedance_probabilities(
    temperatures_in_line_with_assessment,
    exceedance_thresholds_of_interest=exceedance_thresholds_of_interest,
    group_cols=["model", "scenario", "climate_model"],
    unit_col="unit",
    groupby_except_levels="run_id",
)

categories = categorise_scenarios(
    peak_warming_quantiles=peak_warming_quantiles,
    eoc_warming_quantiles=eoc_warming_quantiles,
    group_levels=["climate_model", "model", "scenario"],
    quantile_level="quantile",
)
categories.sort_values()

# %%
exceedance_probabilities.unstack("threshold").sort_values(1.5)

# %% [markdown]
# ### Compile climate output result

# %%
timeseries_run_id = pd.concat([temperatures_in_line_with_assessment])
timeseries_quantile = pd.concat([temperatures_in_line_with_assessment_quantiles])
timeseries_exceedance_probabilities = pd.concat([exceedance_probabilities_over_time])

metadata_run_id = pd.concat(  # type: ignore # pandas-stubs out of date
    [peak_warming, eoc_warming, peak_warming_year]
)
metadata_quantile = pd.concat(  # type: ignore # pandas-stubs out of date
    [
        peak_warming_quantiles,
        eoc_warming_quantiles,
        peak_warming_year_quantiles,
    ]
)
metadata_exceedance_probabilities = exceedance_probabilities
metadata_categories = categories

res = PostProcessingResult(
    timeseries_run_id=timeseries_run_id,
    timeseries_quantile=timeseries_quantile,
    timeseries_exceedance_probabilities=timeseries_exceedance_probabilities,
    metadata_run_id=metadata_run_id,
    metadata_quantile=metadata_quantile,
    metadata_exceedance_probabilities=metadata_exceedance_probabilities,
    metadata_categories=metadata_categories,
)

# %%
res.metadata_categories.unstack(["metric"]).sort_values("category")

# %%
res.metadata_quantile.unstack(["metric", "unit", "quantile"]).loc[
    :,
    [
        (metric, "K", p)
        for (metric, p) in [
            ("max", 0.33),
            ("max", 0.5),
            ("max", 0.67),
            (2100, 0.5),
        ]
    ],
].sort_values(by=[("max", "K", 0.33)]).round(3)

# %% [markdown]
# ## Save

# %%
# Annoying hack, one to fix in future
tmp = res.metadata_quantile.to_frame(name="value").reset_index("metric")
tmp["metric"] = tmp["metric"].astype(str)
tmp = tmp.set_index("metric", append=True)
res.metadata_quantile = tmp

# %%
tmp = res.metadata_run_id.to_frame(name="value").reset_index("metric")
tmp["metric"] = tmp["metric"].astype(str)
tmp = tmp.set_index("metric", append=True)
res.metadata_run_id = tmp

# %%
for df, db in (
    (res.metadata_categories.to_frame(name="value"), POST_PROCESSED_METADATA_CATEGORIES_DB),
    (res.metadata_exceedance_probabilities.to_frame(name="value"), POST_PROCESSED_METADATA_EXCEEDANCE_PROBABILITIES_DB),
    (res.metadata_quantile, POST_PROCESSED_METADATA_QUANTILE_DB),
    (res.metadata_run_id, POST_PROCESSED_METADATA_RUN_ID_DB),
    (res.timeseries_exceedance_probabilities, POST_PROCESSED_TIMESERIES_EXCEEDANCE_PROBABILITIES_DB),
    (res.timeseries_quantile, POST_PROCESSED_TIMESERIES_QUANTILE_DB),
    (res.timeseries_run_id, POST_PROCESSED_TIMESERIES_RUN_ID_DB),
):
    db.save(df, allow_overwrite=True)
