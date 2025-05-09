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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Post-process
#
# Here we post-process the results.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %%

import matplotlib.pyplot as plt
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
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.plotting import create_legend_default

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_HARMONISATION_ID,
    CMIP7_SCENARIOMIP_INFILLING_ID,
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
    SCENARIO_TIME_ID,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
RES_DB_ID_KEY = "_".join(
    [
        COMBINED_HISTORY_ID,
        IAMC_REGION_PROCESSING_ID,
        SCENARIO_TIME_ID,
        CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
        CMIP7_SCENARIOMIP_HARMONISATION_ID,
        CMIP7_SCENARIOMIP_INFILLING_ID,
    ]
)

# %%
RES_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "cmip7-scenariomip-workflow" / "scm-running" / RES_DB_ID_KEY,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

RES_DB.load_metadata().shape

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Complete scenarios

# %%
raw_gsat_variable_in = "Surface Air Temperature Change"

# %%
in_df = RES_DB.load(pix.ismatch(variable=raw_gsat_variable_in))
# in_df

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
tmp = temperatures_in_line_with_assessment.stack().unstack("climate_model")
magicc_diff = (tmp["MAGICCv7.6.0a3"] - tmp["MAGICCv7.5.3"]).unstack("time")
magicc_diff.loc[pix.ismatch(scenario="**Very Low**"), 2000:].openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id",
)

# %%
temperatures_in_line_with_assessment_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(
        temperatures_in_line_with_assessment,
        "run_id",
    ).quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
    new_name="quantile",
)
# temperatures_in_line_with_assessment_quantiles

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
res.metadata_categories.unstack(["metric", "climate_model"])["category_name"].sort_values(by=["MAGICCv7.6.0a3"])

# %%
res.metadata_quantile.unstack(["climate_model", "metric", "unit", "quantile"]).loc[
    :,
    [
        (cm, metric, "K", p)
        for cm in [
            v for v in ["MAGICCv7.6.0a3", "MAGICCv7.5.3"] if v in res.metadata_quantile.pix.unique("climate_model")
        ]
        for (metric, p) in [
            ("max", 0.33),
            ("max", 0.5),
            ("max", 0.67),
            (2100, 0.5),
        ]
    ],
].sort_values(by=[("MAGICCv7.6.0a3", "max", "K", 0.33)]).round(3)

# %%
scratch_selection_l = [
    ("#7f3e3e", ("REMIND-MAgPIE 3.5-4.10", "SSP3 - High Emissions")),
    ("#f7a84f", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Medium Emissions")),
    ("#e1ad01", ("REMIND-MAgPIE 3.5-4.10", "SSP3 - Medium-Low Emissions")),
    ("#586643", ("REMIND-MAgPIE 3.5-4.10", "SSP1 - Medium-Low Emissions")),
    ("#2e9e68", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Low Emissions")),
    ("#4b3d89", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Low Overshoot_d")),
    ("#499edb", ("REMIND-MAgPIE 3.5-4.10", "SSP1 - Very Low Emissions_c")),
]

# %%
ms_separator = " || "
palette = {ms_separator.join(v[1]): v[0] for v in scratch_selection_l}

# %%
scratch_selection = pd.MultiIndex.from_tuples([v[1] for v in scratch_selection_l], names=["model", "scenario"])

# %%
pdf = temperatures_in_line_with_assessment.loc[pix.isin(climate_model="MAGICCv7.6.0a3"), 2000:].openscm.mi_loc(
    scratch_selection
)

ms_level = "model || scenario"

# Push ability to create a new level from multiple other levels into pandas-openscm
new_name = ms_level
new_level = (
    pdf.index.droplevel(pdf.index.names.difference(["model", "scenario"]))
    .drop_duplicates()
    .map(lambda x: ms_separator.join(x))
)

if new_level.shape[0] != pdf.shape[0]:
    dup_level = pdf.index.get_level_values("model") + " || " + pdf.index.get_level_values("scenario")
    new_level = dup_level.unique()
    new_codes = new_level.get_indexer(dup_level)
else:
    new_codes = np.arange(new_level.shape[0])

pdf.index = pd.MultiIndex(
    levels=[*pdf.index.levels, new_level],
    codes=[*pdf.index.codes, new_codes],
    names=[*pdf.index.names, new_name],
)


# %%
fig, axes = plt.subplots(nrows=2, figsize=(4, 8), sharex=True)
for i, (ax, yticks) in enumerate(zip(axes, [np.arange(0.5, 5.01, 0.5), np.arange(0.7, 2.21, 0.1)])):
    pdf.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        hue_var=ms_level,
        style_var="climate_model",
        palette=palette,
        quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.3), ((0.05, 0.95), 0.0)),
        ax=ax,
        create_legend=(lambda x, y: None) if i > 0 else create_legend_default,
    )
    ax.set_xlim([2000, 2100])
    ax.set_yticks(yticks)
    ax.set_ylim(ymin=yticks.min(), ymax=yticks.max())
    # ax.set_ylim(ymax=ymax)
    ax.grid()


# %%
tmp = res.metadata_quantile.unstack("climate_model")
magicc_diff = tmp["MAGICCv7.6.0a3"] - tmp["MAGICCv7.5.3"]
magicc_diff.unstack(["metric", "unit", "quantile"])[
    [(metric, "K", percentile) for metric in ["max", 2100] for percentile in [0.33, 0.5, 0.67, 0.95]]
].sort_values(by=("max", "K", 0.5))

# %% [markdown]
# ## Save

# %%
