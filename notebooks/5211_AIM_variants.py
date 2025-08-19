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
# # AIM variants

# %% [markdown]
# ## Imports

# %%
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns

from emissions_harmonization_historical.constants_5000 import (
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSED_TIMESERIES_DB,
    POST_PROCESSED_TIMESERIES_RUN_ID_DB,
    SCM_OUTPUT_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

# %%
# POST_PROCESSED_METADATA_CATEGORIES_DB.load_metadata().to_frame(index=False)

# %% [markdown]
# ## Load data

# %%
scenarios_to_analyse = [
    # Note: still waiting to decide which variant
    ("AIM 3.0", "SSP2 - Low Overshoot*"),
]

# %%
scenario_locator = None
for model, scenario in scenarios_to_analyse:
    if scenario_locator is None:
        scenario_locator = pix.ismatch(model=model, scenario=scenario)
    else:
        scenario_locator = scenario_locator | pix.ismatch(model=model, scenario=scenario)

# %%
climate_model_locator = pix.ismatch(climate_model="MAGICCv7.6**")

# %%
categories = POST_PROCESSED_METADATA_CATEGORIES_DB.load(scenario_locator & climate_model_locator)["value"]
categories

# %%
metadata_quantile = POST_PROCESSED_METADATA_QUANTILE_DB.load(scenario_locator & climate_model_locator)["value"]
# metadata_quantile

# %%
temperatures_in_line_with_assessment = POST_PROCESSED_TIMESERIES_RUN_ID_DB.load(
    pix.isin(variable="Surface Temperature (GSAT)") & scenario_locator & climate_model_locator
)
# temperatures_in_line_with_assessment

# %%
erfs = SCM_OUTPUT_DB.load(
    pix.ismatch(
        variable=[
            "Effective Radiative Forcing**",
        ]
    )
    & scenario_locator
    & climate_model_locator,
    progress=True,
    max_workers=multiprocessing.cpu_count(),
)
# erfs

# %%
emissions = POST_PROCESSED_TIMESERIES_DB.load(
    scenario_locator,
    progress=True,
    max_workers=multiprocessing.cpu_count(),
)
# emissions

# %% [markdown]
# ## Helper functions


# %%
def add_model_scenario_column(indf: pd.DataFrame, ms_separator: str, ms_level: str, copy: bool = True) -> pd.DataFrame:
    """
    Add a model-scenario column

    TODO: push this to pandas-openscm as something like
    `update_index_levels_multi_input`
    that allows users to updated index levels
    based on the value of multiple other index columns.
    """
    out = indf
    if copy:
        out = out.copy()

    # Push ability to create a new level from multiple other levels into pandas-openscm
    new_name = ms_level
    new_level = (
        indf.index.droplevel(out.index.names.difference(["model", "scenario"]))
        .drop_duplicates()
        .map(lambda x: ms_separator.join(x))
    )

    if new_level.shape[0] != indf.shape[0]:
        dup_level = out.index.get_level_values("model") + ms_separator + out.index.get_level_values("scenario")
        new_level = dup_level.unique()
        new_codes = new_level.get_indexer(dup_level)
    else:
        new_codes = np.arange(new_level.shape[0])

    out.index = pd.MultiIndex(
        levels=[*out.index.levels, new_level],
        codes=[*out.index.codes, new_codes],
        names=[*out.index.names, new_name],
    )

    return out


# %% [markdown]
# ## Plot

# %%
ms_separator = " || "
ms_level = "model || scenario"
# palette = {ms_separator.join(v[1]): v[0] for v in scratch_selection_l}
pdf_temperature = temperatures_in_line_with_assessment.loc[pix.isin(climate_model="MAGICCv7.6.0a3"), 2000:]

pdf_temperature = add_model_scenario_column(pdf_temperature, ms_separator=ms_separator, ms_level=ms_level)
# pdf_temperature

# %%
hue = ms_level


# %% [markdown]
# ### Temperatures


# %%
def create_legend(ax, handles) -> None:
    """Create legend helper"""
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(1.05, -0.2))


fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharex=True)
for i, (ax, yticks) in enumerate(zip(axes, [np.arange(0.5, 4.01, 0.5), np.arange(0.7, 2.21, 0.1)])):
    pdf_temperature.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        hue_var=hue,
        style_var="climate_model",
        # quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.75)),
        quantiles_plumes=((0.5, 1.0),),
        # quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.0), ((0.05, 0.95), 0.0)),
        ax=ax,
        create_legend=(lambda x, y: None) if i > 0 else create_legend,
    )
    ax.set_xlim([2000, 2100])
    ax.set_yticks(yticks)
    ax.set_ylim(ymin=yticks.min(), ymax=yticks.max())
    # ax.set_ylim(ymax=ymax)
    ax.grid()

# %% [markdown]
# ### Peak and 2100 warming

# %%
metadata_quantile.loc[pix.isin(metric=["max", "2100"], quantile=[0.5, 0.33, 0.67]),].unstack(
    ["metric", "quantile"]
).round(3).sort_values(by=[("max", 0.5)], ascending=True)

# %% [markdown]
# ### Categories

# %%
categories.unstack("metric").sort_values("category")

# %% [markdown]
# ### ERFs

# %%
erfs_to_plot = [
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
]
pdf_erfs = add_model_scenario_column(
    erfs.loc[pix.isin(variable=erfs_to_plot)], ms_separator=ms_separator, ms_level=ms_level
).loc[:, 2000:]


# %%
def create_legend(ax, handles) -> None:
    """Create legend helper"""
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5))


ncols = 2
nrows = len(erfs_to_plot) // ncols + len(erfs_to_plot) % ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 5))
axes_flat = axes.flatten()

for i, variable_to_plot in enumerate(erfs_to_plot):
    ax = axes_flat[i]

    vdf = pdf_erfs.loc[pix.isin(variable=variable_to_plot)]
    vdf.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        hue_var=hue,
        style_var="climate_model",
        quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.75)),
        # quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.0), ((0.05, 0.95), 0.0)),
        ax=ax,
        create_legend=create_legend,
    )
    ax.set_title(variable_to_plot, fontdict=dict(fontsize="medium"))

    if i % 2:
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.legend().remove()

    ax.grid()
    # break
# ax.set_xlim([2000, 2100])
# ax.set_yticks(yticks)
# ax.set_ylim(ymin=yticks.min(), ymax=yticks.max())
# # ax.set_ylim(ymax=ymax)
# ax.grid()

# %% [markdown]
# ### Emissions

# %%
emissions_to_plot = [
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|GHG AR6GWP100",
    "Emissions|CO2|AFOLU",
    "Cumulative Emissions|CO2",
    "Emissions|CH4",
    "Emissions|CFC12",
    "Emissions|N2O",
    "Emissions|Sulfur",
    "Emissions|CO",
    "Emissions|BC",
    "Emissions|OC",
    "Emissions|NOx",
    "Emissions|NH3",
    "Emissions|VOC",
]
pdf_emissions = add_model_scenario_column(
    emissions.loc[pix.isin(variable=emissions_to_plot, stage="complete")], ms_separator=ms_separator, ms_level=ms_level
)
# pdf_emissions

# %%
ncols = 2
nrows = len(emissions_to_plot) // ncols + len(emissions_to_plot) % ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 5))
axes_flat = axes.flatten()

for i, variable_to_plot in enumerate(emissions_to_plot):
    ax = axes_flat[i]

    vdf = pdf_emissions.loc[pix.isin(variable=variable_to_plot)].openscm.to_long_data()
    sns.lineplot(
        ax=ax,
        data=vdf,
        x="time",
        y="value",
        hue=hue,
    )
    ax.set_title(variable_to_plot, fontdict=dict(fontsize="medium"))

    if i % 2:
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.legend().remove()

    ax.grid()

# %%
