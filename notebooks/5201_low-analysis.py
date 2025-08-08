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

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["papermill-error-cell-tag"]
# # Analysis of dynamics in low scenarios
#

# %% [markdown]
# ## Imports

# %%
import multiprocessing
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import tqdm.auto

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSED_TIMESERIES_RUN_ID_DB,
    SCM_OUTPUT_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pix.set_openscm_registry_as_default()

# %%
pd.set_option("display.max_rows", 100)

# %% [markdown]
# ## Load data

# %%
scenarios_to_analyse = [
    # ("REMIND*", "*Very Low*"),
    # ("IMAGE*", "*Very Low*"),
    # ("WITCH*", "*- Low*"),
    # ("WITCH*", "*- Very Low*"),
    # ("AIM*", "*Very Low*"),
    ("REMIND*", "SSP1 - Very Low Emissions"),
    # ("IMAGE*", "SSP1 - Very Low Emissions"),
    # ("WITCH*", "SSP1 - Low Overshoot"),
    ("AIM*", "SSP2 - Low Overshoot"),
    ("MESSAGE*", "SSP2 - Low Emissions"),
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
temperatures_in_line_with_assessment_median = temperatures_in_line_with_assessment.openscm.groupby_except(
    "run_id"
).median()
# temperatures_in_line_with_assessment_median

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
erfs_median = erfs.openscm.groupby_except("run_id").median()
# erfs_median

# %%
emissions = SCM_OUTPUT_DB.load(
    pix.ismatch(
        variable=[
            "Emissions**",
        ]
    )
    & scenario_locator
    & climate_model_locator,
    progress=True,
    max_workers=multiprocessing.cpu_count(),
).reset_index("climate_model", drop=True)
# emissions

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

# history.loc[:, :2023]

# %%
scenarios_start_year = emissions.columns.min()

history_to_add = (
    history.openscm.mi_loc(emissions.reset_index(["model", "scenario"], drop=True).drop_duplicates().index)
    .reset_index(["model", "scenario"], drop=True)
    .align(emissions)[0]
    .loc[:, : scenarios_start_year - 1]
)

emissions_complete = pix.concat(
    [
        history_to_add.reorder_levels(emissions.index.names),
        emissions,
    ],
    axis="columns",
)

emissions_complete

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Are any scenarios close to C1?

# %%
categories.unstack(["metric", "climate_model"])["category_name"].sort_values(by=["MAGICCv7.6.0a3"])

# %%
tmp = (
    metadata_quantile.unstack(["metric", "unit", "quantile"])
    .loc[
        :,
        [
            (metric, "K", p)
            for (metric, p) in [
                ("max", 0.33),
                ("max", 0.5),
                ("max", 0.67),
                ("2100", 0.5),
            ]
        ],
    ]
    .reset_index("region", drop=True)
    .reorder_levels(["model", "scenario", "variable", "climate_model"])
    .sort_values(by=[("max", "K", 0.33)])
    .round(3)
)

tmp


# %%
def add_model_scenario_column(indf: pd.DataFrame, ms_separator: str, ms_level: str, copy: bool = True) -> pd.DataFrame:
    """
    Add a model-scenario column

    TODO: use openscm.set_levels
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


# %%
ms_separator = " || "
ms_level = "model || scenario"

# %%
colors = plt.get_cmap("tab20").colors
# colors

palette = {}
for i, (model, scenario) in enumerate(
    categories.index.droplevel(categories.index.names.difference(["model", "scenario"])).values
):
    palette[ms_separator.join([model, scenario])] = colors[i % len(colors)]

# palette

# %%
amsc = partial(
    add_model_scenario_column,
    ms_separator=ms_separator,
    ms_level=ms_level,
)

# %%
start_year = 2000
# start_year = 1750
pdf_emissions = amsc(emissions_complete).loc[:, start_year:]
pdf_erfs = amsc(erfs).loc[:, start_year:]
pdf_temperature = amsc(temperatures_in_line_with_assessment).loc[:, start_year:]

# %%
pdf_emissions

# %%
variables_src = [
    ("Surface Temperature (GSAT)", pdf_temperature, False),
    ("Effective Radiative Forcing", pdf_erfs, False),
    ("Effective Radiative Forcing|Greenhouse Gases", pdf_erfs, False),
    ("Effective Radiative Forcing|Aerosols", pdf_erfs, False),
    ("Effective Radiative Forcing|Aerosols|Direct Effect", pdf_erfs, False),
    ("Effective Radiative Forcing|Aerosols|Direct Effect|BC", pdf_erfs, False),
    ("Effective Radiative Forcing|Aerosols|Direct Effect|OC", pdf_erfs, False),
    ("Effective Radiative Forcing|Aerosols|Direct Effect|SOx", pdf_erfs, False),
    ("Effective Radiative Forcing|Aerosols|Indirect Effect", pdf_erfs, False),
    ("Emissions|CO2|Energy and Industrial Processes", pdf_emissions, True),
    # ("Emissions|GHG", pdf_emissions, True),
    ("Emissions|CO2|AFOLU", pdf_emissions, True),
    ("Emissions|CH4", pdf_emissions, True),
    # ("Emissions|CFC12", pdf_emissions, True),
    ("Emissions|N2O", pdf_emissions, True),
    ("Emissions|Sulfur", pdf_emissions, True),
    ("Emissions|CO", pdf_emissions, True),
    ("Emissions|BC", pdf_emissions, True),
    ("Emissions|OC", pdf_emissions, True),
    ("Emissions|NOx", pdf_emissions, True),
    ("Emissions|NH3", pdf_emissions, True),
    ("Emissions|VOC", pdf_emissions, True),
    # ("Effective Radiative Forcing|CO2", pdf_erfs, False),
    # ("Effective Radiative Forcing|CH4", pdf_erfs, False),
]


ncols = 2
nrows = len(variables_src) // ncols + len(variables_src) % ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 5))
axes_flat = axes.flatten()
hue = ms_level
palette_h = palette

for i, (variable, src, emissions) in tqdm.auto.tqdm(enumerate(variables_src)):
    ax = axes_flat[i]

    if emissions:
        vdf = src.loc[pix.ismatch(variable=f"{variable}*")].openscm.to_long_data()

        sns.lineplot(
            ax=ax,
            data=vdf,
            x="time",
            y="value",
            hue=hue,
            palette=palette_h,
        )
        ax.set_title(vdf["variable"].unique()[0], fontdict=dict(fontsize="medium"))

    else:
        vdf = src.loc[pix.isin(variable=variable)]
        vdf.openscm.plot_plume_after_calculating_quantiles(
            quantile_over="run_id",
            hue_var=hue,
            style_var="climate_model",
            palette=palette_h,
            quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.3), ((0.05, 0.95), 0.0)),
            ax=ax,
        )
        ax.set_title(vdf.pix.unique("variable")[0], fontdict=dict(fontsize="medium"))

    if i % ncols:
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.legend().remove()
    ax.set_xlabel("Year")

    if "Emissions" in variable and ("CO2" in variable or "GHG" in variable):
        ax.axhline(0.0, linestyle="--", color="gray")
        if "Energy" in variable:
            ax.set_yticks(np.arange(-2e4, 6e4 + 1, 1e4))
        elif "GHG" in variable:
            ax.set_yticks(np.arange(-2e4, 9e4 + 1, 1e4))
        else:
            ax.set_yticks(np.arange(-7.5e3, 10e3 + 1, 2.5e3))

    elif "Effective Radiative Forcing" in variable:
        pass

    else:
        ax.set_ylim(ymin=0.0)

    ax.grid()
    # break


# %%
fig.savefig("Global_comparison.png", bbox_inches="tight")

# %%
