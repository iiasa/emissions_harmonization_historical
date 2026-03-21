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
# # Run AR6-style
#
# Here we run the scenarios AR6-style
# and do some comparisons.

# %% [markdown]
# ## Imports

# %%
import multiprocessing
import os
import platform
from functools import partial
from pathlib import Path

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import seaborn as sns
from gcages.ar6 import (
    AR6Harmoniser,
    AR6Infiller,
    AR6PostProcessor,
    AR6PreProcessor,
    AR6SCMRunner,
    get_ar6_full_historical_emissions,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    POST_PROCESSED_TIMESERIES_DB,
    RAW_SCENARIO_DB,
    REPO_ROOT,
    SCM_OUTPUT_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()
# Setup pint
pint.set_application_registry(openscm_units.unit_registry)

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
scenarios_to_analyse = [
    ("REMIND-MAgPIE 3.5-4.11", "SSP1 - Very Low Emissions"),
]
scenarios_to_load = pd.MultiIndex.from_tuples(scenarios_to_analyse, names=["model", "scenario"])

# %%
model_raw = RAW_SCENARIO_DB.load(scenarios_to_load, progress=True)
if model_raw.empty:
    raise AssertionError

# sorted(model_raw.pix.unique("variable"))

# %%
model_raw.loc[pix.ismatch(variable="**CO2|AFOLU", region="World")].sort_index(axis="columns")

# %%
start = model_raw.loc[pix.isin(region="World")]

# %%
relplot_in_emms = partial(
    sns.relplot,
    kind="line",
    linewidth=2.0,
    alpha=0.7,
    facet_kws=dict(sharey=False),
    x="year",
    y="value",
    col="variable",
    col_wrap=2,
)
fg = relplot_in_emms(
    data=start.loc[pix.ismatch(variable=["**CO2|*", "**CH4", "**Sulfur"])]
    .melt(ignore_index=False, var_name="year")
    .reset_index(),
    hue="scenario",
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Pre-process

# %%
pre_processor = AR6PreProcessor.from_ar6_config()

# %%
pre_processed = pre_processor(start)
pre_processed

# %%
pdf = (
    pix.concat(
        [
            start.loc[pix.isin(variable=pre_processed.pix.unique("variable"))].pix.assign(stage="input"),
            start.loc[pix.isin(variable=["Emissions|CO2|AFOLU", "Emissions|CO2|Energy and Industrial Processes"])]
            .openscm.update_index_levels(
                {
                    "variable": partial(
                        convert_variable_name,
                        from_convention=SupportedNamingConventions.IAMC,
                        to_convention=SupportedNamingConventions.GCAGES,
                    )
                }
            )
            .pix.assign(stage="input"),
            pre_processed.pix.assign(stage="pre_processed"),
        ]
    )
    .melt(ignore_index=False, var_name="year")
    .reset_index()
)

fg = relplot_in_emms(
    data=pdf,
    hue="scenario",
    style="stage",
    dashes={
        "input": (1, 1),
        "pre_processed": "",
    },
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Harmonise

# %%
AR6_HISTORICAL_EMISSIONS_FILE = Path("history_ar6.csv")

# %%
harmoniser = AR6Harmoniser.from_ar6_config(
    ar6_historical_emissions_file=AR6_HISTORICAL_EMISSIONS_FILE,
)

# %%
# Strange that this is needed...
tmp = pre_processed.copy()
tmp.columns = tmp.columns.astype("O")
harmonised = harmoniser(tmp)
harmonised

# %%
pdf = (
    pix.concat(
        [
            pre_processed.pix.assign(stage="pre_processed"),
            harmonised.pix.assign(stage="harmonised"),
            harmoniser.historical_emissions.pix.assign(stage="history", scenario="history", model="history").loc[
                pix.isin(variable=pre_processed.pix.unique("variable"))
            ],
        ]
    )
    .melt(ignore_index=False, var_name="year")
    .reset_index()
)

fg = relplot_in_emms(
    data=pdf,
    hue="scenario",
    style="stage",
    dashes={
        "history": (1, 1),
        "pre_processed": (3, 3),
        "harmonised": "",
    },
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Infill

# %%
AR6_INFILLING_DB_FILE = Path("infilling_db_ar6.csv")
AR6_INFILLING_DB_CFCS_FILE = Path("infilling_db_ar6_cfcs.csv")

# %%
infiller = AR6Infiller.from_ar6_config(
    ar6_infilling_db_file=AR6_INFILLING_DB_FILE,
    ar6_infilling_db_cfcs_file=AR6_INFILLING_DB_CFCS_FILE,
    n_processes=None,  # run serially for this demo
    # To make sure that our outputs remain harmonised
    # (also, turns out that the historical emissions
    # are the same as the CFCs database)
    historical_emissions=get_ar6_full_historical_emissions(AR6_INFILLING_DB_CFCS_FILE),
    harmonisation_year=harmoniser.harmonisation_year,
)

# %%
# How strange
harmonised.columns = harmonised.columns.astype(int)
infilled = infiller(harmonised)
infilled

# %%
complete_scenarios = pd.concat([harmonised, infilled])
complete_scenarios

# %% [markdown]
# ## Run MAGICC

# %%
MAGICC_EXE_PATH = REPO_ROOT / "magicc" / "magicc-v7.5.3" / "bin"
MAGICC_AR6_PROBABILISTIC_CONFIG_FILE = (
    REPO_ROOT / "magicc" / "magicc-v7.5.3" / "configs" / "0fd0f62-derived-metrics-id-f023edb-drawnset.json"
)

if platform.system() == "Darwin":
    if platform.processor() == "arm":
        MAGICC_EXE = MAGICC_EXE_PATH / "magicc-darwin-arm64"
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib/gcc/current/"

elif platform.system() == "Linux":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc"

elif platform.system() == "Windows":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc.exe"

# %%
scm_runner = AR6SCMRunner.from_ar6_config(
    # Generally, you want to run SCMs in parallel
    n_processes=multiprocessing.cpu_count(),
    magicc_exe_path=MAGICC_EXE,
    magicc_prob_distribution_path=MAGICC_AR6_PROBABILISTIC_CONFIG_FILE,
    historical_emissions=get_ar6_full_historical_emissions(AR6_INFILLING_DB_CFCS_FILE),
    harmonisation_year=2015,
    # output_variables=(
    #     "Surface Air Temperature Change",
    #     "Effective Radiative Forcing",
    # "Effective Radiative Forcing|Aerosols",
    # "Effective Radiative Forcing|Greenhouse Gases",
    # "Effective Radiative Forcing|CO2",
    # "Effective Radiative Forcing|CH4",
    # "Effective Radiative Forcing|F-Gases",
    # "Effective Radiative Forcing|Solar",
    # "Effective Radiative Forcing|Volcanic",
    #                  ),
)

# %%
scm_runner.output_variables = tuple(
    [*scm_runner.output_variables, "Effective Radiative Forcing|Solar", "Effective Radiative Forcing|Volcanic"]
)
# scm_runner.output_variables

# %%
scm_results = scm_runner(complete_scenarios)
scm_results

# %%
scm_results.loc[pix.ismatch(variable="Effective Radiative Forcing**")].to_feather("erfs-ar6.feather")

# %% [markdown]
# ## Post-process

# %%
post_processor = AR6PostProcessor.from_ar6_config(n_processes=None)
post_processed_results = post_processor(scm_results)

# %% [markdown]
# ## Compare to ScenarioMIP results

# %%
scenario_locator = None
for model, scenario in scenarios_to_analyse:
    if scenario_locator is None:
        scenario_locator = pix.ismatch(model=model, scenario=scenario)
    else:
        scenario_locator = scenario_locator | pix.ismatch(model=model, scenario=scenario)

# %%
erfs_scenariomip = SCM_OUTPUT_DB.load(
    pix.ismatch(
        variable=[
            "Effective Radiative Forcing**",
        ]
    )
    & scenario_locator,
    progress=True,
    max_workers=multiprocessing.cpu_count(),
)
# erfs

# %%
history_scenariomip = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

# %%
emissions_scenariomip = POST_PROCESSED_TIMESERIES_DB.load(
    scenario_locator,
    progress=True,
    max_workers=multiprocessing.cpu_count(),
)
# emissions


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


# %%
ms_separator = " || "
ms_level = "model || scenario"

# %%
emissions_to_plot = [
    "Emissions|CO2|Energy and Industrial Processes",
    # "Emissions|GHG AR6GWP100",
    "Emissions|CO2|AFOLU",
    # "Cumulative Emissions|CO2",
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
    # "Emissions|HFC|HFC125",
    # "Emissions|HFC|HFC134a",
    # "Emissions|HFC|HFC143a",
    # "Emissions|HFC|HFC227ea",
    # "Emissions|HFC|HFC23",
    # "Emissions|HFC|HFC245fa",
    # "Emissions|HFC|HFC32",
    # "Emissions|HFC|HFC43-10",
    # "Emissions|HFC|HFC236fa",
    # "Emissions|HFC|HFC152a",
    # "Emissions|HFC|HFC365mfc",
]
pdf_emissions = (
    add_model_scenario_column(
        pix.concat(
            [
                emissions_scenariomip.loc[pix.isin(variable=emissions_to_plot, stage="complete")]
                .reset_index("stage", drop=True)
                .pix.assign(source="scenariomip"),
                complete_scenarios.openscm.update_index_levels(
                    {
                        "variable": partial(
                            convert_variable_name,
                            from_convention=SupportedNamingConventions.GCAGES,
                            to_convention=SupportedNamingConventions.IAMC,
                        )
                    }
                ).pix.assign(source="ar6"),
                history_scenariomip.pix.assign(source="scenariomip-history"),
            ]
        ),
        ms_separator=ms_separator,
        ms_level=ms_level,
    )
    .sort_index(axis="columns")
    .loc[:, 2014:2100]
)
# pdf_emissions

# %%
import matplotlib.pyplot as plt

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
        hue=ms_level,
        style="source",
        # palette=palette,
    )
    ax.set_title(variable_to_plot, fontdict=dict(fontsize="medium"))

    if i % 2:
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.legend().remove()

    if "CO2" not in variable_to_plot:
        ax.set_ylim(ymin=0)

    ax.grid()

# %%
erfs_to_plot = [
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|Solar",
    "Effective Radiative Forcing|Volcanic",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|F-Gases",
]

pdf_erfs = add_model_scenario_column(
    pix.concat(
        [
            erfs_scenariomip.loc[pix.isin(variable=erfs_to_plot, climate_model="MAGICCv7.6.0a3")].pix.assign(
                source="scenariomip-magicc-v760"
            ),
            scm_results.loc[pix.isin(variable=erfs_to_plot)].pix.assign(source="ar6"),
        ]
    ),
    ms_separator=ms_separator,
    ms_level=ms_level,
)

pdf_erfs = (pdf_erfs.loc[pix.isin(variable=erfs_to_plot)]).loc[:, 2000:2100]


# %%
def create_legend(ax, handles) -> None:
    """Create legend helper"""
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5))


hue = ms_level

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
        style_var="source",
        # palette=palette,
        quantiles_plumes=(
            (0.5, 1.0),
            # ((0.33, 0.67), 0.75),
        ),
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
# fig.savefig("erfs.pdf", format="pdf", bbox_inches="tight")

# %%
pdf_diff = pdf_erfs.openscm.groupby_except("run_id").median().loc[
    pix.isin(source="scenariomip-magicc-v760")
].reset_index(["source", "climate_model"], drop=True) - (
    pdf_erfs.openscm.groupby_except("run_id")
    .median()
    .loc[pix.isin(source="ar6")]
    .reset_index(["source", "climate_model"], drop=True)
)
pdf_diff

# %%
ax = sns.lineplot(
    data=pdf_diff.openscm.to_long_data(),
    x="time",
    y="value",
    style=ms_level,
    hue="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
