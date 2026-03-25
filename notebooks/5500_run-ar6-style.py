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

import matplotlib.pyplot as plt
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import seaborn as sns
from attrs import evolve
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
    AR6_LIKE_SCM_OUTPUT_DB,
    RAW_SCENARIO_DB,
    REPO_ROOT,
)
from emissions_harmonization_historical.scm_running import (
    load_magicc_cfgs,
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

# %%
AR6_LIKE_SCM_OUTPUT_DB.save(complete_scenarios, allow_overwrite=True)

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
scm_runner_magiccv753 = AR6SCMRunner.from_ar6_config(
    # Generally, you want to run SCMs in parallel
    n_processes=multiprocessing.cpu_count(),
    magicc_exe_path=MAGICC_EXE,
    magicc_prob_distribution_path=MAGICC_AR6_PROBABILISTIC_CONFIG_FILE,
    historical_emissions=get_ar6_full_historical_emissions(AR6_INFILLING_DB_CFCS_FILE),
    harmonisation_year=harmoniser.harmonisation_year,
)
scm_runner_magiccv753.output_variables = tuple(
    [
        *scm_runner_magiccv753.output_variables,
        "Effective Radiative Forcing|Solar",
        "Effective Radiative Forcing|Volcanic",
    ]
)
scm_runner_magiccv753.output_variables

# %%
scm_results_magiccv753 = scm_runner_magiccv753(complete_scenarios)
scm_results_magiccv753

# %%
magicc_exe_dir = REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "bin"
magicc_prob_distribution_path = (
    REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

if platform.system() == "Darwin":
    if platform.processor() == "arm":
        magicc_exe_path = magicc_exe_dir / "magicc-darwin-arm64"
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib/gcc/current/"

elif platform.system() == "Linux":
    magicc_exe_path = magicc_exe_dir / "magicc"

elif platform.system() == "Windows":
    magicc_exe_path = magicc_exe_dir / "magicc.exe"

climate_models_cfgs_magiccv60 = load_magicc_cfgs(
    prob_distribution_path=magicc_prob_distribution_path,
    output_variables=scm_runner_magiccv753.output_variables,
    startyear=1750,
)

os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)

scm_runner_magiccv760 = evolve(
    scm_runner_magiccv753,
    climate_models_cfgs=climate_models_cfgs_magiccv60,
)

# %%
scm_results_magiccv760 = scm_runner_magiccv760(complete_scenarios)
scm_results_magiccv760

# %% [markdown]
# ## Post-process

# %%
post_processor = AR6PostProcessor.from_ar6_config(n_processes=None)
post_processed_results = post_processor(scm_results_magiccv753)
post_processed_results_magicc_v76 = post_processor(scm_results_magiccv760)

# %%
for v in [scm_results_magiccv753, scm_results_magiccv760]:
    AR6_LIKE_SCM_OUTPUT_DB.save(v, allow_overwrite=True)

# %%
for v in [post_processed_results, post_processed_results_magicc_v76]:
    AR6_LIKE_SCM_OUTPUT_DB.save(v.timeseries_run_id, allow_overwrite=True)

# %% [markdown]
# ## Compare MAGICC versions

# %%
pdf = pix.concat(
    [
        scm_results_magiccv753,
        scm_results_magiccv760,
    ]
).loc[pix.ismatch(variable="Effective Radiative Forcing|**")]

for variable, vdf in pdf.groupby("variable"):
    ax = vdf.loc[:, 2010:].openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id", hue_var="climate_model", style_var="scenario"
    )
    ax.grid()
    ax.set_title(variable)
    plt.show()

# %%
pdf = pix.concat(
    [
        post_processed_results.timeseries_run_id,
        post_processed_results_magicc_v76.timeseries_run_id,
    ]
)

pdf.loc[:, 2010:].openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id", hue_var="climate_model", style_var="scenario"
).grid()
