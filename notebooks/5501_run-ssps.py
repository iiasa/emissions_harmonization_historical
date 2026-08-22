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
# # Run SSPs
#
# Here we run the SSPs.
# These were harmonised in a separate process
# which is why we use the RCMIP data directly.

# %% [markdown]
# ## Imports

# %%
import multiprocessing
import os
import platform
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import pooch
import seaborn as sns
from attrs import evolve
from gcages.ar6 import (
    AR6PostProcessor,
    AR6SCMRunner,
    get_ar6_full_historical_emissions,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name

from emissions_harmonization_historical.constants_5000 import (
    AR6_LIKE_SCM_OUTPUT_DB,
    REPO_ROOT,
)
from emissions_harmonization_historical.scm_running import (
    COMPLETE_EMISSIONS_INPUT_VARIABLES_GCAGES,
    load_magicc_cfgs,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()
# Setup pint
pint.set_application_registry(openscm_units.unit_registry)
pix.set_openscm_registry_as_default()

# %% [markdown]
# ## Load data

# %%
rcmip_emms_raw = pooch.retrieve(
    "https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="2af9f90c42f9baa813199a902cdd83513fff157a0f96e1d1e6c48b58ffb8b0c1",
)
rcmip_emms_raw

# %%
rcmip_emissions = pandas_openscm.io.load_timeseries_csv(
    rcmip_emms_raw,
    index_columns=[
        "model",
        "scenario",
        "variable",
        "region",
        "unit",
        "mip_era",
        "activity_id",
    ],
    out_columns_name="year",
    out_columns_type=int,
)
rcmip_emissions

# %% [markdown]
# ### Scenarios

# %%
scenarios_to_analyse = [
    ("IMAGE", "ssp119"),
    ("IMAGE", "ssp126"),
    ("MESSAGE-GLOBIOM", "ssp245"),
    ("GCAM4", "ssp434"),
    ("GCAM4", "ssp460"),
    ("AIM/CGE", "ssp370"),
    ("REMIND-MAGPIE", "ssp534-over"),
    ("REMIND-MAGPIE", "ssp585"),
]
scenarios_to_analyse = pd.MultiIndex.from_tuples(scenarios_to_analyse, names=["model", "scenario"])

# %%
scenarios_to_run = rcmip_emissions.openscm.mi_loc(scenarios_to_analyse).loc[pix.isin(region="World"), 2015:]
for y in range(2015, scenarios_to_run.columns.max() + 1):
    if y not in scenarios_to_run:
        scenarios_to_run[y] = np.nan

scenarios_to_run = scenarios_to_run.T.interpolate(method="index").T
scenarios_to_run = scenarios_to_run.loc[
    pix.isin(
        variable=[
            convert_variable_name(
                v, from_convention=SupportedNamingConventions.GCAGES, to_convention=SupportedNamingConventions.RCMIP
            )
            for v in COMPLETE_EMISSIONS_INPUT_VARIABLES_GCAGES
        ]
    ),
    2015:,
].openscm.update_index_levels(
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.RCMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    }
)

scenarios_to_run

# %%
with openscm_units.unit_registry.context("NOx_conversions"):
    pint.set_application_registry(openscm_units.unit_registry)
    start = scenarios_to_run.pix.convert_unit({"Mt NOx/yr": "Mt NO2/yr"}).reset_index(
        ["mip_era", "activity_id"], drop=True
    )

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
    data=start.loc[pix.ismatch(variable=["**CO2|*", "**CH4", "**SOx"])]
    .melt(ignore_index=False, var_name="year")
    .reset_index(),
    hue="scenario",
)

# fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
# fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# Data is already harmonised and infilled, so we can skip those steps.

# %%
complete_scenarios = start
complete_scenarios

# %%
AR6_LIKE_SCM_OUTPUT_DB.save(
    complete_scenarios,
    allow_overwrite=True,
    # groupby=["model", "scenario", "variable"]
)

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
AR6_INFILLING_DB_FILE = Path("infilling_db_ar6.csv")
AR6_INFILLING_DB_CFCS_FILE = Path("infilling_db_ar6_cfcs.csv")

# %%
scm_runner_magiccv753 = AR6SCMRunner.from_ar6_config(
    # Generally, you want to run SCMs in parallel
    n_processes=multiprocessing.cpu_count(),
    magicc_exe_path=MAGICC_EXE,
    magicc_prob_distribution_path=MAGICC_AR6_PROBABILISTIC_CONFIG_FILE,
    historical_emissions=get_ar6_full_historical_emissions(AR6_INFILLING_DB_CFCS_FILE),
    harmonisation_year=2015,
    # Urgh Emissions|HFC245fa
    run_checks=False,
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
scm_results_magiccv753 = scm_runner_magiccv753(complete_scenarios, force_rerun=True)
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
    AR6_LIKE_SCM_OUTPUT_DB.save(
        v,
        allow_overwrite=True,
        # groupby=["model", "scenario", "variable"]
    )

# %%
for v in [post_processed_results, post_processed_results_magicc_v76]:
    AR6_LIKE_SCM_OUTPUT_DB.save(
        v.timeseries_run_id,
        allow_overwrite=True,
        # groupby=["model", "scenario", "variable"]
    )

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
