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

# %% [markdown]
# # Calculate breakdown of deltas between workflows

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing
import os
import platform
import re
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pyam
from gcages.ar6.post_processing import (
    get_temperatures_in_line_with_assessment,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import run_scms
from loguru import logger
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    AR6_LIKE_SCM_OUTPUT_DB,
    DATA_ROOT,
    REPO_ROOT,
    SCM_OUTPUT_DB,
)
from emissions_harmonization_historical.scm_running import load_magicc_cfgs

# %%
pandas_openscm.register_pandas_accessor()

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
# put scenario loading config stuff here so it's easier to see

# %%
model = "REMIND-MAgPIE 3.5-4.11"
scenario = "SSP1 - Very Low Emissions"
climate_model = "MAGICCv7.5.3"

# base_workflow_emissions_db = AR6_LIKE_EMISSIONS_DB
base_workflow_scm_output_db = AR6_LIKE_SCM_OUTPUT_DB
base_workflow_id = "ar6"

# other_workflow_emissions_db = SCM_OUTPUT_DB
other_workflow_scm_output_db = SCM_OUTPUT_DB
other_workflow_id = "scenariomip"

# %%
# SCM_OUTPUT_DB.load_metadata().get_level_values("climate_model").unique()

# %% [markdown]
# ## Have a look at ERF differences

# %%
forcing_breakdown_to_plot = [
    # "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Black Carbon on Snow",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|Ozone",
    # "Effective Radiative Forcing|Aerosols",
    # "Effective Radiative Forcing|Greenhouse Gases",
    # "Effective Radiative Forcing|Ozone",
]

# %%
erfs_base = base_workflow_scm_output_db.load(
    pix.isin(climate_model="MAGICCv7.5.3")
    & pix.isin(variable=forcing_breakdown_to_plot)
    & pix.isin(model=model)
    & pix.isin(scenario=scenario),
    progress=True,
    out_columns_type=int,
).pix.assign(source=base_workflow_id)
erfs_base

# %%
erfs_other = other_workflow_scm_output_db.load(
    pix.isin(climate_model="MAGICCv7.5.3")
    & pix.isin(variable=forcing_breakdown_to_plot)
    & pix.isin(model=model)
    & pix.isin(scenario=scenario),
    progress=True,
    out_columns_type=int,
).pix.assign(source=other_workflow_id)
erfs_other

# %%
erfs_deltas_median = (
    erfs_base.reset_index(["source"], drop=True).openscm.groupby_except("run_id").median()
    - erfs_other.reset_index(["source"], drop=True).openscm.groupby_except("run_id").median()
)
# erfs_deltas_median.sort_index().loc[:, 2005:2020].sort_values(by=2005)#.sum(axis=0)

# %%
plot_years = range(2010, 2100 + 1)
for (model, scenario), msdf in erfs_deltas_median.loc[:, plot_years].groupby(["model", "scenario"]):
    ax = pyam.IamDataFrame(msdf).plot.stack(
        stack="variable",
        title=None,
        total=True,
        # ax=ax,
        # legend=legend,
        cmap="tab20",
    )
    # ax.set_title(f"{model} {scenario}\nrel. to\n{base_model} {base_scenario}")
    ax.axhline(0.0, color="k")
    ax.set_title(f"{base_workflow_id} - {other_workflow_id}")
    fig = ax.get_figure()
    # fig.savefig(f"{model}vs{base_model}_erf_diff.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# %%
# plt_years = [2030, 2040, 2050, 2060, 2080, 2100]
# pdf = erfs_deltas[plt_years].melt(ignore_index=False, var_name="year").reset_index()
# pdf["variable"] = pdf["variable"].str.replace("Effective Radiative Forcing|", "")

# for (model, scenario), msdf in pdf.groupby(["model", "scenario"]):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     sns.boxplot(
#         data=msdf,
#         y="value",
#         x="variable",
#         hue="year",
#         # hue="variable",
#         # x="year",
#     )
#     ax.set_xticks(ax.get_xticks())
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#     ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
#     # ax.set_title(f"{model} {scenario}\nrel. to\n{base_model} {base_scenario}")
#     ax.axhline(0.0, color="tab:gray", zorder=1.2)
#     fig = ax.get_figure()
#     # fig.savefig(f"{model}vs{base_model}_erf_years_diff.pdf", format="pdf")
#     plt.show()


# %% [markdown]
# ## Load complete scenario data

# %%
complete_scenarios_base = base_workflow_scm_output_db.load(
    pix.ismatch(variable="Emissions**") & pix.isin(model=model) & pix.isin(scenario=scenario),
    progress=True,
    out_columns_type=int,
)
# Hmm don't love this, but ok for now
complete_scenarios_base = complete_scenarios_base.openscm.update_index_levels(
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    }
)
# complete_scenarios_base

# %%
complete_scenarios_other = other_workflow_scm_output_db.load(
    pix.ismatch(variable="Emissions**") & pix.isin(model=model) & pix.isin(scenario=scenario),
    progress=True,
    out_columns_type=int,
)
# Hmm don't love this, but ok for now
complete_scenarios_other = (
    complete_scenarios_other.loc[pix.isin(climate_model="MAGICCv7.5.3"), complete_scenarios_base.columns]
    .dropna(how="all", axis="columns")
    .reset_index("climate_model", drop=True)
)
# complete_scenarios_other

# %% [markdown]
# ## Calculate breakdown scenario runs

# %%
to_attribute = [
    # (label, variables to include)
    ("CO2 Fossil", ["Emissions|CO2|Energy and Industrial Processes"]),
    ("CO2 AFOLU", ["Emissions|CO2|AFOLU"]),
    ("CH4", ["Emissions|CH4"]),
    ("N2O", ["Emissions|N2O"]),
    ("BC", ["Emissions|BC"]),
    ("OC", ["Emissions|OC"]),
    ("Sulfur", ["Emissions|Sulfur"]),
    ("NH3", ["Emissions|NH3"]),
    ("NOx", ["Emissions|NOx"]),
    ("CO", ["Emissions|CO"]),
    ("NMVOCs", ["Emissions|VOC"]),
    (
        "Montreal gases",
        [
            *[v for v in complete_scenarios_base.pix.unique("variable") if "Montreal" in v],
            *[
                "Emissions|CH3CCl3",
                "Emissions|CH3Br",
                "Emissions|CH3Cl",
                "Emissions|CCl4",
                "Emissions|CHCl3",
                "Emissions|Halon2402",
                "Emissions|CH2Cl2",
                "Emissions|Halon1202",
                "Emissions|Halon1301",
                "Emissions|HCFC141b",
                "Emissions|HCFC142b",
                "Emissions|Halon1211",
                "Emissions|HCFC22",
            ],
        ],
    ),
    (
        "HFCs PFCs SF6 NF3 SO2F2",
        [
            *[v for v in complete_scenarios_base.pix.unique("variable") if "HFC" in v],
            "Emissions|CF4",
            *[v for v in complete_scenarios_base.pix.unique("variable") if re.match(r"Emissions\|c?C\d*F\d*", v)],
            "Emissions|SF6",
            "Emissions|NF3",
            "Emissions|SO2F2",
        ],
    ),
]
# to_attribute

# %%
not_attributed = set(complete_scenarios_base.pix.unique("variable"))
for v in to_attribute:
    missing = set(v[1]).difference(not_attributed)
    if missing:
        raise AssertionError(missing)

    not_attributed = not_attributed - set(v[1])

if not_attributed:
    raise AssertionError(not_attributed)

# %%
model_tmp = f"{model} {scenario} {base_workflow_id} replaced by {other_workflow_id}".replace(".", "_").replace(" ", "_")

to_run_l = [
    complete_scenarios_base.pix.assign(model=model_tmp, scenario=base_workflow_id),
    complete_scenarios_other.pix.assign(model=model_tmp, scenario=other_workflow_id),
]

for label, emms in to_attribute:
    variable_loc = pix.isin(variable=emms)
    start = complete_scenarios_base.loc[~variable_loc]
    replace = complete_scenarios_other.loc[variable_loc]
    to_run_tmp = pix.concat([start, replace]).pix.assign(
        model=model_tmp,
        scenario=label,
    )
    exp_n_ts = 52
    if to_run_tmp.shape[0] != exp_n_ts:
        raise AssertionError

    to_run_l.append(to_run_tmp)

to_run = pix.concat(to_run_l)
to_run

# %%
to_run.pix.unique(["model", "scenario"]).to_frame(index=False)

# %% [markdown]
# ## Run SCMs

# %%
db_dir = DATA_ROOT / "processed" / "delta-workflow-breakdown" / "zn-002"
db_dir.mkdir(exist_ok=True, parents=True)

db = OpenSCMDB(
    db_dir=db_dir,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)
# # If you need to start again
# db.delete()

# %%
output_variables = (
    # GSAT
    "Surface Air Temperature Change",
)

# %%
if platform.system() == "Darwin":
    if platform.processor() == "arm":
        magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
    else:
        raise NotImplementedError(platform.processor())
elif platform.system() == "Windows":
    raise NotImplementedError(platform.system())
elif platform.system().lower().startswith("linux"):
    magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc"
else:
    raise NotImplementedError(platform.system())

magicc_expected_version = "v7.6.0a3"
magicc_prob_distribution_path = (
    REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

MAGICC_EXE_PATH = REPO_ROOT / "magicc" / "magicc-v7.5.3" / "bin"
if platform.system() == "Darwin":
    if platform.processor() == "arm":
        magicc_exe_path = MAGICC_EXE_PATH / "magicc-darwin-arm64"
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib/gcc/current/"

elif platform.system() == "Linux":
    magicc_exe_path = MAGICC_EXE_PATH / "magicc"

elif platform.system() == "Windows":
    magicc_exe_path = MAGICC_EXE_PATH / "magicc.exe"

magicc_expected_version = "v7.5.3"
magicc_prob_distribution_path = (
    REPO_ROOT / "magicc" / "magicc-v7.5.3" / "configs" / "0fd0f62-derived-metrics-id-f023edb-drawnset.json"
)

os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)

climate_models_cfgs = load_magicc_cfgs(
    prob_distribution_path=magicc_prob_distribution_path,
    output_variables=output_variables,
    startyear=1750,
)
# climate_models_cfgs["MAGICC7"] = climate_models_cfgs["MAGICC7"][:5]

# %%
to_run_openscm_runner = update_index_levels_func(
    to_run,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
        )
    },
)
to_run_openscm_runner

# %%
# If you need a clean start
db.delete()
run_scms(
    scenarios=to_run_openscm_runner,
    climate_models_cfgs=climate_models_cfgs,
    output_variables=output_variables,
    scenario_group_levels=["model", "scenario"],
    n_processes=multiprocessing.cpu_count(),
    db=db,
    verbose=True,
    progress=True,
    batch_size_scenarios=15,
    force_rerun=False,
)

# %%
gsat_out_runs_raw = db.load(
    pix.isin(variable="Surface Air Temperature Change") & pix.isin(model=to_run_openscm_runner.pix.unique("model"))
    # & pix.isin(scenario=[*base.get_level_values("scenario"), *others.get_level_values("scenario")]),
)
# gsat_out_runs_raw


# %%
assessed_gsat_variable = "Surface Temperature (GSAT)"
gsat_assessment_median = 0.85
gsat_assessment_time_period = range(1995, 2014 + 1)
gsat_assessment_pre_industrial_period = range(1850, 1900 + 1)

get_assessed_gsat = partial(
    get_temperatures_in_line_with_assessment,
    assessment_median=gsat_assessment_median,
    assessment_time_period=gsat_assessment_time_period,
    assessment_pre_industrial_period=gsat_assessment_pre_industrial_period,
    group_cols=["climate_model", "model", "scenario"],
)

# %%
gsat_out_runs = update_index_levels_func(
    get_assessed_gsat(gsat_out_runs_raw),
    {"variable": lambda x: assessed_gsat_variable},
)
# gsat_out_runs

# %%
deltas_total = (
    gsat_out_runs.loc[pix.isin(scenario=other_workflow_id)]
    - gsat_out_runs.loc[pix.isin(scenario=base_workflow_id)].reset_index(["model", "scenario"], drop=True)
).pix.assign(model=model, scenario=scenario)
deltas_total

# %%
gsat_out_runs_decomp = gsat_out_runs.pix.format(component="{scenario}").pix.assign(
    model=model,
    scenario=scenario,
)
deltas_components = gsat_out_runs_decomp.loc[
    ~pix.isin(component=[base_workflow_id, other_workflow_id])
] - gsat_out_runs.loc[pix.isin(scenario=base_workflow_id)].reset_index(["model", "scenario"], drop=True)
deltas_components.head(10)

# %%
deltas_components_total = deltas_components.groupby(deltas_components.index.names.difference(["component"])).sum()
deltas_components_total

# %%
deltas_residual = (deltas_total - deltas_components_total).pix.assign(component="residual")
deltas_residual

# %%
deltas_all_components = pix.concat([deltas_residual, deltas_components])
# Sanity check
pd.testing.assert_frame_equal(
    deltas_total,
    deltas_all_components.groupby(deltas_components.index.names.difference(["component"])).sum(),
    check_like=True,
)
# deltas_all_components

# %%
deltas_all_components_median = deltas_all_components.groupby(
    deltas_all_components.index.names.difference(["run_id"])
).median()
deltas_all_components_median  # .max(axis=1)

# %%
plot_years = range(2000, 2100 + 1)
for (model, scenario), msdf in deltas_all_components_median.groupby(["model", "scenario"]):
    ax = pyam.IamDataFrame(msdf.loc[:, plot_years]).plot.stack(
        stack="component",
        title=None,
        total=True,
        # ax=ax,
        # legend=legend,
        cmap="tab20",
    )
    ax.set_title(f"{other_workflow_id} rel. to {base_workflow_id}")
    ax.axhline(0.0, color="k")
    ax.set_yticks(np.arange(-0.2, 0.2, 0.1))
    ax.grid()
    fig = ax.get_figure()
    # fig.savefig(f"{model}vs{base_model}_erf_deltas_components.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# %%
deltas_all_components_median.sort_values(by=2030).loc[:, 2030]

# %%
plot_years = range(2000, 2100 + 1)
for (model, scenario), msdf in deltas_all_components_median.loc[
    pix.isin(component=["BC", "OC", "CH4", "CO", "NMVOCs"])
].groupby(["model", "scenario"]):
    ax = pyam.IamDataFrame(msdf.loc[:, plot_years]).plot.stack(
        stack="component",
        title=None,
        total=True,
        # ax=ax,
        # legend=legend,
        cmap="tab20",
    )
    ax.set_title(f"{other_workflow_id} rel. to {base_workflow_id}")
    ax.axhline(0.0, color="k")
    ax.set_yticks(np.arange(-0.2, 0.2, 0.1))
    ax.grid()
    fig = ax.get_figure()
    # fig.savefig(f"{model}vs{base_model}_erf_deltas_components.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# %%
plot_years = range(2000, 2100 + 1)
for (model, scenario), msdf in erfs_deltas_median.groupby(["model", "scenario"]):
    ax = pyam.IamDataFrame(msdf.loc[:, plot_years]).plot.stack(
        stack="variable",
        title=None,
        total=True,
        # ax=ax,
        # legend=legend,
        cmap="tab20",
    )
    ax.set_title(f"{other_workflow_id} rel. to {base_workflow_id}")
    ax.axhline(0.0, color="k")
    fig = ax.get_figure()
    # fig.savefig(f"{model}vs{base_model}_erf_deltas_median.pdf", format="pdf", bbox_inches="tight")
    plt.show()
