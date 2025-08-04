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
# # Calculate breakdown of deltas between scenarios

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
import seaborn as sns
from gcages.ar6.post_processing import (
    get_temperatures_in_line_with_assessment,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import run_scms
from loguru import logger
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import DATA_ROOT, REPO_ROOT, SCM_OUTPUT_DB
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
base_model = "REMIND-MAgPIE 3.5-4.11"
base_scenario = "SSP1 - Very Low Emissions"
# base_model = "GCAM 7.1 scenarioMIP"
# base_scenario = "SSP3 - High Emissions"

# %%
base = pd.MultiIndex.from_tuples(
    ((base_model, base_scenario),),
    name=["model", "scenario"],
)
base

# %%
others = pd.MultiIndex.from_tuples(
    (
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Emissions"),
        ("AIM 3.0", "SSP2 - Low Overshoot"),
        # ("WITCH 6.0", "SSP5 - Medium-Low Emissions_a"),
    ),
    name=["model", "scenario"],
)
others

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
]

# %%
erfs = SCM_OUTPUT_DB.load(
    pix.isin(climate_model="MAGICCv7.6.0a3")
    & pix.isin(variable=forcing_breakdown_to_plot)
    & pix.isin(model=[*base.get_level_values("model"), *others.get_level_values("model")])
    & pix.isin(scenario=[*base.get_level_values("scenario"), *others.get_level_values("scenario")]),
    progress=True,
    out_columns_type=int,
)
erfs = erfs.openscm.mi_loc(others.join(base, how="outer"))
erfs

# %%
base_loc = pix.isin(scenario=base.get_level_values("scenario"), model=base.get_level_values("model"))
erfs_deltas = erfs[~base_loc] - erfs[base_loc].reset_index(["model", "scenario"], drop=True)
erfs_deltas_median = erfs_deltas.groupby(erfs_deltas.index.names.difference(["run_id"])).median()

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
    ax.set_title(f"{model} {scenario}\nrel. to\n{base_model} {base_scenario}")
    ax.axhline(0.0, color="k")
    plt.show()

# %%
plt_years = [2030, 2040, 2050, 2060, 2080, 2100]
pdf = erfs_deltas[plt_years].melt(ignore_index=False, var_name="year").reset_index()
pdf["variable"] = pdf["variable"].str.replace("Effective Radiative Forcing|", "")

for (model, scenario), msdf in pdf.groupby(["model", "scenario"]):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=msdf,
        y="value",
        x="variable",
        hue="year",
        # hue="variable",
        # x="year",
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.set_title(f"{model} {scenario}\nrel. to\n{base_model} {base_scenario}")
    ax.axhline(0.0, color="tab:gray", zorder=1.2)
    plt.show()

# %% [markdown]
# ## Load complete scenario data

# %%
SCM_OUTPUT_DB.load_metadata().get_level_values("variable")

# %%
complete_scenarios = SCM_OUTPUT_DB.load(
    pix.isin(climate_model="MAGICCv7.6.0a3")
    & pix.ismatch(variable="Emissions**")
    & pix.isin(model=[*base.get_level_values("model"), *others.get_level_values("model")])
    & pix.isin(scenario=[*base.get_level_values("scenario"), *others.get_level_values("scenario")]),
    progress=True,
    out_columns_type=int,
)
complete_scenarios = complete_scenarios.openscm.mi_loc(others.join(base, how="outer"))
complete_scenarios

# %% [markdown]
# ## Calculate breakdown scenario runs

# %%
base_scen = complete_scenarios.openscm.mi_loc(base)
base_model = base.pix.unique("model")[0]
base_scenario = base.pix.unique("scenario")[0]

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
            *[v for v in base_scen.pix.unique("variable") if "Montreal" in v],
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
            *[v for v in base_scen.pix.unique("variable") if "HFC" in v],
            "Emissions|CF4",
            *[v for v in base_scen.pix.unique("variable") if re.match(r"Emissions\|c?C\d*F\d*", v)],
            "Emissions|SF6",
            "Emissions|NF3",
            "Emissions|SO2F2",
        ],
    ),
]
# to_attribute

# %%
not_attributed = set(base_scen.pix.unique("variable"))
for v in to_attribute:
    missing = set(v[1]).difference(not_attributed)
    if missing:
        raise AssertionError(missing)

    not_attributed = not_attributed - set(v[1])

if not_attributed:
    raise AssertionError(not_attributed)

# %%
to_run_l = []
for i in range(others.size):
    other_idx = others[[i]]
    for label, emms in to_attribute:
        model = other_idx[0][other_idx.names.index("model")]
        scenario = other_idx[0][other_idx.names.index("scenario")]

        other_idx = others[[i]]
        variable_loc = pix.isin(variable=emms)
        start = complete_scenarios.openscm.mi_loc(other_idx).loc[~variable_loc]
        replace = base_scen.loc[variable_loc]
        to_run_tmp = pix.concat([start, replace]).pix.assign(
            model=f"{model} -- {scenario} --- {base_model} -- {base_scenario}",
            # model=f"{model} {scenario} - {base_model} {base_scenario}".replace(".", "_").replace(" ", "_"),
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
db_dir = DATA_ROOT / "processed" / "delta-breakdown" / "zn-001"
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

os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)

climate_models_cfgs = load_magicc_cfgs(
    prob_distribution_path=magicc_prob_distribution_path,
    output_variables=output_variables,
    startyear=1750,
)

# %%
to_run_openscm_runner = update_index_levels_func(
    to_run.reset_index("climate_model", drop=True),
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
gsat_out_runs_raw


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
# temperatures_in_line_with_assessment

# %%
gsat_out_normal_workflow_raw = SCM_OUTPUT_DB.load(
    pix.isin(climate_model="MAGICCv7.6.0a3")
    & pix.isin(variable="Surface Air Temperature Change")
    & pix.isin(model=[*base.get_level_values("model"), *others.get_level_values("model")])
    & pix.isin(scenario=[*base.get_level_values("scenario"), *others.get_level_values("scenario")]),
    progress=True,
    out_columns_type=int,
)
gsat_out_normal_workflow_raw = gsat_out_normal_workflow_raw.openscm.mi_loc(others.join(base, how="outer"))
# gsat_out_normal_workflow_raw

# %%
gsat_out_normal_workflow = update_index_levels_func(
    get_assessed_gsat(gsat_out_normal_workflow_raw),
    {"variable": lambda x: assessed_gsat_variable},
)
# temperatures_in_line_with_assessment

# %%
deltas_total = gsat_out_normal_workflow.openscm.mi_loc(others) - gsat_out_normal_workflow.openscm.mi_loc(
    base
).reset_index(["model", "scenario"], drop=True)
deltas_total.head(1)

# %%
gsat_out_runs = gsat_out_runs.pix.format(component="{scenario}").pix.extract(
    model="{model} -- {scenario} --- {model_base} -- {scenario_base}"
)
deltas_components = -1 * (gsat_out_runs - gsat_out_normal_workflow.openscm.mi_loc(others))
deltas_components.head(1)

# %%
deltas_components_total = deltas_components.groupby(deltas_components.index.names.difference(["component"])).sum()
deltas_components_total

# %%
deltas_residual = (deltas_total - deltas_components_total).pix.assign(component="residual")
# deltas_residual

# %%
deltas_all_components = pix.concat([deltas_residual, deltas_components])
# Sanity check
pd.testing.assert_frame_equal(
    deltas_total,
    deltas_all_components.groupby(deltas_components.index.names.difference(["component"]))
    .sum()
    .reset_index(["model_base", "scenario_base"], drop=True),
    check_like=True,
)
# deltas_all_components

# %%
deltas_all_components_median = deltas_all_components.groupby(
    deltas_all_components.index.names.difference(["run_id"])
).median()
# deltas_all_components_median

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
    ax.set_title(f"{model} {scenario}\nrel. to\n{base_model} {base_scenario}")
    ax.axhline(0.0, color="k")
    ax.set_yticks(np.arange(-0.4, 0.6, 0.1))
    ax.grid()
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
    ax.set_title(f"{model} {scenario}\nrel. to\n{base_model} {base_scenario}")
    ax.axhline(0.0, color="k")
    plt.show()

# %%
