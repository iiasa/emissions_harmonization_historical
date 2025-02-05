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
import re

import matplotlib.pyplot as plt
import openscm_runner.adapters
import pandas as pd
import pandas_indexing as pix
import pyam
import seaborn as sns
from gcages.database import GCDB
from gcages.io import load_timeseries_csv
from gcages.pandas_helpers import multi_index_lookup
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.post_processing import AR7FTPostProcessor
from emissions_harmonization_historical.workflow import run_magicc_and_post_processing

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
base_model = "REMIND-MAgPIE 3.4-4.8"
base_scenario = "SSP1 - Very Low Emissions"

# %%
base = pd.MultiIndex.from_tuples(
    ((base_model, base_scenario),),
    name=["model", "scenario"],
)
base

# %%
others = pd.MultiIndex.from_tuples(
    (
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Overshoot"),
        ("IMAGE 3.4", "SSP1 - Low Emissions"),
    ),
    name=["model", "scenario"],
)
others

# %%
n_processes = multiprocessing.cpu_count()
run_checks = False  # TODO: turn on

# %%
INPUT_PATH = DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"

# %%
magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
magicc_expected_version = "v7.6.0a3"
magicc_prob_distribution_path = (
    DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

# %%
os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
if openscm_runner.adapters.MAGICC7.get_version() != magicc_expected_version:
    raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())

# %%
OUTPUT_PATH = (
    INPUT_PATH
    / f"magicc-{magicc_expected_version.replace('.', '-')}_{magicc_prob_distribution_path.stem}_scenario-delta-breakdown"  # noqa: E501
)
OUTPUT_PATH

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
normal_workflow_db = GCDB(INPUT_PATH / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0" / "db")
erfs = normal_workflow_db.load(
    pix.isin(variable=forcing_breakdown_to_plot)
    & pix.isin(model=[*base.to_frame()["model"], *others.to_frame()["model"]])
    & pix.isin(scenario=[*base.to_frame()["scenario"], *others.to_frame()["scenario"]]),
    progress=True,
    out_columns_type=int,
)
erfs = multi_index_lookup(erfs, others.join(base, how="outer"))
erfs

# %%
base_loc = pix.isin(scenario=base.to_frame()["scenario"], model=base.to_frame()["model"])
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
plt_years = [2030, 2040, 2050, 2060]
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
complete_scenarios = load_timeseries_csv(
    INPUT_PATH / "complete_scenarios.csv",
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
complete_scenarios

# %% [markdown]
# ## Calculate breakdown scenario runs

# %%
base_scen = multi_index_lookup(complete_scenarios, base)
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
    ("NH3 NOx CO VOCs", ["Emissions|NH3", "Emissions|NOx", "Emissions|CO", "Emissions|VOC"]),
    ("Montreal gases", [v for v in base_scen.pix.unique("variable") if "Montreal" in v]),
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
to_attribute

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
        start = multi_index_lookup(complete_scenarios, other_idx).loc[~variable_loc]
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
post_processor = AR7FTPostProcessor.from_default_config()

# %%
scm_results = run_magicc_and_post_processing(
    to_run,
    n_processes=n_processes,
    post_processor=post_processor,
    magicc_exe_path=magicc_exe_path,
    magicc_prob_distribution_path=magicc_prob_distribution_path,
    output_path=OUTPUT_PATH,
    data_root=DATA_ROOT,
    batch_size_scenarios=15,
)

# %%
# If you need to re-write.
# Delete OUTPUT_PATH / "db"
# scm_results_db.load_metadata()

# %%
gsat_out_runs_raw = scm_results.post_processed.timeseries

# %%
gsat_out_normal_workflow = normal_workflow_db.load(
    pix.isin(variable="Surface Air Temperature Change")
    & pix.isin(model=[*base.to_frame()["model"], *others.to_frame()["model"]])
    & pix.isin(scenario=[*base.to_frame()["scenario"], *others.to_frame()["scenario"]]),
    progress=True,
    out_columns_type=int,
)
gsat_out_normal_workflow = multi_index_lookup(gsat_out_normal_workflow, others.join(base, how="outer"))
gsat_out_normal_workflow = post_processor(gsat_out_normal_workflow).timeseries
# gsat_out_normal_workflow

# %%
deltas_total = multi_index_lookup(gsat_out_normal_workflow, others) - multi_index_lookup(
    gsat_out_normal_workflow, base
).reset_index(["model", "scenario"], drop=True)
deltas_total

# %%
gsat_out_runs = gsat_out_runs_raw.pix.format(component="{scenario}").pix.extract(
    model="{model} -- {scenario} --- {model_base} -- {scenario_base}"
)
deltas_components = -1 * (gsat_out_runs - multi_index_lookup(gsat_out_normal_workflow, others))
# deltas_components

# %%
deltas_components_total = deltas_components.groupby(deltas_components.index.names.difference(["component"])).sum()
# deltas_components_total

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
    plt.show()
