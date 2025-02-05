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
# # Run workflow - AR6-style
#
# Run the climate assessment workflow as it was run in AR6
# (except for pre-processing fixes which make no sense to leave out).

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing
import os
import platform
import random

import pandas as pd
import pandas_indexing as pix
import pyam
from gcages.ar6 import run_ar6_workflow
from gcages.database import GCDB
from loguru import logger
from nomenclature import DataStructureDefinition

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.post_processing import AR7FTPostProcessor
from emissions_harmonization_historical.pre_processing import AR7FTPreProcessor
from emissions_harmonization_historical.scm_running import SCM_OUTPUT_VARIABLES_DEFAULT

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## Set up

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
OUTPUT_PATH = DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_ar6-workflow"
OUTPUT_PATH_MAGICC = OUTPUT_PATH / "magicc-ar6"
OUTPUT_PATH_MAGICC

# %%
scm_output_variables = SCM_OUTPUT_VARIABLES_DEFAULT

# %%
batch_size_scenarios = 15
n_processes = multiprocessing.cpu_count()
# n_processes = 1

# %%
# Needed for 7.5.3 on a mac
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"

if platform.system() == "Darwin":
    if platform.processor() == "arm":
        magicc_exe = "magicc-darwin-arm64"
    else:
        raise NotImplementedError(platform.processor())
elif platform.system() == "Windows":
        magicc_exe = "magicc.exe"
else:
    raise NotImplementedError(platform.system())

magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "bin" / magicc_exe
magicc_prob_distribution_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json"
scm_results_db = GCDB(OUTPUT_PATH_MAGICC / "db")
scm_results_db

# %%
# # If you need to re-write.
# scm_results_db.delete()

# %% [markdown]
# ## Load scenario data

# %%
scenarios_raw_global = load_global_scenario_data(
    scenario_path=DATA_ROOT / "scenarios" / "data_raw",
    scenario_time_id=SCENARIO_TIME_ID,
    progress=True,
).loc[:, :2100]  # TODO: drop 2100 end once we have usable scenario data post-2100

# %% [markdown]
# ### Hacky pre-processing

# %% [markdown]
# ### Use common definitions to check aggregation issues

# %%
dsd = DataStructureDefinition(DATA_ROOT / ".." / ".." / "common-definitions" / "definitions")

# %%
dsd.variable["Emissions|CO2|Energy"].check_aggregate = True
dsd.variable["Emissions|CO2|Industrial Processes"].check_aggregate = True
dsd.variable["Emissions|CO2|Energy and Industrial Processes"].check_aggregate = True
dsd.variable["Emissions|CO2|AFOLU"].check_aggregate = True
dsd.variable["Emissions|CO2"].check_aggregate = True

# %%
pyam_df = pyam.IamDataFrame(scenarios_raw_global)
pyam_df

# %%
reporting_issues = (
    dsd.check_aggregate(pyam_df)
    .loc[~pix.isin(model=["MESSAGEix-GLOBIOM 2.1-M-R12"])]
    .pix.unique(["model", "scenario", "variable"])
    .to_frame(index=False)
)

# %%
reporting_issues.drop("scenario", axis="columns").drop_duplicates().sort_values("model")

# %%
for (model, variable), mdf in reporting_issues.groupby(["model", "variable"]):
    print(f"Reporting  issues for {model} {variable}")

    if variable == "Emissions|CO2|Energy and Industrial Processes":
        component_vars = dsd.variable[variable].components

    else:
        component_vars = sorted(
            scenarios_raw_global.loc[pix.ismatch(model=model, variable=f"{variable}|*")].pix.unique("variable")
        )

    print(f"{component_vars=}")

    exp_component_vars = dsd.variable[variable].components
    if exp_component_vars is None:
        print("No expected component variables")

    else:
        variables_not_in_exp_component_variables = set(component_vars).difference(set(exp_component_vars))
        print(f"{variables_not_in_exp_component_variables=}")

        components_handled_by_nomenclature = set(exp_component_vars).intersection(set(component_vars))
        print(f"{components_handled_by_nomenclature=}")

    differences_ts = (
        dsd.check_aggregate(pyam.IamDataFrame(pyam_df.filter(model=model)))
        .loc[pix.isin(variable=variable)]
        .melt(ignore_index=False)
        .set_index("variable", append=True)
        .unstack("year")
    )
    display(differences_ts)  # noqa:F821
    # print(differences_ts)

# %%
pre_processed = AR7FTPreProcessor.from_default_config()(scenarios_raw_global)
pre_processed

# %% [markdown]
# ### Down-select scenarios

# %%
# # Randomly select some scenarios
# # (this is how I generated the hard-coded values in the next cell).
# base = pre_pre_processed.pix.unique(["model", "scenario"]).to_frame(index=False)
# base["scenario_group"] = base["scenario"].apply(lambda x: x.split("-")[-1].split("_")[0].strip())

# selected_scenarios_l = []
# selected_models = []
# for scenario_group, sdf in base.groupby("scenario_group"):
#     options = sdf.index.values.tolist()
#     random.shuffle(options)

#     n_selected = 0
#     for option_loc in options:
#         selected_model = sdf.loc[option_loc, :].model
#         if selected_model not in selected_models:
#             selected_scenarios_l.append(sdf.loc[option_loc, :])
#             selected_models.append(selected_model)
#             n_selected += 1
#             if n_selected >= 2:
#                 break

#     else:
#         if n_selected >= 1:
#             selected_scenarios_l.append(sdf.loc[option_loc, :])
#             selected_models.append(selected_model)
#         else:
#             selected_scenarios_l.append(sdf.loc[option_loc, :])
#             selected_models.append(selected_model)

#             option_loc = options[-2]
#             selected_model = sdf.loc[option_loc, :].model
#             selected_scenarios_l.append(sdf.loc[option_loc, :])
#             selected_models.append(selected_model)

# selected_scenarios = pd.concat(selected_scenarios_l, axis="columns").T
# selected_scenarios_idx = selected_scenarios.set_index(["model", "scenario"]).index
# selected_scenarios

# %%
selected_scenarios_idx = pd.MultiIndex.from_tuples(
    (
        ("MESSAGEix-GLOBIOM 2.1-M-R12", "SSP5 - High Emissions"),
        ("IMAGE 3.4", "SSP5 - High Emissions"),
        ("AIM 3.0", "SSP2 - Medium-Low Emissions"),
        ("WITCH 6.0", "SSP2 - Low Emissions"),
        ("REMIND-MAgPIE 3.4-4.8", "SSP2 - Low Overshoot_b"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP5 - Low Overshoot"),
        ("COFFEE 1.5", "SSP2 - Medium Emissions"),
        ("GCAM 7.1 scenarioMIP", "SSP2 - Medium Emissions"),
        ("IMAGE 3.4", "SSP2 - Very Low Emissions"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions"),
    ),
    name=["model", "scenario"],
)
scenarios_run = pre_processed[pre_processed.index.isin(selected_scenarios_idx)]

# scenarios_run = pre_processed.loc[pix.ismatch(scenario=["*Very Low*", "*Overshoot*"], model=["GCAM*", "AIM*", "*"])]
# scenarios_run = pre_processed.loc[pix.ismatch(scenario=["*Very Low*", "*Overshoot*"], model=["GCAM*"])]

# %%
# To run all, just uncomment the line below
# scenarios_run = pre_pre_processed

# %%
scenarios_run.pix.unique(["model", "scenario"]).to_frame(index=False)

# %%
res = run_ar6_workflow(
    input_emissions=scenarios_run,
    magicc_exe_path=magicc_exe_path,
    magicc_prob_distribution_path=magicc_prob_distribution_path,
    batch_size_scenarios=batch_size_scenarios,
    scm_output_variables=scm_output_variables,
    scm_results_db=scm_results_db,
    n_processes=n_processes,
    run_checks=False,  # TODO: turn this back on
)

# %%
post_processor = AR7FTPostProcessor.from_default_config()
post_processor.gsat_variable_name = "AR6 climate diagnostics|Raw Surface Temperature (GSAT)"

# %%
post_processed_updated = post_processor(res.scm_results_raw)
# post_processed_updated.metadata

# %%
pd.testing.assert_series_equal(
    post_processed_updated.metadata["category"],
    res.post_processed_scenario_metadata["category"],
)

# %%
post_processed_updated.metadata.sort_values(["category", "Peak warming 33.0"])

# %%
post_processed_updated.metadata.groupby(["model"])["category"].value_counts().sort_index()

# %%
for full_path, df in (
    (OUTPUT_PATH_MAGICC / "metadata.csv", post_processed_updated.metadata),
    (OUTPUT_PATH / "pre-processed.csv", res.pre_processed_emissions),
    (OUTPUT_PATH / "harmonised.csv", res.harmonised_emissions),
    (OUTPUT_PATH / "infilled.csv", res.infilled_emissions),
    (OUTPUT_PATH / "complete_scenarios.csv", res.complete_scenarios),
    (OUTPUT_PATH_MAGICC / "scm-effective-emissions.csv", res.infilled_emissions),
    (OUTPUT_PATH_MAGICC / "timeseries-percentiles.csv", post_processed_updated.timeseries_percentiles),
    # Don't write this, already in the database
    # ("scm-results.csv", res.scm_results_raw),
    # Can write this, but not using yet so just leave out at the moment
    # because it's slow to write.
    # ("post-processed-timeseries.csv", post_processed_updated.timeseries),
):
    print(f"Writing {full_path}")
    full_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(full_path)
    print(f"Wrote {full_path}")
    print()
