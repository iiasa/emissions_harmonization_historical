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
# # Run workflow - AR6-exact
#
# Run the climate assessment workflow exactly as it was run in AR6
# (including pre-processing errors due to updates in naming standards).

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing
import os
import platform

import pandas as pd
import pandas_indexing as pix
import tqdm.autonotebook as tqdman
from gcages.ar6 import run_ar6_workflow
from gcages.database import GCDB
from gcages.io import load_timeseries_csv
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.post_processing import AR7FTPostProcessor

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
OUTPUT_PATH = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_ar6-workflow-exact"
)
OUTPUT_PATH_MAGICC = OUTPUT_PATH / "magicc-ar6"
OUTPUT_PATH_MAGICC

# %%
scm_output_variables = (
    # GSAT
    "Surface Air Temperature Change",
    # # # GMST
    # # "Surface Air Ocean Blended Temperature Change",
    # # ERFs
    # "Effective Radiative Forcing",
    # "Effective Radiative Forcing|Anthropogenic",
    # "Effective Radiative Forcing|Aerosols",
    # "Effective Radiative Forcing|Aerosols|Direct Effect",
    # "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    # "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    # "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    # "Effective Radiative Forcing|Aerosols|Indirect Effect",
    # "Effective Radiative Forcing|Greenhouse Gases",
    # "Effective Radiative Forcing|CO2",
    # "Effective Radiative Forcing|CH4",
    # "Effective Radiative Forcing|N2O",
    # "Effective Radiative Forcing|F-Gases",
    # "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    # "Effective Radiative Forcing|Ozone",
    # "Effective Radiative Forcing|Aviation|Cirrus",
    # "Effective Radiative Forcing|Aviation|Contrail",
    # "Effective Radiative Forcing|Aviation|H2O",
    # "Effective Radiative Forcing|Black Carbon on Snow",
    # # "Effective Radiative Forcing|CFC11",
    # # "Effective Radiative Forcing|CFC12",
    # # "Effective Radiative Forcing|HCFC22",
    # # "Effective Radiative Forcing|HFC125",
    # # "Effective Radiative Forcing|HFC134a",
    # # "Effective Radiative Forcing|HFC143a",
    # # "Effective Radiative Forcing|HFC227ea",
    # # "Effective Radiative Forcing|HFC23",
    # # "Effective Radiative Forcing|HFC245fa",
    # # "Effective Radiative Forcing|HFC32",
    # # "Effective Radiative Forcing|HFC4310mee",
    # # "Effective Radiative Forcing|CF4",
    # # "Effective Radiative Forcing|C6F14",
    # # "Effective Radiative Forcing|C2F6",
    # # "Effective Radiative Forcing|SF6",
    # # # Heat uptake
    # # "Heat Uptake",
    # # "Heat Uptake|Ocean",
    # # Atmospheric concentrations
    # "Atmospheric Concentrations|CO2",
    # "Atmospheric Concentrations|CH4",
    # "Atmospheric Concentrations|N2O",
    # # # Carbon cycle
    # # "Net Atmosphere to Land Flux|CO2",
    # # "Net Atmosphere to Ocean Flux|CO2",
    # # # permafrost
    # # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    # # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)

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
scenario_files = tuple(SCENARIO_PATH.glob(f"{SCENARIO_TIME_ID}__scenarios-scenariomip__*.csv"))
if not scenario_files:
    msg = f"Check your scenario ID. {list(SCENARIO_PATH.glob('*.csv'))=}"
    raise AssertionError(msg)

scenario_files[:5]

# %%
scenarios_raw = pix.concat(
    [
        load_timeseries_csv(
            f,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_column_type=int,
        )
        for f in tqdman.tqdm(scenario_files)
    ]
).sort_index(axis="columns")

scenarios_raw_global = scenarios_raw.loc[
    pix.ismatch(region="World"),
    # TODO: drop this once we have usable scenario data post 2100
    :2100,
]

scenarios_raw_global

# %% [markdown]
# ### Down-select scenarios

# %%
selected_scenarios_idx = pd.MultiIndex.from_tuples(
    (
        # scenario for which the pre-pre-processing seems to make the biggest difference
        ("GCAM 7.1 scenarioMIP", "SSP1 - Low Overshoot"),
    ),
    name=["model", "scenario"],
)
scenarios_run = scenarios_raw_global[scenarios_raw_global.index.isin(selected_scenarios_idx)]

# scenarios_run = scenarios_raw_global.loc[
# pix.ismatch(scenario=["*Very Low*", "*Overshoot*"], model=["GCAM*", "AIM*", "*"])
# ]

# %%
# # To run all, just uncomment the below
# scenarios_run = scenarios_raw_global

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
res.post_processed_scenario_metadata.value_counts().sort_index()

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

# %%
db_meta_file = DATA_ROOT / "scenarios" / "data_raw" / f"{SCENARIO_TIME_ID}_all-meta.csv"
db_meta_file = DATA_ROOT / "scenarios" / "data_raw" / "20250131-125121_all-meta.csv"
db_meta = pd.read_csv(db_meta_file).set_index(["model", "scenario"])
db_meta

# %% [markdown]
# Comparing the two makes pretty clear that we can reproduce the database.
# The big difference is in the pre-processing,
# which double counts some of the emissions in the scenario.

# %%
post_processed_updated.metadata

# %%
db_meta.loc[db_meta.index.isin(post_processed_updated.metadata.index)][
    [
        "Category",
        "Category_name",
        "median peak warming (MAGICCv7.5.3)",
        "p33 peak warming (MAGICCv7.5.3)",
    ]
]
