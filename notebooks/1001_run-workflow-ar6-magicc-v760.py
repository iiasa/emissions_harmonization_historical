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
# # Run workflow - AR6 with MAGICC v7.6.0
#
# Run the climate assessment workflow with our AR6 processing,
# using MAGICC v7.6.0 as the SCM.

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing
import os

import openscm_runner.adapters
import pandas as pd
from gcages.io import load_timeseries_csv
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.workflow import run_magicc_and_post_processing

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
n_processes = multiprocessing.cpu_count()
run_checks = False  # TODO: turn on

# %%
INPUT_PATH = DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_ar6-workflow"

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
OUTPUT_PATH = INPUT_PATH / f"magicc-{magicc_expected_version.replace('.', '-')}_{magicc_prob_distribution_path.stem}"
OUTPUT_PATH

# %% [markdown]
# ## Load complete scenario data

# %%
complete_scenarios = load_timeseries_csv(
    INPUT_PATH / "complete_scenarios.csv",
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
# Strip prefix
complete_scenarios = complete_scenarios.pix.assign(
    variable=complete_scenarios.index.get_level_values("variable").map(
        lambda x: x.replace("AR6 climate diagnostics|Infilled|", "")
    )
)

# Deal with the HFC245 bug
complete_scenarios = complete_scenarios.pix.assign(
    variable=complete_scenarios.index.get_level_values("variable").str.replace("HFC245ca", "HFC245fa").values,
    unit=complete_scenarios.index.get_level_values("unit").str.replace("HFC245ca", "HFC245fa").values,
)

complete_scenarios

# %% [markdown]
# ## Down-select scenarios

# %%
selected_scenarios_idx = pd.MultiIndex.from_tuples(
    (
        # ("AIM 3.0",	"SSP1 - Very Low Emissions"),
        # ("MESSAGEix-GLOBIOM 2.1-M-R12", "SSP5 - High Emissions"),
        # ("IMAGE 3.4", "SSP5 - High Emissions"),
        # ("AIM 3.0", "SSP2 - Medium-Low Emissions"),
        # ("WITCH 6.0", "SSP2 - Low Emissions"),
        # ("REMIND-MAgPIE 3.4-4.8", "SSP2 - Low Overshoot_b"),
        # ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP5 - Low Overshoot"),
        # ("COFFEE 1.5", "SSP2 - Medium Emissions"),
        # ("GCAM 7.1 scenarioMIP", "SSP2 - Medium Emissions"),
        # ("IMAGE 3.4", "SSP2 - Very Low Emissions"),
        # ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions"),
        ###
        ("REMIND-MAgPIE 3.4-4.8", "SSP1 - Very Low Emissions"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Overshoot"),
        ("IMAGE 3.4", "SSP1 - Low Emissions"),
    ),
    name=["model", "scenario"],
)
scenarios_run = complete_scenarios[complete_scenarios.index.isin(selected_scenarios_idx)]

# scenarios_run = complete_scenarios.loc[
# pix.ismatch(scenario=["*Very Low*", "*Overshoot*"], model=["*", "AIM*", "GCAM*"])]
# scenarios_run = complete_scenarios.loc[pix.ismatch(scenario=["*Very Low*", "*Overshoot*"], model=["GCAM*"])]

# %%
# # To run all, just uncomment the below
# scenarios_run = complete_scenarios

# %%
scenarios_run.pix.unique(["model", "scenario"]).to_frame(index=False)

# %%
scenarios_run


# %% [markdown]
# ## Run SCMs


# %%
def map_ar6_output_names_to_iamc_names(inv: str) -> str:
    """
    Map AR6 output names to IAMC names
    """
    out = inv.replace("|PFC", "")

    if any(
        out.endswith(mg)
        for mg in [
            "CCl4",
            "CFC11",
            "CFC113",
            "CFC114",
            "CFC115",
            "CFC12",
            "CH2Cl2",
            "CH3Br",
            "CH3CCl3",
            "CH3Cl",
            "CHCl3",
            "HCFC141b",
            "HCFC142b",
            "HCFC22",
            "Halon1202",
            "Halon1211",
            "Halon1301",
            "Halon2402",
        ]
    ):
        toks = inv.split("|")
        if "CFC" in out and "HCFC" not in out:
            out = f"Emissions|Montreal Gases|CFC|{toks[-1]}"
        else:
            out = f"Emissions|Montreal Gases|{toks[-1]}"

    return out


# %%
scenarios_run = scenarios_run.pix.assign(
    variable=scenarios_run.index.get_level_values("variable").map(map_ar6_output_names_to_iamc_names),
    unit=scenarios_run.index.get_level_values("unit").str.replace("-", ""),
)

# %%
scm_results = run_magicc_and_post_processing(
    scenarios_run,
    n_processes=n_processes,
    magicc_exe_path=magicc_exe_path,
    magicc_prob_distribution_path=magicc_prob_distribution_path,
    output_path=OUTPUT_PATH,
    data_root=DATA_ROOT,
    batch_size_scenarios=15,
)

# %%
# If you need to re-write.
# remove OUTPUT_PATH / "db"

# %%
scm_results.post_processed.metadata.sort_values(["category", "Peak warming 33.0"])

# %%
scm_results.post_processed.metadata.groupby(["model"])["category"].value_counts().sort_index()

# %%
for out_file, df in (
    ("metadata.csv", scm_results.post_processed.metadata),
    ("scm-effective-emissions.csv", scm_results.scenarios_run),
    ("timeseries-percentiles.csv", scm_results.post_processed.timeseries_percentiles),
    # Don't write this, already in the database
    # ("scm-results.csv", scm_results),
    # Can write this, but not using yet so just leave out at the moment
    # because it's slow to write.
    # ("post-processed-timeseries.csv", post_processed.timeseries),
):
    full_path = OUTPUT_PATH / out_file
    print(f"Writing {full_path}")
    full_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(full_path)
    print(f"Wrote {full_path}")
    print()
