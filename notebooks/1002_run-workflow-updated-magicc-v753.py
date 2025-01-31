# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run workflow - updated MAGICC v7.5.3
#
# Run the climate assessment workflow with our updated processing,
# using MAGICC v7.5.3 as the SCM.

# %% [markdown]
# ## Imports

# %%
import json
import logging
import multiprocessing
import os

import openscm_runner.adapters
import pandas as pd
import pandas_indexing as pix
import pint
from gcages.database import GCDB
from gcages.io import load_timeseries_csv
from gcages.post_processing import PostProcessor
from gcages.scm_running import SCMRunner, convert_openscm_runner_output_names_to_magicc_output_names
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)

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
INPUT_PATH = DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"

# %%
# Needed for 7.5.3 on a mac
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"
magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "bin" / "magicc-darwin-arm64"
magicc_expected_version = "v7.5.3"
magicc_prob_distribution_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json"

# %%
os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
if openscm_runner.adapters.MAGICC7.get_version() != magicc_expected_version:
    raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())

# %%
OUTPUT_PATH = INPUT_PATH / f"magicc-{magicc_expected_version.replace('.', '-')}_{magicc_prob_distribution_path.stem}"
OUTPUT_PATH

# %% [markdown]
# ## Load infilled data

# %%
infilled = load_timeseries_csv(
    INPUT_PATH / "infilled.csv",
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
infilled

# %% [markdown]
# ## Down-select scenarios

# %%
# # Randomly select some scenarios
# # (this is how I generated the hard-coded values in the next cell).
# base = infilled.pix.unique(["model", "scenario"]).to_frame(index=False)
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
# selected_scenarios_idx = pd.MultiIndex.from_tuples(
#     (
#         ("AIM 3.0",	"SSP1 - Very Low Emissions"),
#         # ("MESSAGEix-GLOBIOM 2.1-M-R12", "SSP5 - High Emissions"),
#         # ("IMAGE 3.4", "SSP5 - High Emissions"),
#         # ("AIM 3.0", "SSP2 - Medium-Low Emissions"),
#         # ("WITCH 6.0", "SSP2 - Low Emissions"),
#         # ("REMIND-MAgPIE 3.4-4.8", "SSP2 - Low Overshoot_b"),
#         # ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP5 - Low Overshoot"),
#         # ("COFFEE 1.5", "SSP2 - Medium Emissions"),
#         # ("GCAM 7.1 scenarioMIP", "SSP2 - Medium Emissions"),
#         ("IMAGE 3.4", "SSP2 - Very Low Emissions"),
#         # ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions"),
#     ),
#     name=["model", "scenario"],
# )
# scenarios_run = infilled[infilled.index.isin(selected_scenarios_idx)]

scenarios_run = infilled.loc[pix.ismatch(scenario="*Low*")]

scenarios_run = infilled.loc[pix.ismatch(model="*COFFEE*")]

# %%
# # To run all, just uncomment the below
# scenarios_run = infilled

# %%
scenarios_run.pix.unique(["model", "scenario"]).to_frame(index=False)

# %%
scenarios_run

# %% [markdown]
# ## Run SCMs

# %%
batch_size_scenarios = 15
scm_output_variables = (
    # GSAT
    "Surface Air Temperature Change",
    # # GMST
    # "Surface Air Ocean Blended Temperature Change",
    # ERFs
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    # "Effective Radiative Forcing|CFC11",
    # "Effective Radiative Forcing|CFC12",
    # "Effective Radiative Forcing|HCFC22",
    # "Effective Radiative Forcing|Ozone",
    # "Effective Radiative Forcing|HFC125",
    # "Effective Radiative Forcing|HFC134a",
    # "Effective Radiative Forcing|HFC143a",
    # "Effective Radiative Forcing|HFC227ea",
    # "Effective Radiative Forcing|HFC23",
    # "Effective Radiative Forcing|HFC245fa",
    # "Effective Radiative Forcing|HFC32",
    # "Effective Radiative Forcing|HFC4310mee",
    # "Effective Radiative Forcing|CF4",
    # "Effective Radiative Forcing|C6F14",
    # "Effective Radiative Forcing|C2F6",
    # "Effective Radiative Forcing|SF6",
    # # Heat uptake
    # "Heat Uptake",
    # "Heat Uptake|Ocean",
    # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # # Carbon cycle
    # "Net Atmosphere to Land Flux|CO2",
    # "Net Atmosphere to Ocean Flux|CO2",
    # # permafrost
    # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)

# %% [markdown]
# ### MAGICC stuff

# %%
startyear = 1750
endyear = 2100
out_dynamic_vars = [
    f"DAT_{v}" for v in convert_openscm_runner_output_names_to_magicc_output_names(scm_output_variables)
]

# %%
scm_results_db = GCDB(OUTPUT_PATH / "db")
scm_results_db

# %%
with open(magicc_prob_distribution_path) as fh:
    cfgs_raw = json.load(fh)

base_cfgs = [
    {
        "run_id": c["paraset_id"],
        **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
    }
    for c in cfgs_raw["configurations"]
]


common_cfg = {
    "startyear": startyear,
    "endyear": endyear,
    "out_dynamic_vars": out_dynamic_vars,
    "out_ascii_binary": "BINARY",
    "out_binary_format": 2,
}


run_config = [
    {
        **common_cfg,
        **base_cfg,
    }
    for base_cfg in base_cfgs
]

scm_runner = SCMRunner(
    climate_models_cfgs={"MAGICC7": run_config},
    output_variables=scm_output_variables,
    force_interpolate_to_yearly=True,
    db=scm_results_db,
    res_column_type=int,
    run_checks=False,  # TODO: turn on
    n_processes=n_processes,
)


# %%
# If you need to re-write.
# scm_results_db.delete()
# scm_results_db.load_metadata()


# %%
def transform_rcmip_to_iamc_variable(v):
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (
        ("F-Gases|", ""),
        ("PFC|", ""),
        ("HFC4310mee", "HFC43-10"),
        ("MAGICC AFOLU", "AFOLU"),
        ("MAGICC Fossil and Industrial", "Energy and Industrial Processes"),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
pix.set_openscm_registry_as_default()

# %%
MAGICC_FORCE_START_YEAR = 2015
endyear = 2100

RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"

rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
ar6_harmonisation_points = rcmip_clean.loc[
    pix.ismatch(mip_era="CMIP6")
    & pix.ismatch(scenario="ssp245")
    & pix.ismatch(region="World")
    & pix.ismatch(variable=scenarios_run.pix.unique("variable"))
].reset_index(["mip_era", "activity_id"], drop=True)[MAGICC_FORCE_START_YEAR]
with pint.get_application_registry().context("NOx_conversions"):
    ar6_harmonisation_points = ar6_harmonisation_points.pix.convert_unit(
        {"Mt NOx/yr": "Mt NO2/yr", "kt HFC4310mee/yr": "kt HFC4310/yr"}
    )

expected_n_variables = ar6_harmonisation_points.shape[0]
if ar6_harmonisation_points.shape[0] != expected_n_variables:
    raise AssertionError(ar6_harmonisation_points.shape[0])

ar6_harmonisation_points

# %%
a, b = ar6_harmonisation_points.reset_index(["model", "scenario"], drop=True).align(scenarios_run)
scenarios_run = pix.concat([a.to_frame(), b], axis="columns").sort_index(axis="columns")

# Interpolate so it's clearer what went into MAGICC
scenarios_run = scenarios_run.T.interpolate("index").T
if scenarios_run.isnull().any().any():
    if not scm_runner.force_interpolate_to_yearly:
        raise AssertionError

scenarios_run

# %% [markdown]
# ### Run

# %%
scm_results = scm_runner(scenarios_run, batch_size_scenarios=batch_size_scenarios)
scm_results

# %% [markdown]
# ## Post-process

# %%
post_processor = PostProcessor(
    gsat_variable_name="Surface Air Temperature Change",
    gsat_in_line_with_assessment_variable_name="Assessed Surface Air Temperature Change",
    gsat_assessment_median=0.85,
    gsat_assessment_time_period=range(1995, 2014 + 1),
    gsat_assessment_pre_industrial_period=range(1850, 1900 + 1),
    percentiles_to_calculate=(0.05, 0.33, 0.5, 0.67, 0.95),
    exceedance_global_warming_levels=(1.5, 2.0, 2.5),
    run_checks=run_checks,
)

# %%
post_processed = post_processor(scm_results)
post_processed.metadata.sort_values(["category", "Peak warming 33.0"])

# %%
post_processed.metadata.groupby(["model"])["category"].value_counts().sort_index()

# %%
metadata_out = OUTPUT_PATH / "metadata.csv"
metadata_out.parent.mkdir(exist_ok=True, parents=True)
post_processed.metadata.to_csv(metadata_out)
metadata_out

# %%
for out_file, df in (
    ("metadata.csv", post_processed.metadata),
    ("scm-effective-emissions.csv", scenarios_run),
    ("timeseries-percentiles.csv", post_processed.timeseries_percentiles),
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
