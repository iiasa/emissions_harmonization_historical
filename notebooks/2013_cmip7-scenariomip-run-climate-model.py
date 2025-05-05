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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Run simple climate model
#
# Here we run a simple climate model.
# This is done simple climate model by simple climate model.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %%
import json
import multiprocessing
import os
import platform
from functools import partial

import pandas_indexing as pix
import pandas_openscm
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import convert_openscm_runner_output_names_to_magicc_output_names, run_scms
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_HARMONISATION_ID,
    CMIP7_SCENARIOMIP_INFILLING_ID,
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
    SCENARIO_TIME_ID,
)

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "REMIND-MAgPIE 3.5-4.10"

# %%
model_clean = model.replace(" ", "_").replace(".", "p")
model_clean

# %%
SIMPLE_CLIMATE_MODEL = "magiccv7.6.0a3"
SIMPLE_CLIMATE_MODEL = "magiccv7.5.3"

# %%
pandas_openscm.register_pandas_accessor()

# %%
HARMONISED_ID_KEY = "_".join(
    [
        model_clean,
        COMBINED_HISTORY_ID,
        IAMC_REGION_PROCESSING_ID,
        SCENARIO_TIME_ID,
        CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
        CMIP7_SCENARIOMIP_HARMONISATION_ID,
    ]
)

# %%
HARMONISATION_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonisation-emissions_gcages-conventions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
HARMONISATION_EMISSIONS_FILE

# %%
COMPLETE_ID_KEY = "_".join(
    [
        model_clean,
        COMBINED_HISTORY_ID,
        IAMC_REGION_PROCESSING_ID,
        SCENARIO_TIME_ID,
        CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
        CMIP7_SCENARIOMIP_HARMONISATION_ID,
        CMIP7_SCENARIOMIP_INFILLING_ID,
    ]
)

# %%
COMPLETE_FILE = DATA_ROOT / "cmip7-scenariomip-workflow" / "infilling" / f"complete-emissions_{COMPLETE_ID_KEY}.csv"
COMPLETE_FILE

# %%
OUT_DB_ID_KEY = "_".join(
    [
        COMBINED_HISTORY_ID,
        IAMC_REGION_PROCESSING_ID,
        SCENARIO_TIME_ID,
        CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
        CMIP7_SCENARIOMIP_HARMONISATION_ID,
        CMIP7_SCENARIOMIP_INFILLING_ID,
    ]
)

# %%
RES_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "cmip7-scenariomip-workflow" / "scm-running" / OUT_DB_ID_KEY,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

RES_DB

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Complete scenarios

# %%
complete = load_timeseries_csv(
    COMPLETE_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if complete.empty:
    raise AssertionError

# complete

# %% [markdown]
# ### Model config

# %%
output_variables = (
    # GSAT
    "Surface Air Temperature Change",
    # # GMST
    # "Surface Air Ocean Blended Temperature Change",
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
    # # Heat uptake
    # "Heat Uptake",
    # # "Heat Uptake|Ocean",
    # # Atmospheric concentrations
    # "Atmospheric Concentrations|CO2",
    # "Atmospheric Concentrations|CH4",
    # "Atmospheric Concentrations|N2O",
    # # Carbon cycle
    # "Net Atmosphere to Land Flux|CO2",
    # "Net Atmosphere to Ocean Flux|CO2",
    "CO2_CURRENT_NPP",
    # # Permafrost
    # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)

# %%
if SIMPLE_CLIMATE_MODEL in ["magiccv7.5.3", "magiccv7.6.0a3"]:
    if SIMPLE_CLIMATE_MODEL == "magiccv7.6.0a3":
        magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
        magicc_expected_version = "v7.6.0a3"
        magicc_prob_distribution_path = (
            DATA_ROOT.parents[0]
            / "magicc"
            / "magicc-v7.6.0a3"
            / "configs"
            / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
        )

    elif SIMPLE_CLIMATE_MODEL == "magiccv7.5.3":
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
        magicc_expected_version = "v7.5.3"
        magicc_prob_distribution_path = (
            DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json"
        )

    else:
        raise NotImplementedError(SIMPLE_CLIMATE_MODEL)

    os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)

    with open(magicc_prob_distribution_path) as fh:
        cfgs_raw = json.load(fh)
    cfgs_physical = [
        {
            "run_id": c["paraset_id"],
            **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
        }
        for c in cfgs_raw["configurations"]
    ]

    startyear = 1750
    common_cfg = {
        "startyear": startyear,
        "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(output_variables),
        "out_ascii_binary": "BINARY",
        "out_binary_format": 2,
    }

    run_config = [{**common_cfg, **physical_cfg} for physical_cfg in cfgs_physical]
    # len(run_config)
    climate_models_cfgs = {"MAGICC7": run_config}

    complete_start_year = complete.columns.min()

    harmonisation_emissions_full = load_timeseries_csv(
        HARMONISATION_EMISSIONS_FILE,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
    if harmonisation_emissions_full.empty:
        raise AssertionError

    magicc_start_year = 2015
    harmonisation_emissions = harmonisation_emissions_full
    # harmonisation_emissions

    harmonisation_emissions_to_add = (
        harmonisation_emissions.openscm.mi_loc(
            complete.reset_index(["model", "scenario"], drop=True).drop_duplicates().index
        )
        .reset_index(["model", "scenario"], drop=True)
        .align(complete)[0]
        .loc[:, magicc_start_year : complete_start_year - 1]
    )

    complete_scm = pix.concat(
        [
            harmonisation_emissions_to_add.reorder_levels(complete.index.names),
            complete,
        ],
        axis="columns",
    )
    # Also interpolate for MAGICC
    complete_scm = complete_scm.T.interpolate(method="index").T

else:
    raise NotImplementedError(SIMPLE_CLIMATE_MODEL)

# %%
complete_openscm_runner = update_index_levels_func(
    complete_scm,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
        )
    },
)

# %%
run_scms(
    scenarios=complete_openscm_runner,
    climate_models_cfgs=climate_models_cfgs,
    output_variables=output_variables,
    scenario_group_levels=["model", "scenario"],
    n_processes=multiprocessing.cpu_count(),
    db=RES_DB,
    verbose=True,
    progress=True,
    batch_size_scenarios=15,
    force_rerun=False,
)

# %% [markdown]
# ## Save

# %%
# Nothing to do here, data is in the db

# %% [markdown]
# ## Quick plot

# %%
gsat = RES_DB.load(pix.isin(model=model, variable="Surface Air Temperature Change"))
gsat

# %%
gsat.openscm.plot_plume_after_calculating_quantiles(quantile_over="run_id", style_var="climate_model")

# %%
gsat_delta_tmp = gsat.stack().unstack("climate_model")
gsat_delta = gsat_delta_tmp["MAGICCv7.6.0a3"] - gsat_delta_tmp["MAGICCv7.6.0a3"]

# %%
gsat_delta_tmp = gsat.stack().unstack("climate_model")
gsat_delta = gsat_delta_tmp["MAGICCv7.6.0a3"] - gsat_delta_tmp["MAGICCv7.5.3"]
gsat_delta.unstack("time").openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id",
)
