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
# # Run a simple climate model
#
# Here we run a simple climate model.

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing
import os
import platform
import warnings
from functools import partial

import openscm_units
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import run_scms
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    INFILLED_SCENARIOS_DB,
    MARKERS,
    RCMIP_PROCESSED_DB,
    REPO_ROOT,
    SCM_OUT_DIR,
    SCM_OUTPUT_DB,
)
from emissions_harmonization_historical.scm_running import (
    get_complete_scenarios_for_magicc,
    load_magicc_cfgs,
)

# Suppress expected MAGICC warnings about extending solar forcing to 2500
# The solar RF data is already extended to 2500, but MAGICC's Fortran code
# warns that it's using extrapolated (not observed) data beyond 2100
warnings.filterwarnings("ignore", message=".*Extending solar RF.*")
warnings.filterwarnings("ignore", message=".*magicc logged a WARNING message.*")
warnings.filterwarnings(
    "ignore", message=r"magicc logged a WARNING message\. Check the 'stderr' key.*", category=UserWarning
)
warnings.filterwarnings("ignore", category=UserWarning, module="pymagicc.core")
# Also suppress at the source
logging.getLogger("pymagicc").setLevel(logging.ERROR)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "REMIND"
scm: str = "MAGICCv7.6.0a3"
markers_only: bool = True

# %%
output_dir_model = SCM_OUT_DIR / model
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load data

# %% [markdown]
# ### Complete scenarios (extended to 2500)

# %%
# Load extended scenarios (1750-2500) for climate model runs
# These are the 7 marker scenarios extended beyond 2100
complete_scenarios = INFILLED_SCENARIOS_DB.load(
    pix.isin(stage="extended") & pix.ismatch(model=f"*{model}*")
).reset_index("stage", drop=True)

# Filter out internal diagnostic variables that aren't part of CMIP7 naming convention
internal_variables = [
    "Emissions|CO2|Gross Positive Emissions",
    "Emissions|CO2|Gross Removals",
]
complete_scenarios = complete_scenarios.loc[
    ~complete_scenarios.index.get_level_values("variable").isin(internal_variables)
]

# %%
# Check year range to verify we have extended scenarios
print(f"Year range in complete_scenarios: {complete_scenarios.columns.min()} to {complete_scenarios.columns.max()}")
print(f"Number of scenarios: {len(complete_scenarios.pix.unique('scenario'))}")
print(f"Scenarios: {list(complete_scenarios.pix.unique('scenario'))}")
# %%
if markers_only:
    markers_l = []
    for model, scenario, _ in MARKERS:
        tmp = complete_scenarios.loc[pix.isin(model=model, scenario=scenario)]
        if not tmp.empty:
            markers_l.append(tmp)

    complete_scenarios = pix.concat(markers_l)
    if complete_scenarios.empty:
        raise AssertionError

# %% [markdown]
# ### History
#
# Just in case we need it for MAGICC

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

# history.loc[:, :2023]

# %% [markdown]
# ## Configure SCM

# %%
output_variables = (
    # GSAT
    "Surface Air Temperature Change",
    # # GMST
    "Surface Air Ocean Blended Temperature Change",
    # # ERFs
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
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|Tropospheric Ozone",
    "Effective Radiative Forcing|Stratospheric Ozone",
    "Effective Radiative Forcing|Solar",
    "Effective Radiative Forcing|Volcanic",
    # # Heat uptake
    "Heat Uptake",
    "Heat Uptake|Ocean",
    # # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # # Carbon cycle
    # "Net Atmosphere to Land Flux|CO2",
    # "Net Atmosphere to Ocean Flux|CO2",
    # "CO2_CURRENT_NPP",
    # # Permafrost
    # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
    "Sea Level Rise",
)

# %%
if scm in ["MAGICCv7.5.3", "MAGICCv7.6.0a3"]:
    if scm == "MAGICCv7.6.0a3":
        if platform.system() == "Darwin":
            if platform.processor() == "arm":
                magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
            else:
                raise NotImplementedError(platform.processor())
        elif platform.system() == "Windows":
            raise NotImplementedError(platform.system())
        elif platform.system().lower().startswith("linux"):
            magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc"
            # Set library path for GCC 13.3.0 (MAGICC was built with this version)
            # This ensures gfortran libraries are found even when modules aren't loaded
            gcc_lib_path = "/opt/software/easybuild/software/GCCcore/13.3.0/lib64"
            if "LD_LIBRARY_PATH" in os.environ:
                os.environ["LD_LIBRARY_PATH"] = f"{gcc_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
            else:
                os.environ["LD_LIBRARY_PATH"] = gcc_lib_path

            # Use /scratch instead of /tmp for MAGICC worker temporary directories
            # /tmp is only 10 GB and fills up with 32 parallel MAGICC processes
            os.environ["MAGICC_WORKER_ROOT_DIR"] = "/scratch/bensan"
        else:
            raise NotImplementedError(platform.system())

        magicc_expected_version = "v7.6.0a3"
        magicc_prob_distribution_path = (
            REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
        )

    elif scm == "MAGICCv7.5.3":
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"
        if platform.system() == "Darwin":
            if platform.processor() == "arm":
                magicc_exe = "magicc-darwin-arm64"
            else:
                raise NotImplementedError(platform.processor())
        elif platform.system() == "Windows":
            magicc_exe = "magicc.exe"
        elif platform.system().lower().startswith("linux"):
            magicc_exe = "magicc"
        else:
            raise NotImplementedError(platform.system())

        magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.5.3" / "bin" / magicc_exe
        magicc_expected_version = "v7.5.3"
        magicc_prob_distribution_path = REPO_ROOT / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json"

    else:
        raise NotImplementedError(scm)

    os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)

    climate_models_cfgs = load_magicc_cfgs(
        prob_distribution_path=magicc_prob_distribution_path,
        output_variables=output_variables,
        startyear=1750,
    )

    complete_scm = get_complete_scenarios_for_magicc(
        scenarios=complete_scenarios,
        history=history,
    )

    # Convert year columns from float to int to avoid MAGICC namelist errors
    # MAGICC's Fortran namelist reader expects integer years, not floats
    complete_scm.columns = complete_scm.columns.astype(int)

# Check year range after MAGICC preparation
print(f"Year range in complete_scm: {complete_scm.columns.min()} to {complete_scm.columns.max()}")


# %%


# complete_scm

# %%
climate_models_cfgs["MAGICC7"][0]["out_dynamic_vars"]

# %% [markdown]
# ### If MAGICC, check how yuck the jump will be
#
# Answer: not ideal but we're going to have to live with it.


# %%
if scm.startswith("MAGICC"):
    reporting_to_rcmip = partial(
        convert_variable_name,
        to_convention=SupportedNamingConventions.RCMIP,
        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )
    rcmip_to_reporting = partial(
        convert_variable_name,
        from_convention=SupportedNamingConventions.RCMIP,
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )

    rcmip_hist = RCMIP_PROCESSED_DB.load(
        pix.isin(
            region="World",
            scenario="ssp245",
            variable=complete_scenarios.pix.unique("variable").map(reporting_to_rcmip),
        ),
        progress=True,
    ).loc[:, 1990:2014]
    rcmip_hist = rcmip_hist.openscm.update_index_levels({"variable": rcmip_to_reporting})
    # rcmip_hist

    pdf = pix.concat([rcmip_hist, complete_scm]).loc[:, 1990:2030].openscm.to_long_data().dropna()
    # pdf

    fg = sns.relplot(
        data=pdf,
        x="time",
        y="value",
        hue="scenario",
        col="variable",
        col_order=sorted(pdf["variable"].unique()),
        col_wrap=4,
        kind="line",
        facet_kws=dict(sharey=False),
    )
    for ax in fg.axes.flatten():
        ax.set_ylim(ymin=0.0)

# %% [markdown]
# ## Run SCM

# %% editable=true slideshow={"slide_type": ""}
complete_openscm_runner = update_index_levels_func(
    complete_scm,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
        )
    },
)
# complete_openscm_runner


# %%
class db_hack:
    """Save files in groups while we can't pass groupby through the function below"""

    def __init__(self, actual_db):
        self.actual_db = actual_db

    def load_metadata(self, *args, **kwargs):
        """Load metadata"""
        return self.actual_db.load_metadata(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Save"""
        return self.actual_db.save(
            *args,
            **kwargs,
            groupby=["model", "scenario", "variable"],
            allow_overwrite=True,
        )


# %%
db = db_hack(SCM_OUTPUT_DB)

# %%
# Limit parallel processes to avoid memory issues on high-core-count systems
# Each MAGICC process loads full scenario data, so too many processes causes OOM
# Rule of thumb: ~4-8 GB per MAGICC process for extended scenarios
max_processes = min(multiprocessing.cpu_count(), 32)  # Cap at 32 processes
print(f"Running with {max_processes} parallel processes (system has {multiprocessing.cpu_count()} cores)")

# if scm in ["FAIRv2.2.2"]:
#    some custom code
# else:
run_scms(
    scenarios=complete_openscm_runner,
    climate_models_cfgs=climate_models_cfgs,
    output_variables=output_variables,
    scenario_group_levels=["model", "scenario"],
    n_processes=max_processes,
    db=db,
    verbose=True,
    progress=True,
    batch_size_scenarios=15,
    force_rerun=True,  # CHANGED: Must re-run for extended scenarios to 2500
)

# %%
# Check what was actually saved by run_scms to the database
scm_output_check = SCM_OUTPUT_DB.load(pix.ismatch(model=f"*{model}*", climate_model=f"*{scm}*"))
if not scm_output_check.empty:
    print(f"Year range in SCM_OUTPUT_DB: {scm_output_check.columns.min()} to {scm_output_check.columns.max()}")
    print(f"Variables in SCM_OUTPUT_DB: {sorted(scm_output_check.pix.unique('variable'))}")

    # DIAGNOSTIC: Check scenarios and their year ranges
    print("DIAGNOSTIC: Scenarios in database:")
    for scenario in sorted(scm_output_check.pix.unique("scenario")):
        scenario_data = scm_output_check.loc[scm_output_check.index.get_level_values("scenario") == scenario]
        temp_data = scenario_data.loc[
            scenario_data.index.get_level_values("variable") == "Surface Air Temperature Change"
        ]
        if not temp_data.empty:
            print(f"  {scenario}: {temp_data.columns.min()} to {temp_data.columns.max()}")

    # DIAGNOSTIC: Check year range for Surface Air Temperature Change specifically
    temp_var = scm_output_check.loc[
        scm_output_check.index.get_level_values("variable") == "Surface Air Temperature Change"
    ]
    if not temp_var.empty:
        print(
            f"DIAGNOSTIC: 'Surface Air Temperature Change' year range: "
            f"{temp_var.columns.min()} to {temp_var.columns.max()}"
        )
        # Check if there's a 'stage' index level
        if "stage" in temp_var.index.names:
            print(f"DIAGNOSTIC: 'stage' values in temp data: {sorted(temp_var.pix.unique('stage'))}")
        else:
            print("DIAGNOSTIC: No 'stage' index level found in temperature data!")
    else:
        print("DIAGNOSTIC: 'Surface Air Temperature Change' not found in database!")
else:
    print("No SCM output found in database yet")

# %% [markdown]
# ## Save
#
# The SCM output is already saved in the db.
# Here we also save the emissions that were actually used by the SCM.

# %%
# DIAGNOSTIC: Check what's in complete_scm before saving
print(f"DIAGNOSTIC: complete_scm year range before save: {complete_scm.columns.min()} to {complete_scm.columns.max()}")
print(f"DIAGNOSTIC: complete_scm variables: {sorted(complete_scm.pix.unique('variable')[:5])}...")  # Show first 5

# Check what's already in the database before overwriting
existing_data = SCM_OUTPUT_DB.load(pix.ismatch(model=f"*{model}*", climate_model=f"*{scm}*"))
if not existing_data.empty:
    print(
        f"DIAGNOSTIC: Existing data in SCM_OUTPUT_DB before overwrite: "
        f"{existing_data.columns.min()} to {existing_data.columns.max()}"
    )
    print(f"DIAGNOSTIC: Existing variables: {sorted(existing_data.pix.unique('variable')[:5])}...")

# %% editable=true slideshow={"slide_type": ""}
SCM_OUTPUT_DB.save(complete_scm.pix.assign(climate_model=scm), allow_overwrite=True)

# DIAGNOSTIC: Check what's in the database AFTER saving
final_data = SCM_OUTPUT_DB.load(pix.ismatch(model=f"*{model}*", climate_model=f"*{scm}*"))
print(
    f"DIAGNOSTIC: Final data in SCM_OUTPUT_DB after save: " f"{final_data.columns.min()} to {final_data.columns.max()}"
)
print(f"DIAGNOSTIC: Final variables: {sorted(final_data.pix.unique('variable')[:5])}...")
