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
# # Run MAGICC
#
# TODO: see if we want to also include running FaIR
# here or in another notebook.

# %%
import json
import os
import random

import openscm_runner
import openscm_runner.adapters
import openscm_runner.run
import pandas as pd
import pandas_indexing as pix
import pymagicc.definitions
import scmdata

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    INFILLING_LEFTOVERS_ID,
    INFILLING_SILICONE_ID,
    INFILLING_WMO_ID,
)
from emissions_harmonization_historical.io import load_csv

# %%
SCENARIO_TIME_ID = "20250122-140031"

# %%
complete_scenarios_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "infilled"
    / f"infilled_{SCENARIO_TIME_ID}_{HARMONISATION_ID}_{INFILLING_SILICONE_ID}_{INFILLING_WMO_ID}_{INFILLING_LEFTOVERS_ID}.csv"  # noqa: E501
)
complete_scenarios_file

# %%
N_CFGS_TO_RUN = 5
N_CFGS_TO_RUN = 600

# %%
magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
magicc_expected_version = "v7.6.0a3"
PROBABILISTIC_DISTRIBUTION_FILE = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
# magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "bin" / "magicc-darwin-arm64"
# # Needed for 7.5.3 on a mac
# os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"
# magicc_expected_version = "v7.5.3"
# PROBABILISTIC_DISTRIBUTION_FILE = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json"

# %%
os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)

# %%
magicc7 = openscm_runner.adapters.MAGICC7

# %%
if magicc7.get_version() != magicc_expected_version:
    raise AssertionError(magicc7.get_version())
    
magicc_expected_version

# %%
scenarios_raw = load_csv(complete_scenarios_file)
scenarios_raw

# %%
# # Randomly select some scenarios
# # (this is how I generated the hard-coded values in the next cell).
# base = scenarios_raw.pix.unique(["model", "scenario"]).to_frame(index=False)
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
selected_scenarios_idx = pd.MultiIndex.from_tuples((
    ('MESSAGEix-GLOBIOM 2.1-M-R12', 'SSP5 - High Emissions'),
    ('IMAGE 3.4', 'SSP5 - High Emissions'),
    ('AIM 3.0', 'SSP2 - Medium-Low Emissions'),
    ('WITCH 6.0', 'SSP2 - Low Emissions'),
    ('REMIND-MAgPIE 3.4-4.8', 'SSP2 - Low Overshoot_b'),
    ('MESSAGEix-GLOBIOM-GAINS 2.1-M-R12', 'SSP5 - Low Overshoot'),
    ('COFFEE 1.5', 'SSP2 - Medium Emissions'),
    ('GCAM 7.1 scenarioMIP', 'SSP2 - Medium Emissions'),
    ('IMAGE 3.4', 'SSP2 - Very Low Emissions'),
    ('MESSAGEix-GLOBIOM-GAINS 2.1-M-R12', 'SSP1 - Very Low Emissions')
    ), 
    name=["model", "scenario"],
)
selected_scenarios_idx

# %%
scenarios_run = scenarios_raw[scenarios_raw.index.isin(selected_scenarios_idx)]
scenarios_run


# %%
def transform_iacm_to_openscm_runner_variable(v):
    """Transform IAMC variables to OpenSCM-Runner variables"""
    res = v

    replacements = (
        ("CFC|", ""),
        ("HFC|", ""),
        ("|Montreal Gases", ""),
        ("HFC43-10", "HFC4310mee", ),
        ("AFOLU", "MAGICC AFOLU", ),
        ("Energy and Industrial Processes", "MAGICC Fossil and Industrial", ),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
scenarios_run_openscmrunner = scenarios_run.copy()
scenarios_run_openscmrunner = scenarios_run_openscmrunner.pix.assign(
    variable=scenarios_run_openscmrunner.index.get_level_values("variable").map(transform_iacm_to_openscm_runner_variable).values
)
# Have to interpolate too before passing to OpenSCM-Runner, particularly MAGICC
scenarios_run_openscmrunner = scenarios_run_openscmrunner.T.interpolate("index").T
scenarios_run_openscmrunner

# %%
with open(PROBABILISTIC_DISTRIBUTION_FILE) as fh:
    cfgs_raw = json.load(fh)

base_cfgs = [
    {
        "run_id": c["paraset_id"],
        **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
    }
    for c in cfgs_raw["configurations"]
]

# %%
startyear = 1750
endyear = 2105  # add the 'MAGICC' buffer
common_cfg = {
    "startyear": startyear,
    "endyear": endyear,
    "out_dynamic_vars": [
        "DAT_SURFACE_TEMP",
        'DAT_SURFACE_MIXEDLAYERTEMP',
        'DAT_TOTAL_INCLVOLCANIC_ERF',
        'DAT_TOTAL_ANTHRO_ERF',
        'DAT_AEROSOL_ERF',
        'DAT_TOTAER_DIR_ERF',
        'DAT_BCT_ERF',
        'DAT_OCT_ERF',
        'DAT_SOXT_ERF',
        'DAT_CLOUD_TOT_ERF',
        'DAT_GHG_ERF',
        'DAT_CO2_ERF',
        'DAT_CH4_ERF',
        'DAT_N2O_ERF',
        'DAT_FGASSUM_ERF',
        'DAT_MHALOSUM_ERF',
        'DAT_CFC11_ERF',
        'DAT_CFC12_ERF',
        'DAT_HCFC22_ERF',
        'DAT_OZTOTAL_ERF',
        'DAT_HFC125_ERF',
        'DAT_HFC134A_ERF',
        'DAT_HFC143A_ERF',
        'DAT_HFC227EA_ERF',
        'DAT_HFC23_ERF',
        'DAT_HFC245FA_ERF',
        'DAT_HFC32_ERF',
        'DAT_HFC4310_ERFmee',
        'DAT_CF4_ERF',
        'DAT_C6F14_ERF',
        'DAT_C2F6_ERF',
        'DAT_SF6_ERF',
        'DAT_HEATUPTK_EARTH',
        'DAT_CO2_CONC',
        'DAT_CH4_CONC',
        'DAT_N2O_CONC',
        'DAT_CO2_AIR2LAND_FLUX',
        'DAT_CO2_AIR2OCEAN_FLUX',
        'DAT_CO2PF_EMIS',
        'DAT_CH4PF_EMIS'
    ],
    "out_ascii_binary": "BINARY",
    "out_binary_format": 2,
}


# %%
def get_openscm_runner_output_names(magicc_names):
    return [
        pymagicc.definitions.convert_magicc7_to_openscm_variables(
            magiccvarname
        ).replace("DAT_", "")
        for magiccvarname in magicc_names
    ]



# %%
openscm_runner_output_variables = get_openscm_runner_output_names(
    common_cfg["out_dynamic_vars"]
)
openscm_runner_output_variables


# %%
run_config = [
    {
        **common_cfg,
        **base_cfg,
    }
    for base_cfg in base_cfgs[:N_CFGS_TO_RUN]
]
len(run_config)


# %%
# TODO: refactor all this into a model runner
# that hides the model configuration.
# TODO: add caching/saving along the way and batching.
magicc_res_full = openscm_runner.run.run(
    scenarios=scmdata.ScmRun(scenarios_run_openscmrunner, copy_data=True),
    climate_models_cfgs={"MAGICC7": run_config},
    output_variables=openscm_runner_output_variables,
)
magicc_res_full

# %%
magicc_res = magicc_res_full.timeseries(time_axis="year").loc[pix.isin(region=["World"])]
magicc_res

# %%
sorted(magicc_res.pix.unique("variable"))

# %%
temperature_raw = magicc_res.loc[pix.isin(variable="Surface Air Temperature Change")]
temperature_raw

# %%
target_median = 0.85
target_median_years = range(1995, 2014 + 1)
pi_years = range(1850, 1900 + 1)

# %%
temperature_rel_pi = temperature_raw.subtract(temperature_raw.loc[:, pi_years].mean(axis="columns"), axis="rows")
temperature_shift = target_median - temperature_rel_pi.loc[:, target_median_years].mean(axis="columns").groupby(["model", "scenario"]).median()
temperature_match_historical_assessment = temperature_rel_pi.add(temperature_shift, axis="rows")
temperature_match_historical_assessment.loc[:, target_median_years].mean(axis="columns").groupby(["model", "scenario"]).median()

# %%
peak_warming_quantiles = temperature_match_historical_assessment.max(axis="columns").groupby(["model", "scenario"]).quantile([0.05, 0.17, 0.5, 0.83, 0.95]).unstack().sort_values(by=0.5)
peak_warming_quantiles

# %%
eoc_warming_quantiles = temperature_match_historical_assessment[2100].groupby(["model", "scenario"]).quantile([0.05, 0.17, 0.5, 0.83, 0.95]).unstack().sort_values(by=0.5)
eoc_warming_quantiles

# %%
categories = pd.Series(
    "C8: Above 4.0°C",
    index=peak_warming_quantiles.index.
    name="Category_name",
)

categories[peak_warming_quantiles[0.5] < 4.0] = "C7: Below 4.0°C"
categories[peak_warming_quantiles[0.5] < 3.0] = "C6: Below 3.0°C"
categories[peak_warming_quantiles[0.5] < 2.5] = "C5: Below 2.5°C"
categories[peak_warming_quantiles[0.5] < 2.0] = "C4: Below 2.0°C"
categories[peak_warming_quantiles[0.67] < 2.0] = "C3: Likely below 2°C"
categories[(peak_warming_quantiles[0.33] > 1.5) & (eoc_warming_quantiles[0.5] < 1.5)] = "C2: Below 1.5°C with high OS"
categories[(peak_warming_quantiles[0.33] <= 1.5) & (eoc_warming_quantiles[0.5] < 1.5)] = "C1b: Below 1.5°C with low OS"
categories[peak_warming_quantiles[0.5] < 1.5] = "C1a: Below 1.5°C with no OS"

categories


# %%
def get_exceedance_probability(indf: pd.DataFrame, warming_level: float) -> float:
    peaks = indf.max(axis="columns")
    ep = (peaks > warming_level) / peaks.shape[0] * 100

    return ep


# %%
exceedance_probabilities_l = [
    temperature_match_historical_assessment.groupby(["model", "scenario"]).apply(get_exceedance_probability, warming_level=gwl)
    for gwl in [1.5, 2.0]
]
pix.concat(exceedance_probabilities_l)


# %%
def get_exceedance_probability_over_time(indf: pd.DataFrame, warming_level: float) -> pd.Series:
    gt_wl = (indf > warming_level).sum(axis="rows")
    ep = gt_wl / indf.shape[0] * 100

    return ep


# %%
exceedance_probabilities_over_time_l = [
    temperature_match_historical_assessment.groupby(["model", "scenario"]).apply(get_exceedance_probability_over_time, warming_level=gwl)
    for gwl in [1.5, 2.0]
]
pix.concat(exceedance_probabilities_over_time_l)

# %%
