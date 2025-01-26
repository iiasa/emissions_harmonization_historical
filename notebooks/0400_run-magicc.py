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

import numpy as np
import openscm_runner
import openscm_runner.adapters
import openscm_runner.run
import pandas as pd
import pandas_indexing as pix
import pint
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
pix.set_openscm_registry_as_default()

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
PROBABILISTIC_DISTRIBUTION_FILE = (
    DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

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
# TODO: remove once we have data post 2100
scenarios_raw = scenarios_raw.loc[:, :2100]
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
selected_scenarios_idx = pd.MultiIndex.from_tuples(
    (
        # ('MESSAGEix-GLOBIOM 2.1-M-R12', 'SSP5 - High Emissions'),
        # ('IMAGE 3.4', 'SSP5 - High Emissions'),
        # ('AIM 3.0', 'SSP2 - Medium-Low Emissions'),
        # ('WITCH 6.0', 'SSP2 - Low Emissions'),
        # ('REMIND-MAgPIE 3.4-4.8', 'SSP2 - Low Overshoot_b'),
        # ('MESSAGEix-GLOBIOM-GAINS 2.1-M-R12', 'SSP5 - Low Overshoot'),
        # ('COFFEE 1.5', 'SSP2 - Medium Emissions'),
        # ('GCAM 7.1 scenarioMIP', 'SSP2 - Medium Emissions'),
        ("IMAGE 3.4", "SSP2 - Very Low Emissions"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions"),
    ),
    name=["model", "scenario"],
)
selected_scenarios_idx

# %%
# scenarios_run = scenarios_raw[scenarios_raw.index.isin(selected_scenarios_idx)]
scenarios_run = scenarios_raw.loc[pix.ismatch(scenario="*Very Low*")]
scenarios_run


# %%
def transform_iacm_to_openscm_runner_variable(v):
    """Transform IAMC variables to OpenSCM-Runner variables"""
    res = v

    replacements = (
        ("CFC|", ""),
        ("HFC|", ""),
        ("|Montreal Gases", ""),
        (
            "HFC43-10",
            "HFC4310mee",
        ),
        (
            "AFOLU",
            "MAGICC AFOLU",
        ),
        (
            "Energy and Industrial Processes",
            "MAGICC Fossil and Industrial",
        ),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
scenarios_run_openscmrunner = scenarios_run.copy()
scenarios_run_openscmrunner = scenarios_run_openscmrunner.pix.assign(
    variable=scenarios_run_openscmrunner.index.get_level_values("variable")
    .map(transform_iacm_to_openscm_runner_variable)
    .values
)
# Have to interpolate too before passing to OpenSCM-Runner, particularly MAGICC
scenarios_run_openscmrunner = scenarios_run_openscmrunner.T.interpolate("index").T
scenarios_run_openscmrunner


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
MAGICC_FORCE_START_YEAR = 2015
endyear = 2105  # add the 'MAGICC' buffer

RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"

rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable")
    .map(transform_rcmip_to_iamc_variable)
    .map(transform_iacm_to_openscm_runner_variable)
)
ar6_harmonisation_points = rcmip_clean.loc[
    pix.ismatch(mip_era="CMIP6")
    & pix.ismatch(scenario="ssp245")
    & pix.ismatch(region="World")
    & pix.ismatch(variable=scenarios_run_openscmrunner.pix.unique("variable"))
].reset_index(["mip_era", "activity_id"], drop=True)[MAGICC_FORCE_START_YEAR]
with pint.get_application_registry().context("NOx_conversions"):
    ar6_harmonisation_points = ar6_harmonisation_points.pix.convert_unit(
        {"Mt NOx/yr": "Mt NO2/yr", "kt HFC4310mee/yr": "kt HFC4310/yr"}
    )

ar6_harmonisation_points

# %%
# Also have to add AR6 historical values in 2015 and interpolate,
# because we haven't recalibrated MAGICC.
a, b = ar6_harmonisation_points.reset_index(["model", "scenario"], drop=True).align(scenarios_run_openscmrunner)
scenarios_run_openscmrunner = pix.concat([a.to_frame(), b], axis="columns").sort_index(axis="columns")
for y in range(MAGICC_FORCE_START_YEAR, endyear + 1):
    if y not in scenarios_run_openscmrunner:
        scenarios_run_openscmrunner[y] = np.nan

scenarios_run_openscmrunner = scenarios_run_openscmrunner.sort_index(axis="columns").T.interpolate("index").T
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
common_cfg = {
    "startyear": startyear,
    "endyear": endyear,
    "out_dynamic_vars": [
        "DAT_SURFACE_TEMP",
        "DAT_SURFACE_MIXEDLAYERTEMP",
        "DAT_TOTAL_INCLVOLCANIC_ERF",
        "DAT_TOTAL_ANTHRO_ERF",
        "DAT_AEROSOL_ERF",
        "DAT_TOTAER_DIR_ERF",
        "DAT_BCT_ERF",
        "DAT_OCT_ERF",
        "DAT_SOXT_ERF",
        "DAT_CLOUD_TOT_ERF",
        "DAT_GHG_ERF",
        "DAT_CO2_ERF",
        "DAT_CH4_ERF",
        "DAT_N2O_ERF",
        "DAT_FGASSUM_ERF",
        "DAT_MHALOSUM_ERF",
        "DAT_CFC11_ERF",
        "DAT_CFC12_ERF",
        "DAT_HCFC22_ERF",
        "DAT_OZTOTAL_ERF",
        "DAT_HFC125_ERF",
        "DAT_HFC134A_ERF",
        "DAT_HFC143A_ERF",
        "DAT_HFC227EA_ERF",
        "DAT_HFC23_ERF",
        "DAT_HFC245FA_ERF",
        "DAT_HFC32_ERF",
        "DAT_HFC4310_ERFmee",
        "DAT_CF4_ERF",
        "DAT_C6F14_ERF",
        "DAT_C2F6_ERF",
        "DAT_SF6_ERF",
        "DAT_HEATUPTK_EARTH",
        "DAT_CO2_CONC",
        "DAT_CH4_CONC",
        "DAT_N2O_CONC",
        "DAT_CO2_AIR2LAND_FLUX",
        "DAT_CO2_AIR2OCEAN_FLUX",
        "DAT_CO2PF_EMIS",
        "DAT_CH4PF_EMIS",
    ],
    "out_ascii_binary": "BINARY",
    "out_binary_format": 2,
}


# %%
def get_openscm_runner_output_names(magicc_names):
    """
    Get output names for the call to OpenSCM-Runner
    """
    return [
        pymagicc.definitions.convert_magicc7_to_openscm_variables(magiccvarname).replace("DAT_", "")
        for magiccvarname in magicc_names
    ]


# %%
openscm_runner_output_variables = get_openscm_runner_output_names(common_cfg["out_dynamic_vars"])
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
# sorted(magicc_res.pix.unique("variable"))

# %%
temperature_raw = magicc_res.loc[pix.isin(variable="Surface Air Temperature Change")]
temperature_raw

# %%
target_median = 0.85
target_median_years = range(1995, 2014 + 1)
pi_years = range(1850, 1900 + 1)

# %%
temperature_rel_pi = temperature_raw.subtract(temperature_raw.loc[:, pi_years].mean(axis="columns"), axis="rows")
temperature_shift = (
    target_median
    - temperature_rel_pi.loc[:, target_median_years].mean(axis="columns").groupby(["model", "scenario"]).median()
)
temperature_match_historical_assessment = temperature_rel_pi.add(temperature_shift, axis="rows")
temperature_match_historical_assessment.loc[:, target_median_years].mean(axis="columns").groupby(
    ["model", "scenario"]
).median()

# %%
ax = (
    temperature_match_historical_assessment.loc[:, 2000:2100]
    .groupby(["model", "scenario", "region", "variable", "unit"])
    .median()
    .reset_index("region", drop=True)
    .T.plot()
)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.grid()

# %%
cumulative_emms = (
    scenarios_run_openscmrunner.loc[pix.ismatch(variable="**CO2**"), 2020:]
    .groupby(scenarios_run.index.names.difference(["variable"]))
    .sum(min_count=2)
    .reset_index("region", drop=True)
    .T.cumsum()
    * 1.65
    * 12
    / 44
    / 1e6
)

ax = cumulative_emms.plot()
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
# ax.set_ylim([0, 1e6])

# %%
# Really want warming decomposition here
methane_emms = scenarios_run_openscmrunner.loc[pix.ismatch(variable="**CH4"), 2020:].reset_index("region", drop=True).T

ax = methane_emms.plot()
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.set_ylim(ymin=0)

# %%
# Really want warming decomposition here
sulfur_emms = (
    scenarios_run_openscmrunner.loc[pix.ismatch(variable="**Sulfur"), 2020:].reset_index("region", drop=True).T
)

ax = sulfur_emms.plot()
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.set_ylim(ymin=0)

# %%
peak_warming_quantiles = (
    temperature_match_historical_assessment.max(axis="columns")
    .groupby(["model", "scenario"])
    .quantile([0.05, 0.17, 0.33, 0.5, 0.67, 0.83, 0.95])
    .unstack()
    .sort_values(by=0.33)
)
peak_warming_quantiles

# %%
eoc_warming_quantiles = (
    temperature_match_historical_assessment[2100]
    .groupby(["model", "scenario"])
    .quantile([0.05, 0.17, 0.5, 0.83, 0.95])
    .unstack()
    .sort_values(by=0.5)
)
eoc_warming_quantiles

# %%
categories = pd.Series(
    "C8: Above 4.0°C",
    index=peak_warming_quantiles.index,
    name="Category_name",
)

categories[peak_warming_quantiles[0.5] < 4.0] = "C7: Below 4.0°C"  # noqa: PLR2004
categories[peak_warming_quantiles[0.5] < 3.0] = "C6: Below 3.0°C"  # noqa: PLR2004
categories[peak_warming_quantiles[0.5] < 2.5] = "C5: Below 2.5°C"  # noqa: PLR2004
categories[peak_warming_quantiles[0.5] < 2.0] = "C4: Below 2.0°C"  # noqa: PLR2004
categories[peak_warming_quantiles[0.67] < 2.0] = "C3: Likely below 2°C"  # noqa: PLR2004
categories[(peak_warming_quantiles[0.33] > 1.5) & (eoc_warming_quantiles[0.5] < 1.5)] = "C2: Below 1.5°C with high OS"  # noqa: PLR2004
categories[(peak_warming_quantiles[0.33] <= 1.5) & (eoc_warming_quantiles[0.5] < 1.5)] = "C1b: Below 1.5°C with low OS"  # noqa: PLR2004
categories[peak_warming_quantiles[0.5] < 1.5] = "C1a: Below 1.5°C with no OS"  # noqa: PLR2004

categories


# %%
def get_exceedance_probability(indf: pd.DataFrame, warming_level: float) -> float:
    """
    Get exceedance probability

    For exceedance probability over time
    (i.e. at each timestep, rather than at any point in the simulation),
    see `get_exceedance_probability_over_time`
    """
    peaks = indf.max(axis="columns")
    n_above_level = (peaks > warming_level).sum(axis="rows")
    ep = n_above_level / peaks.shape[0] * 100

    return ep


# %%
exceedance_probabilities_l = []
for gwl in [1.5, 2.0, 2.5]:
    gwl_exceedance_probabilities_l = []
    for (model, scenario), msdf in temperature_match_historical_assessment.groupby(["model", "scenario"]):
        ep = get_exceedance_probability(msdf, warming_level=gwl)
        ep_s = pd.Series(
            ep,
            name=f"{gwl:.2f}°C exceedance probability",
            index=pd.MultiIndex.from_tuples(((model, scenario),), names=["model", "scenario"]),
        )
        gwl_exceedance_probabilities_l.append(ep_s)

    exceedance_probabilities_l.append(pix.concat(gwl_exceedance_probabilities_l))

exceedance_probabilities = (
    pix.concat(exceedance_probabilities_l, axis="columns").melt(ignore_index=False).pix.assign(unit="%")
)
exceedance_probabilities = exceedance_probabilities.pivot_table(
    values="value", columns="variable", index=exceedance_probabilities.index.names
).sort_values("1.50°C exceedance probability")
exceedance_probabilities


# %%
def get_exceedance_probability_over_time(indf: pd.DataFrame, warming_level: float) -> pd.Series:
    """
    Get exceedance probability over time

    For exceedance probability at any point in the simulation,
    see `get_exceedance_probability`
    """
    gt_wl = (indf > warming_level).sum(axis="rows")
    ep = gt_wl / indf.shape[0] * 100

    return ep


# %%
exceedance_probabilities_l = []
for gwl in [1.5, 2.0, 2.5]:
    ep = (
        temperature_match_historical_assessment.groupby(
            temperature_match_historical_assessment.index.names.difference(["variable", "unit", "run_id"])
        )
        .apply(get_exceedance_probability_over_time, warming_level=gwl)
        .pix.assign(unit="%", variable=f"{gwl:.2f}°C exceedance probability")
    )

    exceedance_probabilities_l.append(ep)

exceedance_probabilities_over_time = pix.concat(exceedance_probabilities_l)
exceedance_probabilities_over_time

# %%
ax = (
    exceedance_probabilities_over_time.loc[pix.ismatch(variable="1.50*"), 2000:2100]
    .reset_index(["region", "unit"], drop=True)
    .T.plot()
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.grid()
