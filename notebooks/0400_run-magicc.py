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

# %% [markdown]
# ## Imports

# %%
import multiprocessing
import os

import numpy as np
import pandas as pd
import pandas_indexing as pix
import platform
import pint
from gcages.ar6 import AR6PostProcessor, AR6SCMRunner
from gcages.ar6.post_processing import get_temperatures_in_line_with_assessment
from gcages.database import GCDB

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    INFILLING_LEFTOVERS_ID,
    INFILLING_SILICONE_ID,
    INFILLING_WMO_ID,
    MAGICC_RUN_ID,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.io import load_csv

# %% [markdown]
# ## Setup

# %%
pix.set_openscm_registry_as_default()

# %% [markdown]
# ## Define inputs

# %%
complete_scenarios_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "infilled"
    / f"infilled_{SCENARIO_TIME_ID}_{HARMONISATION_ID}_{INFILLING_SILICONE_ID}_{INFILLING_WMO_ID}_{INFILLING_LEFTOVERS_ID}.csv"  # noqa: E501
)
complete_scenarios_file

# %% [markdown]
# ## Set up runner

# %%
# AR6-like
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


scm_runner = AR6SCMRunner.from_ar6_like_config(
    magicc_exe_path=DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "bin" / magicc_exe,
    magicc_prob_distribution_path=DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json",
    db=GCDB(
        DATA_ROOT
        / "climate-assessment-workflow"
        / "scm-output"
        / "ar6-like-magicc"
        / f"{SCENARIO_TIME_ID}_{HARMONISATION_ID}_{INFILLING_SILICONE_ID}_{INFILLING_WMO_ID}_{INFILLING_LEFTOVERS_ID}_{MAGICC_RUN_ID}"  # noqa: E501
    ),
    run_checks=False,  # TODO: turn on
    n_processes=multiprocessing.cpu_count(),
)

# Depends on use case, but this is fine for now
scm_runner.output_variables = ("Surface Air Temperature Change",)
scm_runner

# %%
# magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
# magicc_expected_version = "v7.6.0a3"
# probabilistic_distribution_file = (
#     DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
# )
# startyear = 1750
# endyear = 2100
# output_variables = ("Surface Air Temperature Change",)
# out_dynamic_vars = [f"DAT_{v}" for v in convert_openscm_runner_output_names_to_magicc_output_names(output_variables)]


# os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
# if openscm_runner.adapters.MAGICC7.get_version() != magicc_expected_version:
#     raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())


# with open(probabilistic_distribution_file) as fh:
#     cfgs_raw = json.load(fh)

# base_cfgs = [
#     {
#         "run_id": c["paraset_id"],
#         **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
#     }
#     for c in cfgs_raw["configurations"]
# ]


# common_cfg = {
#     "startyear": startyear,
#     "endyear": endyear,
#     "out_dynamic_vars": out_dynamic_vars,
#     "out_ascii_binary": "BINARY",
#     "out_binary_format": 2,
# }


# run_config = [
#     {
#         **common_cfg,
#         **base_cfg,
#     }
#     for base_cfg in base_cfgs
# ]

# scm_runner = AR6SCMRunner(
#     climate_models_cfgs={"MAGICC7": run_config},
#     output_variables=output_variables,
#     force_interpolate_to_yearly=True,
#     db=GCDB(
#         DATA_ROOT
#         / "climate-assessment-workflow"
#         / "scm-output"
#         / "ar7-ft-magicc"
#         / f"{SCENARIO_TIME_ID}_{HARMONISATION_ID}_{INFILLING_SILICONE_ID}_{INFILLING_WMO_ID}_{INFILLING_LEFTOVERS_ID}_{magicc_expected_version.replace('.', '-')}_{probabilistic_distribution_file.stem}"  # noqa: E501
#     ),
#     run_checks=False,  # TODO: turn on
#     n_processes=multiprocessing.cpu_count(),
# )

# # Depends on use case, but this is fine for now
# scm_runner.output_variables=("Surface Air Temperature Change",)
# scm_runner

# %% [markdown]
# ## Set up post-processor

# %%
post_processor = AR6PostProcessor.from_ar6_like_config(
    run_checks=False,  # TODO: turn back on
)
post_processor

# %% [markdown]
# ## Load scenarios

# %%
scenarios_raw = load_csv(complete_scenarios_file)
# TODO: remove once we have data post 2100
scenarios_raw = scenarios_raw.loc[:, :2100]
scenarios_raw

# %% [markdown]
# ### Sub-select scenarios for testing
#
# Optional section

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
# selected_scenarios_idx = pd.MultiIndex.from_tuples(
#     (
#         ('MESSAGEix-GLOBIOM 2.1-M-R12', 'SSP5 - High Emissions'),
#         ('IMAGE 3.4', 'SSP5 - High Emissions'),
#         ('AIM 3.0', 'SSP2 - Medium-Low Emissions'),
#         ('WITCH 6.0', 'SSP2 - Low Emissions'),
#         ('REMIND-MAgPIE 3.4-4.8', 'SSP2 - Low Overshoot_b'),
#         ('MESSAGEix-GLOBIOM-GAINS 2.1-M-R12', 'SSP5 - Low Overshoot'),
#         ('COFFEE 1.5', 'SSP2 - Medium Emissions'),
#         ('GCAM 7.1 scenarioMIP', 'SSP2 - Medium Emissions'),
#         ("IMAGE 3.4", "SSP2 - Very Low Emissions"),
#         ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions"),
#     ),
#     name=["model", "scenario"],
# )
# scenarios_run = scenarios_raw[scenarios_raw.index.isin(selected_scenarios_idx)]

scenarios_run = scenarios_raw.loc[pix.ismatch(scenario="*Very Low*")]

scenarios_run.pix.unique(["model", "scenario"]).to_frame(index=False)


# %% [markdown]
# ### Hack in values from 2015, as that is what MAGICCv7 needs
#
# Obviously delete if not running MAGICCv7.


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

if scenarios_run.isnull().any().any():
    if not scm_runner.force_interpolate_to_yearly:
        raise AssertionError

scenarios_run

# %% [markdown]
# ## Run

# %%
res_full = scm_runner(scenarios_run, batch_size_scenarios=5)
res_full

# %%
post_processed = post_processor(res_full)
post_processed

# %%
temperature_match_historical_assessment = get_temperatures_in_line_with_assessment(
    res_full.loc[pix.isin(variable=["AR6 climate diagnostics|Raw Surface Temperature (GSAT)"])],
    assessment_median=post_processor.gsat_assessment_median,
    assessment_time_period=post_processor.gsat_assessment_time_period,
    assessment_pre_industrial_period=post_processor.gsat_assessment_pre_industrial_period,
).pix.assign(variable="AR6 climate diagnostics|Surface Temperature (GSAT)")

temperature_match_historical_assessment

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
scenarios_plt = scenarios_run.copy()
for y in range(scenarios_plt.columns.min(), scenarios_plt.columns.max()):
    if y not in scenarios_plt:
        scenarios_plt[y] = np.nan

scenarios_plt = scenarios_plt.sort_index(axis="columns").T.interpolate("index").T
scenarios_plt

# %%
cumulative_emms = (
    scenarios_plt.loc[pix.ismatch(variable="**CO2**"), 2020:]
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
methane_emms = scenarios_plt.loc[pix.ismatch(variable="**CH4"), 2020:].reset_index("region", drop=True).T

ax = methane_emms.plot()
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.set_ylim(ymin=0)

# %%
# Really want warming decomposition here
sulfur_emms = scenarios_plt.loc[pix.ismatch(variable="**Sulfur"), 2020:].reset_index("region", drop=True).T

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
category_join = (
    post_processed.index.to_frame(index=False)[["model", "scenario", "category"]]
    .drop_duplicates()
    .sort_values("category")
    .set_index(["model", "scenario"])
)
peak_warming_quantiles_join = peak_warming_quantiles[[0.33, 0.5, 0.67]].copy().round(3)
peak_warming_quantiles_join.columns = peak_warming_quantiles_join.columns.map(lambda x: f"Peak {x * 100:.1f}th")

eoc_quantiles_join = eoc_warming_quantiles[[0.5]].copy().round(3)
eoc_quantiles_join.columns = eoc_quantiles_join.columns.map(lambda x: f"2100 {x * 100:.1f}th")

pd.concat(
    [
        category_join,
        peak_warming_quantiles_join,
        eoc_quantiles_join,
    ],
    axis="columns",
).sort_values(["category", "Peak 50.0th"])

# %%
category_join.groupby(["model"]).value_counts().sort_index()


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
        .pix.assign(unit="%", variable=f"{gwl:.2f}°C exceedancse probability")
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
