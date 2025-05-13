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
# # Compile historical emissions for the global workflow
#
# Here we compile the historical emissions needed for the global workflow.

# %% [markdown]
# ## Imports

# %%
from functools import partial

import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    ADAM_ET_AL_2024_PROCESSING_ID,
    CMIP_CONCENTRATION_INVERSION_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    GCB_PROCESSING_ID,
    HISTORY_SCENARIO_NAME,
    IAMC_REGION_PROCESSING_ID,
    VELDERS_ET_AL_2022_PROCESSING_ID,
    WMO_2022_PROCESSING_ID,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
HARMONISATION_GRIDDING_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"gridding-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
# HARMONISATION_GRIDDING_EMISSIONS_FILE

# %%
OUT_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"global-workflow-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_PATH

# %% [markdown]
# ## Load data

# %%
gridding_harmonisation_emissions = load_timeseries_csv(
    HARMONISATION_GRIDDING_EMISSIONS_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if gridding_harmonisation_emissions.empty:
    raise AssertionError

gridding_harmonisation_emissions.columns.name = "year"
# gridding_harmonisation_emissions

# %% [markdown]
# ## Process

# %% [markdown]
# ## Global workflow emissions
#
# This replaces notebook 0201 because it ensures consistency
# between smoothing of biomass burning at the regional and global level.

# %% [markdown]
# To check: are the sums the same for all IAM model regions?
# The reason this is worth checking is that, if the IAM is missing a country,
# then you could easily be harmonising to global totals which are effectively different.

# %%
# For now, assume the choice of IAM regions doesn't mater
model_regions = [r for r in gridding_harmonisation_emissions.index.get_level_values("region").unique() if "REMIND" in r]
# model_regions
global_workflow_harmonisation_emissions_from_gridding = to_global_workflow_emissions(
    gridding_harmonisation_emissions.loc[pix.isin(region=["World", *model_regions])]
).pix.assign(scenario="history", model="gridding-emissions")

# %%
load_csv = partial(
    load_timeseries_csv,
    index_columns=["model", "scenario", "variable", "region", "unit"],
    out_column_type=int,
)

# %%
adam_et_al_2024 = load_csv(
    DATA_ROOT
    / "global"
    / "adam-et-al-2024"
    / "processed"
    / f"adam-et-al-2024_cmip7_global_{ADAM_ET_AL_2024_PROCESSING_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
adam_et_al_2024 = update_index_levels_func(
    adam_et_al_2024,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.IAMC,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    },
)

# adam_et_al_2024

# %%
# To create these, run notebook "0110_cmip-conc-inversions"
cmip_inversions = load_csv(
    DATA_ROOT / "global" / "esgf" / "CR-CMIP-1-0-0" / f"inverse_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)


def fix_variable_name(v: str) -> str:
    """Fix the variable name given the naming mess we have"""
    if v in [
        "Emissions|cC4F8",
        "Emissions|CF4",
        "Emissions|C2F6",
        "Emissions|C3F8",
        "Emissions|C4F10",
        "Emissions|C5F12",
        "Emissions|C6F14",
        "Emissions|C7F16",
        "Emissions|C8F18",
    ]:
        toks = v.split("|")
        v = "|".join([toks[0], "PFC", toks[1]])
    if v in [
        "Emissions|Montreal Gases|CFC|CFC11",
        "Emissions|Montreal Gases|CFC|CFC12",
        "Emissions|Montreal Gases|CFC|CFC113",
        "Emissions|Montreal Gases|CFC|CFC114",
        "Emissions|Montreal Gases|CFC|CFC115",
    ]:
        v = v.replace("Montreal Gases|", "").replace("CFC|", "")

    v = v.replace("Montreal Gases|", "")

    v = v.replace("Emissions|HFC|HFC245fa", "Emissions|HFC|HFC245ca")

    return convert_variable_name(
        v,
        from_convention=SupportedNamingConventions.IAMC,
        to_convention=SupportedNamingConventions.GCAGES,
    )


cmip_inversions = update_index_levels_func(
    cmip_inversions,
    {"variable": fix_variable_name},
)
# cmip_inversions

# %%
gcb_afolu = load_csv(
    DATA_ROOT / "global" / "gcb" / "processed" / f"gcb-afolu_cmip7_global_{GCB_PROCESSING_ID}.csv",
).pix.assign(scenario="history")
gcb_afolu = update_index_levels_func(
    gcb_afolu,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.IAMC,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    },
)
# gcb_afolu

# %%
velders_et_al_2022 = load_csv(
    DATA_ROOT
    / "global"
    / "velders-et-al-2022"
    / "processed"
    / f"velders-et-al-2022_cmip7_global_{VELDERS_ET_AL_2022_PROCESSING_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
velders_et_al_2022 = update_index_levels_func(
    velders_et_al_2022,
    {
        "variable": lambda x: convert_variable_name(
            x.replace("HFC245fa", "HFC245ca"),
            from_convention=SupportedNamingConventions.IAMC,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    },
)

# velders_et_al_2022

# %%
wmo_2022 = load_csv(
    DATA_ROOT / "global" / "wmo-2022" / "processed" / f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)


def fix_variable_name(v: str) -> str:
    """Fix the variable name given the naming mess we have"""
    if v in [
        "Emissions|cC4F8",
        "Emissions|CF4",
        "Emissions|C2F6",
        "Emissions|C3F8",
        "Emissions|C4F10",
        "Emissions|C5F12",
        "Emissions|C6F14",
        "Emissions|C7F16",
        "Emissions|C8F18",
    ]:
        toks = v.split("|")
        v = "|".join([toks[0], "PFC", toks[1]])
    if v in [
        "Emissions|Montreal Gases|CFC|CFC11",
        "Emissions|Montreal Gases|CFC|CFC12",
        "Emissions|Montreal Gases|CFC|CFC113",
        "Emissions|Montreal Gases|CFC|CFC114",
        "Emissions|Montreal Gases|CFC|CFC115",
    ]:
        v = v.replace("Montreal Gases|", "").replace("CFC|", "")

    v = v.replace("Montreal Gases|", "")

    v = v.replace("Emissions|HFC|HFC245fa", "Emissions|HFC|HFC245ca")

    return convert_variable_name(
        v,
        from_convention=SupportedNamingConventions.IAMC,
        to_convention=SupportedNamingConventions.GCAGES,
    )


wmo_2022 = update_index_levels_func(
    wmo_2022,
    {"variable": fix_variable_name},
)

# wmo_2022

# %%
all_sources = pix.concat(
    [
        global_workflow_harmonisation_emissions_from_gridding,
        adam_et_al_2024,
        cmip_inversions,
        gcb_afolu,
        velders_et_al_2022,
        wmo_2022,
    ]
)
# all_sources

# %%
global_variable_sources = {
    "Emissions|BC": "gridding-emissions",
    "Emissions|CF4": "WMO 2022 AGAGE inversions",
    "Emissions|C2F6": "WMO 2022 AGAGE inversions",
    "Emissions|C3F8": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|cC4F8": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|C4F10": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|C5F12": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|C7F16": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|C8F18": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|C6F14": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|CH4": "gridding-emissions",
    "Emissions|CO": "gridding-emissions",
    "Emissions|CO2|Biosphere": "Global Carbon Budget",
    "Emissions|CO2|Fossil": "gridding-emissions",
    "Emissions|HFC125": "Velders et al., 2022",
    "Emissions|HFC134a": "Velders et al., 2022",
    "Emissions|HFC143a": "Velders et al., 2022",
    "Emissions|HFC152a": "Velders et al., 2022",
    "Emissions|HFC227ea": "Velders et al., 2022",
    "Emissions|HFC23": "Adam et al., 2024",
    "Emissions|HFC236fa": "Velders et al., 2022",
    "Emissions|HFC245fa": "Velders et al., 2022",
    "Emissions|HFC32": "Velders et al., 2022",
    "Emissions|HFC365mfc": "Velders et al., 2022",
    "Emissions|HFC4310mee": "Velders et al., 2022",
    "Emissions|CCl4": "WMO 2022",
    "Emissions|CFC11": "WMO 2022",
    "Emissions|CFC113": "WMO 2022",
    "Emissions|CFC114": "WMO 2022",
    "Emissions|CFC115": "WMO 2022",
    "Emissions|CFC12": "WMO 2022",
    "Emissions|CH2Cl2": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|CH3Br": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|CH3CCl3": "WMO 2022",
    "Emissions|CH3Cl": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|CHCl3": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|HCFC141b": "WMO 2022",
    "Emissions|HCFC142b": "WMO 2022",
    "Emissions|HCFC22": "WMO 2022",
    "Emissions|Halon1202": "WMO 2022",
    "Emissions|Halon1211": "WMO 2022",
    "Emissions|Halon1301": "WMO 2022",
    "Emissions|Halon2402": "WMO 2022",
    "Emissions|N2O": "gridding-emissions",
    "Emissions|NF3": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|NH3": "gridding-emissions",
    "Emissions|NOx": "gridding-emissions",
    "Emissions|OC": "gridding-emissions",
    "Emissions|SF6": "WMO 2022 AGAGE inversions",
    "Emissions|SO2F2": "CR-CMIP-1-0-0-inverse-smooth",
    "Emissions|Sulfur": "gridding-emissions",
    "Emissions|VOC": "gridding-emissions",
}

# %%
global_workflow_harmonisation_emissions_l = []
for variable, source in global_variable_sources.items():
    source_loc = pix.isin(variable=[variable]) & pix.isin(model=[source])
    to_keep = all_sources.loc[source_loc]
    if to_keep.empty:
        msg = f"{variable} not available from {source}"
        raise AssertionError(msg)

    global_workflow_harmonisation_emissions_l.append(to_keep)

global_workflow_harmonisation_emissions = (
    pix.concat(global_workflow_harmonisation_emissions_l).sort_index().sort_index(axis="columns")
)


def update_variable_name(v: str) -> str:
    """Fix the variable name given the naming mess we have"""
    v = v.replace("Sulfur", "SOx").replace("VOC", "NMVOC")

    return convert_variable_name(
        v,
        from_convention=SupportedNamingConventions.GCAGES,
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )


global_workflow_harmonisation_emissions = update_index_levels_func(
    global_workflow_harmonisation_emissions, {"variable": update_variable_name}
)
exp_n_timeseries = 52
if global_workflow_harmonisation_emissions.shape[0] != exp_n_timeseries:
    raise AssertionError

# global_workflow_harmonisation_emissions

# %%
global_workflow_harmonisation_emissions_rcmip_like = update_index_levels_func(
    global_workflow_harmonisation_emissions,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.RCMIP,
        )
    },
)

# %%
rcmip = load_timeseries_csv(
    RCMIP_PATH,
    index_columns=["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"],
    out_column_type=int,
)
ssp245 = rcmip.loc[
    pix.isin(
        scenario="ssp245",
        region="World",
        variable=global_workflow_harmonisation_emissions_rcmip_like.pix.unique("variable"),
    )
]
ssp245 = ssp245.T.interpolate(method="index").T

full_var_set = ssp245.pix.unique("variable")
n_variables_in_full_scenario = 52
if len(ssp245) != n_variables_in_full_scenario:
    raise AssertionError

# ssp245

# %%
tmp = pix.concat(
    [
        global_workflow_harmonisation_emissions_rcmip_like.pix.assign(scenario="CMIP7 ScenarioMIP"),
        ssp245.reset_index(["activity_id", "mip_era"], drop=True),
    ]
).loc[:, 2000:2025]
pdf = tmp.melt(ignore_index=False, var_name="year").reset_index()

fg = sns.relplot(
    data=pdf,
    x="year",
    y="value",
    hue="model",
    style="scenario",
    col="variable",
    col_wrap=3,
    col_order=sorted(pdf["variable"].unique()),
    facet_kws=dict(sharey=False),
    kind="line",
    alpha=0.5,
)

for ax in fg.figure.axes:
    ax.set_ylim(0)

# %% [markdown]
# ## Combine and save

# %%
global_workflow_harmonisation_emissions.to_csv(OUT_FILE)
OUT_FILE
