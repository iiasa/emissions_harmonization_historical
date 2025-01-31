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
# # Create composite history
#
# Here we create our composite history timeseries,
# based on the various inputs we have.

# %% [markdown]
# ## Imports

# %%
from enum import StrEnum, auto
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import seaborn as sns

from emissions_harmonization_historical.constants import (
    ADAM_ET_AL_2024_PROCESSING_ID,
    CEDS_PROCESSING_ID,
    CMIP_CONCENTRATION_INVERSION_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    EDGAR_PROCESSING_ID,
    GCB_PROCESSING_ID,
    GFED_PROCESSING_ID,
    HISTORY_SCENARIO_NAME,
    VELDERS_ET_AL_2022_PROCESSING_ID,
    WMO_2022_PROCESSING_ID,
)
from emissions_harmonization_historical.io import load_csv
from emissions_harmonization_historical.units import assert_units_match_wishes

# %% [markdown]
# ## Set up the unit registry

# %%
pix.units.set_openscm_registry_as_default()


# %% [markdown]
# ## Config


# %%
class CEDSOption(StrEnum):
    """CEDS options"""

    Zenodo_2024_07_08 = auto()
    # esgf_gridded_yyyy_mm_dd = auto()


# %%
CEDS_SOURCE = CEDSOption.Zenodo_2024_07_08
CEDS_SOURCE


# %%
class BiomassBurningOption(StrEnum):
    """Biomass burning data options"""

    GFED4_1s = auto()
    esgf_gridded_dres_cmip_bb4cmip7_1_0 = auto()


# %%
BIOMASS_BURNING_SOURCE = BiomassBurningOption.GFED4_1s
BIOMASS_BURNING_SOURCE

# %%
combined_processed_output_file = DATA_ROOT / Path(
    "combined-processed-output", f"cmip7_history_{COMBINED_HISTORY_ID}.csv"
)
combined_processed_output_file_world_only = DATA_ROOT / Path(
    "global-composite", f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"
)

# %% [markdown]
# ## Process national data

# %%
if CEDS_SOURCE == CEDSOption.Zenodo_2024_07_08:
    ceds = pix.concat(
        [
            load_csv(
                DATA_ROOT / Path("national", "ceds", "processed", f"ceds_cmip7_national_{CEDS_PROCESSING_ID}.csv")
            ),
            load_csv(DATA_ROOT / Path("national", "ceds", "processed", f"ceds_cmip7_global_{CEDS_PROCESSING_ID}.csv")),
        ]
    )
else:
    raise NotImplementedError(CEDS_SOURCE)

ceds

# %%
if BIOMASS_BURNING_SOURCE == BiomassBurningOption.GFED4_1s:
    biomass_burning = pix.concat(
        [
            load_csv(
                DATA_ROOT / Path("national", "gfed", "processed", f"gfed_cmip7_national_{GFED_PROCESSING_ID}.csv")
            ),
            load_csv(DATA_ROOT / Path("national", "gfed", "processed", f"gfed_cmip7_World_{GFED_PROCESSING_ID}.csv")),
        ]
    )
else:
    raise NotImplementedError(BIOMASS_BURNING_SOURCE)

biomass_burning

# %%
cmip7combined_data = pix.concat([ceds, biomass_burning])
cmip7combined_data

# %%
assert_units_match_wishes(cmip7combined_data)

# %%
combined_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
cmip7combined_data.to_csv(combined_processed_output_file)
combined_processed_output_file

# %% [markdown]
# ## Process global data

# %% [markdown]
# ### Create CEDS-BB4CMIP composite

# %%
ceds_world = ceds.loc[pix.isin(region=["World"])]
ceds_world

# %%
ceds_sum = (
    ceds_world.pix.extract(variable="Emissions|{species}|{sector}", dropna=False)
    .groupby([*(set(ceds_world.index.names) - {"variable"}), "species"])
    .sum()
    .pix.format(variable="Emissions|{species}", drop=True)
    .pix.format(variable="{variable}|{model}", drop=False)
    # CEDS CO2 does not include any AFOLU
    .rename(index=lambda v: v.replace("CO2|", "CO2|Energy and Industrial Processes|"))
)
ceds_sum

# %%
ceds_iteration = ceds_world.pix.unique("model")
if len(ceds_iteration) > 1:
    raise AssertionError(ceds_iteration)

ceds_iteration = ceds_iteration[0]

# %%
ceds_co2 = ceds_sum.loc[pix.ismatch(variable="Emissions|CO2|Energy and Industrial Processes|*")]
ceds_co2 = ceds_co2.rename(index=lambda x: x.replace(f"|{ceds_iteration}", ""))
ceds_co2

# %%
biomass_burning_world = biomass_burning.loc[pix.isin(region=["World"])]
biomass_burning_world

# %%
biomass_burning_sum = (
    biomass_burning_world.pix.extract(variable="Emissions|{species}|{sector}", dropna=False)
    .groupby([*(set(biomass_burning_world.index.names) - {"variable"}), "species"])
    .sum()
    .pix.format(variable="Emissions|{species}", drop=True)
    .pix.format(variable="{variable}|{model}", drop=False)
    # Use CO2 AFOLU from GCB
    .loc[~pix.ismatch(variable="Emissions|CO2|**")]
)
biomass_burning_sum

# %%
ceds_biomass_burning_composite = (
    pix.concat(
        [
            # Keep CEDS CO2 separate
            ceds_sum.loc[~pix.ismatch(variable="Emissions|CO2**")],
            biomass_burning_sum,
        ]
    )
    .pix.extract(variable="Emissions|{species}|{source}")
    .pix.assign(model="CEDS-BB4CMIP")
    .groupby([*(set(ceds_sum.index.names) - {"variable"}), "species"])
    .sum(min_count=2)  # will give weird output if any units differ
    .pix.format(variable="Emissions|{species}", drop=True)
)
ceds_biomass_burning_composite

# %% [markdown]
# ### Create our global composite

# %%
adam_et_al_2024 = load_csv(
    DATA_ROOT
    / "global"
    / "adam-et-al-2024"
    / "processed"
    / f"adam-et-al-2024_cmip7_global_{ADAM_ET_AL_2024_PROCESSING_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
adam_et_al_2024

# %%
edgar = load_csv(
    DATA_ROOT / "global" / "edgar" / "processed" / f"edgar_cmip7_global_{EDGAR_PROCESSING_ID}.csv",
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
edgar

# %%
gcb_afolu = load_csv(
    DATA_ROOT / "global" / "gcb" / "processed" / f"gcb-afolu_cmip7_global_{GCB_PROCESSING_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
gcb_afolu

# %%
velders_et_al_2022 = load_csv(
    DATA_ROOT
    / "global"
    / "velders-et-al-2022"
    / "processed"
    / f"velders-et-al-2022_cmip7_global_{VELDERS_ET_AL_2022_PROCESSING_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
velders_et_al_2022

# %%
wmo_2022 = load_csv(
    DATA_ROOT / "global" / "wmo-2022" / "processed" / f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
wmo_2022

# %%
cmip_inversions = load_csv(
    DATA_ROOT / "global" / "esgf" / "CR-CMIP-0-4-0" / f"inverse_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.csv"
).pix.assign(scenario=HISTORY_SCENARIO_NAME)

# %%
all_sources = pix.concat(
    [
        adam_et_al_2024,
        ceds_biomass_burning_composite,
        ceds_co2,
        cmip_inversions,
        edgar,
        gcb_afolu,
        velders_et_al_2022,
        wmo_2022,
    ]
)
all_sources

# %%
wmo_2022.pix.unique(["model", "variable"]).to_frame(index=False)

# %%
global_variable_sources = {
    "Emissions|BC": "CEDS-BB4CMIP",
    "Emissions|CF4": "WMO 2022 AGAGE inversions",
    "Emissions|C2F6": "WMO 2022 AGAGE inversions",
    "Emissions|C3F8": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|cC4F8": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|C4F10": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|C5F12": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|C7F16": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|C8F18": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|C6F14": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|CH4": "CEDS-BB4CMIP",
    "Emissions|CO": "CEDS-BB4CMIP",
    "Emissions|CO2|AFOLU": "Global Carbon Budget",
    "Emissions|CO2|Energy and Industrial Processes": ceds_iteration,
    "Emissions|HFC|HFC125": "Velders et al., 2022",
    "Emissions|HFC|HFC134a": "Velders et al., 2022",
    "Emissions|HFC|HFC143a": "Velders et al., 2022",
    "Emissions|HFC|HFC152a": "Velders et al., 2022",
    "Emissions|HFC|HFC227ea": "Velders et al., 2022",
    "Emissions|HFC|HFC23": "Adam et al., 2024",
    "Emissions|HFC|HFC236fa": "Velders et al., 2022",
    "Emissions|HFC|HFC245fa": "Velders et al., 2022",
    "Emissions|HFC|HFC32": "Velders et al., 2022",
    "Emissions|HFC|HFC365mfc": "Velders et al., 2022",
    "Emissions|HFC|HFC43-10": "Velders et al., 2022",
    "Emissions|Montreal Gases|CCl4": "WMO 2022",
    "Emissions|Montreal Gases|CFC|CFC11": "WMO 2022",
    "Emissions|Montreal Gases|CFC|CFC113": "WMO 2022",
    "Emissions|Montreal Gases|CFC|CFC114": "WMO 2022",
    "Emissions|Montreal Gases|CFC|CFC115": "WMO 2022",
    "Emissions|Montreal Gases|CFC|CFC12": "WMO 2022",
    "Emissions|Montreal Gases|CH2Cl2": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|Montreal Gases|CH3Br": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|Montreal Gases|CH3CCl3": "WMO 2022",
    "Emissions|Montreal Gases|CH3Cl": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|Montreal Gases|CHCl3": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|Montreal Gases|HCFC141b": "WMO 2022",
    "Emissions|Montreal Gases|HCFC142b": "WMO 2022",
    "Emissions|Montreal Gases|HCFC22": "WMO 2022",
    "Emissions|Montreal Gases|Halon1202": "WMO 2022",
    "Emissions|Montreal Gases|Halon1211": "WMO 2022",
    "Emissions|Montreal Gases|Halon1301": "WMO 2022",
    "Emissions|Montreal Gases|Halon2402": "WMO 2022",
    "Emissions|N2O": "CEDS-BB4CMIP",
    "Emissions|NF3": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|NH3": "CEDS-BB4CMIP",
    "Emissions|NOx": "CEDS-BB4CMIP",
    "Emissions|OC": "CEDS-BB4CMIP",
    "Emissions|SF6": "WMO 2022 AGAGE inversions",
    "Emissions|SO2F2": "CR-CMIP-0-4-0-inverse-smooth",
    "Emissions|Sulfur": "CEDS-BB4CMIP",
    "Emissions|VOC": "CEDS-BB4CMIP",
}

# %%
global_composite_l = []
for variable, source in global_variable_sources.items():
    source_loc = pix.isin(variable=[variable]) & pix.isin(model=[source])
    to_keep = all_sources.loc[source_loc]
    if to_keep.empty:
        msg = f"{variable} not available from {source}"
        raise AssertionError(msg)

    global_composite_l.append(to_keep)

global_composite = pix.concat(global_composite_l).sort_index().sort_index(axis="columns")
global_composite

# %%
assert_units_match_wishes(global_composite)

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_PATH


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
rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
ar6_history = rcmip_clean.loc[pix.isin(mip_era=["CMIP6"], scenario=["ssp245"], region=["World"])]
ar6_history = (
    ar6_history.loc[
        ~pix.ismatch(
            variable=[
                f"Emissions|{stub}|**" for stub in ["BC", "CH4", "CO", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC"]
            ]
        )
        & ~pix.ismatch(variable=["Emissions|CO2|*|**"])
        & ~pix.isin(variable=["Emissions|CO2"])
    ]
    .T.interpolate("index")
    .T
)
full_var_set = ar6_history.pix.unique("variable")
n_variables_in_full_scenario = 52
if len(full_var_set) != n_variables_in_full_scenario:
    raise AssertionError

# %%
tmp = pix.concat(
    [
        global_composite,
        ar6_history.reset_index(["activity_id", "mip_era"], drop=True),
    ]
).loc[:, 1900:2025]
pdf = tmp.melt(ignore_index=False, var_name="year").reset_index()

sns.relplot(
    data=pdf,
    x="year",
    y="value",
    hue="model",
    style="scenario",
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
    kind="line",
    alpha=0.5,
)

# %%
combined_processed_output_file_world_only.parent.mkdir(exist_ok=True, parents=True)
global_composite.to_csv(combined_processed_output_file_world_only)
combined_processed_output_file_world_only
