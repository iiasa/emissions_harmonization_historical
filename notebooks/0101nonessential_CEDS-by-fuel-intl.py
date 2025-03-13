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
# # Additional separate processing file: extract CEDS emissions by fuel for international aviation and international shipping
#
# Prepare data from [CEDS](https://github.com/JGCRI/CEDS).

# %%
# import external packages and functions
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_indexing as pix
from pandas_indexing.core import isna

from emissions_harmonization_historical.ceds import add_global, get_map, read_CEDS
from emissions_harmonization_historical.constants import (
    CEDS_EXPECTED_NUMBER_OF_REGION_VARIABLE_PAIRS_IN_GLOBAL_HARMONIZATION,
    CEDS_PROCESSING_ID,
    DATA_ROOT,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# set unit registry
pix.units.set_openscm_registry_as_default()


# %% [markdown]
# Set paths

# %%
ceds_release = "2024_07_08"
ceds_data_folder = DATA_ROOT / Path("national", "ceds", "data_raw", "global_with_fuel_information")
ceds_sector_mapping_file = DATA_ROOT / Path("national", "ceds", "data_aux", "sector_mapping.xlsx")
ceds_processed_output_file_global = DATA_ROOT / Path(
    "global", "ceds", "processed", f"ceds_cmip7_intlAviationShipping_{CEDS_PROCESSING_ID}.csv"
)

# %% [markdown]
# Specify species to processes

# %%
# use all species covered in CEDS
species = [
    "BC",
    "CH4",
    "CO",
    "CO2",
    "N2O",  # new, to have regional, was global in CMIP6
    "NH3",
    "NMVOC",  # assumed to be equivalent to IAMC-style reported VOC
    "NOx",
    "OC",
    "SO2",
]
# select sectors that we want to keep
sectors = [
    "1A3ai_International-aviation",
    "1A3aii_Domestic-aviation",
    "1A3di_International-shipping"
]

# %% [markdown]
# Load sector mapping of emissions species

# %%
ceds_mapping = pd.read_excel(ceds_sector_mapping_file, sheet_name="CEDS Mapping 2024")
ceds_map = get_map(ceds_mapping, "59_Sectors_2024")  # note; with 7BC now added it is actually 60 sectors, not 59?!
ceds_map.to_frame(index=False)

# %% [markdown]
# Read CEDS emissions data

# %%
ceds = pd.concat(
    read_CEDS(Path(ceds_data_folder) / f"{s}_CEDS_global_emissions_by_sector_fuel_v{ceds_release}.csv") for s in species
)
ceds = pix.assignlevel(ceds, region="World")
ceds = ceds.pix.semijoin(ceds_map, how="outer")
ceds.loc[isna].pix.unique(["sector_59", "sector"])  # print sectors with NAs

# %%
# '6B_Other-not-in-total' is not assigned, and normally not used by CEDS. To be certain that we notice it when something
# ... changes, we check that it is indeed zero, such that we are not missing anything.
year_cols = ceds.columns.astype(int)
first_year = year_cols[0]
last_year = year_cols[-1]
sum_of_6B_other = (
    ceds.loc[pix.ismatch(sector_59="6B_Other-not-in-total")]
    .sum(axis=1)  # sum across years
    .sum(axis=0)  # sum across species and countries
)
assert sum_of_6B_other == 0

# %%
ceds

# %%
# change units to align with IAM scenario data
# adjust units; change column 'units' to 'unit' and add '/yr'
ceds = ceds.pix.dropna(subset=["units"]).pix.format(unit="{units}/yr", drop=True)
# adjust units; change all to values to Mt instead of kt
ceds = pix.units.convert_unit(ceds, lambda x: "Mt " + x.removeprefix("kt").strip())
# exception for N2O/yr, which should remain in kt following https://github.com/IAMconsortium/common-definitions/
ceds = pix.units.convert_unit(ceds, lambda x: "kt " + x.removeprefix("Mt").strip() if x == "Mt N2O/yr" else x)
# unit of BC from C to BC
ceds.index = pd.MultiIndex.from_tuples(
    [
        (em, region, fuel, sector_59, sector, "Mt BC/yr" if unit == "Mt C/yr" and em == "BC" else unit)
        for em, region, fuel, sector_59, sector, unit in ceds.index
    ],
    names=ceds.index.names,
)
# unit of OC from C to OC
ceds.index = pd.MultiIndex.from_tuples(
    [
        (em, region, fuel, sector_59, sector, "Mt OC/yr" if unit == "Mt C/yr" and em == "OC" else unit)
        for em, region, fuel, sector_59, sector, unit in ceds.index
    ],
    names=ceds.index.names,
)
# change name(s) of emissions species
# use 'Sulfur' instead of 'SO2'
ceds.index = pd.MultiIndex.from_tuples(
    [
        ("Sulfur" if em == "SO2" else em, region, fuel, sector_59, sector, unit)
        for em, region, fuel, sector_59, sector, unit in ceds.index
    ],
    names=ceds.index.names,
)
ceds

# %% [markdown]
# Filter only desired sectors

# %%
ceds = ceds.loc[pix.isin(sector_59=sectors)]
ceds

# %% [markdown]
# Other fixes

# %%
ceds = ceds.groupby(["region", "em", "unit", "sector", "fuel"]).sum().pix.fixna()  # group and fix NAs
ceds

# %%
# Rename NMVOC
ceds = ceds.rename(index=lambda v: v.replace("NMVOC", "VOC"))

# %%
ceds_reformatted = ceds.rename_axis(index={"em": "variable"})
ceds_reformatted

# %%
# rename to IAMC-style variable names including standard index order
ceds_reformatted_iamc = (
    ceds_reformatted.pix.format(variable="Emissions|{variable}|{sector}|{fuel}", drop=True)
    .pix.assign(scenario=HISTORY_SCENARIO_NAME, model=f"CEDSv{ceds_release}")
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
).sort_values(by=["variable"])
ceds_reformatted_iamc

# %%
assert_units_match_wishes(ceds_reformatted_iamc)

# %% [markdown]
# Save formatted CEDS data

# %%
out_global = ceds_reformatted_iamc.loc[pix.isin(region="World")]  # only the added "World" region

# %%
# international only CEDS data (aircraft and international shipping)
ceds_processed_output_file_global.parent.mkdir(exist_ok=True, parents=True)
out_global.to_csv(ceds_processed_output_file_global)
ceds_processed_output_file_global
