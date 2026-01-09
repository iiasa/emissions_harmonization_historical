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
# # Prepare FAO agriculture emissions
#
# Prepare data from FAOstat
# that has already been prepared in CEDS-aligned sectors (currently as provided by Steve Smith).
#
# This notebook is used solely to serve the vetting of IAM scenarios
# based on national/regional-level harmonization sectors in the same format as what is produced in CEDS-prepare.py

# %%
# import external packages and functions
import os
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
from pandas_indexing.core import isna

from emissions_harmonization_historical.ceds import add_global, get_map, read_CEDS
from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    FAO_PROCESSING_ID,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# set unit registry
pix.units.set_openscm_registry_as_default()


# %% [markdown]
# Set paths

# %%
fao_data_folder = DATA_ROOT / Path("national", "fao", "data_raw")

fao_csv_files = [f for f in os.listdir(fao_data_folder) if f.endswith(".csv")]

fao_sector_mapping_file = DATA_ROOT / Path("national", "fao", "data_aux", "sector_mapping_fao.xlsx")

fao_processed_output_file_national = DATA_ROOT / Path(
    "national", "fao", "processed", f"fao_national_{FAO_PROCESSING_ID}.csv"
)
fao_processed_output_file_global = DATA_ROOT / Path(
    "national", "fao", "processed", f"fao_global_{FAO_PROCESSING_ID}.csv"
)


# %%
fao_csv_files

# %% [markdown]
# Specify species to processes

# %%
# FAO in this dataset from Steve has CH4 agriculture and N2O agriculture
species = [
    # "BC",
    "CH4",
    # "CO",
    # "CO2",
    "N2O",  # new, to have regional, was global in CMIP6
    # "NH3",
    # "NMVOC",  # assumed to be equivalent to IAMC-style reported VOC
    # "NOx",
    # "OC",
    # "SO2",
]

# %% [markdown]
# Load sector mapping of emissions species

# %%
fao_mapping = pd.read_excel(fao_sector_mapping_file, sheet_name="FAO mapping")
fao_map = get_map(
    fao_mapping, sector_column="FAO_sectors", sector_output_column_name="fao_sectors"
)  # note; no clear description provided by steve
fao_map.to_frame(index=False)

# %% [markdown]
# Read EDGAR emissions data

# %%
fao = pd.concat(
    read_CEDS(Path(fao_data_folder) / f"C.{s}_NC_emissions_agriculture.csv").pix.assign(em=s, units=f"kt{s}")
    for s in species
).rename_axis(index={"iso": "country", "sector_59": "fao_sectors"})
fao = fao.pix.semijoin(fao_map, how="outer")

assert len(fao.loc[isna].pix.unique(["fao_sectors", "sector"])) == 0  # check no sectors are not covered

# %%
# some tests could be added here
# fao.pix.unique(['units'])

# %%
# change units to align with IAM scenario data
# adjust units; change column 'units' to 'unit' and add '/yr'
fao = fao.pix.dropna(subset=["units"]).pix.format(unit="{units}/yr", drop=True)
# adjust units; change all to values to Mt instead of kt
fao = pix.units.convert_unit(fao, lambda x: "Mt " + x.removeprefix("kt").strip())
# exception for N2O/yr, which should remain in kt following https://github.com/IAMconsortium/common-definitions/
fao = pix.units.convert_unit(fao, lambda x: "kt " + x.removeprefix("Mt").strip() if x == "Mt N2O/yr" else x)
# unit of NOx from NOx to NO2
fao.index = pd.MultiIndex.from_tuples(
    [
        (
            country,
            fao_sectors,
            sector_description,
            em,
            sector,
            "Mt NO2/yr" if unit == "Mt NOx/yr" and em == "NOx" else unit,
        )
        for country, fao_sectors, sector_description, em, sector, unit in fao.index
    ],
    names=fao.index.names,
)
# change name(s) of emissions species
# use 'Sulfur' instead of 'SO2'
fao.index = pd.MultiIndex.from_tuples(
    [
        (country, fao_sectors, sector_description, "Sulfur" if em == "SO2" else em, sector, unit)
        for country, fao_sectors, sector_description, em, sector, unit in fao.index
    ],
    names=fao.index.names,
)
fao.index.to_frame(index=False)

# %%
fao = fao.groupby(["em", "country", "unit", "sector"]).sum().pix.fixna()  # group and fix NAs

# %%
# aggregate countries where this is necessary, e.g. because of specific other data (like SSP socioeconomic driver data)
# based on the new SSP data, we only need to aggregate Serbia and Kosovo
country_combinations = {
    # "isr_pse": ["isr", "pse"], "sdn_ssd": ["ssd", "sdn"],
    "srb_ksv": ["srb", "srb (kosovo)"]
}
fao = fao.pix.aggregate(country=country_combinations)

# %%
# add global
fao = add_global(fao)

# %%
# # Rename NMVOC [not in this FAOstat data]
# fao = fao.rename(index=lambda v: v.replace("NMVOC", "VOC"))

# %%
if FAO_PROCESSING_ID == "0010":
    fao = fao.loc[
        :, fao.columns.get_level_values(0) != 2023  # noqa: PLR2004
    ]  # drop year 2023 which does not have complete data for N2O agriculture

# %%
fao_reformatted = fao.rename_axis(index={"em": "variable", "country": "region"})
fao_reformatted

# %%
# rename to IAMC-style variable names including standard index order
fao_reformatted_iamc = (
    fao_reformatted.pix.format(variable="Emissions|{variable}|{sector}", drop=True)
    .pix.assign(scenario=HISTORY_SCENARIO_NAME, model="FAO")
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
).sort_values(by=["region", "variable"])
fao_reformatted_iamc

# %%
fao_reformatted_iamc.pix.unique(["variable"])

# %%
assert_units_match_wishes(fao_reformatted_iamc)

# %% [markdown]
# Save formatted EDGAR data

# %%
out_global = fao_reformatted_iamc.loc[pix.isin(region="World")]  # only the added "World" region
out_national = fao_reformatted_iamc.loc[~pix.isin(region="World")]

# %% [markdown]
# Check that national sums equal global total.

# %%
# Check that `out_national_with_global` totals (all countries in iso3c + CEDS 'global' region)
# ... are the same as `out_global` totals ("World")
national_sums_checker = (
    pix.assignlevel(out_national.groupby(["model", "scenario", "variable", "unit"]).sum(), region="World")
    .reset_index()
    .set_index(out_global.index.names)
)
national_sums_checker.columns = national_sums_checker.columns.astype(int)
national_sums_checker
pd.testing.assert_frame_equal(out_global, national_sums_checker, check_like=True)

# %%
# national CEDS data
fao_processed_output_file_national.parent.mkdir(exist_ok=True, parents=True)
out_national.to_csv(fao_processed_output_file_national)
fao_processed_output_file_national

# %%
# globally aggregated data (all emissions)
fao_processed_output_file_global.parent.mkdir(exist_ok=True, parents=True)
out_global.to_csv(fao_processed_output_file_global)
fao_processed_output_file_global
