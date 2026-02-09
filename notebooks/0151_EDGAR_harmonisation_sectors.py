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
# # Prepare EDGAR
#
# Prepare data from EDGAR that has already been prepared in CEDS-aligned sectors
# (currently as provided by Steve Smith).
#
# We also have 0109_edgar-prepare.py, which is used especially for minor gases (F-gases),
# on the global level, for climate emulator input.
#
# Instead, this notebook is used to serve the vetting of IAM scenarios based on national
# level harmonization sectors in the same format as what is produced in CEDS-prepare.py

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
    EDGAR_PROCESSING_ID,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# set unit registry
pix.units.set_openscm_registry_as_default()


# %% [markdown]
# Set paths

# %%
edgar_data_folder = DATA_ROOT / Path("national", "edgar", "data_raw")

edgar_csv_files = [f for f in os.listdir(edgar_data_folder) if f.endswith(".csv")]

edgar_sector_mapping_file = DATA_ROOT / Path("national", "edgar", "data_aux", "sector_mapping_edgar.xlsx")

edgar_processed_output_file_national = DATA_ROOT / Path(
    "national", "edgar", "processed", f"edgar_national_{EDGAR_PROCESSING_ID}.csv"
)
edgar_processed_output_file_global = DATA_ROOT / Path(
    "national", "edgar", "processed", f"edgar_global_{EDGAR_PROCESSING_ID}.csv"
)


# %%
edgar_csv_files

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

# %% [markdown]
# Load sector mapping of emissions species

# %%
edgar_mapping = pd.read_excel(edgar_sector_mapping_file, sheet_name="EDGAR mapping")
edgar_map = get_map(
    edgar_mapping, sector_column="EDGAR_sectors", sector_output_column_name="edgar_sectors"
)  # note; with 7BC now added it is actually 60 sectors, not 59?!
edgar_map.to_frame(index=False)

# %% [markdown]
# Read EDGAR emissions data

# %%
edgar = pd.concat(
    read_CEDS(Path(edgar_data_folder) / f"E.{s}_EDGAR.csv").pix.assign(em=s, units=f"kt{s}") for s in species
).rename_axis(index={"iso": "country", "sector_59": "edgar_sectors"})
edgar = edgar.pix.semijoin(edgar_map, how="outer")
edgar.loc[isna].pix.unique(["edgar_sectors", "sector", "sector_description"])  # print sectors with NAs

# %%
# some tests could be added here
# edgar.pix.unique(['units'])

# %%
# change units to align with IAM scenario data
# adjust units; change column 'units' to 'unit' and add '/yr'
edgar = edgar.pix.dropna(subset=["units"]).pix.format(unit="{units}/yr", drop=True)
# adjust units; change all to values to Mt instead of kt
edgar = pix.units.convert_unit(edgar, lambda x: "Mt " + x.removeprefix("kt").strip())
# exception for N2O/yr, which should remain in kt following https://github.com/IAMconsortium/common-definitions/
edgar = pix.units.convert_unit(edgar, lambda x: "kt " + x.removeprefix("Mt").strip() if x == "Mt N2O/yr" else x)
# unit of NOx from NOx to NO2
edgar.index = pd.MultiIndex.from_tuples(
    [
        (
            country,
            edgar_sectors,
            sector_description,
            em,
            sector,
            "Mt NO2/yr" if unit == "Mt NOx/yr" and em == "NOx" else unit,
        )
        for country, edgar_sectors, sector_description, em, sector, unit in edgar.index
    ],
    names=edgar.index.names,
)
# change name(s) of emissions species
# use 'Sulfur' instead of 'SO2'
edgar.index = pd.MultiIndex.from_tuples(
    [
        (country, edgar_sectors, sector_description, "Sulfur" if em == "SO2" else em, sector, unit)
        for country, edgar_sectors, sector_description, em, sector, unit in edgar.index
    ],
    names=edgar.index.names,
)
edgar.index.to_frame(index=False)

# %%
edgar = edgar.groupby(["em", "country", "unit", "sector"]).sum().pix.fixna()  # group and fix NAs

# %%
# aggregate countries where this is necessary, e.g. because of specific other data (like SSP socioeconomic driver data)
# based on the new SSP data, we only need to aggregate Serbia and Kosovo
country_combinations = {
    # "isr_pse": ["isr", "pse"], "sdn_ssd": ["ssd", "sdn"],
    "srb_ksv": ["srb", "srb (kosovo)"]
}
edgar = edgar.pix.aggregate(country=country_combinations)

# %%
# add global
edgar = add_global(edgar)

# %%
# Rename NMVOC
edgar = edgar.rename(index=lambda v: v.replace("NMVOC", "VOC"))

# %%
edgar_reformatted = edgar.rename_axis(index={"em": "variable", "country": "region"})
edgar_reformatted

# %%
# rename to IAMC-style variable names including standard index order
edgar_reformatted_iamc = (
    edgar_reformatted.pix.format(variable="Emissions|{variable}|{sector}", drop=True)
    .pix.assign(scenario=HISTORY_SCENARIO_NAME, model="EDGAR")
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
).sort_values(by=["region", "variable"])
edgar_reformatted_iamc

# %%
edgar_reformatted_iamc.pix.unique(["variable"])

# %%
assert_units_match_wishes(edgar_reformatted_iamc)

# %% [markdown]
# Save formatted EDGAR data

# %%
out_global = edgar_reformatted_iamc.loc[pix.isin(region="World")]  # only the added "World" region
out_national = edgar_reformatted_iamc.loc[~pix.isin(region="World")]

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
edgar_processed_output_file_national.parent.mkdir(exist_ok=True, parents=True)
out_national.to_csv(edgar_processed_output_file_national)
edgar_processed_output_file_national

# %%
# globally aggregated data (all emissions)
edgar_processed_output_file_global.parent.mkdir(exist_ok=True, parents=True)
out_global.to_csv(edgar_processed_output_file_global)
edgar_processed_output_file_global

# %%
