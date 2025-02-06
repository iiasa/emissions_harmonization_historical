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
# # Prepare CEDS
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
ceds_data_folder = DATA_ROOT / Path("national", "ceds", "data_raw")
ceds_sector_mapping_file = DATA_ROOT / Path("national", "ceds", "data_aux", "sector_mapping.xlsx")
ceds_processed_output_file_national = DATA_ROOT / Path(
    "national", "ceds", "processed", f"ceds_cmip7_national_{CEDS_PROCESSING_ID}.csv"
)
ceds_processed_output_file_international = DATA_ROOT / Path(
    "national", "ceds", "processed", f"ceds_cmip7_international_{CEDS_PROCESSING_ID}.csv"
)
# (by default we do not save this one below to avoid data duplication)
# ceds_processed_output_file_national_and_international = DATA_ROOT / Path(
#     "national", "ceds", "processed", f"ceds_cmip7_national_and_international_{CEDS_PROCESSING_ID}.csv"
# )
ceds_processed_output_file_global = DATA_ROOT / Path(
    "national", "ceds", "processed", f"ceds_cmip7_global_{CEDS_PROCESSING_ID}.csv"
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
    read_CEDS(Path(ceds_data_folder) / f"{s}_CEDS_emissions_by_country_sector_v{ceds_release}.csv") for s in species
).rename_axis(index={"region": "country"})
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
        (em, country, sector_59, sector, "Mt BC/yr" if unit == "Mt C/yr" and em == "BC" else unit)
        for em, country, sector_59, sector, unit in ceds.index
    ],
    names=ceds.index.names,
)
# unit of OC from C to OC
ceds.index = pd.MultiIndex.from_tuples(
    [
        (em, country, sector_59, sector, "Mt OC/yr" if unit == "Mt C/yr" and em == "OC" else unit)
        for em, country, sector_59, sector, unit in ceds.index
    ],
    names=ceds.index.names,
)
# change name(s) of emissions species
# use 'Sulfur' instead of 'SO2'
ceds.index = pd.MultiIndex.from_tuples(
    [
        ("Sulfur" if em == "SO2" else em, country, sector_59, sector, unit)
        for em, country, sector_59, sector, unit in ceds.index
    ],
    names=ceds.index.names,
)
ceds

# %%
ceds = ceds.groupby(["em", "country", "unit", "sector"]).sum().pix.fixna()  # group and fix NAs

# %%
# aggregate countries where this is necessary, e.g. because of specific other data (like SSP socioeconomic driver data)
# based on the new SSP data, we only need to aggregate Serbia and Kosovo
country_combinations = {
    # "isr_pse": ["isr", "pse"], "sdn_ssd": ["ssd", "sdn"],
    "srb_ksv": ["srb", "srb (kosovo)"]
}
ceds = ceds.pix.aggregate(country=country_combinations)

# %%
# add global
ceds = add_global(ceds)

# %%
# Rename NMVOC
ceds = ceds.rename(index=lambda v: v.replace("NMVOC", "VOC"))

# %%
ceds_reformatted = ceds.rename_axis(index={"em": "variable", "country": "region"})
ceds_reformatted

# %%
# rename to IAMC-style variable names including standard index order
ceds_reformatted_iamc = (
    ceds_reformatted.pix.format(variable="Emissions|{variable}|{sector}", drop=True)
    .pix.assign(scenario=HISTORY_SCENARIO_NAME, model=f"CEDSv{ceds_release}")
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
).sort_values(by=["region", "variable"])
ceds_reformatted_iamc

# %%
ceds_reformatted_iamc

# %%
assert_units_match_wishes(ceds_reformatted_iamc)

# %% [markdown]
# Save formatted CEDS data

# %%
out_global = ceds_reformatted_iamc.loc[pix.isin(region="World")]  # only the added "World" region
out_national_with_global = ceds_reformatted_iamc.loc[
    ~pix.isin(region="World")
]  # remove the added "World" region, and the CEDS "global" region
out_national_without_global = ceds_reformatted_iamc.loc[
    ~pix.isin(region=["World", "global"])
]  # remove the added "World" region, and the CEDS "global" region
out_national_only_global = ceds_reformatted_iamc.loc[
    pix.isin(region="global")
]  # only the CEDS "global" region; which represents "international" emissions

# %%
assert out_national_with_global.shape[0] + out_global.shape[0] == ceds_reformatted_iamc.shape[0]
assert (
    out_national_without_global.shape[0] + out_national_only_global.shape[0] + out_global.shape[0]
    == ceds_reformatted_iamc.shape[0]
)

# %% [markdown]
# Check that national sums equal global total.

# %%
# Check that `out_national_with_global` totals (all countries in iso3c + CEDS 'global' region)
# ... are the same as `out_global` totals ("World")
national_sums_checker = (
    pix.assignlevel(out_national_with_global.groupby(["model", "scenario", "variable", "unit"]).sum(), region="World")
    .reset_index()
    .set_index(out_global.index.names)
)
national_sums_checker.columns = national_sums_checker.columns.astype(int)
national_sums_checker
pd.testing.assert_frame_equal(out_global, national_sums_checker, check_like=True)

# %%
# Check that `out_national_without_global` totals (all countries in iso3c) are the same as `out_global` totals ("World")
# - this should not hold for aircraft and international shipping

# without aircraft and international shipping:
global_checker = out_global.loc[~pix.ismatch(variable=["**Aircraft", "**Shipping"])]
national_sums_checker = (
    pix.assignlevel(
        out_national_without_global.groupby(["model", "scenario", "variable", "unit"]).sum(), region="World"
    )
    .loc[~pix.ismatch(variable=["**Aircraft", "**Shipping"])]
    .reset_index()
    .set_index(out_global.index.names)
)
national_sums_checker.columns = national_sums_checker.columns.astype(int)
pd.testing.assert_frame_equal(global_checker, national_sums_checker, check_like=True)

# check that aircraft and international shipping are zero:
filtered_rows = out_national_without_global[
    out_national_without_global.index.to_frame().apply(
        lambda row: row.str.contains("Aircraft|Shipping", case=False).any(), axis=1
    )
]
numeric_values = filtered_rows.select_dtypes(include=[np.number])
non_zero_values = numeric_values[numeric_values != 0].dropna(how="all")
if non_zero_values.empty:
    print("✅ Test passed: All numeric values for 'Aircraft' or 'Shipping' rows are zero.")
else:
    # For version 2024_07_08 of CEDS, we expect this to show BC and OC emissions for USA.
    # ... the reason is that for 'International Shipping', we also include the CEDS sector '1A3di_Oil_Tanker_Loading'
    # ... which only reports non-zero values for USA for BC and OC.
    print("⚠️ Found non-zero values in the following rows:")
    nz = non_zero_values.pix.to_tidy()
    print(nz[["region", "variable"]].drop_duplicates())

    assert (
        len(nz[["region", "variable"]].drop_duplicates())
        == CEDS_EXPECTED_NUMBER_OF_REGION_VARIABLE_PAIRS_IN_GLOBAL_HARMONIZATION
    )  # ... if the CEDS version is not 2024_07_08, this assert statement may need to be updated

# %%
# national CEDS data
ceds_processed_output_file_national.parent.mkdir(exist_ok=True, parents=True)
out_national_without_global.to_csv(ceds_processed_output_file_national)
ceds_processed_output_file_national

# %%
# international only CEDS data (aircraft and international shipping)
ceds_processed_output_file_international.parent.mkdir(exist_ok=True, parents=True)
out_national_only_global.to_csv(ceds_processed_output_file_international)
ceds_processed_output_file_international

# %%
# # national+international CEDS data (by default we do not save this to avoid data duplication)
# ceds_processed_output_file_national_and_international.parent.mkdir(exist_ok=True, parents=True)
# out_national_without_global.to_csv(ceds_processed_output_file_national_and_international)
# ceds_processed_output_file_national_and_international

# %%
# globally aggregated data (all emissions)
ceds_processed_output_file_global.parent.mkdir(exist_ok=True, parents=True)
out_global.to_csv(ceds_processed_output_file_global)
ceds_processed_output_file_global
