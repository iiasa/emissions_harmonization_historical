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
# # Prepare GFED
#
# Prepare data from [GFED](https://www.globalfiredata.org/).

# %%
# import external packages and functions
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pint
import ptolemy
import xarray as xr
from pandas_indexing import set_openscm_registry_as_default

from emissions_harmonization_historical.constants import DATA_ROOT, GFED_PROCESSING_ID, HISTORY_SCENARIO_NAME
from emissions_harmonization_historical.gfed import (
    add_global,
    read_cell_area,
    # read_var,
    # read_coords,
    # concat_group,
    # read_monthly,
    # month_to_cftime,
    read_year,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# set unit registry
ur = set_openscm_registry_as_default()
pix.units.set_openscm_registry_as_default()


# %% [markdown]
# Set paths

# %%
gfed_release = "GFED4.1s"

gfed_data_folder = DATA_ROOT / Path("national", "gfed", "data_raw")

gfed_data_aux_folder = DATA_ROOT / Path("national", "gfed", "data_aux")
gfed_emission_factors = Path(
    gfed_data_aux_folder, "GFED4_Emission_Factors.txt"
)  # emission factors of burning of different biomes
gfed_nmvoc_aux_conversion = Path(
    gfed_data_aux_folder, "NMVOC-species.xlsx"
)  # for unit conversion of volatile organic compounds
voc_unit = "kg VOC"  # or kg C
gfed_isomask = Path(gfed_data_aux_folder, "iso_mask.nc")  # for aggregating to countries
gfed_grid_template = Path(
    gfed_data_aux_folder, "BC-em-openburning_input4MIPs_emissions_CMIP_REMIND-MAGPIE-SSP5-34-OS-V1_gn_201501-210012.nc"
)  # for country-level grid emissions reporting template

gfed_processed_output_file = DATA_ROOT / Path("national", "gfed", "processed", f"gfed_cmip7_{GFED_PROCESSING_ID}.csv")

gfed_processed_output_file_national = DATA_ROOT / Path(
    "national", "gfed", "processed", f"gfed_cmip7_national_{GFED_PROCESSING_ID}.csv"
)

gfed_processed_output_file_World = DATA_ROOT / Path(
    "national", "gfed", "processed", f"gfed_cmip7_World_{GFED_PROCESSING_ID}.csv"
)


gfed_temp_file = DATA_ROOT / Path("national", "gfed", "processed", "gfed_temporaryfile.csv")

# %% [markdown]
# Specify gases to processes

# %%
# use all gases covered in CEDS
gases = [
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
# renaming GFED variables to longer IAMC-style naming (renaming happens only at the end of this script)
sector_mapping = {
    "AGRI": "Agricultural Waste Burning",
    "BORF": "Forest Burning",
    "DEFO": "Forest Burning",
    "PEAT": "Peat Burning",
    "SAVA": "Grassland Burning",
    "TEMF": "Forest Burning",
}

# %% [markdown]
# Load raw emissions data

# %%
# load raw emissions data
emissions = xr.concat(
    [read_year(filename) for filename in sorted(gfed_data_folder.glob("*.hdf5"), key=lambda p: p.stem)],
    dim="time",
)
# add the cell_area data variable
emissions["cell_area"] = read_cell_area(next(iter(gfed_data_folder.glob("*.hdf5"))))
# set unit attributes for the DM (dry matter) and C (Carbon)
emissions["DM"].attrs.update(dict(unit="kg DM m-2 / month"))
emissions["C"].attrs.update(dict(unit="g C m-2 / month"))
# show xarray
emissions

# %% [markdown]
# Get emissions factor for different species

# %%
_, marker, *sectors = pd.read_csv(gfed_emission_factors, sep=r"\s+", skiprows=15, nrows=1, header=None).iloc[0]
assert marker == "SPECIE", f"header in {gfed_emission_factors} is not in line 16 anymore or looks different"

ef = pd.read_csv(gfed_emission_factors, sep=r"\s+", header=None, names=sectors, comment="#").rename_axis(
    index="em", columns="sector"
)

# Aggregate NMVOC to a total in terms of kgC
nmvoc_species = pd.read_excel(gfed_nmvoc_aux_conversion, index_col=0)
nmvoc_species = nmvoc_species.set_axis(
    nmvoc_species.index.str.replace(r"^(\w+) *\(.*\)$", r"\1", regex=True).rename("em")
)

# Whether to convert to kgC
if voc_unit == "kg C":
    nmvoc_factors = (
        nmvoc_species.loc[nmvoc_species["NMVOC"].isin(["y"])]
        .pipe(lambda df: df["Carbon weight"] / df["Molecular weight"])
        .astype(float)
    )
elif voc_unit == "kg VOC":
    nmvoc_factors = pd.Series(1, nmvoc_species.index[nmvoc_species["NMVOC"].isin(["y"])])
else:
    raise NotImplementedError(f"voc_unit must be 'kg VOC' or 'kg C', not: '{voc_unit}'")

ef.loc["NMVOC"] = ef.multiply(nmvoc_factors, axis=0).sum()

ef_per_DM = ef.loc[gases] / ef.loc["DM"]
# in kg {species} / kg DM
ef_per_DM

# %% [markdown]
# Aggregate to countries

# %%
# Step 1: Load the country (ISO) NetCDF mask with 0.5-degree resolution.
# This mask assigns each grid cell an ISO country code, allowing emissions to be aggregated by country.
# 'chunks={"iso": 1}' uses Dask to enable chunking for memory efficiency, loading one ISO code at a time.
idxr = xr.open_dataarray(gfed_isomask, chunks={"iso": 1})

# Step 2: Open a NetCDF file to use as a grid template for latitude and longitude coordinates.
# The template file provides the lat/lon grid for regridding the emissions data.
with xr.open_dataset(gfed_grid_template) as template:
    # Interpolate the "DM" (Dry Matter) emissions data to the lat/lon grid from the template,
    # using linear interpolation. This matches the emissions data to the same grid resolution.
    dm_regrid = emissions["DM"].interp(lon=template.lon, lat=template.lat, method="linear")

# Step 3: Compute the area of each grid cell using the 'ptolemy.cell_area' function.
# This function calculates the area of each grid cell based on the interpolated lat/lon grid.
# The resulting cell areas are stored in an xarray DataArray, with units of square meters ("m2").
cell_area = xr.DataArray(ptolemy.cell_area(lats=dm_regrid.lat, lons=dm_regrid.lon), attrs=dict(unit="m2"))

# calculate emissions by country by:
# taking the country cell IDs (idxr), multiplying it by the area (cell_area),
# and by the regridded lat/lon grid resummed to per year (dm_regrid.groupby("time.year").sum())
# Step 4: Calculate emissions by country.
# This is done by multiplying the regridded Dry Matter (DM) emissions data by the grid cell areas (cell_area)
# and by the ISO country cell IDs (idxr).
# The emissions are then summed over the lat/lon grid for each year (grouping by "time.year").
# Finally, we get the emissions (in unit kg DM / a) by multiplying by the emissions factor (ef_per_DM).
country_emissions = (
    (dm_regrid.groupby("time.year").sum() * cell_area * idxr).sum(["lat", "lon"]).assign_attrs(dict(unit="kg DM / a"))
    * xr.DataArray(ef_per_DM)
).compute()
country_emissions

# %% [markdown]
# Convert units from Dry Matter to emissions

# %%
country_emissions

# %%
units = pd.MultiIndex.from_tuples(
    [
        ("BC", "kg C"),
        ("OC", "kg OC"),
        ("CO", "kg CO"),
        ("CO2", "kg CO2"),
        ("CH4", "kg CH4"),
        ("N2O", "kg N2O"),
        ("NH3", "kg NH3"),
        # NOx units in GFED are NO by default,
        # whereas NOx in openscm-units means
        # NO2.
        ("NOx", "kg NO"),
        ("NMVOC", voc_unit),
        ("SO2", "kg SO2"),
    ],
    names=["em", "unit"],
)
emissions_df = (
    country_emissions.to_series()
    .unstack("year")
    .rename_axis(index={"iso": "country"})
    .pix.semijoin(units, how="left")
    .pix.convert_unit(lambda u: u.replace("kg", "kt"))
)
emissions_df

# %%
country_combinations = {"srb_ksv": ["srb", "srb (kosovo)"]}
emissions_df = emissions_df.pix.aggregate(country=country_combinations)

# %% [markdown]
# Intermediary save before reformatting

# %%
# intermediary save
emissions_df.to_csv(gfed_temp_file)

# %% [markdown]
# Reformat, including updated variable naming

# %%
burningCMIP7 = (
    pd.read_csv(
        gfed_temp_file,
        index_col=list(range(4)),
    )
    .rename(columns=int)
    .rename_axis(columns="year")
)

# burningCMIP7.pix

# %%
# set units
unit = pd.MultiIndex.from_tuples(
    [
        ("BC", "kt BC/yr"),
        ("OC", "kt OC/yr"),
        ("CO", "kt CO/yr"),
        ("CO2", "kt CO2/yr"),
        ("CH4", "kt CH4/yr"),
        ("N2O", "kt N2O/yr"),
        ("NH3", "kt NH3/yr"),
        ("NOx", "kt NO/yr"),
        ("NMVOC", "kt VOC/yr"),
        ("SO2", "kt SO2/yr"),
    ],
    names=["em", "unit"],
)

# %%
# TODO: update unit conversions as necessary, e.g. by following IAMC units.
# Could also be done in a IAMC-preprocessing script.

# %%
# reformat
burningCMIP7_ref = (
    burningCMIP7.pix.convert_unit(lambda u: u.replace("kt", "Mt"))
    .rename(index=sector_mapping, level="sector")
    .groupby(["em", "sector", "country", "unit"])
    .sum()
)

# rename to IAMC-style variable names
burningCMIP7_ref = (
    burningCMIP7_ref.rename(index={"SO2": "Sulfur", "NMVOC": "VOC"}, level="em")
    .pix.format(variable="Emissions|{em}|{sector}", drop=True)
    .pix.assign(scenario=HISTORY_SCENARIO_NAME, model=gfed_release)
)

# add global level aggregation ("World")
burningCMIP7_ref = add_global(burningCMIP7_ref, groups=["model", "scenario", "variable", "unit"])

# %%
# Various fixes
burningCMIP7_ref = burningCMIP7_ref.rename_axis(index={"em": "variable", "country": "region"})
burningCMIP7_ref = burningCMIP7_ref.pix.format(unit="{unit}/yr")

with pint.get_application_registry().context("NOx_conversions"):
    burningCMIP7_ref = burningCMIP7_ref.pix.convert_unit({"Mt N2O/yr": "kt N2O/yr", "Mt NO/yr": "Mt NO2/yr"})

tmp = burningCMIP7_ref.reset_index("unit")
tmp["unit"] = tmp["unit"].str.replace("Mt C/yr", "Mt BC/yr")
burningCMIP7_ref = tmp.set_index("unit", append=True).reorder_levels(burningCMIP7_ref.index.names)
burningCMIP7_ref

# %%
# Missing (NM)VOC at the moment from this processing...
assert_units_match_wishes(burningCMIP7_ref)

# %%
out_global = burningCMIP7_ref.loc[pix.isin(region="World")]  # only the added "World" region
out_national_without_World = burningCMIP7_ref.loc[~pix.isin(region="World")]  # remove the added "World" region
out_only_World = burningCMIP7_ref.loc[
    pix.isin(region="World")
]  # only the GFED "global" region; which represents "international" emissions

# %%
out_national_without_World.to_csv(gfed_processed_output_file_national)
out_only_World.to_csv(gfed_processed_output_file_World)

# %% [markdown]
# Save formatted GFED data

# %%
burningCMIP7_ref.to_csv(gfed_processed_output_file)
