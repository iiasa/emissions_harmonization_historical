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

# %%
# import external packages and functions
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
from pandas_indexing.core import isna

from emissions_harmonization_historical.ceds import add_global, get_map, read_CEDS
from emissions_harmonization_historical.constants import DATA_ROOT

# set unit registry
ur = pix.units.set_openscm_registry_as_default()


# %% [markdown]
# Set paths

# %%
ceds_release = "2024_07_08"
ceds_data_folder = DATA_ROOT / Path("national", "ceds", "data_raw")
ceds_sector_mapping_file = DATA_ROOT / Path("national", "ceds", "data_aux", "sector_mapping.xlsx")
ceds_processed_output_file = DATA_ROOT / Path("national", "ceds", "processed", "ceds_cmip7_national_alpha.csv")

# %% [markdown]
# Specify gases to processes

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
ceds_map = get_map(ceds_mapping, "59_Sectors_2024")
ceds_map.to_frame(index=False)

# %% [markdown]
# Read CEDS emissions data

# %%
ceds = pd.concat(
    read_CEDS(Path(ceds_data_folder) / f"{specie}_CEDS_emissions_by_country_sector_v{ceds_release}.csv")
    for specie in species
).rename_axis(index={"region": "country"})
ceds.attrs["name"] = "CEDS21"
ceds = ceds.pix.semijoin(ceds_map, how="outer")
ceds.loc[isna].pix.unique(["sector_59", "sector"])  # print sectors with NAs

# %%
ceds = ceds.pix.dropna(subset=["units"]).pix.format(unit="{units}/yr", drop=True)  # adjust units
ceds = pix.units.convert_unit(ceds, lambda x: "Mt " + x.removeprefix("kt").strip())  # adjust units

# %%
ceds = ceds.groupby(["em", "country", "unit", "sector"]).sum().pix.fixna()  # group and fix NAs

# %%
# aggregate countries where this is necessary, e.g. because of specific other data (like SSP socioeconomic driver data)
# TODO: check this based on the new SSP data (because old SSP data only had the sum); so recheck.
country_combinations = {"isr_pse": ["isr", "pse"], "sdn_ssd": ["ssd", "sdn"], "srb_ksv": ["srb", "srb (kosovo)"]}
ceds = ceds.pix.aggregate(country=country_combinations)

# %%
# add global
ceds = add_global(ceds)

# %%
# See Hoesly et al. 2018 plots of NOx for suggestion that NOx units is NO2
# https://gmd.copernicus.org/articles/11/369/2018/gmd-11-369-2018.pdf

unit_wishes = pd.MultiIndex.from_tuples(
    [
        ("BC", "Mt BC/yr"),
        ("CH4", "Mt CH4/yr"),
        ("CO", "Mt CO/yr"),
        ("CO2", "Mt CO2/yr"),
        ("N2O", "Mt N2O/yr"),
        ("NH3", "Mt NH3/yr"),
        ("NMVOC", "Mt NMVOC/yr"),
        ("NOx", "Mt NO2/yr"),  # CEDS is reported in NO2 units
        ("OC", "Mt OC/yr"),
        ("SO2", "Mt SO2/yr"),
    ],
    names=["em", "unit"],
)

# %%
ceds.pix.unique(unit_wishes.names)

# %%
ceds.pix.unique(unit_wishes.names).symmetric_difference(unit_wishes)

# %%
# reformat
ceds_ref = (
    ceds.droplevel("unit")
    .pix.semijoin(unit_wishes, how="left")
    .rename_axis(index={"em": "variable", "country": "region"})
)

# %%
# rename to IAMC-style variable names including standard index order
ceds_ref = (
    ceds_ref.rename(index={"SO2": "Sulfur"}, level="variable")
    .pix.format(variable="CMIP7 History|Emissions|{variable}|{sector}", drop=True)
    .pix.assign(model="History", scenario=f"CEDSv{ceds_release}")
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
)

# %% [markdown]
# Save formatted CEDS data

# %%
ceds_ref

# %%
(ceds_ref.to_csv(ceds_processed_output_file))

# %%
