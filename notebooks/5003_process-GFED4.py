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
# # Process GFED
#
# Process data from [GFED](https://www.globalfiredata.org/).

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import tqdm.auto
import xarray as xr
import xarray_regrid  # noqa: F401

from emissions_harmonization_historical.constants_5000 import GFED4_PROCESSED_DB, GFED4_RAW_PATH, HISTORY_SCENARIO_NAME
from emissions_harmonization_historical.gfed import (
    read_cell_area,
    read_year,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# %% [markdown]
# ## Set up

# %%
# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
pandas_openscm.register_pandas_accessor()

# %%
# use all species covered in CEDS
species = [
    "BC",
    "CH4",
    "CO",
    "CO2",
    "N2O",  # was global in CMIP6, so having this regionally is new
    "NH3",
    "NMVOC",  # assumed to be equivalent to IAMC-style reported VOC
    "NOx",
    "OC",
    "SO2",
]
# mapping from GFED variables to longer IAMC-style naming
sector_mapping = {
    "AGRI": "Agricultural Waste Burning",
    "BORF": "Forest Burning",
    "DEFO": "Forest Burning",
    "PEAT": "Peat Burning",
    "SAVA": "Grassland Burning",
    "TEMF": "Forest Burning",
}

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Emissions factors info

# %%
nmvoc_aux_conversion = pd.read_excel(GFED4_RAW_PATH / "NMVOC-species.xlsx", index_col=0)
nmvoc_species = nmvoc_aux_conversion.set_axis(
    nmvoc_aux_conversion.index.str.replace(r"^(\w+) *\(.*\)$", r"\1", regex=True).rename("em")
)
# nmvoc_species
nmvoc_factors = pd.Series(1, nmvoc_species.index[nmvoc_species["NMVOC"].isin(["y"])])
# nmvoc_factors

# %%
GFED_EMISSIONS_FACTORS_FILE = GFED4_RAW_PATH / "GFED4_Emission_Factors.txt"

_, marker, *sectors = pd.read_csv(GFED_EMISSIONS_FACTORS_FILE, sep=r"\s+", skiprows=15, nrows=1, header=None).iloc[0]
if marker != "SPECIE":
    msg = "Unexpected header, check reading"
    raise AssertionError(msg)

emissions_factors = pd.read_csv(
    GFED_EMISSIONS_FACTORS_FILE, sep=r"\s+", header=None, names=sectors, comment="#"
).rename_axis(index="em", columns="sector")
# emissions_factors

# %% [markdown]
# ### Country mask

# %%
country_mask = xr.open_dataarray(GFED4_RAW_PATH / "iso_mask.nc").compute()
# If you want to check that the mask sums to 1
# np.unique(country_mask.sum(["iso"]))

# %% [markdown]
# ## Process

# %% [markdown]
# ### Get emissions factors of interest

# %%
nmvoc_aux_conversion = pd.read_excel(GFED4_RAW_PATH / "NMVOC-species.xlsx", index_col=0)
nmvoc_species = nmvoc_aux_conversion.set_axis(
    nmvoc_aux_conversion.index.str.replace(r"^(\w+) *\(.*\)$", r"\1", regex=True).rename("em")
)
# nmvoc_species
nmvoc_factors = pd.Series(1, nmvoc_species.index[nmvoc_species["NMVOC"].isin(["y"])])
# nmvoc_factors

# %%
emissions_factors.loc["NMVOC"] = emissions_factors.multiply(nmvoc_factors, axis=0).sum()
emissions_factors_per_DM = emissions_factors.loc[species] / emissions_factors.loc["DM"]
# in kg {species} / kg DM
# emissions_factors_per_DM

# %% [markdown]
# ### Aggregate to countries

# %% [markdown]
# Do this year by year
# because dask explodes with the complexity of reading the hdf5 files.

# %%
COMPUTE = False  # Set this to True to trigger computation and basically disable dask


def manage_compute(obj):
    """Manage dask compute"""
    return obj.compute() if COMPUTE else obj


# %%
res_l = []
for filename in tqdm.auto.tqdm(sorted(GFED4_RAW_PATH.glob("*.hdf5"))):
    emissions = read_year(filename)
    emissions["cell_area"] = read_cell_area(next(iter(GFED4_RAW_PATH.glob("*.hdf5"))))
    emissions["DM"].attrs.update(dict(unit="kg DM m-2 / month"))
    emissions["C"].attrs.update(dict(unit="g C m-2 / month"))
    emissions = manage_compute(emissions[["DM", "cell_area"]])

    dry_matter_regridded = emissions["DM"].regrid.linear(country_mask)
    # Super hacky and only works because the iso mask
    # is half the resolution of GFED, anyway.
    # (I couldn't get conservative regridding to do what I expected,
    # so I moved on).
    cell_area_regridded = (
        emissions["cell_area"]
        .rolling(lat=2, lon=2)
        .sum()
        .sel(lat=np.arange(89.625, -90, -0.5), lon=np.arange(-179.625, 180, 0.5))
        .assign_coords(
            lat=(emissions.lat[::2].values + emissions.lat[1::2].values) / 2.0,
            lon=(emissions.lon[::2].values + emissions.lon[1::2].values) / 2.0,
        )
        .sel(lat=country_mask.lat, lon=country_mask.lon)
    )

    emissions_per_cell_regridded = dry_matter_regridded * cell_area_regridded

    # Get dry matter by country per year
    dry_matter_by_country = manage_compute((emissions_per_cell_regridded * country_mask).sum(["lat", "lon"]))

    dry_matter_by_country_per_year = (
        dry_matter_by_country.groupby("time.year").sum().assign_attrs(dict(unit="kg DM / a"))
    )
    # Get emissions by country per year
    emissions_by_country = manage_compute(dry_matter_by_country_per_year * xr.DataArray(emissions_factors_per_DM))
    res_l.append(emissions_by_country)

emissions_by_country = xr.concat(res_l, dim="year")
# emissions_by_country

# %%
# A quick check from a number that matters a lot
np.testing.assert_allclose(
    28.05 * 1e9, emissions_by_country.sum(["sector", "iso"]).sel(em="OC", year=2023).values, atol=0.01 * 1e9
)

# %% [markdown]
# #### Convert units

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
        ("NMVOC", "kg VOC"),
        ("SO2", "kg SO2"),
    ],
    names=["em", "unit"],
)
emissions_df = (
    emissions_by_country.to_series()
    .unstack("year")
    .rename_axis(index={"iso": "country"})
    .pix.semijoin(units, how="left")
    .pix.convert_unit(lambda u: u.replace("kg", "kt"))
)
emissions_df

# %% [markdown]
# #### Combine countries that are not reported separately elsewhere

# %%
country_combinations = {"srb_ksv": ["srb", "srb (kosovo)"]}
emissions_df = emissions_df.pix.aggregate(country=country_combinations)

# %% [markdown]
# ### Reformat

# %%
res = emissions_df.copy()
# res

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
# Move to IAMC format
res = (
    res.pix.convert_unit(lambda u: u.replace("kt", "Mt"))
    .rename(index=sector_mapping, level="sector")
    .groupby(["em", "sector", "country", "unit"])
    .sum()
)

# rename to IAMC-style variable names
res = (
    res.rename(index={"SO2": "Sulfur", "NMVOC": "VOC"}, level="em")
    .pix.format(variable="Emissions|{em}|{sector}", drop=True)
    .pix.assign(scenario=HISTORY_SCENARIO_NAME, model="GFED4")
)
res


# %%
# Other fixes
res = res.rename_axis(index={"em": "variable", "country": "region"})
res = res.pix.format(unit="{unit}/yr")

with pint.get_application_registry().context("NOx_conversions"):
    res = res.pix.convert_unit({"Mt N2O/yr": "kt N2O/yr", "Mt NO/yr": "Mt NO2/yr"})

tmp = res.reset_index("unit")
tmp["unit"] = tmp["unit"].str.replace("Mt C/yr", "Mt BC/yr")
res = tmp.set_index("unit", append=True).reorder_levels(res.index.names)
res

# %%
# Another handy quick check
res.groupby(["variable", "unit"]).sum().loc[pix.ismatch(variable="**|OC|**")].sum().round(2)

# %%
assert_units_match_wishes(res)

# %% [markdown]
# ## Save formatted GFED data

# %%
GFED4_PROCESSED_DB.save(res.pix.assign(stage="iso3c"), allow_overwrite=True)
