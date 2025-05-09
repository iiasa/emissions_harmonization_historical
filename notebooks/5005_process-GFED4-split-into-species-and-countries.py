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
# # Process GFED4 into species and countries
#
# Split data from [GFED4](https://www.globalfiredata.org/related.html)
# into species and countries.

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import tqdm.auto
import xarray as xr
import xarray_regrid  # noqa: F401

from emissions_harmonization_historical.constants_5000 import (
    GFED4_PROCESSED_DB,
    GFED4_RAW_PATH,
    GFED4_REGRIDDING_SMOOTHING_EXTENSION_OUTPUT_DIR,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.gfed import (
    load_emissions_factors_per_dry_matter,
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
# ### Smoothed and extended data

# %%
emissions_dry_full = xr.open_dataarray(
    GFED4_REGRIDDING_SMOOTHING_EXTENSION_OUTPUT_DIR / "emissions_dry_matter_smooth_extended.nc"
)
# emissions_dry_full

# %% [markdown]
# ### Country mask

# %%
country_mask = xr.open_dataarray(GFED4_RAW_PATH / "iso_mask.nc").compute()
# If you want to check that the mask sums to 1
# np.unique(country_mask.sum(["iso"]))

# %% [markdown]
# ### Emissions factors info

# %%
emissions_factors_per_dry_matter = load_emissions_factors_per_dry_matter(GFED4_RAW_PATH)
# emissions_factors_per_dry_matter

# %% [markdown]
# ## Process

# %%
# emissions_dry_full * emissions_factors_per_dry_matter

# %%
emissions_dry_country_l = []
for iso in tqdm.auto.tqdm(country_mask["iso"].values):
    emissions_dry_country_l.append((emissions_dry_full * country_mask.sel(iso=[iso])).sum(["lat", "lon"]).compute())

emissions_dry_country = xr.concat(emissions_dry_country_l, "iso")
emissions_dry_country

# %% [markdown]
# ### Split to individual species

# %%
emissions_country = emissions_dry_country * emissions_factors_per_dry_matter
# emissions_country

# %%
emissions_country.sum(["sector", "iso"]).sel(em="OC", year=2021)

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
    emissions_country.to_series()
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
