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
# # Process GFED4 regridding, smoothing and extension
#
# Regrid, smooth and extend data from
# [GFED4](https://www.globalfiredata.org/related.html).
# We do it in this order so it is easier to make sure
# that we preserved the totals in the right places.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import xarray as xr
import xarray_regrid  # noqa: F401

from emissions_harmonization_historical.constants_5000 import (
    GFED4_ANNUAL_SUM_OUTPUT_DIR,
    GFED4_RAW_PATH,
    GFED4_REGRIDDING_SMOOTHING_EXTENSION_OUTPUT_DIR,
)
from emissions_harmonization_historical.gfed import load_emissions_factors_per_dry_matter
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Set up

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Annual totals

# %%
emissions_dry_matter_per_yr = xr.open_dataarray(GFED4_ANNUAL_SUM_OUTPUT_DIR / "emissions_dry_matter_per_yr.nc")
# emissions_dry_matter_per_yr

# %% [markdown]
# ### Cell area

# %%
cell_area = xr.load_dataarray(GFED4_ANNUAL_SUM_OUTPUT_DIR / "cell_area.nc")
# cell_area

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

# %% [markdown]
# Firstly make sure that the global sum is correct.

# %%
np.testing.assert_allclose(
    (emissions_dry_matter_per_yr.sel(year=2023) * emissions_factors_per_dry_matter * cell_area)
    .sum(["lat", "lon", "sector"])
    .sel(em="OC"),
    28.05 * 1e9,
    atol=0.01 * 1e9,
)

# %% [markdown]
# ### Get emissions per cell

# %%
emissions_dry_matter_per_yr_per_cell = (emissions_dry_matter_per_yr * cell_area).assign_attrs(dict(unit="kg DM / yr"))
# emissions_dry_matter_per_yr_per_cell

# %% [markdown]
# Make sure that the global sum is still correct.

# %%
np.testing.assert_allclose(
    (emissions_dry_matter_per_yr_per_cell.sel(year=2023) * emissions_factors_per_dry_matter)
    .sum(["lat", "lon", "sector"])
    .sel(em="OC"),
    28.05 * 1e9,
    atol=0.01 * 1e9,
)

# %% [markdown]
# ### Regrid onto country mask

# %%
# Super hacky and only works because the iso mask
# is half the resolution of GFED, anyway.
# (I couldn't get conservative regridding to do what I expected,
# so I moved on).

resolution_decrease_factor = 2
rdf = resolution_decrease_factor

in_lat = emissions_dry_matter_per_yr.lat
in_lon = emissions_dry_matter_per_yr.lon

if rdf != 2:  # noqa: PLR2004
    # The below will explode
    raise NotImplementedError(rdf)

target_lat = (in_lat[::rdf].values + in_lat[1::rdf].values) / rdf
target_lon = (in_lon[::rdf].values + in_lon[1::rdf].values) / rdf

emissions_dry_matter_per_yr_per_cell_regridded = (
    emissions_dry_matter_per_yr_per_cell.rolling(lat=rdf, lon=rdf)
    .sum()
    .sel(lat=in_lat[1::rdf], lon=in_lon[1::rdf])
    .assign_coords(
        lat=target_lat,
        lon=target_lon,
    )
    .sel(lat=country_mask.lat, lon=country_mask.lon)
)
emissions_dry_matter_per_yr_per_cell_regridded

# %% [markdown]
# Make sure that the global sum is still correct.

# %%
np.testing.assert_allclose(
    (emissions_dry_matter_per_yr_per_cell_regridded.sel(year=2023) * emissions_factors_per_dry_matter)
    .sum(["lat", "lon", "sector"])
    .sel(em="OC"),
    28.05 * 1e9,
    atol=0.01 * 1e9,
)

# %% [markdown]
# ### Smooth
#
# Centred five-year rolling mean, as described
# [here](https://input4mips-cvs.readthedocs.io/en/latest/dataset-overviews/open-biomass-burning-emissions/#calculation-of-the-smoothed-dataset).
#
# It turns out that this doesn't actually work,
# because the GFED4 data we downloaded isn't the same as what is in CMIP7 for 2021.
# So, we're going to have to process the raw CMIP data.
# Not doing that now as I don't want to deal with the downloading.

# %%
emissions_dry_matter_per_yr_per_cell_regridded_smoothed = (
    emissions_dry_matter_per_yr_per_cell_regridded.rolling(year=5, center=True).mean().dropna("year")
)
# emissions_dry_matter_per_yr_per_cell_regridded_smoothed

# %%
# This does not match the smoothed CMIP7 dataset
# because the GFED4 data is not the same as what is used for CMIP.
(emissions_dry_matter_per_yr_per_cell_regridded_smoothed.sel(year=2021) * emissions_factors_per_dry_matter).sum(
    ["lat", "lon", "sector"]
).sel(em="OC")

# %% [markdown]
# ### Linearly extrapolate
#
# We do this on the grid cell level.
# Someone else can think about
# whether this should be a multivariate regression
# to better capture spatial covariance.

# %%
n_years_for_regression = 10
coeffs_ds = emissions_dry_matter_per_yr_per_cell_regridded_smoothed.polyfit("year", deg=1, skipna=True)
# coeffs_ds

# %%
years_interp = np.arange(
    emissions_dry_matter_per_yr_per_cell_regridded_smoothed["year"].max() + 1,
    HARMONISATION_YEAR + 1,
)
year_factor = xr.DataArray(years_interp, dims=("year",), coords=dict(year=years_interp))
# year_factor

# %%
extrapolated = coeffs_ds["polyfit_coefficients"].sel(degree=1) * year_factor + coeffs_ds["polyfit_coefficients"].sel(
    degree=0
)
# extrapolated

# %%
full = xr.concat([emissions_dry_matter_per_yr_per_cell_regridded_smoothed, extrapolated], dim="year")
# full

# %%
selectors = dict(lat=8.125, lon=0.0, method="nearest")
full.sel(**selectors).plot.line(hue="sector")

# %% [markdown]
# ## Save

# %%
GFED4_REGRIDDING_SMOOTHING_EXTENSION_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
full.to_netcdf(GFED4_REGRIDDING_SMOOTHING_EXTENSION_OUTPUT_DIR / "emissions_dry_matter_smooth_extended.nc")
