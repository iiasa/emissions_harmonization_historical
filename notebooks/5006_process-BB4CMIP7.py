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
# # Process BB4CMIP7 data
#
# Process the biomass burning emissions
# from the BB4CMIP7 data on ESGF.
# In theory, these are the same as GFED4.
# In practice, they're not so we have to do it this way.
# This is extremely painful, we have to deal with a stupid amount of data.
# We download the data, process it, then delete it
# to avoid filling up our disks with 50GB of (mostly) zeros.
# We also run this species by species,
# again to avoid overloading our disks.

# %% [markdown]
# ## Imports

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pooch
import tqdm.auto
import xarray as xr
from dask.diagnostics import ProgressBar

from emissions_harmonization_historical.constants_5000 import (
    BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR,
    GFED4_RAW_PATH,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Setup

# %% [markdown]
# this notebook needs to be run for all the following emission species by hand:
#
# "BC", "CH4", "CO", "CO2", "N2O", "NH3", "NMVOC"/"NMVOCbulk", "NOx", "OC", "SO2"

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
species: str = "NMVOC"
species_esgf: str = "NMVOCbulk"


# %% [markdown]
# ## Load data

# %%
country_mask = (
    xr.open_dataarray(GFED4_RAW_PATH / "iso_mask.nc").compute().rename({"lat": "latitude", "lon": "longitude"})
)
# If you want to check that the mask sums to 1
# np.unique(country_mask.sum(["iso"]))
# country_mask

# %% [markdown]
# ## Download data


# %%
def get_download_urls(species: str) -> tuple[str, ...]:
    """
    Get URLs to download
    """
    total_files = (
        f"https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/mon/{species}smoothed/gn/v20250403/{species}smoothed_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_gn_{tp}.nc"
        for tp in [
            "175001-189912",
            "190001-202112",
        ]
    )
    sector_files = (
        f"https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/mon/{species}{suffix}/gn/v20250403/{species}{suffix}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_gn_175001-202112.nc"
        for suffix in [
            "smoothedpercentageAGRI",
            "smoothedpercentageBORF",
            "smoothedpercentageDEFO",
            "smoothedpercentagePEAT",
            "smoothedpercentageSAVA",
            "smoothedpercentageTEMF",
        ]
    )

    res = tuple(
        (
            *total_files,
            *sector_files,
            "https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/fx/gridcellarea/gn/v20250227/gridcellarea_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_gn.nc",
        )
    )

    return res


# %%
downloaded_files_l = []
for download_url in get_download_urls(species_esgf):
    downloaded_files_l.append(
        pooch.retrieve(
            download_url,
            known_hash=None,  # from ESGF, assume safe
            fname=download_url.split("/")[-1],
            progressbar=True,
        )
    )

# downloaded_files_l

# %%
alldat = xr.open_mfdataset(downloaded_files_l)
cell_area = alldat["gridcellarea"]


# %%
def to_annual_sum(da: xr.DataArray, time_bounds: xr.DataArray) -> xr.DataArray:
    """
    Convert to an annual sum
    """
    # Much easier with cf-python
    seconds_per_step = (time_bounds.sel(bound=1) - time_bounds.sel(bound=0)).compute()

    # Yuck but ok
    if str(seconds_per_step.values.dtype) != "timedelta64[ns]":
        raise AssertionError

    seconds_per_step = seconds_per_step / 1e9

    if " s-1" not in da.attrs["units"]:
        raise AssertionError

    out_units = da.attrs["units"].replace(" s-1", "")

    res = (da * seconds_per_step.astype(int)).groupby("time.year").sum().assign_attrs(dict(units=out_units))

    return res


def to_annual_global_sum(ds: xr.Dataset, variable_of_interest: str, cell_area: xr.DataArray) -> xr.DataArray:
    """
    Convert to an annual- global-sum
    """
    # Much easier with cf-python
    annual_sum = to_annual_sum(
        ds[variable_of_interest],
        time_bounds=ds["time_bnds"],
    ).compute()

    return (annual_sum * cell_area).sum(["latitude", "longitude"])


# %%
latest_file_l = [f for f in downloaded_files_l if "2021" in str(f) and f"{species_esgf}smoothed_" in str(f)]
if len(latest_file_l) != 1:
    raise AssertionError(latest_file_l)

latest_file = latest_file_l[0]
# latest_file
exp_last_year_annual_total_Tg = float(xr.open_dataset(latest_file).attrs["annual_total_last_year_Tg_yr"])
exp_last_year_annual_total_Tg

# %%
# Make sure we can sum correctly
np.testing.assert_allclose(
    np.round(
        to_annual_global_sum(
            alldat.isel(time=range(-12, 0)), variable_of_interest=f"{species_esgf}smoothed", cell_area=cell_area
        )
        .sel(year=2021)
        .compute()
        .values
        / 1e9,
        2,
    ),
    exp_last_year_annual_total_Tg,
    rtol=1e-3,
)

# %%
total_var = alldat[f"{species_esgf}smoothed"]
# total_var

# %%
sector_variables = [v for v in alldat.data_vars if "percentage" in v]
# sector_variables


# %%
def to_regridded_per_cell(
    da: xr.DataArray,
    cell_area: xr.DataArray,
    resolution_decrease_factor: int = 2,
) -> xr.DataArray:
    """
    Convert from data per area to regridded data per cell

    The regridding is very basic,
    just upscaling to a grid that is less fine.
    (We tried other tools for this,
    but they weren't conserving mass).
    """
    if "m^2" not in cell_area.attrs["long_name"]:
        raise AssertionError

    if "m-2" not in da.attrs["units"]:
        raise AssertionError

    da_per_cell = da * cell_area

    da_per_cell = da_per_cell.assign_attrs(dict(units=da.attrs["units"].replace(" m-2", "")))

    rdf = resolution_decrease_factor
    in_lat = da_per_cell.latitude
    in_lon = da_per_cell.longitude

    if rdf != 2:  # noqa: PLR2004
        # The below will explode
        raise NotImplementedError(rdf)

    target_lat = (in_lat[::rdf].values + in_lat[1::rdf].values) / rdf
    target_lon = (in_lon[::rdf].values + in_lon[1::rdf].values) / rdf

    da_per_cell_regridded = (
        da_per_cell.rolling(latitude=rdf, min_periods=1, center=False)
        .sum()
        .sel(latitude=in_lat[1::rdf])
        .rolling(longitude=rdf, min_periods=1, center=False)
        .sum()
        .sel(longitude=in_lon[1::rdf])
        .assign_coords(
            latitude=target_lat,
            longitude=target_lon,
        )
    )

    return da_per_cell_regridded


# %%
total_v_per_cell_regridded = to_regridded_per_cell(
    da=total_var,
    cell_area=cell_area,
)
total_v_per_cell_regridded_per_year = to_annual_sum(
    da=total_v_per_cell_regridded,
    time_bounds=alldat["time_bnds"],
)
# Make sure we can regrid correctly
np.testing.assert_allclose(
    np.round(
        total_v_per_cell_regridded_per_year.sel(year=2021).sum(["latitude", "longitude"]).values / 1e9,
        2,
    ),
    exp_last_year_annual_total_Tg,
    rtol=1e-3,
)

# %%
for sv_name in tqdm.auto.tqdm(sector_variables):
    sv = alldat[sv_name]
    if sv.attrs["units"] != "%":
        raise AssertionError

    sector_per_area = (total_var * sv / 100.0).assign_attrs(dict(units=total_var.attrs["units"]))

    sector_per_cell_regridded = to_regridded_per_cell(
        da=sector_per_area,
        cell_area=cell_area,
    )
    sector_per_cell_regridded_per_year = to_annual_sum(
        da=sector_per_cell_regridded,
        time_bounds=alldat["time_bnds"],
    )

    n_years_for_regression = 10
    coeffs_ds = sector_per_cell_regridded_per_year.isel(year=range(-10, 0)).polyfit("year", deg=1, skipna=True)

    years_interp = np.arange(
        sector_per_cell_regridded_per_year["year"].max() + 1,
        HARMONISATION_YEAR + 1,
    )
    year_factor = xr.DataArray(years_interp, dims=("year",), coords=dict(year=years_interp))

    extrapolated = coeffs_ds["polyfit_coefficients"].sel(degree=1) * year_factor + coeffs_ds[
        "polyfit_coefficients"
    ].sel(degree=0).assign_attrs(dict(units=sector_per_cell_regridded_per_year.attrs["units"]))

    full = xr.concat([sector_per_cell_regridded_per_year, extrapolated], dim="year")

    selectors = dict(latitude=8.125, longitude=0.0, method="nearest")
    # Plot extrapolation
    fig, ax = plt.subplots()
    full.sel(**selectors).sel(year=range(2013, 2023 + 1)).plot.line(ax=ax)
    ax.set_title(sv_name)

    plt.show()

    sector = sv_name.split("percentage")[-1]
    # sector

    full_by_country = (full * country_mask).sum(["latitude", "longitude"]).assign_attrs(dict(units=full.attrs["units"]))
    # # Comment this out to crunch full dataset
    # full_by_country = full_by_country.sel(year=[2020, 2021, 2022, 2023])
    print("Starting to compute data by country")
    with ProgressBar(dt=5.0):
        full_by_country_res = full_by_country.compute().assign_coords(sector=sector)

    print("Finished computing data by country")
    # full_by_country

    out_file = BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR / f"{species}_{sector}.nc"
    BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Writing {out_file}")
    full_by_country_res.to_netcdf(out_file)
    print(f"Wrote {out_file}")

# %% [markdown]
# ## Delete raw files
#
# Save our disks

# %%
for fp in downloaded_files_l:
    os.unlink(fp)
