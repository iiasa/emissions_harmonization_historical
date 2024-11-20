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
from collections import defaultdict
from pathlib import Path

import numpy as np
import ptolemy
import scmdata
import xarray as xr
from tqdm import tqdm

from emissions_harmonization_historical.constants import DATA_ROOT

# %%
raw_data_path = DATA_ROOT / "national/gfed-bb4cmip/data_raw/data"
raw_data_path

# %%
gfed_processed_output_file = DATA_ROOT / Path("national", "gfed", "processed", "gfed-bb4cmip_cmip7_national_alpha.csv")

# %%
gfed_data_aux_folder = DATA_ROOT / Path("national", "gfed", "data_aux")
gfed_isomask = Path(gfed_data_aux_folder, "iso_mask.nc")  # for aggregating to countries

# %% [markdown]
# Group raw data into variable groups which can be used for processing.

# %%
bb4cmip_files = list(raw_data_path.rglob("*.nc"))
bb4cmip_file_groups = defaultdict(list)

for file in bb4cmip_files:
    variable = file.name.split("_")[0]
    bb4cmip_file_groups[variable].append(file)

# %%
species_data = {
    "BC": {
        "unit_label": "Mt BC / yr",
        "filename_label": "BC",
    },
    "NMVOC": {
        "unit_label": "Mt NMVOC / yr",
        "filename_label": "NMVOC_bulk",
    },
    "CO2": {
        "unit_label": "Mt CO2 / yr",
        "filename_label": "CO2",
    },
    "CH4": {
        "unit_label": "Mt CH4 / yr",
        "filename_label": "CH4",
    },
    "N2O": {
        "unit_label": "Mt N2O / yr",
        "filename_label": "N2O",
    },
    "OC": {
        "unit_label": "Mt OC / yr",
        "filename_label": "OC",
    },
    "NH3": {
        "unit_label": "Mt NH3 / yr",
        "filename_label": "NH3",
    },
    "NOx": {
        "unit_label": "Mt NO / yr",
        "filename_label": "NOx",
    },
    "SO2": {
        "unit_label": "Mt SO2 / yr",
        "filename_label": "SO2",
    },
}

# %%
# cell areas of GFED map
cell_area_gfed = xr.open_mfdataset(bb4cmip_file_groups["gridcellarea"]).rename({"latitude": "lat", "longitude": "lon"})
# not sure needed

# get iso map at half degree
idxr = xr.open_dataarray(gfed_isomask, chunks="auto")  # chunks={"iso": 1})


# %%
def gfed_to_scmrun(in_da: xr.DataArray, *, unit_label: str, world: bool = False) -> scmdata.ScmRun:
    """
    Convert an `xr.DataArray` to `scmdata.ScmRun`

    Custom to this notebook

    Parameters
    ----------
    in_da
        Input `xr.DataArray` to convert

    unit_label
        The label to apply to the unit as it will be handled in the output.

        This typically includes the species, e.g. "Mt BC / yr"

    world
        Do global emissions

    Returns
    -------
    :
        `scmdata.ScmRun`
    """
    df = in_da.to_numpy()

    if world:
        region = "World"
    else:
        region = in_da.iso.to_numpy()

    out = scmdata.ScmRun(
        df,
        columns=dict(
            model="GFED-BB4CMIP",
            scenario="historical-CMIP6Plus",
            region=region,
            unit=unit_label,
            variable=f"Emissions|{species}|BB4CMIP",
        ),
        index=np.arange(1750, 2023, dtype=int),
    )  # .interpolate(target_times=np.arange(1750, 2023, dtype=int)).timeseries(time_axis="year")

    return out


# %%
# out = []
for species in tqdm(species_data):
    species_ds = xr.open_mfdataset(bb4cmip_file_groups[species], combine_attrs="drop_conflicts").rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    # species_ds

    # First get the emissions on to an annual grid
    emissions_da = species_ds[species].chunk("auto")  # kg / m2 / s

    # weight by month length
    seconds_per_month = (
        np.tile([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], len(emissions_da.time) // 12) * 24 * 60 * 60
    )
    weights = xr.DataArray(seconds_per_month, dims=("time",), coords=emissions_da.coords["time"].coords)

    # get annual sum emissions in each grid cell
    gridded_annual_emissions_rate = (emissions_da * weights).groupby("time.year").sum().compute()  # kg / m2 / yr

    # regrid to half degree
    regridded_annual_emissions_rate = gridded_annual_emissions_rate.regrid.conservative(
        idxr
    )  # still kg / m2 / yr, now coarser grid

    # get cell areas of new grid
    cell_area = xr.DataArray(
        ptolemy.cell_area(lats=regridded_annual_emissions_rate.lat, lons=regridded_annual_emissions_rate.lon)
    )  # m2

    # emissions in each country, annual time series
    country_emissions = ((regridded_annual_emissions_rate * cell_area * idxr).sum(["lat", "lon"])).compute() / 1e9
    country_emissions  # Mt / yr in each country, the 1e9 is kg to Mt
    world_emissions = (regridded_annual_emissions_rate * cell_area).sum(["lat", "lon"]).compute() / 1e9

    out_world = gfed_to_scmrun(world_emissions, unit_label=species_data[species]["unit_label"], world=True)
    out_country = gfed_to_scmrun(country_emissions, unit_label=species_data[species]["unit_label"])

    # write out temporary files to save RAM and process combined dataset later
    out_world = out_world.append(out_country)

    gfed_temp_file = DATA_ROOT / Path("national", "gfed-bb4cmip", "processed", f"{species}.csv")
    out_world = out_world.interpolate(target_times=np.arange(1750, 2023, dtype=int)).timeseries(time_axis="year")
    out_world.to_csv(gfed_temp_file)


# %%
# out = scmdata.run_append(out).interpolate(target_times=np.arange(1750, 2023, dtype=int)).timeseries(time_axis="year")

# %%
# out.to_csv(gfed_processed_output_file)
