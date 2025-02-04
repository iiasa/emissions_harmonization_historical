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
# # Prepare GFED BB4CMIP
#
# Prepare GFED data as submitted to
# [ESGF](https://aims2.llnl.gov/search?project=input4MIPs&activeFacets=%7B%22mip_era%22%3A%22CMIP6Plus%22%2C%22institution_id%22%3A%22DRES%22%7D).

# %%
from collections import defaultdict
from pathlib import Path

import numpy as np
import ptolemy
import scmdata
import xarray as xr
import xarray_regrid  # noqa: F401
from tqdm import tqdm

from emissions_harmonization_historical.constants import DATA_ROOT, GFED_PROCESSING_ID

# %%
raw_data_path = DATA_ROOT / "national/gfed-bb4cmip/data_raw/data"
raw_data_path

# %%
gfed_processed_output_dir = DATA_ROOT / Path("national", "gfed-bb4cmip", "processed")
gfed_processed_output_dir

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
    if variable == "Higher":
        # My bad, we didn't catch this in validation
        variable = "_".join(file.name.split("_")[:2])

    bb4cmip_file_groups[variable].append(file)

# %%
bb4cmip_file_groups.keys()

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
    "CO": {
        "unit_label": "Mt CO / yr",
        "filename_label": "CO",
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
# Get this into memory so it doesn't load every time
cell_area_gfed = cell_area_gfed.compute()
# not sure needed

# get iso map at half degree
idxr = xr.open_dataarray(gfed_isomask, chunks="auto")  # chunks={"iso": 1})
# Get this into memory so it doesn't load every time
idxr = idxr.compute()

# %%
idxr.sel(iso="aus").plot()


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
        Do global emissions (rather than national)

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
            scenario="historical",
            region=region,
            unit=unit_label,
            variable=f"Emissions|{species}|BB4CMIP",
        ),
        index=in_da.year,
    )

    return out


# %%
gfed_processed_output_dir.mkdir(exist_ok=True, parents=True)
for species, species_info in tqdm(species_data.items(), desc="Species"):
    print(f"Processing {species=}")

    species_ds = xr.open_mfdataset(bb4cmip_file_groups[species], combine_attrs="drop_conflicts").rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    # # Handy for testing
    # species_ds = species_ds.isel(time=range(12 * 3))
    # species_ds = species_ds.isel(time=species_ds["time"].dt.year.isin([1750, 1899, 1900, 2022]))

    # # Handy for seeing what's going on
    # display(species_ds)

    # First get the emissions on an annual grid
    emissions_da = species_ds[species].chunk("auto")
    assert emissions_da.units == "kg m-2 s-1"

    # get annual sum emissions in each grid cell
    ### weight by month length
    seconds_per_month = (
        np.tile([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], len(emissions_da.time) // 12) * 24 * 60 * 60
    )
    weights = xr.DataArray(seconds_per_month, dims=("time",), coords=emissions_da.coords["time"].coords)
    gridded_annual_emissions_rate = (emissions_da * weights).groupby("time.year").sum().compute()  # kg / m2 / yr
    ### no weighting by month length
    ## (testing in case it lines up better with data provider, doesn't seem to matter)
    # seconds_per_year = 365 * 24 * 60 * 60
    # gridded_annual_emissions_rate = (
    #    emissions_da.groupby("time.year").mean().compute()
    # ) * seconds_per_year  # kg / m2 / yr

    # Split into chunks from here to avoid destroying laptops
    n_yrs_to_process = 10
    out_world_l = []
    out_country_l = []
    # Convert to list to help tqdm out
    yr_starts = list(gridded_annual_emissions_rate.year[::n_yrs_to_process].values)
    for yr_start in tqdm(yr_starts, desc=f"{species} chunks"):
        yr_end = yr_start + n_yrs_to_process
        if yr_end > gridded_annual_emissions_rate.year.max():
            yr_end = int(gridded_annual_emissions_rate.year.max()) + 1

        chunk_to_process = gridded_annual_emissions_rate.sel(year=range(yr_start, yr_end))

        # regrid to half degree
        regridded_annual_emissions_rate = chunk_to_process.regrid.conservative(
            idxr
        )  # still kg / m2 / yr, now coarser grid

        # get cell areas of new grid
        cell_area = xr.DataArray(
            ptolemy.cell_area(lats=regridded_annual_emissions_rate.lat, lons=regridded_annual_emissions_rate.lon)
        )  # m2

        # emissions in each country, annual time series
        # Mt / yr in each country, the 1e9 is kg to Mt
        country_emissions = ((regridded_annual_emissions_rate * cell_area * idxr).sum(["lat", "lon"])).compute() / 1e9
        world_emissions = (regridded_annual_emissions_rate * cell_area).sum(["lat", "lon"]).compute() / 1e9

        out_world_chunk = gfed_to_scmrun(world_emissions, unit_label=species_info["unit_label"], world=True)
        out_country_chunk = gfed_to_scmrun(country_emissions, unit_label=species_info["unit_label"])

        out_world_l.append(out_world_chunk)
        out_country_l.append(out_country_chunk)
        # break

    out_world = scmdata.run_append(out_world_l)
    out_country = scmdata.run_append(out_country_l)

    # Last checks with what data provider thinks
    for original_file in bb4cmip_file_groups[species]:
        orig_ds = xr.open_dataset(original_file)
        timerange = original_file.stem.split("_")[-1].split("-")
        compare_unit = species_info["unit_label"].replace("Mt", "Tg")

        for i, timestamp in enumerate(timerange):
            year = int(timestamp[:4])
            if i < 1:
                compare_val = float(orig_ds.attrs["annual_total_first_year_Tg_yr"])
            elif i == 1:
                compare_val = float(orig_ds.attrs["annual_total_last_year_Tg_yr"])
            else:
                raise NotImplementedError()

            try:
                np.testing.assert_allclose(
                    np.round(out_world.filter(year=year).convert_unit(compare_unit).values.squeeze(), 2),
                    compare_val,
                    rtol=7.5e-3,
                )
            except AssertionError as exc:
                print(f"Difference from what the data provider thinks the global value should be in {year}")
                print(exc)

    # Write out temporary files to save RAM and process combined dataset in 0104
    gfed_temp_file_world = gfed_processed_output_dir / f"{species}_world_{GFED_PROCESSING_ID}.csv"
    out_world.timeseries(time_axis="year").to_csv(gfed_temp_file_world)
    print(f"Wrote {gfed_temp_file_world}")
    gfed_temp_file_country = gfed_processed_output_dir / f"{species}_national_{GFED_PROCESSING_ID}.csv"
    out_country.timeseries(time_axis="year").to_csv(gfed_temp_file_country)
    print(f"Wrote {gfed_temp_file_country}")

    # break
