# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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

# %%
# import external packages and functions
from pathlib import Path

# just for debugging
import numpy as np
import pandas as pd
import pandas_indexing as pix
import xarray as xr
from pandas_indexing import set_openscm_registry_as_default

from emissions_harmonization_historical.constants import CAMS_PROCESSING_ID, DATA_ROOT

# set unit registry
ur = set_openscm_registry_as_default()
pix.units.set_openscm_registry_as_default()

# %%
gfed_data_folder = DATA_ROOT / Path("national", "gfed", "data_raw")
gfed_data_aux_folder = DATA_ROOT / Path("national", "gfed", "data_aux")

cams_data_folder = DATA_ROOT / Path("national", "cams", "data_raw")

cams_country_temp_file = DATA_ROOT / Path("national", "cams", "processed", "cams_country_temporaryfile.csv")
cams_world_temp_file = DATA_ROOT / Path("national", "cams", "processed", "cams_world_temporaryfile.csv")

cams_country_proc_file = DATA_ROOT / Path("national", "cams", "processed", f"cams_country_{CAMS_PROCESSING_ID}.csv")
cams_world_proc_file = DATA_ROOT / Path("national", "cams", "processed", f"cams_world_{CAMS_PROCESSING_ID}.csv")

# use the iso mask from gfed
isomask = Path(gfed_data_aux_folder, "iso_mask.nc")
# idxr = xr.open_dataarray(isomask, chunks={"iso": 1})
idxr = xr.open_dataarray(isomask)

# %% [markdown]
# The CAMS emissions are given as absolute values on each cell (Tg=Mt), not as surface concentrations.
# So they just need to be summed, not averaged. Since the iso3 mask is available at coarser resolution
# than the data, the data is coarsened to the same resolution.

# %%
iso3_list = idxr.iso.to_numpy()

gases = [
    "bc",
    "ch4",
    "co",
    "nh3",
    "nmvocs",
    "nox",
    "oc",
    "so2",
    "n2o",
    "co2_excl_short-cycle_org_C",
    "co2_short-cycle_org_C",
]
years = np.arange(2000, 2026)
# gases = ["bc", "ch4", "nh3"]
# years = np.arange(2000, 2003)  # for quick debugging
#
for i, gas in enumerate(gases):
    for j, y in enumerate(years):
        folderName = "CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_" + gas + "_v6.2_yearly"
        fileName = "CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_" + gas + "_v6.2_yearly_" + str(y) + ".nc"
        fullName = folderName + "/" + fileName
        print(fullName)
        cams_data_file = Path(cams_data_folder, fullName)
        emissions = xr.open_dataset(cams_data_file)
        # remove the gridcell_area and sum variables
        emissions = emissions.drop_vars(["gridcell_area", "sum"])
        # compute world total for ships
        world_emissions_ant_yr = emissions.sum(["lat", "lon"])
        # coarsen to the resolution of the iso3 mask
        # we need to go from a 0.1x0.1 grid to a 0.5x0.5 grid
        emissions = emissions.coarsen(lat=5, lon=5, boundary="trim").sum()
        print("coarsened")
        # align grids by interpolation (is this ok?)
        emissions = emissions.interp(lon=idxr.lon, lat=idxr.lat, method="linear")
        print("interpolated")
        for k, reg in enumerate(iso3_list):
            country_emissions_reg = (emissions * idxr.sel(iso=reg)).sum(["lat", "lon"])
            if k == 0:
                # if the array does not exist, create it
                country_emissions_yr = country_emissions_reg
            else:
                country_emissions_yr = xr.concat([country_emissions_yr, country_emissions_reg], dim="iso")
        if j == 0:
            country_emissions_gas = country_emissions_yr
            world_emissions_ant_gas = world_emissions_ant_yr
        else:
            country_emissions_gas = xr.concat([country_emissions_gas, country_emissions_yr], dim="time")
            world_emissions_ant_gas = xr.concat([world_emissions_ant_gas, world_emissions_ant_yr], dim="time")
        del country_emissions_yr
        del world_emissions_ant_yr
        # print("time for concatenation ", perf_counter()-t0)
    # turn "nmvocs" to "nmvoc" for consistency
    if gas == "nmvocs":
        country_emissions_gas = country_emissions_gas.assign_coords(gas="NMVOC")
        world_emissions_ant_gas = world_emissions_ant_gas.assign_coords(gas="NMVOC")
    elif gas == "nox":
        country_emissions_gas = country_emissions_gas.assign_coords(gas="NOx")
        world_emissions_ant_gas = world_emissions_ant_gas.assign_coords(gas="NOx")
    elif gas == "co2_excl_short-cycle_org_C":
        country_emissions_gas = country_emissions_gas.assign_coords(gas="CO2_excl_short-cycle_org_C")
        world_emissions_ant_gas = world_emissions_ant_gas.assign_coords(gas="CO2_excl_short-cycle_org_C")
    elif gas == "co2_short-cycle_org_C":
        country_emissions_gas = country_emissions_gas.assign_coords(gas="CO2_short-cycle_org_C")
        world_emissions_ant_gas = world_emissions_ant_gas.assign_coords(gas="CO2_short-cycle_org_C")
    else:
        # default: just convert to uppercase
        country_emissions_gas = country_emissions_gas.assign_coords(gas=gas.upper())
        world_emissions_ant_gas = world_emissions_ant_gas.assign_coords(gas=gas.upper())
    print("assigned")
    if i == 0:
        country_emissions = country_emissions_gas
        world_emissions_ant = world_emissions_ant_gas
    else:
        country_emissions = xr.concat([country_emissions, country_emissions_gas], dim="gas")
        world_emissions_ant = xr.concat([world_emissions_ant, world_emissions_ant_gas], dim="gas")
    del country_emissions_gas
    del world_emissions_ant_gas

# %% [markdown]
# Now the CAMS-GLOB-AIR data is imported.
#
# **TODO**: the sum sector of CAMS-GLOB-ANT does not contain air and so could be misleading, should we remove it?

# %%
gases_air = ["bc", "co", "nh3", "nmvoc", "nox", "oc", "so2", "co2"]
years_air = np.arange(2000, 2023)
# gases_air = ["bc", "co", "nh3"]
# years_air = np.arange(2000, 2003)

print("Importing CAMS-GLOB-AIR data")
for i, gas in enumerate(gases_air):
    for j, y in enumerate(years_air):
        print(gas, y)
        folderName = "CAMS-GLOB-AIR_Glb_0.5x0.5_anthro_" + gas + "_v2.1_yearly"
        fileName = "CAMS-GLOB-AIR_Glb_0.5x0.5_anthro_" + gas + "_v2.1_yearly_" + str(y) + ".nc"
        fullName = folderName + "/" + fileName
        print(fullName)
        cams_data_file = Path(cams_data_folder, fullName)
        emissions = xr.open_dataset(cams_data_file)
        # remove the gridcell_area variable
        emissions = emissions.drop_vars("gridcell_area")
        # nox has the "level" coordinate, but it seems spurious
        if "level" in list(emissions.coords):
            emissions = emissions.drop_vars("level")
        # todo: compute world value before aligning
        world_emissions_air_yr = emissions.sum(["lat", "lon"])
        # air emissions are already on the same grid as idxr: no need to coarsen,
        # but idxr is clipping the poles, so we need to align
        emissions, _ = xr.align(emissions, idxr, join="inner", exclude=["iso", "time"])
        for k, reg in enumerate(iso3_list):
            country_emissions_reg = (emissions * idxr.sel(iso=reg)).sum(["lat", "lon"])
            if k == 0:
                # if the array does not exist, create it
                country_emissions_yr = country_emissions_reg
            else:
                country_emissions_yr = xr.concat([country_emissions_yr, country_emissions_reg], dim="iso")
        if j == 0:
            country_emissions_gas = country_emissions_yr
            world_emissions_air_gas = world_emissions_air_yr
        else:
            country_emissions_gas = xr.concat([country_emissions_gas, country_emissions_yr], dim="time")
            world_emissions_air_gas = xr.concat([world_emissions_air_gas, world_emissions_air_yr], dim="time")
        del country_emissions_yr
        del world_emissions_air_yr
    if gas == "nox":
        country_emissions_gas = country_emissions_gas.assign_coords(gas="NOx")
        world_emissions_air_gas = world_emissions_air_gas.assign_coords(gas="NOx")
    else:
        country_emissions_gas = country_emissions_gas.assign_coords(gas=gas.upper())
        world_emissions_air_gas = world_emissions_air_gas.assign_coords(gas=gas.upper())
    if i == 0:
        country_emissions_air = country_emissions_gas
        world_emissions_air = world_emissions_air_gas
    else:
        country_emissions_air = xr.concat([country_emissions_air, country_emissions_gas], dim="gas")
        world_emissions_air = xr.concat([world_emissions_air, world_emissions_air_gas], dim="gas")
    del country_emissions_gas
    del world_emissions_air_gas

# %% [markdown]
# Now that all data have been imported, convert to pandas dataframe and to IAMC style names.
# Intermediate files are exported for easier import into R for comparison with data aggregated using other methods.

# %%

# make model map
varlist = list(country_emissions.variables)
varlist.remove("time")
varlist.remove("iso")
varlist.remove("gas")
models = pd.MultiIndex.from_tuples(
    [(var, "CAMS-GLOB-ANT-v6.2") for var in varlist] + [("avi", "CAMS-GLOB-AIR-v2.1")], names=["sector", "model"]
)

country_emissions = xr.merge([country_emissions, country_emissions_air])

world_emissions = xr.merge([world_emissions_ant, world_emissions_air])

country_emissions_df = country_emissions.to_array(name="emissions").to_dataframe().reset_index()
world_emissions_df = world_emissions.to_array(name="emissions").to_dataframe().reset_index()

# clean up dates to have just years
country_emissions_df["time"] = country_emissions_df["time"].dt.to_period("Y")
world_emissions_df["time"] = world_emissions_df["time"].dt.to_period("Y")

# add missing columns
country_emissions_df["scenario"] = "historical"  # this is debatable: CAMS has some extrapolation
world_emissions_df["scenario"] = "historical"
world_emissions_df["region"] = "World"

# now pivot to have years as columns, as required
country_emissions_df = country_emissions_df.pivot(
    columns="time", index=("variable", "gas", "iso", "scenario"), values="emissions"
)
world_emissions_df = world_emissions_df.pivot(
    columns="time", index=("variable", "gas", "scenario", "region"), values="emissions"
)

# add units
units = pd.MultiIndex.from_tuples(
    [
        ("BC", "Mt BC/yr"),
        ("CH4", "Mt CH4/yr"),
        ("CO", "Mt CO/yr"),
        ("NH3", "Mt NH3/yr"),
        ("NMVOC", "Mt NMVOC/yr"),
        ("NOx", "Mt NO/yr"),
        ("OC", "Mt OC/yr"),
        ("SO2", "Mt SO2/yr"),
        ("N2O", "Mt N2O/yr"),
        ("CO2", "Mt CO2/yr"),
        ("CO2_excl_short-cycle_org_C", "Mt CO2/yr"),
        ("CO2_short-cycle_org_C", "Mt CO2/yr"),
    ],
    names=["gas", "unit"],
)

country_emissions_df = country_emissions_df.rename_axis(index={"iso": "region", "variable": "sector"}).pix.semijoin(
    units, how="left"
)
world_emissions_df = world_emissions_df.pix.semijoin(units, how="left").rename_axis(index={"variable": "sector"})

# add models
country_emissions_df = country_emissions_df.pix.semijoin(models, how="left")
world_emissions_df = world_emissions_df.pix.semijoin(models, how="left")

# aggregate Serbia and Kosovo as for gfed
country_combinations = {"srb_ksv": ["srb", "srb (kosovo)"]}
country_emissions_df = country_emissions_df.pix.aggregate(region=country_combinations)

country_emissions_df.to_csv(cams_country_temp_file)
world_emissions_df.to_csv(cams_world_temp_file)

# TODO: check! This is a tentative mapping! To which sectors should we map exactly?
# These are the sector descriptions from eccad
sector_mapping = {
    "awb": "Agricultural Waste Burning",
    "com": "Commercial",
    "ene": "Power generation",
    "fef": "Fugitives",
    "ind": "Industrial process",
    "ref": "Refineries",
    "res": "Residential",
    "shp": "Ships",
    "sum": "Sum Sectors",
    "swd": "Solid waste and waste water",
    "tnr": "Off Road transportation",
    "tro": "Road transportation",
    "agl": "Agriculture livestock",
    "ags": "Agriculture soils",
    "fef_coal": "Fugitive coal",
    "fef_gas": "Fugitive gas",
    "fef_oil": "Fugitive oil",
    "slv": "Solvents",
    "avi": "Aviation",
}

# replace the sector mapping
country_emissions_df = country_emissions_df.reset_index().replace({"sector": sector_mapping})
world_emissions_df = world_emissions_df.reset_index().replace({"sector": sector_mapping})

# rename to IAMC-style variable names
country_emissions_df.insert(
    2, "variable", "Emissions|" + country_emissions_df["gas"] + "|" + country_emissions_df["sector"]
)
world_emissions_df.insert(1, "variable", "Emissions|" + world_emissions_df["gas"] + "|" + world_emissions_df["sector"])

country_emissions_df = country_emissions_df.drop(columns=["sector", "gas"])
world_emissions_df = world_emissions_df.drop(columns=["sector", "gas"])

country_emissions_df = country_emissions_df.set_index(["model", "scenario", "region", "variable", "unit"])
world_emissions_df = world_emissions_df.set_index(["model", "scenario", "region", "variable", "unit"])

country_emissions_df.to_csv(cams_country_proc_file)
world_emissions_df.to_csv(cams_world_proc_file)
