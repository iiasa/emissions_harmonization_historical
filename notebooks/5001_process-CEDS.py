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
# # Process CEDS
#
# Process data from [CEDS](https://github.com/JGCRI/CEDS).

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
from gcages.index_manipulation import split_sectors
from pandas_indexing.core import isna
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.ceds import get_map, read_CEDS
from emissions_harmonization_historical.constants_5000 import (
    CEDS_PROCESSED_DB,
    CEDS_RAW_PATH,
    CEDS_TOP_LEVEL_RAW_PATH,
    CEDS_VERSION_ID,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# %%
RUN_OPTIONAL_BYFUEL_PROCESSING_FOR_AVIATION_AND_SHIPPING = False

# %% [markdown]
# ## Set up

# %%
# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
pandas_openscm.register_pandas_accessor()

# %%
CEDS_SECTOR_MAPPING_FILE = CEDS_TOP_LEVEL_RAW_PATH / "auxilliary" / "sector_mapping.xlsx"

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

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Sector mapping

# %%
ceds_mapping = pd.read_excel(
    CEDS_SECTOR_MAPPING_FILE,
    sheet_name="CEDS Mapping 2024",
)
ceds_map = get_map(
    ceds_mapping,
    "59_Sectors_2024",  # note; with 7BC and 2L now added it is actually 61 sectors, not 59 anymore
)
# ceds_map.to_frame(index=False)

# %% [markdown]
# ### CEDS emissions data

# %%
ceds = (
    pd.concat(
        # for all earlier versions processed here
        # (i.e. Drive_2025_03_18, Drive_2025_03_11, Zenodo_2024_07_08)
        # read_CEDS(
        #     Path(CEDS_RAW_PATH) / f"{s}_CEDS_emissions_by_country_sector_v{ceds_release}.csv"
        # )
        read_CEDS(  # for Zenodo_2025_03_18
            CEDS_RAW_PATH
            / f"CEDS_{CEDS_VERSION_ID}_aggregate"
            / f"{s}_CEDS_estimates_by_country_sector_{CEDS_VERSION_ID}.csv"
        )
        for s in species
    )
    .rename_axis(index={"region": "country"})
    .pix.semijoin(ceds_map, how="outer")
)

ceds.loc[isna].pix.unique(["sector_59", "sector"])  # print sectors with NAs
# check that all emissions are mapped properly
assert ceds.loc[isna].reset_index().sector_59.unique().tolist() == ["6B_Other-not-in-total"]

# ceds

# %%
ceds_totals = ceds.openscm.groupby_except(["sector_59", "sector", "country"]).sum()
# ceds_totals.loc[:, 2020:]

# %% [markdown]
# ## Process

# %% [markdown]
# ### Ensure unmapped sectors are handled

# %%
# '6B_Other-not-in-total' is not assigned, and normally not used by CEDS.
# To be certain that we notice it when something changes,
# we check that it is indeed zero.
year_cols = ceds.columns.astype(int)
first_year = year_cols[0]
last_year = year_cols[-1]
sum_of_6B_other = (
    ceds.loc[pix.ismatch(sector_59="6B_Other-not-in-total")]
    .sum(axis=1)  # sum across years
    .sum(axis=0)  # sum across species and countries
)
assert sum_of_6B_other == 0

# %% [markdown]
# ### Update units and names

# %%
ceds = ceds.pix.dropna(subset=["units"]).pix.format(unit="{units}/yr", drop=True)

# adjust units; change all to values to Mt instead of kt
ceds = pix.units.convert_unit(ceds, lambda x: "Mt " + x.removeprefix("kt").strip())
# exception for N2O/yr, which should remain in kt following https://github.com/IAMconsortium/common-definitions/
ceds = pix.units.convert_unit(ceds, lambda x: "kt " + x.removeprefix("Mt").strip() if x == "Mt N2O/yr" else x)


# unit updates
def update_unit(unit, em):
    """Update unit from raw CEDS units"""
    if unit == "Mt C/yr" and em == "BC":
        return "Mt BC/yr"

    if unit == "Mt C/yr" and em == "OC":
        return "Mt OC/yr"

    if unit == "Mt NMVOC/yr" and em == "NMVOC":
        return "Mt VOC/yr"

    return unit


ceds.index = pd.MultiIndex.from_tuples(
    [(em, country, sector_59, sector, update_unit(unit, em)) for em, country, sector_59, sector, unit in ceds.index],
    names=ceds.index.names,
)

# change name(s) of emissions species
# use 'Sulfur' instead of 'SO2'
ceds = update_index_levels_func(ceds, {"em": lambda x: x.replace("SO2", "Sulfur").replace("NMVOC", "VOC")})

ceds

# %%
# TODO: add CEDS extension back to 1750 somewhere
# ceds.loc[["CH4", "N2O"]]

# %% [markdown]
# ### Calculate aggregates of interest

# %%
ceds = ceds.groupby(["em", "country", "unit", "sector"]).sum().pix.fixna()  # group and fix NAs
# ceds

# %%
# aggregate countries where this is necessary, e.g. because of specific other data (like SSP socioeconomic driver data)
# based on the new SSP data, we only need to aggregate Serbia and Kosovo
country_combinations = {
    # "isr_pse": ["isr", "pse"], "sdn_ssd": ["ssd", "sdn"],
    "kos": ["srb (kosovo)"]
}
ceds = ceds.pix.aggregate(country=country_combinations)
# ceds

# %% [markdown]
# ### Format in IAMC style

# %%
ceds_reformatted = ceds.rename_axis(index={"em": "variable", "country": "region"})
ceds_reformatted

# %%
# rename to IAMC-style variable names including standard index order
ceds_reformatted_iamc = (
    ceds_reformatted.pix.format(variable="Emissions|{variable}|{sector}", drop=True)
    .pix.assign(scenario=HISTORY_SCENARIO_NAME, model=f"CEDS_{CEDS_VERSION_ID}")
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
).sort_values(by=["region", "variable"])
# ceds_reformatted_iamc

# %% [markdown]
# ### Check units

# %%
assert_units_match_wishes(ceds_reformatted_iamc)

# %% [markdown]
# ### Map national aviation emissions to global
#
# Since release 2025_03_18, CEDS splits up aviation between its 'global' region (for international aircraft) and other territories. Before it was all allocated to the 'global' region, which is what we were expecting, and what we want to continue to use here.
#
# 1. take all national aircraft emissions and aggregate and rename to global
# 2. aggregate global emissions + national without aircraft + national aircraft aggregated to global

# %%
ceds_reformatted_iamc.loc[pix.ismatch(variable="Emissions|*|Aircraft")].loc[
    pix.isin(region=["global", "usa"])
]  # show that there is both international and domestic aviation emissions in this data

# %%
out_no_aircraft = ceds_reformatted_iamc.loc[~pix.ismatch(variable="**|Aircraft")]

# %%
out_aircraft = (
    ceds_reformatted_iamc.loc[pix.ismatch(variable="**|Aircraft")]
    .groupby(["model", "scenario", "variable", "unit"])
    .sum(numeric_only=True)
    .pix.assign(region="global")
)
out_aircraft = out_aircraft.reorder_levels(out_no_aircraft.index.names)

# %%
out = pd.concat([out_aircraft, out_no_aircraft], axis=0)

# %%
assert (
    out.loc[~pix.ismatch(variable="**Aircraft")].shape[0]
    == ceds_reformatted_iamc.loc[~pix.ismatch(variable="**Aircraft")].shape[0]
)

# %% [markdown]
# ### Check we didn't lose any mass

# %%
res_sum = split_sectors(out).openscm.groupby_except(["region", "sectors"]).sum().pix.project(["species", "unit"])
res_sum.index = res_sum.index.rename({"species": "em", "unit": "units"})
res_sum = pix.units.convert_unit(
    res_sum,
    lambda x: x.replace("Mt", "kt"),
    level="units",
)
res_sum = update_index_levels_func(
    res_sum,
    {
        "em": lambda x: x.replace("Sulfur", "SO2").replace("VOC", "NMVOC"),
        "units": lambda x: x.replace("/yr", "").replace("kt ", "kt").replace("VOC", "NMVOC"),
    },
)


def update_units_r(em, units):
    """Update units back to raw CEDS units"""
    if em in ("BC", "OC"):
        return "ktC"

    return units


res_sum.index = pd.MultiIndex.from_tuples(
    [(em, update_units_r(em, units)) for em, units in res_sum.index],
    names=res_sum.index.names,
)

pd.testing.assert_frame_equal(
    res_sum,
    ceds_totals,
    check_like=True,
)

# %%
ceds_totals.loc[:, 2020:]

# %%
res_sum.loc[:, 2020:]

# %% [markdown]
# ## Save formatted CEDS data

# %%
CEDS_PROCESSED_DB.save(out.pix.assign(stage="iso3c_ish"), allow_overwrite=True)

# %% [markdown]
# ----------
#
#
# # Optional: run also global, by fuel (aircraft and international shipping)
# Has been used as additional information for IAM modellers.
# Available from CEDS Zenodo as well, to download from here (if you haven't already done that): https://zenodo.org/records/15059443

# %%
CEDS_RAW_PATH

# %%
CEDS_VERSION_ID

# %%
if RUN_OPTIONAL_BYFUEL_PROCESSING_FOR_AVIATION_AND_SHIPPING:
    ceds_by_fuel = pd.concat(
        # for all earlier versions processed here
        # (i.e. Drive_2025_03_18, Drive_2025_03_11, Zenodo_2024_07_08)
        # read_CEDS(
        #     Path(CEDS_RAW_PATH) / f"{s}_CEDS_emissions_by_country_sector_v{ceds_release}.csv"
        # )
        read_CEDS(  # for Zenodo_2025_03_18
            CEDS_RAW_PATH
            / f"CEDS_{CEDS_VERSION_ID}_aggregate"
            / f"{s}_CEDS_global_estimates_by_sector_fuel_{CEDS_VERSION_ID}.csv"
        )
        for s in species
    ).pix.semijoin(ceds_map, how="outer")

    # only aircraft and shipping
    ceds_by_fuel_aviation_shipping = ceds_by_fuel.loc[pix.isin(sector=["Aircraft", "International Shipping"])]

    # assign all emissions to the global level
    ceds_by_fuel_aviation_shipping = pix.assignlevel(ceds_by_fuel_aviation_shipping, region="World")

    # aggregate
    ceds_by_fuel_aviation_shipping = (
        ceds_by_fuel_aviation_shipping.groupby(["em", "region", "fuel", "units", "sector"]).sum().pix.fixna()
    )  # group and fix NAs
    ceds_by_fuel_aviation_shipping

# %% [markdown]
# ### update units and names

# %%
if RUN_OPTIONAL_BYFUEL_PROCESSING_FOR_AVIATION_AND_SHIPPING:
    ceds_by_fuel_aviation_shipping = ceds_by_fuel_aviation_shipping.pix.dropna(subset=["units"]).pix.format(
        unit="{units}/yr", drop=True
    )

    # adjust units; change all to values to Mt instead of kt
    ceds_by_fuel_aviation_shipping = pix.units.convert_unit(
        ceds_by_fuel_aviation_shipping, lambda x: "Mt " + x.removeprefix("kt").strip()
    )
    # exception for N2O/yr, which should remain in kt following https://github.com/IAMconsortium/common-definitions/
    ceds_by_fuel_aviation_shipping = pix.units.convert_unit(
        ceds_by_fuel_aviation_shipping, lambda x: "kt " + x.removeprefix("Mt").strip() if x == "Mt N2O/yr" else x
    )

    ceds_by_fuel_aviation_shipping.index = pd.MultiIndex.from_tuples(
        [
            (em, country, sector_59, sector, update_unit(unit, em))
            for em, country, sector_59, sector, unit in ceds_by_fuel_aviation_shipping.index
        ],
        names=ceds_by_fuel_aviation_shipping.index.names,
    )

    # change name(s) of emissions species
    # use 'Sulfur' instead of 'SO2'
    ceds_by_fuel_aviation_shipping = update_index_levels_func(
        ceds_by_fuel_aviation_shipping, {"em": lambda x: x.replace("SO2", "Sulfur").replace("NMVOC", "VOC")}
    )

    ceds_by_fuel_aviation_shipping

# %% [markdown]
# ### Format in IAMC style

# %%
if RUN_OPTIONAL_BYFUEL_PROCESSING_FOR_AVIATION_AND_SHIPPING:
    ceds_by_fuel_aviation_shipping_iamdf = ceds_by_fuel_aviation_shipping.rename_axis(
        index={"em": "variable", "country": "region"}
    )
    ceds_by_fuel_aviation_shipping_iamdf

    # rename to IAMC-style variable names including standard index order
    ceds_by_fuel_aviation_shipping_iamdf = (
        ceds_by_fuel_aviation_shipping_iamdf.pix.format(variable="Emissions|{variable}|{sector}|{fuel}", drop=True)
        .pix.assign(scenario=HISTORY_SCENARIO_NAME, model=f"CEDS_{CEDS_VERSION_ID}")
        .reorder_levels(["model", "scenario", "region", "variable", "unit"])
    ).sort_values(by=["region", "variable"])
    ceds_by_fuel_aviation_shipping_iamdf


# %%
if RUN_OPTIONAL_BYFUEL_PROCESSING_FOR_AVIATION_AND_SHIPPING:
    # check units
    assert_units_match_wishes(ceds_by_fuel_aviation_shipping_iamdf)

# %% [markdown]
# ### Save the file as CSV (it is not used elsewhere in the processing chain)

# %%
if RUN_OPTIONAL_BYFUEL_PROCESSING_FOR_AVIATION_AND_SHIPPING:
    ceds_by_fuel_aviation_shipping_filename = "ceds_cmip7_Aircraft_intlShipping_byfuel_" + CEDS_VERSION_ID + ".csv"
    ceds_by_fuel_aviation_shipping_iamdf.to_csv(CEDS_TOP_LEVEL_RAW_PATH / ceds_by_fuel_aviation_shipping_filename)
