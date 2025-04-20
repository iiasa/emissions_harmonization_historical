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
# # Scratch - regional-sectoral harmonisation
#
# An attempt to see if we can make this run.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.units_helpers import strip_pint_incompatible_characters_from_units
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.io import load_raw_scenario_data

# %% [markdown]
# ## Set up

# %%
pd.set_option("display.max_colwidth", None)

# %%
pandas_openscm.register_pandas_accessor()

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
scenarios_raw = load_raw_scenario_data(
    scenario_path=DATA_ROOT / "scenarios" / "data_raw",
    scenario_time_id=SCENARIO_TIME_ID,
    progress=True,
).loc[:, :2100]  # TODO: drop 2100 end once we have usable scenario data post-2100
scenarios_raw.columns.name = "year"
scenarios_raw

# %% [markdown]
# ### Historical emissions

# %%
HISTORICAL_EMISSIONS_FILE = DATA_ROOT / Path(
    "combined-processed-output", f"iamc_regions_cmip7_history_{IAMC_REGION_PROCESSING_ID}.csv"
)
HISTORICAL_EMISSIONS_FILE

# %%
history = strip_pint_incompatible_characters_from_units(
    load_timeseries_csv(
        HISTORICAL_EMISSIONS_FILE,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
)
history

# %% [markdown]
# ### Pre-processor

# %%
pre_processor = CMIP7ScenarioMIPPreProcessor()

# %% [markdown]
# ## Process

# %% [markdown]
# Simplify to one scenario for now.

# %%
model = "REMIND-MAgPIE 3.5-4.10"
scenario = "SSP1 - Low Emissions_c"

# %%
sdf = scenarios_raw.loc[pix.isin(model=model, scenario=scenario)].dropna(how="all", axis="columns")
# Also simplify to just the model regions and get rid of Kyoto variables
sdf = sdf.loc[pix.ismatch(region=["World", f"{model.split(' ')[0]}*|*"])].loc[~pix.ismatch(variable="**Kyoto**")]
# Also simplify to just data from 2015 onwards,
# we definitely don't care about data before this
sdf = sdf.loc[:, 2015:]
sdf

# %% [markdown]
# Raw pre-processing doesn't work
# because of missing reporting.

# %%
pre_processor(sdf)

# %% [markdown]
# Effectively rewriting gcages from here on.
# TODO: push this back into gcages

# %%
from gcages.cmip7_scenariomip.pre_processing import REQUIRED_WORLD_VARIABLES_IAMC
from pandas_openscm.indexing import multi_index_lookup, multi_index_match

# %%
REQUIRED_GRIDDING_SPECIES_IAMC: tuple[str, ...] = (
    "CO2",
    "CH4",
    "N2O",
    "BC",
    "CO",
    "NH3",
    "OC",
    "NOx",
    "Sulfur",
    "VOC",
)

# %%
REQUIRED_GRIDDING_SECTORS_WORLD_IAMC: tuple[str, ...] = (
    "Energy|Demand|Bunkers|International Aviation",
    "Energy|Demand|Bunkers|International Shipping",
)


# %%
REQUIRED_WORLD_VARIABLES_IAMC: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_IAMC
    for sector in REQUIRED_GRIDDING_SECTORS_WORLD_IAMC
)

# %%
REQUIRED_WORLD_INDEX_IAMC = pd.MultiIndex(
    names=["variable"],
    levels=[REQUIRED_WORLD_VARIABLES_IAMC],
    codes=[np.arange(len(REQUIRED_WORLD_VARIABLES_IAMC))],
)

# %%
REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC: tuple[str, ...] = (
    "Energy|Supply",
    # Components of industrial
    "Energy|Demand|Industry",
    "Energy|Demand|Other Sector",
    "Industrial Processes",
    "Other",
    "Energy|Demand|Residential and Commercial and AFOFI",
    "Product Use",
    # Technically, domestic aviation could be reported just
    # at the world level and it would be fine.
    # In practice, no-one does that and the logic is much simpler
    # if we assume it has to be reported regionally
    # (because then domestic aviation and transport are on the same regional 'grid')
    # so do that for now.
    "Energy|Demand|Transportation|Domestic Aviation",
    "Energy|Demand|Transportation",
    # The rest of TRANSPORTATION_SECTOR_REQUIRED_REPORTING_IAMC
    # can be reported at the world level only and it is fine
    "Waste",
    # Note: AFOLU|Agriculture is the only compulsory component for agriculture
    # which is why we don't use
    # *AGRICULTURE_SECTOR_COMPONENTS_IAMC
    "AFOLU|Agriculture",
    "AFOLU|Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning",
)

NOT_CO2_REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC: tuple[str, ...] = (
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning",
)
"""
Sectors that aren't required for CO2

These are burning sectors because burning should come from the model internally
to avoid double counting.
In general, this is super messy but I think this is the right interpretation.
"""

# %%
REQUIRED_REGIONAL_VARIABLES_IAMC: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_IAMC
    for sector in REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC
    if not (species == "CO2" and sector in NOT_CO2_REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC)
)
REQUIRED_REGIONAL_INDEX_IAMC = pd.MultiIndex(
    names=["variable"],
    levels=[REQUIRED_REGIONAL_VARIABLES_IAMC],
    codes=[np.arange(len(REQUIRED_REGIONAL_VARIABLES_IAMC))],
)

# %%
OPTIONAL_GRIDDING_SECTORS_REGIONAL_IAMC: tuple[str, ...] = (
    "Other Capture and Removal",
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
)

# %%
OPTIONAL_REGIONAL_VARIABLES_IAMC: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_IAMC
    for sector in OPTIONAL_GRIDDING_SECTORS_REGIONAL_IAMC
)
OPTIONAL_REGIONAL_INDEX_IAMC = pd.MultiIndex(
    names=["variable"],
    levels=[OPTIONAL_REGIONAL_VARIABLES_IAMC],
    codes=[np.arange(len(OPTIONAL_REGIONAL_VARIABLES_IAMC))],
)


# %%
def add_regions_to_required_index(ri: pd.MultiIndex, regions: list[str], region_level: str = "region") -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [
            *ri.levels,
            regions,
        ],
        names=[*ri.names, region_level],
    )


# %%
def get_missing_index(ri: pd.MultiIndex, sdf_index: pd.MultiIndex) -> pd.MultiIndex:
    return ri[~multi_index_match(ri, sdf_index.droplevel(sdf_index.names.difference(ri.names)))]


# %% [markdown]
# Figure out what the model hasn't reported at the World level.

# %%
required_world_index = add_regions_to_required_index(REQUIRED_WORLD_INDEX_IAMC, ["World"])
# required_world_index.to_frame()

# %%
missing_world_index = get_missing_index(required_world_index, sdf.index)
# missing_world_index

# %% [markdown]
# Figure out what the model hasn't reported at the regional level.

# %%
model_regions = [r for r in sdf.pix.unique("region") if r.startswith(model.split(" ")[0])]
# model_regions

# %%
required_regional_index = add_regions_to_required_index(REQUIRED_REGIONAL_INDEX_IAMC, model_regions)
# required_regional_index.to_frame(index=False).sort_values("variable").iloc[:20, :]

# %%
missing_regional_index = get_missing_index(required_regional_index, sdf.index)
# tmp = missing_regional_index.to_frame(index=False).sort_values("variable")
# tmp[tmp["region"].str.contains("Australia")]

# %%
variable_unit_map = {v: u for v, u in sdf.loc[pix.ismatch(variable="Emissions|*")].pix.unique(["variable", "unit"])}


def guess_unit(variable: str) -> str:
    for k, v in variable_unit_map.items():
        if f"{k}|" in variable:
            return v


variable_unit_map

# %%
missing_index = missing_world_index.append(missing_regional_index)
missing_index = missing_index.pix.assign(unit=missing_index.get_level_values("variable").map(guess_unit))
missing_index

# %% [markdown]
# Double check to see if the model is reporting anything else that we might be missing
# before continuing.

# %%
from collections import defaultdict
from collections.abc import Iterable


def group_into_levels(v_list: Iterable[str], level_separator: str = "|") -> dict[int, list[str]]:
    n_sector_groupings = defaultdict(list)
    for v in v_list:
        n_sectors = v.count(level_separator) + 1
        n_sector_groupings[n_sectors].append(v)

    return dict(n_sector_groupings)


# %%
required_sector_grouped = group_into_levels(REQUIRED_REGIONAL_VARIABLES_IAMC)
optional_sector_grouped = group_into_levels(OPTIONAL_REGIONAL_VARIABLES_IAMC)
# optional_sector_grouped


# %%
def get_higher_lower_detail(level: int, ind: dict[int, str]) -> tuple[list[str], list[str]]:
    higher_detail = []
    lower_detail = []
    for k, v in ind.items():
        if k > level:
            higher_detail.extend(v)
        elif k < level:
            lower_detail.extend(v)

    return higher_detail, lower_detail


# %%
import textwrap

level_separator = "|"

verbose = False
# verbose = True
for n_sectors in sorted(required_sector_grouped.keys()):
    exp_variables = required_sector_grouped[n_sectors]
    if n_sectors in optional_sector_grouped:
        opt_variables = optional_sector_grouped[n_sectors]
    else:
        opt_variables = []

    higher_detail_required, lower_detail_required = get_higher_lower_detail(n_sectors, required_sector_grouped)
    higher_detail_optional, lower_detail_optional = get_higher_lower_detail(n_sectors, optional_sector_grouped)
    higher_detail = [*higher_detail_required, *higher_detail_optional]
    lower_detail = [*lower_detail_required, *lower_detail_optional]

    exp_variables_s = set(exp_variables)
    optional_variables_s = set(opt_variables)
    sdf_variables = set(sdf.loc[pix.ismatch(variable=level_separator.join(["*"] * n_sectors))].pix.unique("variable"))
    not_exp = sdf_variables - exp_variables_s - optional_variables_s

    # optional = not_exp.intersection(set(OPTIONAL_REGIONAL_VARIABLES_IAMC))
    reported_at_higher_detail = {v for v in not_exp if any(v in hd for hd in higher_detail)}
    reported_at_lower_detail = {v for v in not_exp if any(ld in v for ld in lower_detail)}
    # No idea how to express the Bunkers exception well
    reported_at_world_level = not_exp.intersection(
        {*REQUIRED_WORLD_VARIABLES_IAMC, *[v.split("|International")[0] for v in REQUIRED_WORLD_VARIABLES_IAMC]}
    )
    extras = not_exp - reported_at_higher_detail - reported_at_lower_detail - reported_at_world_level

    print(f"Level: {n_sectors}")

    def wrap_h(vs: set[str]) -> None:
        return textwrap.indent("\n".join(sorted(vs)), prefix="  - ")

    if verbose:
        print(f"Reported:\n{wrap_h(exp_variables_s.intersection(sdf_variables))}")
        print(f"Reported optional:\n{wrap_h(optional_variables_s.intersection(sdf_variables))}")
    print(f"Missing:\n{wrap_h(exp_variables_s - sdf_variables)}")
    if verbose:
        print(f"Missing optional:\n{wrap_h(optional_variables_s - sdf_variables)}")
        print(f"Reported at higher detail:\n{wrap_h(reported_at_higher_detail)}")
        print(f"Reported at lower detail:\n{wrap_h(reported_at_lower_detail)}")
        print(f"Reported at world level:\n{wrap_h(reported_at_world_level)}")
    print(f"Extras:\n{wrap_h(extras)}")
    print("")
    # break

# %% [markdown]
# Think carefully about whether the below will work or not.
# For REMIND, we can just add on the zeroes
# as we expect everything else to be reported correctly.

# %%
missing_timeseries = pd.DataFrame(
    np.zeros((missing_index.shape[0], sdf.shape[1])), index=missing_index, columns=sdf.columns
).pix.assign(model=model, scenario=scenario)
# missing_timeseries

# %%
sdf_take_2 = pix.concat([sdf, missing_timeseries])
# sdf_take_2.loc[pix.ismatch(variable="Emissions|CO2")]


# %%
def split_sectors(indf: pd.DataFrame, dropna: bool = True):
    return indf.pix.extract(variable="{table}|{species}|{sectors}", dropna=dropna)


def split_species(indf: pd.DataFrame, dropna: bool = True):
    return indf.pix.extract(variable="{table}|{species}", dropna=dropna)


# %% [markdown]
# The data essentially breaks down into four tables (although even this isn't really correct):
#
# 1. totals (no region or sector dimension)
# 2. totals at the region level (region dimension, no sector dimension)
# 3. totals at the sector level (no region dimension, sector dimension)
# 4. region-sector information (region and sector dimensions)
#
# The consistency between these tables is tricky, however.
# You can't just blindly add across the dimensions
# and expect to get the equivalent table.
# The reason is that sectors aren't independent
# (so you have to drop carefully before summing)
# and some things are only reported at the higher level
# (i.e. there isn't information in the lower level).
#
# If you're careful, you can:
#
# 1. get the totals from the totals at the sector level (3 -> 1)
#     - you have to be careful that you don't double count the sectors
#     - and you obviously won't get the totals for species
#       that don't report sectoral information (e.g. HFCs)
# 1. get the totals at the regional level from the region-sector information (4 -> 2)
#     - you have to be careful that you don't double count the sectors
#     - and you obviously won't get the totals for species
#       that don't report sectoral information
#       (e.g. HFCs although these are not always reported at the regional total level)
# 1. get the totals at the sectoral level from the region-sector information (4 -> 3)
#     - you obviously won't get the totals for sectors
#       that don't report regional information (e.g. international aviation and shipping)
#
# However, in general, even if you're careful, you cannot:
#
# 1. get the totals from the totals at the regional level alone (2 -> 1)
#    because some variables are only reported as a regional sum (e.g. bunkers).
#     - you have to add on the data for the variables that only report as a regional sum

# %%
from gcages.testing import assert_frame_equal


def rul(idf: pd.DataFrame) -> pd.DataFrame:
    idf.index = idf.index.remove_unused_levels()
    return idf


indf = sdf_take_2
indf = sdf
world_reported_locator = pix.ismatch(region="World")
world_reported = rul(indf.loc[world_reported_locator].reset_index("region", drop=True))
not_world_reported = rul(indf.loc[~world_reported_locator])

has_sector_locator = pix.ismatch(variable="*|*|**")
sector_dim = rul(split_sectors(world_reported.loc[has_sector_locator]))  # no region dimension, has sector dimension
total = rul(split_species(world_reported.loc[~has_sector_locator]))  # no region or sector dimension

region_sector_dim = rul(split_sectors(not_world_reported.loc[has_sector_locator]))  # region and sector dimensions
region_dim = rul(split_species(not_world_reported.loc[~has_sector_locator]))  # region dimension, no sector dimension

# %%
DOMESTIC_AVIATION_SECTOR_IAMC: str = "Energy|Demand|Transportation|Domestic Aviation"
ALL_CONSIDERED_GRIDDING_SECTORS = [
    *REQUIRED_GRIDDING_SECTORS_WORLD_IAMC,
    *REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC,
    *OPTIONAL_GRIDDING_SECTORS_REGIONAL_IAMC,
]
ALL_CONSIDERED_GRIDDING_SECTORS_WITHOUT_DOUBLE_COUNTING = list(
    set(ALL_CONSIDERED_GRIDDING_SECTORS) - {DOMESTIC_AVIATION_SECTOR_IAMC}
)
# ALL_CONSIDERED_GRIDDING_SECTORS_WITHOUT_DOUBLE_COUNTING

# %%
# # Getting the totals from the totals at the sector level (3 -> 1)
# # blindly does not work
# assert_frame_equal(total, sector_dim.openscm.groupby_except("sectors").sum())

# Getting the totals from the totals at the sector level (3 -> 1)
# works if we're careful.
# Avoid double counting
sector_dim_sector_sum_components = sector_dim.loc[
    pix.isin(sectors=ALL_CONSIDERED_GRIDDING_SECTORS_WITHOUT_DOUBLE_COUNTING)
]
sector_dim_sector_sum = sector_dim_sector_sum_components.openscm.groupby_except("sectors").sum()
assert_frame_equal(
    sector_dim_sector_sum,
    # Only compare with species that report sectoral information
    total.loc[pix.isin(species=sector_dim_sector_sum.pix.unique("species"))],
    rtol=1e-3,  # have to be relatively loose here because of rounding issues
    atol=1.0,  # typically, differences of 1.0 aren't of interest to us in the reporting we use (although could also just re-aggregate)
)

# %%
# # Getting the totals at the regional level from the region-sector information (4 -> 2)
# # blindly does not work
# assert_frame_equal(region_dim, region_sector_dim.openscm.groupby_except("sectors").sum())

# Getting the totals at the regional level from the region-sector information (4 -> 2)
# works if we're careful.
# Avoid double counting
region_sector_dim_sector_sum_components = region_sector_dim.loc[
    pix.isin(sectors=ALL_CONSIDERED_GRIDDING_SECTORS_WITHOUT_DOUBLE_COUNTING)
]
region_sector_dim_sector_sum = region_sector_dim_sector_sum_components.openscm.groupby_except("sectors").sum()
assert_frame_equal(
    region_sector_dim_sector_sum,
    # Only compare with species that report sectoral information
    region_dim.loc[pix.isin(species=region_sector_dim_sector_sum.pix.unique("species"))],
    rtol=1e-3,  # have to be relatively loose here because of rounding issues
    atol=1.0,  # typically, differences of 1.0 aren't of interest to us in the reporting we use (although could also just re-aggregate)
)

# %%
# # Getting the totals at the sectoral level from the region-sector information (4 -> 3)
# # blindly does not work
# assert_frame_equal(sector_dim, region_sector_dim.openscm.groupby_except("region").sum(), rtol=1e-1, atol=1.0)

# Getting the totals at the sectoral level from the region-sector information (4 -> 3)
# works if we're careful.
# We just have to avoid any part of the sectoral hierarchy
# that includes variables that are only reported at the World level.
# There must be ways to program this,
# but I can't think through the logic right now
# (and I'm also not sure the REMIND data actually follows any logic
# because the variables that are only reported at the World level
# seem to vary randomly),
# so here is a hard-coded example.
locator = ~pix.ismatch(sectors="Energy**")
assert_frame_equal(
    sector_dim.loc[locator],
    region_sector_dim.openscm.groupby_except("region").sum().loc[locator],
    rtol=1e-3,
    atol=1.0,
)

# %%
# # Getting the totals from the totals at the regional level alone (2 -> 1)
# # blindly does not work
# assert_frame_equal(total, region_dim.openscm.groupby_except("region").sum())

# Getting the totals from the totals at the regional level alone (2 -> 1)
# works if we're careful.

# Get information that's available at the world level only.
# There must be a way to do this programatically,
# but I can't tell what it is.
# I tried this, but it doesn't work because many sectors are filled with zeros in the reported data.
# not_regional_sectors = list(set(sector_dim.pix.unique("sectors")) - set(region_sector_dim.pix.unique("sectors")))
not_regional_sectors = [
    "Energy|Demand|Bunkers|International Aviation",
    "Energy|Demand|Bunkers|International Shipping",
]
available_at_world_only = sector_dim.loc[pix.isin(sectors=not_regional_sectors)].openscm.groupby_except("sectors").sum()
# PIK reports international shipping and aviation
# at the regional level for these species,
# so we have to undo the double counting.
available_at_world_only.loc[pix.isin(species=["CO2", "N2O"])] = 0.0

# Add it onto the naive sum
# (being careful to only use species of interest)
region_level_sum = region_dim.openscm.groupby_except("region").sum()
region_level_sum_comparable = (
    region_level_sum.loc[pix.isin(species=available_at_world_only.pix.unique("species"))] + available_at_world_only
)

assert_frame_equal(
    region_level_sum_comparable,
    # Only compare with species that report sectoral information
    total.loc[pix.isin(species=region_level_sum_comparable.pix.unique("species"))],
    rtol=1e-4,  # have to be relatively loose here because of rounding issues
    atol=1.0,  # typically, differences of 1.0 aren't of interest to us in the reporting we use (although could also just re-aggregate)
)

# %% [markdown]
# If we are confident that our data obeys the relationships above,
# we can do the processing.

# %%

# %%
assert False

# %%
region_sector_sum = region_sector_sum_components.groupby(region_sector.index.names.difference(["sectors"])).sum()

# %%
world_sector

# %%
split_sectors(sdf_take_2)

# %%
pre_processor(sdf_take_2)

# %%
sdf_totals = sdf.loc[pix.ismatch(region="World", variable="Emissions|*")]
sdf_totals
sdf_complete = pix.concat([sdf, missing_timeseries])
# TODO: make gcages more robust to extra data
sdf_complete_required_only = multi_index_lookup(sdf_complete, required_index)
sdf_take_2 = pix.concat([sdf_complete_required_only, sdf_totals])
sdf_take_2

# %%
assert False

# %%
from gcages.cmip7_scenariomip.pre_processing import (
    AGRICULTURE_SECTOR_COMPONENTS_IAMC,
    REQUIRED_GRIDDING_SPECIES_IAMC,
    REQUIRED_REGIONAL_INDEX_IAMC,
    REQUIRED_WORLD_INDEX_IAMC,
)
from pandas_openscm.indexing import multi_index_lookup

# %% [markdown]
# Check for completeness.


# %%
def add_regions_to_required_index(ri: pd.MultiIndex, regions: list[str], region_level: str = "region") -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [
            *ri.levels,
            regions,
        ],
        names=[*ri.names, region_level],
    )


# %%
required_world_index = add_regions_to_required_index(REQUIRED_WORLD_INDEX_IAMC, ["World"])


# %%
def get_missing_index(ri: pd.MultiIndex, sdf_index: pd.MultiIndex) -> pd.MultiIndex:
    return ri[~multi_index_match(ri, sdf_index.droplevel(sdf_index.names.difference(ri.names)))]


# %%
missing_world_index = get_missing_index(required_world_index, sdf.index)
missing_world_index

# %%
model_regions = [r for r in sdf.pix.unique("region") if r.startswith(model.split(" ")[0])]
model_regions

# %%
REQUIRED_REGIONAL_INDEX_IAMC

# %% [markdown]
# Add on optional sectors too.

# %%
import itertools

optional_sectors = [
    f"Emissions|{species}|{sector}"
    for species, sector in itertools.product(REQUIRED_GRIDDING_SPECIES_IAMC, AGRICULTURE_SECTOR_COMPONENTS_IAMC)
]

# %%
full_regional_index_iamc = REQUIRED_REGIONAL_INDEX_IAMC.append(
    pd.MultiIndex.from_product([optional_sectors], names=["variable"])
)
full_regional_index_iamc

# %%
full_regional_index = add_regions_to_required_index(full_regional_index_iamc, model_regions)
[v for v in full_regional_index.to_frame(index=False)["variable"].unique() if "CO" in v]

# %%
missing_regional_index = get_missing_index(required_regional_index, sdf.index)
missing_regional_index

# %%
missing_index = missing_regional_index.append(missing_world_index)
missing_index

# %%
variable_unit_map = {v: u for v, u in sdf.loc[pix.ismatch(variable="Emissions|*")].pix.unique(["variable", "unit"])}


def guess_unit(variable: str) -> str:
    for k, v in variable_unit_map.items():
        if f"{k}|" in variable:
            return v


variable_unit_map

# %%
missing_index = missing_index.pix.assign(unit=missing_index.get_level_values("variable").map(guess_unit))
missing_index

# %%
missing_timeseries = pd.DataFrame(
    np.zeros((missing_index.shape[0], sdf.shape[1])), index=missing_index, columns=sdf.columns
).pix.assign(model=model, scenario=scenario)
missing_timeseries

# %%
required_index = required_regional_index.append(required_world_index)
# required_index

# %%
sdf_totals = sdf.loc[pix.ismatch(region="World", variable="Emissions|*")]
sdf_totals

# %%
sdf_complete = pix.concat([sdf, missing_timeseries])
# TODO: make gcages more robust to extra data
sdf_complete_required_only = multi_index_lookup(sdf_complete, required_index)
sdf_take_2 = pix.concat([sdf_complete_required_only, sdf_totals])
sdf_take_2

# %%
pre_processor(sdf_take_2)

# %% [markdown]
# Still failing because of inconsistencies in the sums.
# Hence also reaggregate.

# %%
tmp = sdf_complete_required_only.pix.extract(variable="{table}|{species}|{sector}")
# We have started from required only, so a straight sum should be fine
reaggreated_totals = (
    tmp.groupby(tmp.index.names.difference(["region", "sector"]))
    .sum()
    .pix.format(variable="{table}|{species}", drop=True)
    .pix.assign(region="World")
    .reorder_levels(sdf.index.names)
)

# %% [markdown]
# We can see how different our reaggregation is from the reported totals.
# Some alarmingly big differences.

# %%
(
    ((reaggreated_totals - sdf_totals.loc[reaggreated_totals.index]) / reaggreated_totals)
    .abs()
    .sort_values(2005, ascending=False)
    * 100
).round(1)

# %%
sdf_complete_required_only.loc[pix.ismatch(variable="Emissions|CO2|**", region="**Australi**")]

# %%
sdf.loc[
    pix.ismatch(variable="Emissions|CO2|*", region="World") & ~pix.ismatch(variable="**Energy and Industrial**")
]  # .groupby(sdf.index.names.difference(["variable"])).sum()

# %%
sdf.loc[pix.ismatch(variable="Emissions|CO2", region="World")]

# %%
pix.concat([reaggreated_totals.pix.assign(source="reaggregated"), sdf_totals.pix.assign(source="model_reported")]).loc[
    pix.ismatch(variable="**CO2")
].pix.project(["variable", "region", "source"]).T.plot()

# %%
