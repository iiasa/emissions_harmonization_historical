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
# Simplify to one model for now.

# %%
sorted(scenarios_raw.pix.unique("model"))

# %%
model = "AIM 3.0"
scenario = "SSP2 - Low Overshoot_a"
scenario = None
model = "COFFEE 1.6"
scenario = None
# model = "IMAGE 3.4"
# scenario = None
model = "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"
scenario = None
model = "REMIND-MAgPIE 3.5-4.10"
scenario = None
# model = "WITCH 6.0"
# scenario = None

# %%
sdf = scenarios_raw.loc[pix.isin(model=model)].dropna(how="all", axis="columns")
# sdf.pix.unique("scenario")

# %%
if scenario is not None:
    sdf = sdf.loc[pix.isin(scenario=scenario)]

# %%
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
# # Comment out for now so we can run all
# pre_processor(sdf)

# %% [markdown]
# Effectively rewriting gcages from here on.
# TODO: push this back into gcages

# %%
from gcages.cmip7_scenariomip.pre_processing import REQUIRED_WORLD_VARIABLES_IAMC
from pandas_openscm.indexing import multi_index_match

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
# missing_index

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
len(sdf.pix.unique("scenario"))

# %%
missing_index_incl_scenarios = None
for scenario in sdf.pix.unique("scenario"):
    # Hacking through this as speed not an issue
    very_slow = pd.DataFrame(
        [scenario] * missing_index.shape[0], columns=["scenario"], index=missing_index
    ).reset_index()
    new_levels = pd.MultiIndex.from_frame(very_slow)
    if missing_index_incl_scenarios is None:
        missing_index_incl_scenarios = new_levels
    else:
        missing_index_incl_scenarios = missing_index_incl_scenarios.append(new_levels)

# %%
missing_timeseries = pd.DataFrame(
    np.zeros((missing_index_incl_scenarios.shape[0], sdf.shape[1])),
    index=missing_index_incl_scenarios,
    columns=sdf.columns,
).pix.assign(model=model)
missing_timeseries.loc[pix.ismatch(variable="**CH4**")]

# %% [markdown]
# Also include world sum.

# %%
missing_timeseries_region_sum = (
    missing_timeseries.loc[~pix.isin(region="World")].openscm.groupby_except("region").sum().pix.assign(region="World")
)
missing_timeseries_including_region_sum = pix.concat([missing_timeseries, missing_timeseries_region_sum])
# # missing_timeseries_including_region_sum

# %%
# Only append timeseries that aren't already in sdf
# (we can end up with duplicates when missing sectors are only missing for some regions
# because of the aggregation done above)
missing_timeseries_to_append = missing_timeseries_including_region_sum.loc[
    ~multi_index_match(missing_timeseries_including_region_sum.index, sdf.index)
]
# missing_timeseries_to_append

# %%
sdf_take_2 = pix.concat([sdf, missing_timeseries_to_append])
# sdf_take_2.loc[pix.ismatch(variable="Emissions|BC**Aviation**", region="World")]


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
# indf

# %%
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string
from pandas_openscm.index_manipulation import update_index_levels_func


# %%
def fix_sector_info(v_in: str) -> str:
    out = v_in
    if "|HFC|" in out:
        # This isn't a sector level
        out = out.replace("|HFC|", "|")

    return out


def rebreak_sector_info(v_in: str) -> str:
    out = v_in
    if "|HFC" in out:
        # Put the broken sector level back in
        out = out.replace("|HFC", "|HFC|HFC")

    return out


# TODO: include unit fix in name too
indf_sector_info_fixed = update_index_levels_func(
    indf,
    {"variable": fix_sector_info, "unit": strip_pint_incompatible_characters_from_unit_string},
)

# %%
world_reported_locator = pix.ismatch(region="World")
world_reported = rul(indf_sector_info_fixed.loc[world_reported_locator].reset_index("region", drop=True))
not_world_reported = rul(indf_sector_info_fixed.loc[~world_reported_locator])

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
    # Ideally set variable specific tolerance
    rtol=5e-2,  # shouldn't need to be so loose here
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
    # Ideally set variable specific tolerance
    rtol=5e-2,  # shouldn't need to be so loose here
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
    # Ideally set variable specific tolerance
    rtol=5e-2,  # shouldn't need to be so loose here
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
    # Ideally set variable specific tolerance
    rtol=5e-4,  # this seems to be much easier for models to get right
)

# %% [markdown]
# If we are confident that our data obeys the relationships above,
# we can do the processing.

# %%
reported_times = indf.dropna(how="all", axis="columns")
# reported_times

# %% [markdown]
# We only work with the following three tables


# %%
def rul(idf: pd.DataFrame) -> pd.DataFrame:
    idf.index = idf.index.remove_unused_levels()
    return idf


world_reported_locator = pix.ismatch(region="World")
world_reported = indf_sector_info_fixed.loc[world_reported_locator].reset_index("region", drop=True)
not_world_reported = indf_sector_info_fixed.loc[~world_reported_locator]

has_sector_locator = pix.ismatch(variable="*|*|**")

total = rul(split_species(world_reported.loc[~has_sector_locator]))  # for verification and stuff that is only global
sector_dim = rul(
    split_sectors(world_reported.loc[has_sector_locator])
)  # things that we're only interested in at the global level
region_sector_dim = rul(split_sectors(not_world_reported.loc[has_sector_locator]))  # for the gridding level

# %% [markdown]
# Move into columns being sectors while we shuffle things around.

# %%
region_sector_dim_sector_cols = region_sector_dim.stack().unstack("sectors")
# region_sector_dim_sector_cols

# %%
try:
    sector_dim_sectors_cols = sector_dim.stack().unstack("sectors")
except ValueError:
    display(sector_dim.loc[sector_dim.index.duplicated(keep=False)].sort_index())
    raise

# sector_dim_sectors_cols

# %% [markdown]
# Re-classify aviation

# %%
INTERNATIONAL_AVIATION_SECTOR_IAMC: str = "Energy|Demand|Bunkers|International Aviation"
AVIATION_SECTOR_CEDS: str = "Aircraft"
TRANSPORTATION_SECTOR_IAMC: str = "Energy|Demand|Transportation"
TRANSPORTATION_SECTOR_CEDS: str = "Transportation Sector"

# %%
from pandas_openscm.grouping import groupby_except

# %%
region_sector_dim_domestic_aviation = region_sector_dim_sector_cols[DOMESTIC_AVIATION_SECTOR_IAMC]
# Remove emissions from default reporting, creating a new sector in the process
region_sector_dim_sector_cols[TRANSPORTATION_SECTOR_CEDS] = (
    region_sector_dim_sector_cols[TRANSPORTATION_SECTOR_IAMC] - region_sector_dim_domestic_aviation
)
# Create the new sector
sector_dim_sectors_cols[AVIATION_SECTOR_CEDS] = (
    sector_dim_sectors_cols[INTERNATIONAL_AVIATION_SECTOR_IAMC]
    + groupby_except(region_sector_dim_domestic_aviation, "region").sum()
)

# Drop out now redundant sectors
region_sector_dim_sector_cols = region_sector_dim_sector_cols.drop(
    [
        DOMESTIC_AVIATION_SECTOR_IAMC,
    ],
    axis="columns",
)
sector_dim_sectors_cols = sector_dim_sectors_cols.drop(
    [
        TRANSPORTATION_SECTOR_IAMC,
        INTERNATIONAL_AVIATION_SECTOR_IAMC,
    ],
    axis="columns",
)

sector_dim_sectors_cols  # .stack().unstack("year")


# %% [markdown]
# Aggregate industry


# %%
def aggregate_sector(
    indf: pd.DataFrame,  # assumes sector columns
    sector_out: str,
    sector_components: list[str],
    allow_missing: list[str] | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    res = indf
    if copy:
        res = res.copy()

    to_sum = sector_components
    if allow_missing is not None:
        missing = {c for c in to_sum if c not in indf}
        # Anything which is missing and allowed to be missing,
        # we can drop from to_sum
        to_drop_from_sum = missing.intersection(set(allow_missing))
        to_sum = list(set(to_sum) - to_drop_from_sum)

        # Also make sure that missing values in optional columns
        # don't break things
        allow_missing_in_indf = indf.columns.intersection(allow_missing)
        res[allow_missing_in_indf] = res[allow_missing_in_indf].fillna(0.0)

    res[sector_out] = res[to_sum].sum(axis="columns", min_count=len(to_sum))
    res = res.drop(to_sum, axis="columns")

    return res


# %%
INDUSTRIAL_SECTOR_CEDS: str = "Industrial Sector"
INDUSTRIAL_SECTOR_CEDS_COMPONENTS_IAMC: tuple[str, ...] = (
    "Energy|Demand|Industry",
    "Energy|Demand|Other Sector",
    "Industrial Processes",
    "Other",
)

# %%
region_sector_dim_sectors_cols_industry = aggregate_sector(
    region_sector_dim_sector_cols,
    sector_out=INDUSTRIAL_SECTOR_CEDS,
    sector_components=list(INDUSTRIAL_SECTOR_CEDS_COMPONENTS_IAMC),
)

# %%
AGRICULTURE_SECTOR_CEDS: str = "Agriculture"
AGRICULTURE_SECTOR_COMPONENTS_IAMC: tuple[str, ...] = (
    "AFOLU|Agriculture",
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
)

# %%
region_sector_dim_sectors_cols_industry_agriculture = aggregate_sector(
    region_sector_dim_sectors_cols_industry,
    sector_out=AGRICULTURE_SECTOR_CEDS,
    sector_components=list(AGRICULTURE_SECTOR_COMPONENTS_IAMC),
    allow_missing=list(OPTIONAL_GRIDDING_SECTORS_REGIONAL_IAMC),
)

# %%
GRIDDING_SECTORS_WORLD_REAGGREGATED: tuple[str, ...] = ("Aircraft", "Energy|Demand|Bunkers|International Shipping")
GRIDDING_SECTORS_REGIONAL_REAGGREGATED: tuple[str, ...] = (
    "Energy|Supply",
    "Industrial Sector",
    "Energy|Demand|Residential and Commercial and AFOFI",
    "Product Use",
    "Transportation Sector",
    "Waste",
    "Agriculture",
    "AFOLU|Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning",
)

# %%
REAGGREGATED_TO_GRIDDING_SECTOR_MAP: dict[str, str] = {
    "Energy|Supply": "Energy Sector",
    INDUSTRIAL_SECTOR_CEDS: INDUSTRIAL_SECTOR_CEDS,
    "Energy|Demand|Residential and Commercial and AFOFI": "Residential Commercial Other",
    "Product Use": "Solvents Production and Application",
    TRANSPORTATION_SECTOR_CEDS: TRANSPORTATION_SECTOR_CEDS,
    "Waste": "Waste",
    AVIATION_SECTOR_CEDS: AVIATION_SECTOR_CEDS,
    "Energy|Demand|Bunkers|International Shipping": "International Shipping",
    AGRICULTURE_SECTOR_CEDS: AGRICULTURE_SECTOR_CEDS,
    "AFOLU|Agricultural Waste Burning": "Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning": "Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning": "Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning": "Peat Burning",
}

# %%
time_name = "year"


# %%
def rename_to_gridding_sectors(idf: pd.DataFrame, sectors_to_grab: tuple[str, ...], time_name: str) -> pd.DataFrame:
    return idf[list(sectors_to_grab)].rename(REAGGREGATED_TO_GRIDDING_SECTOR_MAP, axis="columns")


sector_dim_sectors_cols_gridding = rename_to_gridding_sectors(
    sector_dim_sectors_cols, GRIDDING_SECTORS_WORLD_REAGGREGATED, time_name=time_name
)
region_sector_dim_sectors_cols_gridding = rename_to_gridding_sectors(
    region_sector_dim_sectors_cols_industry_agriculture, GRIDDING_SECTORS_REGIONAL_REAGGREGATED, time_name=time_name
)


# %%
def combine_sectors(indf: pd.DataFrame, dropna: bool = True):
    return indf.pix.format(variable="{table}|{species}|{sectors}", drop=True)


def combine_species(indf: pd.DataFrame, dropna: bool = True):
    return indf.pix.format(variable="{table}|{species}", drop=True)


# %%
global_workflow_emissions_not_from_gridding_emissions = combine_species(
    total.loc[
        ~pix.isin(species=sector_dim_sectors_cols_gridding.pix.unique("species"))
        # Not handled for now
        & ~pix.ismatch(unit="**equiv**")
    ]
)
global_workflow_emissions_not_from_gridding_emissions = update_index_levels_func(
    global_workflow_emissions_not_from_gridding_emissions,
    {"variable": rebreak_sector_info},
)
global_workflow_emissions_not_from_gridding_emissions


# %%
def get_global_workflow_emissions_from_gridding_emissions_sector_cols(
    *,
    gridding_emissions_sector_dim_sector_cols: pd.DataFrame,
    gridding_emissions_region_sector_dim_sector_cols: pd.DataFrame,
    co2_fossil_sectors: tuple[str, ...] = (
        "Energy Sector",
        "Industrial Sector",
        "Residential Commercial Other",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
        "Aircraft",
        "International Shipping",
    ),
    co2_biosphere_sectors: tuple[str, ...] = (
        "Agriculture",
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ),
    co2_fossil_sector: str = "Energy and Industrial Processes",
    co2_biosphere_sector: str = "AFOLU",
) -> pd.DataFrame:
    global_totals = pd.concat(
        [
            gridding_emissions_sector_dim_sector_cols,
            gridding_emissions_region_sector_dim_sector_cols.openscm.groupby_except("region").sum(),
        ],
        axis="columns",
    )
    co2_locator = pix.isin(species="CO2")
    non_co2 = global_totals.loc[~co2_locator].sum(axis="columns").unstack()

    co2_fossil = (
        global_totals.loc[co2_locator, list(co2_fossil_sectors)]
        .sum(axis="columns")
        .unstack()
        .pix.assign(sectors=co2_fossil_sector)
    )
    co2_biosphere = (
        global_totals.loc[co2_locator, list(co2_biosphere_sectors)]
        .sum(axis="columns")
        .unstack()
        .pix.assign(sectors=co2_biosphere_sector)
    )

    global_workflow_emissions = pix.concat(
        [
            combine_species(non_co2),
            combine_sectors(pix.concat([co2_fossil, co2_biosphere])),
        ]
    )

    return global_workflow_emissions


# %%
global_workflow_emissions_from_gridding_emissions = get_global_workflow_emissions_from_gridding_emissions_sector_cols(
    gridding_emissions_sector_dim_sector_cols=sector_dim_sectors_cols_gridding,
    gridding_emissions_region_sector_dim_sector_cols=region_sector_dim_sectors_cols_gridding,
)

# %%
global_workflow_emissions_raw_names = pix.concat(
    [global_workflow_emissions_from_gridding_emissions, global_workflow_emissions_not_from_gridding_emissions]
)
# global_workflow_emissions_raw_names

# %%
from functools import partial

from gcages.renaming import SupportedNamingConventions, convert_variable_name

# %%
global_workflow_emissions = update_index_levels_func(
    global_workflow_emissions_raw_names,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    },
)

global_workflow_emissions

# %%
time_name = "year"
gridding_workflow_emissions = pix.concat(
    [
        combine_sectors(region_sector_dim_sectors_cols_gridding.stack().unstack(time_name)),
        combine_sectors(sector_dim_sectors_cols_gridding.stack().unstack(time_name)).pix.assign(region="World"),
    ]
)
# gridding_workflow_emissions.loc[pix.ismatch(variable="**BC**Aircraft")]

# %%
import itertools

from gcages.completeness import assert_all_groups_are_complete

# %%
gridding_required_index_world = pd.MultiIndex.from_product(
    [
        [
            f"Emissions|{species}|{REAGGREGATED_TO_GRIDDING_SECTOR_MAP[sector]}"
            for species, sector in itertools.product(
                REQUIRED_GRIDDING_SPECIES_IAMC, GRIDDING_SECTORS_WORLD_REAGGREGATED
            )
        ],
        ["World"],
    ],
    names=["variable", "region"],
)
gridding_required_index_regional = pd.MultiIndex.from_product(
    [
        [
            f"Emissions|{species}|{REAGGREGATED_TO_GRIDDING_SECTOR_MAP[sector]}"
            for species, sector in itertools.product(
                REQUIRED_GRIDDING_SPECIES_IAMC, GRIDDING_SECTORS_REGIONAL_REAGGREGATED
            )
            if not (species == "CO2" and sector in NOT_CO2_REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC)
        ],
        region_sector_dim.pix.unique("region"),
    ],
    names=["variable", "region"],
)
gridding_required_index = gridding_required_index_world.append(gridding_required_index_regional)
# gridding_required_index

# %%
assert_all_groups_are_complete(gridding_workflow_emissions, complete_index=gridding_required_index)

# %%
from gcages.aneris_helpers import harmonise_all


# %%
def interpolate_to_yearly(indf: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    res = indf
    if copy:
        res = res.copy()

    for y in np.arange(indf.columns.min(), indf.columns.max()):
        if y not in indf:
            res[y] = np.nan

    res = res.sort_index(axis="columns")
    res = res.T.interpolate("index").T

    return res


# %%
to_try_locator = ~pix.ismatch(variable=["**Aircraft"])
gridding_workflow_emissions_to_try = interpolate_to_yearly(gridding_workflow_emissions.loc[to_try_locator])
gridding_workflow_emissions_to_try = gridding_workflow_emissions_to_try.dropna()
gridding_workflow_emissions_to_try  # .loc[pix.ismatch(variable="Emissions|BC|Waste", region="**South Africa")]

# %%
# history.loc[pix.ismatch(region=f"{model.split(' ')[0]}**")].pix.unique("region")

# %%
history_fixed = update_index_levels_func(
    history,
    {
        "region": lambda x: x.replace("REMIND-MAgPIE 3.4-4.8", "REMIND-MAgPIE 3.5-4.10").replace(
            "COFFEE 1.5", "COFFEE 1.6"
        )
    },
)

history_shipping = (
    history_fixed.loc[
        pix.ismatch(
            variable=["**Shipping**"],  # TODO: add aviation and remove this hack
            region=gridding_workflow_emissions_to_try.pix.unique("region"),
        )
    ]
    .openscm.groupby_except("region")
    .sum()
    .pix.assign(region="World")
)

history_model_gridding_relevant = pix.concat(
    [
        history_fixed.loc[
            multi_index_match(
                history_fixed.index, gridding_workflow_emissions_to_try.index.droplevel(["model", "scenario"])
            )
        ],
        history_shipping,
    ]
)
history_model_gridding_relevant

# %%
harmonisation_year = 2022
gridding_workflow_emissions_harmonised = harmonise_all(
    scenarios=gridding_workflow_emissions_to_try,
    history=history_model_gridding_relevant,
    year=harmonisation_year,
)
gridding_workflow_emissions_harmonised


# %%
def get_global_workflow_emissions_incl_region_col_from_gridding_emissions(
    gridding_emissions: pd.DataFrame,
    co2_fossil_sectors: tuple[str, ...] = (
        "Energy Sector",
        "Industrial Sector",
        "Residential Commercial Other",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
        "Aircraft",
        "International Shipping",
    ),
    co2_biosphere_sectors: tuple[str, ...] = (
        "Agriculture",
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ),
    co2_fossil_sector: str = "Energy and Industrial Processes",
    co2_biosphere_sector: str = "AFOLU",
    region_out: str = "World",
) -> pd.DataFrame:
    world_locator = pix.isin(region="World")

    gridding_emissions_sector_dim_sector_cols = (
        split_sectors(gridding_emissions.loc[world_locator].reset_index("region", drop=True)).stack().unstack("sectors")
    )
    gridding_emissions_region_sector_dim_sector_cols = (
        split_sectors(gridding_emissions.loc[~world_locator]).stack().unstack("sectors")
    )

    return get_global_workflow_emissions_from_gridding_emissions_sector_cols(
        gridding_emissions_sector_dim_sector_cols=gridding_emissions_sector_dim_sector_cols,
        gridding_emissions_region_sector_dim_sector_cols=gridding_emissions_region_sector_dim_sector_cols,
        co2_fossil_sectors=co2_fossil_sectors,
        co2_biosphere_sectors=co2_biosphere_sectors,
        co2_fossil_sector=co2_fossil_sector,
        co2_biosphere_sector=co2_biosphere_sector,
    ).pix.assign(region=region_out)


# %%
ggwe = partial(
    get_global_workflow_emissions_incl_region_col_from_gridding_emissions,
    co2_fossil_sectors=(
        "Energy Sector",
        "Industrial Sector",
        "Residential Commercial Other",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
        # Don't have this harmonised for now
        # "Aircraft",
        "International Shipping",
    ),
)

# %%
tmp = history_model_gridding_relevant.pix.assign(model="history")
tmp.columns.name = "year"
global_workflow_history = ggwe(tmp)
# global_workflow_history

# %%
global_workflow = ggwe(gridding_workflow_emissions)
# global_workflow

# %%
global_workflow_harmonised = ggwe(gridding_workflow_emissions_harmonised)
# global_workflow_harmonised

# %%
import seaborn as sns

# %%
# sdf.loc[pix.ismatch(variable="Emissions|CO2|Energy|Demand|Industry", region="World")].pix.project("scenario").T.plot()

# %% [markdown]
# International shipping from sulfur and NOx missing.

# %%
locator = pix.ismatch(variable="Emissions|*|International Shipping")
model_shipping = (
    gridding_workflow_emissions_harmonised.loc[
        locator & pix.isin(scenario=gridding_workflow_emissions_harmonised.pix.unique("scenario")[0]),
        :harmonisation_year,
    ]
).reset_index(["model", "scenario"], drop=True)[harmonisation_year]
model_shipping.compare(
    history_model_gridding_relevant.loc[locator, harmonisation_year]
    .reset_index(["model", "scenario"], drop=True)
    .reorder_levels(model_shipping.index.names)
    .loc[model_shipping.index],
    result_names=("model", "history"),
)

# %%
locator = pix.isin(region="World") & pix.ismatch(
    variable=[
        "*|*",
        "**CO2|Energy and Industrial Processes",
        # # Not AFOLU because we're not harmonising to the right target
        # # and it's not used by ESMs
        # "**CO2|AFOLU",
    ]
)

sns_df = (
    pix.concat(
        [
            global_workflow_history.pix.assign(stage="history").loc[:, 2000:harmonisation_year],
            sdf.pix.assign(stage="raw"),
            global_workflow.pix.assign(stage="pre-processed"),
            global_workflow_harmonised.pix.assign(stage="harmonised"),
        ]
    )
    .loc[locator]
    .openscm.to_long_data()
)
if sns_df.empty:
    raise AssertionError

fg = sns.relplot(
    data=sns_df,
    x="time",
    y="value",
    hue="scenario",
    style="stage",
    dashes={
        "history": "",
        "harmonised": "",
        "pre-processed": (1, 1),
        "raw": (3, 3),
    },
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
    kind="line",
    linewidth=2,
)
fg.figure.suptitle(model, y=1.01)
for ax in fg.axes:
    if "CO2" in ax.get_title():
        ax.axhline(0.0, color="k", alpha=0.3, zorder=1.0)
    else:
        ax.set_ylim(0.0)

# %% [markdown]
# Notes:
#
# - need to check internal consistency of history
#   (if we harmonise at all levels, total should be harmonised too)
# - Sulfur emissions harmonised negative for IMAGE, not ideal
# - negative CO2 fossil emissions harmonised out for WITCH, not ideal
# - CO2 industry not reported correctly by WITCH,
#   which makes things look weird

# %%
# gridding_workflow_emissions_harmonised_reaggregated.loc[pix.ismatch(variable="**Sulfur**")]

# %%
# gridding_workflow_emissions_harmonised.pix.unique("region")

# %%
if "AIM" in model:
    locator = pix.ismatch(region="**|EU & UK")
elif "COFFEE" in model:
    locator = pix.ismatch(region="**|Europe")
elif "IMAGE" in model:
    locator = pix.ismatch(region="**|Western Europe")
elif "MESSAGE" in model:
    locator = pix.ismatch(region="**|Western Europe")
elif "REMIND" in model:
    locator = pix.ismatch(region="**|EU 28")
elif "WITCH" in model:
    locator = pix.ismatch(region="**|Europe")
else:
    raise NotImplementedError(model)

# locator = locator & pix.ismatch(variable="*|CO2|*")
locator = locator & pix.ismatch(variable="*|Sulfur|*")

sns_df = (
    pix.concat(
        [
            # sdf.pix.assign(stage="raw"),
            history_model_gridding_relevant.loc[locator].pix.assign(stage="history").loc[:, 2000:harmonisation_year],
            gridding_workflow_emissions.pix.assign(stage="pre_processed"),
            gridding_workflow_emissions_harmonised.pix.assign(stage="harmonised"),
        ]
    )
    .loc[locator]
    .openscm.to_long_data()
)
if sns_df.empty:
    raise AssertionError

fg = sns.relplot(
    data=sns_df,
    x="time",
    y="value",
    hue="scenario",
    style="stage",
    dashes=dict(
        harmonised="",
        history="",
        pre_processed=(1, 1),
        raw=(3, 3),
    ),
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
    kind="line",
    linewidth=2,
)
fg.figure.suptitle(model, y=1.01)
for ax in fg.axes:
    if "CO2" in ax.get_title():
        ax.axhline(0.0, color="k", alpha=0.3, zorder=1.0)
    else:
        ax.set_ylim(0.0)

# %%
variable = "Emissions|CO2|Energy Sector"
variable = "Emissions|CO2|Industrial Sector"

locator = pix.ismatch(variable=variable)

sns_df = (
    pix.concat(
        [
            # sdf.pix.assign(stage="raw"),
            history_model_gridding_relevant.loc[locator].pix.assign(stage="history").loc[:, 2000:harmonisation_year],
            gridding_workflow_emissions.pix.assign(stage="pre_processed"),
            gridding_workflow_emissions_harmonised.pix.assign(stage="harmonised"),
        ]
    )
    .loc[locator]
    .openscm.to_long_data()
)
if sns_df.empty:
    raise AssertionError

fg = sns.relplot(
    data=sns_df,
    x="time",
    y="value",
    hue="scenario",
    style="stage",
    dashes=dict(
        harmonised="",
        history="",
        pre_processed=(1, 1),
        raw=(3, 3),
    ),
    col="region",
    col_wrap=3,
    facet_kws=dict(sharey=True),
    kind="line",
    linewidth=2,
)
fg.figure.suptitle(f"{model} - {variable}", y=1.01)
for ax in fg.axes:
    ax.axhline(0.0, color="k", alpha=0.3, zorder=1.0)

# %%
sns_df

# %%
locator = pix.ismatch(region="World")

sns_df = split_sectors(
    pix.concat(
        [
            # sdf.pix.assign(stage="raw"),
            history_model_gridding_relevant.loc[locator].pix.assign(stage="history").loc[:, 2000:harmonisation_year],
            gridding_workflow_emissions.pix.assign(stage="pre_processed"),
            gridding_workflow_emissions_harmonised.pix.assign(stage="harmonised"),
        ]
    ).loc[locator]
).openscm.to_long_data()
if sns_df.empty:
    raise AssertionError

fg = sns.relplot(
    data=sns_df,
    x="time",
    y="value",
    hue="scenario",
    style="stage",
    dashes=dict(
        harmonised="",
        history="",
        pre_processed=(1, 1),
        raw=(3, 3),
    ),
    col="sectors",
    row="species",
    facet_kws=dict(sharey=False),
    kind="line",
    linewidth=2,
)
fg.figure.suptitle(model, y=1.01)
for ax in fg.axes.flatten():
    ax.axhline(0.0, color="k", alpha=0.3, zorder=1.0)
