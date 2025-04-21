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
# # Model reporting
#
# Here we check the reporting of a given model.

# %% [markdown]
# ## Imports

# %%
import textwrap

import pandas as pd
import pandas_indexing as pix
from gcages.completeness import get_missing_levels

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.io import load_raw_scenario_data

# %% [markdown]
# ## Set up

# %%
pd.set_option("display.max_colwidth", None)

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"
output_dir: str = "data/reporting-checking"

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
# ### Stuff to move to gcages

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
def get_required_world_index_iamc(
    regions: tuple[str, ...] = ("World",),
    region_level: str = "region",
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [REQUIRED_WORLD_VARIABLES_IAMC, regions],
        names=["variable", "region"],
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
    # # Optional in reporting
    # "AFOLU|Land|Fires|Peat Burning",
)

REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC_CO2_EXCEPTIONS: tuple[str, ...] = (
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

REQUIRED_REGIONAL_VARIABLES_IAMC: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_IAMC
    for sector in REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC
    if not (species == "CO2" and sector in REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC_CO2_EXCEPTIONS)
)


# %%
def get_required_region_index_iamc(
    model_regions: tuple[str, ...],
    region_level: str = "region",
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [REQUIRED_REGIONAL_VARIABLES_IAMC, model_regions],
        names=["variable", "region"],
    )


# %% [markdown]
# ## Check completeness

# %% [markdown]
# ### Pick model

# %%
# for m in sorted(scenarios_raw.pix.unique("model")):
#     print(f"model = '{m}'")

# %% editable=true slideshow={"slide_type": ""}
model = "AIM 3.0"
model = "COFFEE 1.6"
# model = 'GCAM 7.1 scenarioMIP'
# model = 'IMAGE 3.4'
model = "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"
# model = 'REMIND-MAgPIE 3.5-4.10'
# model = 'WITCH 6.0'

# %% [markdown]
# Extract the model data, keeping:
#
# - only reported timesteps
# - only data from 2015 onwards (we don't care about data before this)

# %%
model_df = scenarios_raw.loc[pix.isin(model=model), 2015:].dropna(how="all", axis="columns")
if model_df.empty:
    raise AssertionError
# model_df

# %% [markdown]
# ### Figure out the model-specific regions

# %%
model_regions = [r for r in model_df.pix.unique("region") if r.startswith(model.split(" ")[0])]
if not model_regions:
    raise AssertionError
# model_regions

# %%
missing_world = get_missing_levels(model_df.index, get_required_world_index_iamc(), unit_col="unit")
if missing_world.empty:
    print("Nothing missing at the World level")

else:
    print("The following timeseries are missing at the World level")
    # Could save this to CSV
    display(missing_world.to_frame(index=False))

# %%
regional_missing = get_missing_levels(model_df.index, get_required_region_index_iamc(model_regions), unit_col="unit")

if regional_missing.empty:
    print("Nothing missing at the regional level")

else:
    # TODO: call function here
    pass

# Could save this to CSV
regional_missing.to_frame(index=False)

# %%
missing_by_region = regional_missing.to_frame(index=False).groupby("region")["variable"].apply(lambda x: x.values)

# %%
most_missing = []
most_missing_region = None
for region, mr in missing_by_region.items():
    if len(mr) > len(most_missing):
        most_missing = mr.tolist()
        most_missing_region = region

most_missing_region

# %%
all_regions_missing_the_same = all(
    set(mr) == set(missing_by_region.loc[most_missing_region]) for mr in missing_by_region
)

# %%
if all_regions_missing_the_same:
    print("All regions are missing the same variables")

# %%
missing_in_all = set(most_missing)
for mr in missing_by_region:
    missing_in_all = missing_in_all.intersection(set(mr))

print("Missing in all regions")
missing_in_all

# %%
for region, mr in missing_by_region.items():
    print(region)
    region_missing_specific = set(mr) - missing_in_all
    if region_missing_specific:
        print(textwrap.indent("\n".join(region_missing_specific), prefix="  - "))
    else:
        print("  - Nothing missing beyond what is missing in all regions")

    print()
