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
# # Compile historical emissions for gridding
#
# Here we compile the historical emissions needed for gridding.

# %% [markdown]
# ## Imports

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas_indexing as pix
import pandas_openscm
import tqdm.auto
from gcages.cmip7_scenariomip.gridding_emissions import get_complete_gridding_index
from gcages.completeness import assert_all_groups_are_complete
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    HISTORY_SCENARIO_NAME,
    IAMC_REGION_PROCESSING_ID,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
HISTORICAL_EMISSIONS_COUNTRY_WORLD_FILE = (
    DATA_ROOT / "combined-processed-output" / f"cmip7_history_{COMBINED_HISTORY_ID}.csv"
)
HISTORICAL_EMISSIONS_COUNTRY_WORLD_FILE

# %%
HISTORICAL_EMISSIONS_MODEL_REGION_FILE = (
    DATA_ROOT / "combined-processed-output" / f"iamc_regions_cmip7_history_{IAMC_REGION_PROCESSING_ID}.csv"
)
# HISTORICAL_EMISSIONS_MODEL_REGION_FILE

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_DB = OpenSCMDB(
    db_dir=SCENARIO_PATH / SCENARIO_TIME_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

SCENARIO_DB.load_metadata().shape

# %%
OUT_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"gridding-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load data

# %%
historical_emissions_world = (
    load_timeseries_csv(
        HISTORICAL_EMISSIONS_COUNTRY_WORLD_FILE,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
    .loc[pix.isin(region="World")]
    .pix.assign(model="history", scenario="history")
)
if historical_emissions_world.empty:
    raise AssertionError

# historical_emissions_world

# %%
historical_emissions_regional = load_timeseries_csv(
    HISTORICAL_EMISSIONS_MODEL_REGION_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
).pix.assign(model="history", scenario="history")
if historical_emissions_regional.empty:
    raise AssertionError

# historical_emissions_regional

# %% [markdown]
# ## Process

# %% [markdown]
# ### Sectors we grid at the global level

# %%
world_gridding_index = get_complete_gridding_index(model_regions=["not", "used"])
world_gridding_index = world_gridding_index[world_gridding_index.get_level_values("region") == "World"]
if world_gridding_index.empty:
    raise AssertionError

world_gridding_index

# %%
gridding_harmonisation_emissions_world = historical_emissions_world.openscm.mi_loc(world_gridding_index)
assert_all_groups_are_complete(gridding_harmonisation_emissions_world, world_gridding_index)

# gridding_harmonisation_emissions_world

# %% [markdown]
# No high variability headaches here so nothing more to do.

# %% [markdown]
# ### Sectors we grid at the model region level
#
# Much more complicated because we have to do this model by model.

# %% [markdown]
# Get the models regions from their reporting.

# %%
scenario_db_metadata = SCENARIO_DB.load_metadata()

# This double looping thing is yuck,
# but speed isn't an issue so I am moving on.
model_regions_all = defaultdict(list)
for model, region in scenario_db_metadata.droplevel(
    scenario_db_metadata.names.difference(["model", "region"])
).drop_duplicates():
    model_regions_all[model].append(region)

model_regions = defaultdict(list)
for model in scenario_db_metadata.get_level_values("model").unique():
    model_tok = model.split(" ")[0]
    for region in model_regions_all[model]:
        if region.startswith(model_tok):
            model_regions[model].append(region)

    if model not in model_regions:
        msg = f"Regions for {model} were not extracted {model_regions_all[model]=}"
        raise AssertionError(msg)

# model_regions

# %% [markdown]
# Extract the relevant historical data.

# %%
potential_model_version_issues = {
    "REMIND-MAgPIE 3.5-4.10": "REMIND-MAgPIE 3.4-4.8",
    "COFFEE 1.6": "COFFEE 1.5",
}


# %%
def try_retrieving_with_regions(regions):
    """
    Try retrieving model region data for a given set of regions
    """
    model_region_gridding_index = get_complete_gridding_index(model_regions=regions)
    model_region_gridding_index = model_region_gridding_index[
        model_region_gridding_index.get_level_values("region") != "World"
    ]

    gridding_historical_emissions_model_specific = historical_emissions_regional.openscm.mi_loc(
        model_region_gridding_index
    )
    assert_all_groups_are_complete(
        gridding_historical_emissions_model_specific, model_region_gridding_index, group_keys=None
    )

    return gridding_historical_emissions_model_specific


# %%
gridding_historical_emissions_model_region_l = []
for model, regions in model_regions.items():
    try:
        gridding_historical_emissions_model_specific = try_retrieving_with_regions(regions)
    except ValueError:
        model_regions_missing_from_history = [
            r for r in regions if r not in historical_emissions_regional.index.get_level_values("region").unique()
        ]
        if not model_regions_missing_from_history:
            # No idea what has gone wrong
            raise

        if model not in potential_model_version_issues:
            # No idea what has gone wrong
            raise

        # Try with different region names
        model_version_to_try = potential_model_version_issues[model]
        regions_renamed = [r.replace(model, model_version_to_try) for r in regions]
        gridding_historical_emissions_model_specific = try_retrieving_with_regions(regions_renamed)
        gridding_historical_emissions_model_specific = (
            gridding_historical_emissions_model_specific.openscm.update_index_levels(
                {"region": lambda r: r.replace(model_version_to_try, model)}, copy=False
            )
        )

    gridding_historical_emissions_model_region_l.append(gridding_historical_emissions_model_specific)

gridding_historical_emissions_model_region = pix.concat(gridding_historical_emissions_model_region_l)

# Double check
res_regions = gridding_historical_emissions_model_region.index.get_level_values("region").unique()
for model, regions in model_regions.items():
    missing = [r for r in regions if r not in res_regions]
    if missing:
        msg = f"Missing regions in output for {model}. {missing=}"
        raise AssertionError(msg)

gridding_historical_emissions_model_region

# %% [markdown]
# ## Smooth high variability variables

# %%
# TODO: decide which variables exactly to use averaging with
high_variability_variables = sorted(
    [
        v
        for v in gridding_historical_emissions_model_region.pix.unique("variable")
        if "Burning" in v
        and any(
            s in v
            for s in (
                "BC",
                "CO",
                "CH4",
                # # Having looked at the data, I'm not sure I would do this for CO2
                # "CO2",
                "N2O",
                "NH3",
                "NOx",
                "OC",
                "VOC",
            )
        )
    ]
)
high_variability_variables

# %%
[r for r in gridding_historical_emissions_model_region.pix.unique("region") if "REMIND" in r]

# %%
# Match biomass burning smoothing
n_years_for_average = 5
plot_regions = ["AIM 3.0|Brazil", "REMIND-MAgPIE 3.5-4.10|Canada, Australia, New Zealand"]
plot_regions = []

gridding_harmonisation_emissions_model_region_l = []
for (variable, region), vrdf in tqdm.auto.tqdm(
    gridding_historical_emissions_model_region.groupby(["variable", "region"])
):
    if variable in high_variability_variables:
        tmp = vrdf.copy()
        harmonisation_value = tmp.loc[:, HARMONISATION_YEAR - n_years_for_average + 1 : HARMONISATION_YEAR].mean(
            axis="columns"
        )

        if region in plot_regions:
            ax = vrdf.pix.project(["variable", "region"]).loc[:, 1990:].T.plot()
            ax.scatter(HARMONISATION_YEAR, float(harmonisation_value.iloc[0]), marker="x", color="tab:orange")
            ax.grid(which="major")
            # ax.set_xticks(regress_vals.columns, minor=True)
            # ax.grid(which="minor")
            plt.show()

        tmp[HARMONISATION_YEAR] = harmonisation_value

    else:
        tmp = vrdf

    gridding_harmonisation_emissions_model_region_l.append(tmp)

gridding_harmonisation_emissions_model_region = pix.concat(gridding_harmonisation_emissions_model_region_l)

# %% [markdown]
# ## Combine and save

# %%
gridding_harmonisation_emissions = pix.concat(
    [
        gridding_harmonisation_emissions_world,
        gridding_harmonisation_emissions_model_region,
    ]
).pix.assign(scenario=HISTORY_SCENARIO_NAME)
gridding_harmonisation_emissions.columns.name = "year"
# gridding_historical_emissions

# %%
gridding_harmonisation_emissions.to_csv(OUT_FILE)
OUT_FILE
