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
# # Harmonisation
#
# Here we harmonise the data for each model
# (has to be model specific as each model has different regions).
#
# You will need to run notebook `2000*` before this one.

# %% [markdown]
# ## Imports

# %%
import pandas_indexing as pix
import pandas_openscm
from gcages.aneris_helpers import harmonise_all
from gcages.units_helpers import strip_pint_incompatible_characters_from_units
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_HARMONISATION_ID,
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    HARMONISATION_VALUES_ID,
    IAMC_REGION_PROCESSING_ID,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
HISTORICAL_EMISSIONS_COUNTRY_WORLD_FILE = (
    DATA_ROOT / "combined-processed-output" / f"cmip7_history_{COMBINED_HISTORY_ID}.csv"
)
# HISTORICAL_EMISSIONS_COUNTRY_WORLD_FILE

# %%
HISTORICAL_EMISSIONS_MODEL_REGION_FILE = (
    DATA_ROOT / "combined-processed-output" / f"iamc_regions_cmip7_history_{IAMC_REGION_PROCESSING_ID}.csv"
)
# HISTORICAL_EMISSIONS_MODEL_REGION_FILE

# %%
HISTORICAL_EMISSIONS_GLOBAL_WORKFLOW_FILE = (
    DATA_ROOT
    / "global-composite"
    / f"cmip7-harmonisation-history_world_{COMBINED_HISTORY_ID}_{HARMONISATION_VALUES_ID}.csv"
)
# HISTORICAL_EMISSIONS_GLOBAL_WORKFLOW_FILE

# %%
model: str = "REMIND-MAgPIE 3.5-4.10"

# %%
out_dir = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / "_".join([CMIP7_SCENARIOMIP_PRE_PROCESSING_ID, CMIP7_SCENARIOMIP_HARMONISATION_ID])
)
out_dir.mkdir(exist_ok=True, parents=True)

# %%
out_db = OpenSCMDB(
    db_dir=out_dir,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# %%
in_dir = DATA_ROOT / "cmip7-scenariomip-workflow" / "pre-processing" / CMIP7_SCENARIOMIP_PRE_PROCESSING_ID

# %%
in_db = OpenSCMDB(
    db_dir=in_dir,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

in_db.load_metadata().shape

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Pre-processed gridding emissions

# %%
model_pre_processed = in_db.load(pix.isin(model=model, stage="gridding_emissions"), progress=True).reset_index(
    "stage", drop=True
)
if model_pre_processed.empty:
    raise AssertionError

# model_pre_processed

# %%
model_pre_processed_global_workflow = in_db.load(
    pix.isin(model=model, stage="global_workflow_emissions"), progress=True
).reset_index("stage", drop=True)
if model_pre_processed_global_workflow.empty:
    raise AssertionError

# model_pre_processed_global_workflow

# %% [markdown]
# ## Harmonise

# %%
harmonisation_year = 2022

# %% [markdown]
# ### Historical emissions

# %%
history = pix.concat(
    [
        strip_pint_incompatible_characters_from_units(
            load_timeseries_csv(
                f,
                index_columns=["model", "scenario", "region", "variable", "unit"],
                out_column_type=int,
            )
        )
        for f in [HISTORICAL_EMISSIONS_COUNTRY_WORLD_FILE, HISTORICAL_EMISSIONS_MODEL_REGION_FILE]
    ]
)
# Cut just to period of interest
history = history.loc[:, 1990:harmonisation_year]

history

# %% [markdown]
# Temporary hack: rename variables until we update common definitions.

# %%
history = update_index_levels_func(
    history,
    {
        "region": lambda x: x.replace("REMIND-MAgPIE 3.4-4.8", "REMIND-MAgPIE 3.5-4.10").replace(
            "COFFEE 1.5", "COFFEE 1.6"
        )
    },
)

# %% [markdown]
# Strip down to just things relevant for the model.

# %%
history_model_relevant = multi_index_lookup(
    history,
    model_pre_processed.index.droplevel(
        model_pre_processed.index.names.difference(["variable", "region"])
    ).drop_duplicates(),
)
history_model_relevant = history_model_relevant

history_model_relevant

# %%
from aneris.methods import default_methods
from gcages.aneris_helpers import _convert_units_to_match

# %%
from gcages.index_manipulation import split_sectors
from pandas_indexing import isin

# %%
aneris_default_methods_l = []
for (model, scenario), msdf in model_pre_processed.groupby(["model", "scenario"]):
    hist_msdf = history.loc[
        isin(region=msdf.pix.unique("region"))  # type: ignore
        & isin(variable=msdf.pix.unique("variable"))  # type: ignore
    ]
    # _check_data(hist_msdf, msdf, year)

    hist_msdf = _convert_units_to_match(start=hist_msdf, match=msdf)

    # need to convert to aneris' internal datastructure
    level_order = ["model", "scenario", "region", "variable", "unit"]
    msdf_aneris = msdf.reorder_levels(level_order)
    # Drop out any years that are all nan before passing to aneris
    msdf_aneris = msdf_aneris.dropna(how="all", axis="columns")
    msdf_aneris = split_sectors(msdf_aneris, middle_level="gas", bottom_level="sector")
    msdf_aneris.index = msdf_aneris.index.droplevel(msdf_aneris.index.names.difference(["region", "gas", "sector"]))

    # Convert to format expected by aneris
    hist_msdf_aneris = hist_msdf.pix.assign(model="history", scenario="scen").reorder_levels(level_order)
    hist_msdf_aneris = split_sectors(hist_msdf_aneris, middle_level="gas", bottom_level="sector")
    hist_msdf_aneris.index = hist_msdf_aneris.index.droplevel(
        hist_msdf_aneris.index.names.difference(["region", "gas", "sector"])
    )

    aneris_defaults_ms = default_methods(
        hist_msdf_aneris,
        msdf_aneris,
        base_year=harmonisation_year,
    )

    aneris_default_methods_l.append(
        aneris_defaults_ms[0].to_frame("default_method").pix.assign(model=model, scenario=scenario)
    )

aneris_default_methods = pix.concat(aneris_default_methods_l)
aneris_default_methods

# %%
gridding_workflow_emissions_harmonised = harmonise_all(
    scenarios=model_pre_processed,
    history=history_model_relevant,
    year=harmonisation_year,
)

gridding_workflow_emissions_harmonised

# %%
history_global_workflow = load_timeseries_csv(
    HISTORICAL_EMISSIONS_GLOBAL_WORKFLOW_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
).loc[:, 1990:harmonisation_year]

history_global_workflow

# %%
from gcages.renaming import SupportedNamingConventions, convert_variable_name


# %%
# TODO: go back and update naming used for history
def hack_update(v: str) -> str:
    if v == "Emissions|CF4":
        v = "Emissions|PFC|CF4"

    if v == "Emissions|C2F6":
        v = "Emissions|PFC|C2F6"

    if v == "Emissions|C3F8":
        v = "Emissions|PFC|C3F8"

    if v == "Emissions|C4F10":
        v = "Emissions|PFC|C4F10"

    if v == "Emissions|C5F12":
        v = "Emissions|PFC|C5F12"

    if v == "Emissions|C6F14":
        v = "Emissions|PFC|C6F14"

    if v == "Emissions|C7F16":
        v = "Emissions|PFC|C7F16"

    if v == "Emissions|C8F18":
        v = "Emissions|PFC|C8F18"

    if v == "Emissions|cC4F8":
        v = "Emissions|PFC|cC4F8"

    if v == "Emissions|HFC|HFC245fa":
        v = "Emissions|HFC|HFC245ca"

    if "Montreal Gases" in v:
        v = v.replace("Montreal Gases|", "")

    if "CFC" in v:
        v = v.replace("CFC|", "")

    return convert_variable_name(
        v,
        from_convention=SupportedNamingConventions.IAMC,
        to_convention=SupportedNamingConventions.GCAGES,
    )


history_global_workflow_renamed = update_index_levels_func(history_global_workflow, {"variable": hack_update})


# %%
history_model_relevant_global = multi_index_lookup(
    history_global_workflow_renamed,
    model_pre_processed_global_workflow.index.droplevel(
        model_pre_processed.index.names.difference(["variable", "region"])
    ).drop_duplicates(),
)

history_model_relevant_global

# %%
# TODO: reprocess to get all the way through this
harmonisation_year = 2021
global_workflow_emissions_harmonised = harmonise_all(
    scenarios=model_pre_processed_global_workflow,
    history=history_model_relevant_global,
    year=harmonisation_year,
)

global_workflow_emissions_harmonised

# %%
# TODO: save data

# %%
from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions

# %%
# to_global_workflow_emissions(history_model_relevant)

# %%
pdf = pix.concat(
    [
        to_global_workflow_emissions(gridding_workflow_emissions_harmonised.pix.assign(stage="harmonised_gridded")),
        global_workflow_emissions_harmonised.pix.assign(stage="harmonised_global"),
        model_pre_processed_global_workflow.pix.assign(stage="raw"),
        history_global_workflow_renamed.pix.assign(stage="history_global"),
    ]
)
pdf = pdf.openscm.to_long_data(time_col_name="year")

pdf

# %%
import seaborn as sns

# %%
sns.relplot(
    data=pdf[pdf["variable"].isin(["Emissions|BC", "Emissions|CO2|Fossil", "Emissions|CH4"])],
    x="year",
    y="value",
    col="variable",
    hue="scenario",
    style="stage",
    kind="line",
    col_wrap=2,
    facet_kws=dict(sharey=False),
)
