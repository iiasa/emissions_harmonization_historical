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
# # Create history for gridding

# %% [markdown]
# ## Imports

# %%
import sys

import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import tqdm.auto
from gcages.cmip7_scenariomip.gridding_emissions import get_complete_gridding_index
from gcages.completeness import assert_all_groups_are_complete
from gcages.index_manipulation import split_sectors
from loguru import logger

from emissions_harmonization_historical.constants_5000 import (
    BB4CMIP7_PROCESSED_DB,
    CEDS_PROCESSED_DB,
    COUNTRY_LEVEL_HISTORY,
    HISTORY_FOR_HARMONISATION_ID,
    HISTORY_HARMONISATION_DB,
    HISTORY_HARMONISATION_DIR,
    REGION_MAPPING_FILE,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR
from emissions_harmonization_historical.zenodo import upload_to_zenodo

# %%
COUNTRY_LEVEL_HISTORY

# %% [markdown]
# ## Setup

# %%
pandas_openscm.register_pandas_accessor()

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Region mapping

# %%
region_mapping = pd.read_csv(REGION_MAPPING_FILE)
region_mapping = region_mapping.rename(columns={"name": "model_region", "hierarchy": "model", "iso3": "iso_list"})
region_mapping = region_mapping[["model_region", "model", "iso_list"]]
region_mapping = region_mapping.dropna(
    subset=["iso_list"]
)  # don't try to aggregate anything if there are no countries defined for a specific region
region_mapping["iso_list"] = region_mapping["iso_list"].str.lower()
region_mapping["iso_list"] = region_mapping["iso_list"].apply(
    lambda x: x.strip("[]").replace("'", "").split(", ")
)  # transform from Series/string to list-like object which is iterable
region_mapping

# %%
# region_mapping[region_mapping["model"].str.startswith("GCAM")]

# %% [markdown]
# ### History data

# %%
ceds_processed_data = CEDS_PROCESSED_DB.load(pix.isin(stage="iso3c_ish")).reset_index("stage", drop=True)
# ceds_processed_data.loc[pix.ismatch(variable=["**CH4**", "**N2O**"])]

# %%
gfed4_processed_data = BB4CMIP7_PROCESSED_DB.load(pix.isin(stage="iso3c")).reset_index("stage", drop=True)
# gfed4_processed_data

# %% [markdown]
# ### export country-level interim history for use in gridding repo

# %%
country_history = pix.concat(
    [
        ceds_processed_data,
        gfed4_processed_data,
    ]
)

# %%
country_history.to_csv(COUNTRY_LEVEL_HISTORY)

# %% [markdown]
# ## Identify ISO codes that won't map correctly

# %%
history_included_codes = {}
for model, mdf in country_history.groupby("model"):
    model_iso_codes = set(mdf.pix.unique("region"))
    history_included_codes[model] = model_iso_codes

# %%
model_included_codes = {}
for model, mdf in region_mapping.groupby("model"):
    model_iso_codes = set([r for v in mdf["iso_list"] for r in v])
    model_included_codes[model] = model_iso_codes

# %%
missing_isos_l = []
for model, iso_codes in model_included_codes.items():
    model_row = {"model": model}
    for history_source, history_iso_codes in history_included_codes.items():
        model_row[f"missing_vs_{history_source}"] = sorted(history_iso_codes - iso_codes - {"global"})
        model_row[f"extra_vs_{history_source}"] = sorted(iso_codes - history_iso_codes)

    missing_isos_l.append(model_row)

missing_isos = pd.DataFrame(missing_isos_l)
missing_isos  # .set_index("model").loc["AIM 3.0"]["missing_vs_GFED4"]

# %% [markdown]
# ## Aggregate into regions we need for harmonising gridding emissions

# %%
history_for_gridding_l = [
    # Start with our World data
    ceds_processed_data.loc[
        pix.isin(region="global") & pix.ismatch(variable=["**Aircraft", "**International Shipping"])
    ].pix.assign(region="World"),
]
for model_region, iso_list in tqdm.auto.tqdm(region_mapping[["model_region", "iso_list"]].to_records(index=False)):
    history_for_model_region = country_history.loc[pix.isin(region=iso_list)]
    history_model_region = (
        history_for_model_region.openscm.groupby_except("region").sum(min_count=1).pix.assign(region=model_region)
    )
    history_for_gridding_l.append(history_model_region)

history_for_gridding = pix.concat(history_for_gridding_l).rename_axis("year", axis="columns")
history_for_gridding

# %%
split_sectors(history_for_gridding.loc[pix.ismatch(variable="**|OC|**Burning", region="AIM**")]).openscm.groupby_except(
    ["region", "sectors"]
).sum(min_count=1)

# %%
split_sectors(history_for_gridding.loc[pix.ismatch(variable="**|CH4|**", region="AIM**")]).openscm.groupby_except(
    ["region", "sectors"]
).sum(min_count=1)

# %% [markdown]
# ### Add synthetic history for CDR sectors
#
# We assume that all CDR sectors had a value of zero in the historical period.

# %%
cdr_sectors_template = history_for_gridding.loc[pix.ismatch(variable=["Emissions|CO2|Energy Sector"])].pix.assign(
    model="Synthetic", variable="CDR template"
)
# Assume zero for this history
cdr_sectors_template.loc[:, :] = 0.0

history_for_gridding_incl_cdr = pix.concat(
    [
        history_for_gridding,
        cdr_sectors_template.pix.assign(variable="Emissions|CO2|BECCS"),
        cdr_sectors_template.pix.assign(variable="Emissions|CO2|Other non-Land CDR"),
    ]
)

history_for_gridding_incl_cdr

# %% [markdown]
# ## Last checks

# %%
model_regions = [r for r in history_for_gridding_incl_cdr.pix.unique("region") if r != "World"]
complete_gridding_index = get_complete_gridding_index(model_regions=model_regions)
# complete_gridding_index
assert_all_groups_are_complete(
    history_for_gridding_incl_cdr.pix.assign(model="history"),
    complete_gridding_index,
)

# %%
if history_for_gridding_incl_cdr[HARMONISATION_YEAR].isnull().any():
    missing = history_for_gridding_incl_cdr.loc[history_for_gridding_incl_cdr[HARMONISATION_YEAR].isnull()]

    display(missing)  # noqa: F821
    raise AssertionError

# %% [markdown]
# ## Save

# %%
HISTORY_HARMONISATION_DB.save(
    history_for_gridding_incl_cdr.pix.assign(purpose="gridding_emissions"), allow_overwrite=True
)

# %% [markdown]
# ## Upload to Zenodo

# %%
# Rewrite as single file
out_file_grid = HISTORY_HARMONISATION_DIR / f"history-for-gridding-workflow_{HISTORY_FOR_HARMONISATION_ID}.csv"
gridding = HISTORY_HARMONISATION_DB.load(pix.isin(purpose="gridding_emissions")).loc[:, :HARMONISATION_YEAR]
gridding.to_csv(out_file_grid)
out_file_grid

# %%
logger.configure(handlers=[dict(sink=sys.stderr, level="INFO")])
logger.enable("openscm_zenodo")

# %%
upload_to_zenodo([out_file_grid], remove_existing=False, update_metadata=True)
