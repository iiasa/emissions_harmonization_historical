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
from functools import partial

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pandas_openscm.comparison
import tqdm.auto
from gcages.cmip7_scenariomip.gridding_emissions import get_complete_gridding_index
from gcages.completeness import assert_all_groups_are_complete

from emissions_harmonization_historical.constants_5000 import (
    BB4CMIP7_PROCESSED_DB,
    CEDS_PROCESSED_DB,
    COUNTRY_LEVEL_HISTORY,
    CREATE_HISTORY_FOR_GRIDDING_ID,
    HISTORY_HARMONISATION_INTERIM_DIR,
    MARKERS,
    REGION_MAPPING_FILE,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

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
common_definitions_mapping = {"MESSAGEix-GLOBIOM-GAINS 2.1-M-R12": "MESSAGEix-GLOBIOM-GAINS 2.1-R12"}
marker_models = [common_definitions_mapping[v[0]] if v[0] in common_definitions_mapping else v[0] for v in MARKERS]
# Not going to fix up various errors for non-markers
region_mapping = region_mapping[region_mapping["model"].isin(marker_models)]

region_mapping

# %%
region_mapping["model"].unique()

# %%
# Manual hacks
# TODO: push upstream into common-definitions
for model, iso_to_add, iso_group_to_match in (
    ("MESSAGEix-GLOBIOM-GAINS 2.1-R12", "sxm", "jam"),
    ("IMAGE 3.4", "sjm", "nor"),
    ("AIM 3.0", "ggy", "gbr"),
    ("AIM 3.0", "imn", "gbr"),
    ("AIM 3.0", "jey", "gbr"),
):
    region_mapping_model = region_mapping[region_mapping["model"] == model]
    if region_mapping_model.empty:
        raise AssertionError

    for model_region, mrdf in region_mapping_model.groupby("model_region"):
        if iso_group_to_match in mrdf["iso_list"].values[0]:
            model_region_to_add_to = model_region
            break

    else:
        msg = f"{model} {iso_group_to_match}"
        raise AssertionError(msg)

    region_mapping.loc[region_mapping["model_region"] == model_region_to_add_to, "iso_list"].iloc[0].append(iso_to_add)
    print(f"Added {iso_to_add} to {model_region_to_add_to}")

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
    # ISO3-ish regions
    country_history.pix.format(region="iso3ish|{region}"),
    # Include our World only data too
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
country_history_global = country_history.loc[pix.isin(region="global")]
country_history_not_global = country_history.loc[~pix.isin(region="global")]

zero_tolerance = 1e-6
country_history_global_non_zero = country_history_global[country_history_global.sum(axis="columns") > zero_tolerance]
country_history_not_global_non_zero = country_history_not_global[
    country_history_not_global.sum(axis="columns") > zero_tolerance
]

non_zero_in_both_global_and_country = (
    country_history_not_global_non_zero.reset_index("region")
    .index.drop_duplicates()
    .intersection(country_history_global_non_zero.reset_index("region").index.drop_duplicates())
)
if not non_zero_in_both_global_and_country.empty:
    raise AssertionError(non_zero_in_both_global_and_country)

# %%
# Check we don't lose any mass compared to country files
country_history_not_global_sum = country_history_not_global.openscm.groupby_except("region").sum()
country_history_not_global_sum.columns.name = "year"

for region_prefix in set([v.split("|")[0] for v in history_for_gridding.pix.unique("region")]):
    history_prefix_sum = (
        history_for_gridding.loc[pix.ismatch(region=f"{region_prefix}**")].openscm.groupby_except("region").sum()
    )

    if region_prefix in ["iso3ish", "World"]:
        compare_against = (
            country_history.openscm.groupby_except("region").sum().openscm.mi_loc(history_prefix_sum.index)
        )
    else:
        compare_against = country_history_not_global_sum.openscm.mi_loc(history_prefix_sum.index)

    comparison = pandas_openscm.comparison.compare_close(
        left=history_prefix_sum,
        right=compare_against,
        left_name="history_prefix_sum",
        right_name="country_history_sum",
        isclose=partial(np.isclose, atol=1e-12, rtol=1e-10),
    )
    if not comparison.empty:
        print(region_prefix)
        print("Problem")
        display(comparison)  # noqa: F821
        raise AssertionError(region_prefix)

# %%
# ceds_processed_data.loc[pix.isin(region="global") & pix.ismatch(variable=["**Aircraft", "**International Shipping"])]

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
        cdr_sectors_template.pix.assign(variable="Emissions|CO2|Enhanced Weathering"),
        cdr_sectors_template.pix.assign(variable="Emissions|CO2|Direct Air Capture"),
        cdr_sectors_template.pix.assign(variable="Emissions|CO2|Ocean"),
    ]
)

# %%
# Emissions|*|Other CDR
species_list = [
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
]

other_cdr_sector_templates = []
for sp in species_list:
    # Use any existing sector as a template for structure
    template = history_for_gridding.loc[pix.ismatch(variable=[f"Emissions|{sp}|Energy Sector"])].copy()
    template = template.pix.assign(model="Synthetic", variable=f"Emissions|{sp}|Other CDR")
    template.loc[:, :] = 0.0  # or your actual "Other" values
    other_cdr_sector_templates.append(template)

# Add all "Other" sectors to the dataset
history_for_gridding_incl_cdr = pix.concat([history_for_gridding_incl_cdr, *other_cdr_sector_templates])

# %% [markdown]
# ## Last checks

# %%
model_regions = [
    r for r in history_for_gridding_incl_cdr.pix.unique("region") if r != "World" and not r.startswith("iso3ish")
]
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
out_file = HISTORY_HARMONISATION_INTERIM_DIR / f"gridding-history_{CREATE_HISTORY_FOR_GRIDDING_ID}.feather"
out_file.parent.mkdir(exist_ok=True, parents=True)

history_for_gridding_incl_cdr.to_feather(out_file)
