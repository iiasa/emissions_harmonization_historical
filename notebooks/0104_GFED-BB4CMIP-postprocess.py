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
# # Post-process prepared GFED BB4CMIP data
#
# The 0103 script dumped the files into separate CSVs; here we combine and make consistenct with other emissions files
# and IAMC format

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import pandas_indexing as pix
import pint

from emissions_harmonization_historical.constants import DATA_ROOT, GFED_PROCESSING_ID
from emissions_harmonization_historical.units import assert_units_match_wishes

# %%
# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
data_path = DATA_ROOT / "national/gfed-bb4cmip/processed"

# %%
out_path_global = data_path / f"gfed-bb4cmip_cmip7_global_{GFED_PROCESSING_ID}.csv"
out_path_national = data_path / f"gfed-bb4cmip_cmip7_national_{GFED_PROCESSING_ID}.csv"

# %%
species = [
    "BC",
    "NMVOC",
    "CO",
    "CO2",
    "CH4",
    "N2O",
    "OC",
    "NH3",
    "NOx",
    "SO2",
]

# %%
df_list = []
# Rename variable in place
for s in species:
    for suffix in [f"_world_{GFED_PROCESSING_ID}", f"_national_{GFED_PROCESSING_ID}"]:
        df_in = pd.read_csv(data_path / f"{s}{suffix}.csv")
        df_in.variable = f"Emissions|{s}|Biomass Burning"
        df_list.append(df_in)
        # display(df_in)

# %%
df = pd.concat(df_list)

# %%
df["model"] = "BB4CMIP"
df

# %%
# sort order: region, variable
df_sorted = df.sort_values(["region", "variable"])

# %%
df_sorted

# %%
# fix column order
df_reordered = df_sorted.set_index(["model", "scenario", "region", "variable", "unit"])

# %%
df_renamed = df_reordered.rename(
    index={
        "Emissions|SO2|Biomass Burning": "Emissions|Sulfur|Biomass Burning",
        "Emissions|NMVOC|Biomass Burning": "Emissions|VOC|Biomass Burning",
    },
    level="variable",
)
df_renamed

# %%
with pint.get_application_registry().context("NOx_conversions"):
    df_renamed_desired_units = pix.units.convert_unit(
        df_renamed,
        {"Mt NO / yr": "Mt NO2/yr"},
    )

# %%
df_renamed_desired_units = pix.units.convert_unit(
    df_renamed_desired_units,
    lambda x: x.replace(" / yr", "/yr"),
)
df_renamed_desired_units = pix.units.convert_unit(
    df_renamed_desired_units,
    {"Mt N2O/yr": "kt N2O/yr", "Mt NMVOC/yr": "Mt VOC/yr"},
)

# %%
assert_units_match_wishes(df_renamed_desired_units)
df_renamed_desired_units.columns = df_renamed_desired_units.columns.astype(int)

# %%
out_global = df_renamed_desired_units.loc[pix.ismatch(region="World")]
out_global

# %%
out_national = df_renamed_desired_units.loc[~pix.ismatch(region="World")]
out_national

# %%
assert out_national.shape[0] + out_global.shape[0] == df_renamed_desired_units.shape[0]

# %% [markdown]
# Check that national sums equal global total.

# %%
national_sums_checker = (
    pix.assignlevel(out_national.groupby(["model", "scenario", "variable", "unit"]).sum(), region="World")
    .reset_index()
    .set_index(out_global.index.names)
)
national_sums_checker.columns = national_sums_checker.columns.astype(int)
national_sums_checker

# %%
out_global

# %%
pd.testing.assert_frame_equal(out_global, national_sums_checker, check_like=True)

# %%
out_global.to_csv(out_path_global)
out_path_global

# %%
out_national.to_csv(out_path_national)
out_path_national
