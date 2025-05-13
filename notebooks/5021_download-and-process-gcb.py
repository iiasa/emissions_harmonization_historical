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
# # Download and process GCB
#
# Download and process data from the
# [global carbon budget (GCB)](https://globalcarbonbudgetdata.org/latest-data.html).
#
# If you want to process the fossil data (currently we don't),
# use the version from [10.5281/zenodo.14106218](https://zenodo.org/records/14106218),
# since the Excel sheet of fossil fuel production by country may have errors
# (in any case the sum of country emissions and bunkers
# does not equal the global total in the Excel sheets).
# For details on this, see https://bsky.app/profile/cjsmith.be/post/3lbhxt4chqc2x.

# %% [markdown]
# ## Imports

# %%
from functools import partial
from pathlib import Path

import openscm_units
import pandas as pd
import pandas_indexing  # noqa: F401
import pooch
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    GCB_PROCESSED_DB,
    GCB_RAW_PATH,
    GCB_VERSION,
    HISTORY_SCENARIO_NAME,
)

# %% [markdown]
# ## Setup

# %%
lulucf_filename = f"National_LandUseChange_Carbon_Emissions_{GCB_VERSION}.xlsx"
lulucf_filename


# %% [markdown]
# ## Helper functions


# %%
def read_lu_sheet(filepath: Path, sheet_name: str, raw_units: str) -> pd.DataFrame:
    """Read a land-use sheet from GCB"""
    res = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        skiprows=7,
        index_col=0,
    )
    if res.index.name != f"unit: {raw_units}":
        msg = f"Check units. Got units={res.index.name!r}"
        raise AssertionError(msg)

    res.index.name = "year"

    return res


# %% [markdown]
# ## Download

# %%
downloaded_files = pooch.retrieve(
    url=f"https://globalcarbonbudgetdata.org/downloads/jGJH0-data/{lulucf_filename}",
    fname=lulucf_filename,
    path=GCB_RAW_PATH,
    known_hash="e22321b7457823650357ec19b25f570adf15646e5e1944a5b14f893bcc9e1874",
    progressbar=True,
)

# %% [markdown]
# ## Process

# %%
rlus = partial(read_lu_sheet, filepath=GCB_RAW_PATH / lulucf_filename)

raw_units = "Tg C/year"
df_lu = (
    pd.concat(
        [
            rlus(sheet_name="BLUE", raw_units=raw_units),
            rlus(sheet_name="H&C2023", raw_units=raw_units),
            rlus(sheet_name="OSCAR", raw_units=raw_units),
            rlus(sheet_name="LUCE", raw_units=raw_units),
        ]
    )
    .groupby("year")
    .mean()
)
df_lu

# %%
df_lu_global = df_lu[["Global"]].rename(columns={"Global": "World"})
# # TODO: ask Chris what was going on with this, emissions aren't 600 in 1850...
# df_lu_extended = df_lu_global.copy()
# df_lu_extended.loc[1750:1849, "World"] = np.linspace(3, 597, 100)
# df_lu_global

# %%
out_units = "Mt CO2/yr"
conversion_factor = openscm_units.unit_registry.Quantity(1, raw_units).to(out_units).m
conversion_factor

# %%
df_lu_global_converted = df_lu_global * conversion_factor
# df_lu_global_converted

# %%
df_lu_out = df_lu_global_converted.rename_axis("region", axis="columns").T.pix.assign(
    variable="Emissions|CO2|Biosphere",
    unit=out_units,
    model="Global Carbon Budget",
    scenario=HISTORY_SCENARIO_NAME,
)
df_lu_out

# %% [markdown]
# Make sure we can convert the variable names

# %%
update_index_levels_func(
    df_lu_out,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %% [markdown]
# ## Save

# %%
GCB_PROCESSED_DB.save(df_lu_out, allow_overwrite=True)
