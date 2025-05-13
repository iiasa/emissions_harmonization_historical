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
# # Download and process RCMIP data
#
# Download the [RCMIP](rcmip.org) data and process it,
# the simplest way to get the values we used in CMIP6.

# %% [markdown]
# ## Imports

# %%
import multiprocessing

import pandas_indexing as pix
import pooch
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants_5000 import (
    RCMIP_PROCESSED_DB,
    RCMIP_RAW_PATH,
    RCMIP_VERSION_ID,
)

# %% [markdown]
# ## Download

# %%
url_to_hit = f"https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/{RCMIP_VERSION_ID}/rcmip-emissions-annual-means-{RCMIP_VERSION_ID.replace('.', '-')}.csv"  # noqa: E501
url_to_hit

# %%
downloaded_file = pooch.retrieve(
    url_to_hit,
    known_hash=None,
    path=RCMIP_RAW_PATH,
    progressbar=True,
)

# %% [markdown]
# ## Processs

# %%
res = (
    load_timeseries_csv(
        downloaded_file,
        lower_column_names=True,
        index_columns=["model", "scenario", "variable", "region", "unit", "mip_era", "activity_id"],
        out_columns_type=int,
    )
    .reset_index(["mip_era", "activity_id"], drop=True)
    .loc[pix.ismatch(scenario=["ssp*", "rcp*", "hist*"])]
)
# res

# %%
RCMIP_PROCESSED_DB.save(
    res, groupby=["model", "scenario", "region"], progress=True, max_workers=multiprocessing.cpu_count()
)
