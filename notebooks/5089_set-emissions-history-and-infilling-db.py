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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Set emissions history and infilling DB
#
# Here we download the emissions history and infilling DB from Zenodo.
# This is taken from Zenodo to ensure that the workflow
# is reproducible, stable
# and doesn't require users to re-process all the input data.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import sys
import tempfile

import pandas as pd
import pandas_indexing  # noqa: F401
from loguru import logger

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    HISTORY_ZENODO_RECORD_ID,
    INFILLING_DB,
    INFILLING_DB_ZENODO_RECORD_ID,
)
from emissions_harmonization_historical.zenodo import download_zenodo_url, get_zenodo_interactor

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Setup

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]

# %%
logger.configure(handlers=[dict(sink=sys.stderr, level="INFO")])
logger.enable("openscm_zenodo")

# %% editable=true slideshow={"slide_type": ""}
zenodo_interactor = get_zenodo_interactor()

# %% [markdown]
# ## History

# %%
zenodo_response_for_record = zenodo_interactor.get_record(HISTORY_ZENODO_RECORD_ID)
zenodo_response_for_record.raise_for_status()

# %%
download_urls = {
    f["key"]: {"url": f["links"]["self"], "size": f["size"]} for f in zenodo_response_for_record.json()["files"]
}
# download_urls

# %%
for prefix, purpose in (
    (
        "global-workflow-history",
        "global_workflow_emissions",
    ),
    (
        "gridding-history",
        "gridding_emissions",
    ),
):
    file_info_l = [download_urls[k] for k in download_urls if (k.startswith(prefix) and k.endswith(".parquet.gzip"))]
    if len(file_info_l) != 1:
        raise AssertionError(file_info_l)

    file_info = file_info_l[0]

    print(f"Downloading {purpose} emissions from https://zenodo.org/uploads/{HISTORY_ZENODO_RECORD_ID}")
    with tempfile.NamedTemporaryFile(suffix=".parquet.gzip") as tf:
        download_zenodo_url(
            file_info["url"],
            # We require the interactor while the record's files are embargoed.
            zenodo_interactor,
            fh=tf,
            size=file_info["size"],
        )
        df = pd.read_parquet(tf.name)

    print(f"Adding {purpose} emissions to the history for harmonisation database")
    HISTORY_HARMONISATION_DB.save(
        df.pix.assign(purpose=purpose),
        allow_overwrite=True,
    )

# %% [markdown]
# ## Infilling database

# %%
zenodo_response_for_record = zenodo_interactor.get_record(INFILLING_DB_ZENODO_RECORD_ID)
zenodo_response_for_record.raise_for_status()

# %%
download_urls = {
    f["key"]: {"url": f["links"]["self"], "size": f["size"]} for f in zenodo_response_for_record.json()["files"]
}
# download_urls

# %% editable=true slideshow={"slide_type": ""}
file_info_l = [download_urls[k] for k in download_urls if (k.startswith("infiling-db") and k.endswith(".parquet.gzip"))]
if len(file_info_l) != 1:
    raise AssertionError(file_info_l)

file_info = file_info_l[0]

print(f"Downloading infilling database from https://zenodo.org/uploads/{INFILLING_DB_ZENODO_RECORD_ID}")
with tempfile.NamedTemporaryFile(suffix=".parquet.gzip") as tf:
    download_zenodo_url(
        file_info["url"],
        # We require the interactor while the record's files are embargoed.
        zenodo_interactor,
        fh=tf,
        size=file_info["size"],
    )
    df = pd.read_parquet(tf.name)

print("Saving infilling database")
INFILLING_DB.save(
    df,
    allow_overwrite=True,
)
