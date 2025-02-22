# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# import external packages and functions
from pathlib import Path

import pandas as pd
import pandas_indexing as pix

from emissions_harmonization_historical.constants import (
    CEDS_PROCESSING_ID,
    DATA_ROOT,
)

# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
# choose data set options to combine
CEDS_OPTION = (
    "CEDS-Zenodo-national"  # options: ["CEDS-Zenodo-national"] # later, something like "CEDS-ESGF-gridded" may be added
)
BURNING_OPTION = "GFED4_1s"  # options: ["GFED4_1s", "GFED4-BB4CMIP"]

# file name for output
# TODO: add versioning / ID to this file
combined_processed_output_file = DATA_ROOT / Path("combined-processed-output", "cmip7_history.csv")

# %%
if CEDS_OPTION == "CEDS-Zenodo-national":
    ceds_processed_national = DATA_ROOT / Path(
        "national", "ceds", "processed", f"ceds_cmip7_national_{CEDS_PROCESSING_ID}.csv"
    )
    ceds_processed_global = DATA_ROOT / Path(
        "national", "ceds", "processed", f"ceds_cmip7_global_{CEDS_PROCESSING_ID}.csv"
    )
    ceds = pd.concat([pd.read_csv(ceds_processed_national), pd.read_csv(ceds_processed_global)])

if BURNING_OPTION == "GFED4_1s":
    gfed_processed_national = DATA_ROOT / Path("national", "gfed", "processed", "gfed_cmip7_national_alpha.csv")
    gfed_processed_global = DATA_ROOT / Path("national", "gfed", "processed", "gfed_cmip7_global_alpha.csv")
    gfed = pd.concat([pd.read_csv(gfed_processed_national), pd.read_csv(gfed_processed_global)])

    ## TODO: all code in this if-statement after this comment is some ad-hoc
    ## ... temporary formatting fixes (should be fixed in the 0102 script, and then deleted here)
    # add /yr
    gfed["unit"] = gfed["unit"].apply(lambda x: x if x.endswith("/yr") else f"{x}/yr")

    # change Mt N2O to kt N2O
    rowidx_mask = gfed["unit"] == "Mt N2O/yr"
    numeric_datacols = gfed.select_dtypes(include="number").columns
    gfed.loc[rowidx_mask, numeric_datacols] *= 1000  # Multiply all numeric columns by 1000
    gfed.loc[rowidx_mask, "unit"] = "kt N2O/yr"  # Change 'unit' to "kt N2O/yr"

    # C to BC
    gfed["unit"] = gfed["unit"].apply(lambda x: "Mt BC/yr" if x == "Mt C/yr" else x)

    # change names to align with 0101_CEDS
    gfed.rename(columns={"country": "region"}, inplace=True)


cmip7combined_data = pd.concat([ceds, gfed])

# %%
cmip7combined_data

# %%
combined_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
cmip7combined_data.to_csv(combined_processed_output_file, index=False)
combined_processed_output_file
