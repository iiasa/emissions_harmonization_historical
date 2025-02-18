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
# # Prepare PRIMAP-histTP
#
# Prepare data from PRIMAP-histTP.

# %%
# import external packages and functions
from pathlib import Path

import numpy as np
import pandas as pd

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    PRIMAP_HIST_PROCESSING_ID,
)

# %%
primap_input_file = DATA_ROOT / Path(
    "national", "primap-hist", "data_raw", "Guetschow_et_al_2024a-PRIMAP-hist_v2.6_final_no_rounding_13-Sep-2024.csv"
)
primap_processed_output_file = DATA_ROOT / Path(
    "national", "primap-hist", "processed", f"primap-hist-tp_cmip7_global_{PRIMAP_HIST_PROCESSING_ID}.csv"
)

# %%
# only care about N2O and CH4
species = [
    "CH4",
    "N2O",
]

# %%
primap = pd.read_csv(primap_input_file)

# %%
primap

# %%
ch4 = (
    primap.loc[
        (primap.entity == "CH4")
        & (primap["scenario (PRIMAP-hist)"] == "HISTTP")
        & (primap["area (ISO3)"] == "EARTH")
        & (primap["category (IPCC2006_PRIMAP)"] == "M.0.EL"),
        "1750":,
    ].values.squeeze()
    / 1000
)

# %%
n2o = (
    primap.loc[
        (primap.entity == "N2O")
        & (primap["scenario (PRIMAP-hist)"] == "HISTTP")
        & (primap["area (ISO3)"] == "EARTH")
        & (primap["category (IPCC2006_PRIMAP)"] == "M.0.EL"),
        "1750":,
    ].values.squeeze()
    / 1000
)

# %%
indices = [
    ("PRIMAP-HistTP", "historical", "World", "Emissions|CH4|Fossil, industrial and agriculture", "Mt CH4/yr"),
    ("PRIMAP-HistTP", "historical", "World", "Emissions|N2O|Fossil, industrial and agriculture", "Mt N2O/yr"),
]

# %%
index = pd.MultiIndex.from_tuples(indices, names=["model", "scenario", "region", "variable", "unit"])

# %%
primap_out = pd.DataFrame([ch4, n2o], index=index, columns=np.arange(1750, 2024))

# %%
primap_out

# %%
primap_out.to_csv(primap_processed_output_file)

# %%
