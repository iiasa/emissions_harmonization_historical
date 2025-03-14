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
ch4 = primap.loc[
    (primap.entity == "CH4")
    & (primap.unit == "CH4 * gigagram / a")  # converted with 1000 below
    & (primap["scenario (PRIMAP-hist)"] == "HISTTP")
    & (primap["area (ISO3)"] == "EARTH")
    & (primap["category (IPCC2006_PRIMAP)"] == "M.0.EL"),
]
ch4

# %%
n2o = primap.loc[
    (primap.entity == "N2O")
    & (primap.unit == "N2O * gigagram / a")
    & (primap["scenario (PRIMAP-hist)"] == "HISTTP")
    & (primap["area (ISO3)"] == "EARTH")
    & (primap["category (IPCC2006_PRIMAP)"] == "M.0.EL"),
]
n2o

# %%
primap_out = pd.concat([ch4, n2o])
primap_out["model"] = "PRIMAP-HistTP"
primap_out["scenario"] = "historical"
primap_out["region"] = "World"
primap_out["variable"] = "Emissions|" + primap_out["entity"] + "|Fossil, industrial and agriculture"
primap_out["unit"] = primap_out["unit"].map(
    {
        "CH4 * gigagram / a": "kt CH4/yr",
        "N2O * gigagram / a": "kt N2O/yr",
    }
)
primap_out = primap_out.set_index(["model", "scenario", "region", "variable", "unit"])
primap_out = primap_out.drop(
    ["source", "scenario (PRIMAP-hist)", "provenance", "area (ISO3)", "entity", "category (IPCC2006_PRIMAP)"],
    axis="columns",
)

ch4_loc = primap_out.index.get_level_values("variable").str.startswith("Emissions|CH4")
primap_out.loc[ch4_loc, :] /= 1000

primap_out = primap_out.reset_index("unit")
ch4_loc = primap_out.index.get_level_values("variable").str.startswith("Emissions|CH4")
assert primap_out.loc[ch4_loc, "unit"].values.tolist() == ["kt CH4/yr"]
primap_out.loc[ch4_loc, "unit"] = "Mt CH4/yr"  # divided by 1000 above
primap_out = primap_out.set_index("unit", append=True)

primap_out

# %%
primap_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
primap_out.to_csv(primap_processed_output_file)
primap_processed_output_file
