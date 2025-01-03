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

# %% [markdown]
# The 0103 script dumped the files into separate CSVs; here we combine and make consistenct with other emissions files
# and IAMC format

# %%
import pandas as pd

from emissions_harmonization_historical.constants import DATA_ROOT

# %%
data_path = DATA_ROOT / "national/gfed-bb4cmip/processed"

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
for specie in species:
    df_in = pd.read_csv(data_path / f"{specie}.csv")
    df_in.variable = f"CMIP7 History|Emissions|{specie}|Biomass Burning"
    df_list.append(df_in)

# %%
df = pd.concat(df_list)

# %%
df["model"] = "History"
df

# %%
# sort order: region, variable
df_sorted = df.sort_values(["region", "variable"])

# %%
df_sorted

# %%
# fix column order
df_reordered = df_sorted.reindex(
    columns=["model", "scenario", "region", "variable", "unit"] + [str(i) for i in range(1750, 2101)]
)

# %%
df_reordered

# %%
df_reordered.to_csv(data_path / "gfed-bb4cmip_cmip7_national_alpha.csv", index=False)

# %%
