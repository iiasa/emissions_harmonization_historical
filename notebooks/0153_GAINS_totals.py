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
# # GAINS processing
#
# Process data supplied by Shaohui Zhao.
#
# Now only global totals, in the future sectoral disaggregation can be supplied.

# %%
from pathlib import Path

import pandas as pd
import pandas_indexing as pix

from emissions_harmonization_historical.constants import GAINS_PROCESSING_ID, DATA_ROOT

# %%
pix.set_openscm_registry_as_default()

# %%
raw_data_path = DATA_ROOT / "national/gains/data_raw/"
raw_data_path

# %%
adam_et_al_2022_processed_output_file = DATA_ROOT / Path(
    "national",
    "gains",
    "processed",
    f"gains-total_{GAINS_PROCESSING_ID}.csv",
)
adam_et_al_2022_processed_output_file

# %%
in_file = raw_data_path / "country_hist_emissions.xlsx"

# %%
raw_df = pd.read_excel(in_file, sheet_name="Export Worksheet")
raw_df

# %%
# find potential duplicates
dupls = raw_df.reset_index().duplicated(subset=["COUNTRY", "IDPOLLUTANT_FRACTIONS", "UNIT", "IDYEARS"], keep=False)
raw_df[dupls]

# %%
df = raw_df.rename(
    {
        "COUNTRY": "region",
        "IDPOLLUTANT_FRACTIONS": "variable",
        "UNIT": "unit",
        "IDYEARS": "year",
        "EMISSION": "value",
    }, axis="columns"
)
df["region"] = df["region"].str.lower()
df["model"] = "GAINS"
df["scenario"] = "historical"
# df = df.set_index(["model", "scenario", "region", "variable", "unit"])
df = df.pivot(index=["model", "scenario", "region", "variable", "unit"], columns="year", values="value")
df

# %%
# change units to align with IAM scenario data
# adjust units; change column 'units' to 'unit' and add '/yr'
df = df.pix.dropna(subset=["unit"]).pix.format(unit="{unit}/yr", drop=True)
# # adjust units; change all to values to Mt instead of kt
# fao = pix.units.convert_unit(fao, lambda x: "Mt " + x.removeprefix("kt").strip())
# # exception for N2O/yr, which should remain in kt following https://github.com/IAMconsortium/common-definitions/
# fao = pix.units.convert_unit(fao, lambda x: "kt " + x.removeprefix("Mt").strip() if x == "Mt N2O/yr" else x)
# # unit of NOx from NOx to NO2
# fao.index = pd.MultiIndex.from_tuples(
#     [
#         (country, fao_sectors, sector_description, em, sector, "Mt NO2/yr" if unit == "Mt NOx/yr" and em == "NOx" else unit)
#         for country, fao_sectors, sector_description, em, sector, unit in fao.index
#     ],
#     names=fao.index.names,
# )
# # change name(s) of emissions species
# # use 'Sulfur' instead of 'SO2'
# fao.index = pd.MultiIndex.from_tuples(
#     [
#         (country, fao_sectors, sector_description, "Sulfur" if em == "SO2" else em, sector, unit)
#         for country, fao_sectors, sector_description, em, sector, unit in fao.index
#     ],
#     names=fao.index.names,
# )
# fao.index.to_frame(index=False)
df

# %%
adam_et_al_2022_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(adam_et_al_2022_processed_output_file)
adam_et_al_2022_processed_output_file
