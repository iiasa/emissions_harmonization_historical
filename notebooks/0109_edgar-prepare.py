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
# # EDGAR - process
#
# Process data from [EDGAR](https://edgar.jrc.ec.europa.eu/dataset_ghg2024).

# %%
from pathlib import Path

import pandas as pd
import pandas_indexing as pix

from emissions_harmonization_historical.constants import DATA_ROOT, EDGAR_PROCESSING_ID

# %%
pix.set_openscm_registry_as_default()

# %%
edgar_processed_output_file = DATA_ROOT / Path(
    "global",
    "edgar",
    "processed",
    f"edgar_cmip7_global_{EDGAR_PROCESSING_ID}.csv",
)
edgar_processed_output_file

# %%
raw_data_path = DATA_ROOT / "global/edgar/data_raw/"
raw_data_path

# %%
raw_data_file = raw_data_path / "EDGAR_F-gases_1990_2023.zip.unzip/EDGAR_F-gases_1990_2023.xlsx"
raw_data_file

# %%
raw = pd.read_excel(raw_data_file, sheet_name="TOTALS BY COUNTRY")
units = raw.loc[raw["Content:"] == "Unit:", "Emissions by country and main source category"].values.tolist()
if len(units) != 1:
    raise AssertionError(units)

units = units[0]
units

# %%
cleaner = raw.iloc[8:, :]
cleaner.columns = cleaner.iloc[0, :]
cleaner.columns.name = None
cleaner = cleaner.iloc[1:, :]
cleaner.columns = cleaner.columns.str.replace("Y_", "")
cleaner = cleaner.drop(["IPCC_annex", "C_group_IM24_sh", "Name"], axis="columns")


hfc_map = {"HFC4310mee": "HFC43-10"}


def format_substance(s):
    """
    Format a substance into an IAMC-style variable
    """
    s = s.replace("-", "")
    if s.startswith("HFC"):
        try:
            return f"Emissions|HFC|{hfc_map[s]}"
        except KeyError:
            return f"Emissions|HFC|{s}"

    return f"Emissions|{s}"


cleaner["variable"] = cleaner["Substance"].map(format_substance)
cleaner["unit"] = f"{units} " + cleaner["variable"].apply(lambda x: x.split("|")[-1].replace("-", "")) + "/yr"
cleaner = cleaner.drop("Substance", axis="columns")
cleaner = cleaner.set_index(["Country_code_A3", "variable", "unit"])

cleaner

# %%
global_totals = cleaner.groupby(cleaner.index.names.difference(["Country_code_A3"])).sum()
global_totals = global_totals.pix.convert_unit({u: u.replace("Gg", "kt") for u in global_totals.pix.unique("unit")})
global_totals

# %%
out = global_totals.pix.assign(scenario="history", model="EDGAR", region="World").reorder_levels(
    ["model", "scenario", "variable", "region", "unit"]
)
out

# %%
edgar_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(edgar_processed_output_file)
edgar_processed_output_file
