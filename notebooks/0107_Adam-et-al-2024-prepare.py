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
# # Adam et al., 2024 - process
#
# Process data from [Adam et al., 2024](https://doi.org/10.1038/s43247-024-01946-y).

# %%
from pathlib import Path

import pandas as pd
import pandas_indexing as pix

from emissions_harmonization_historical.constants import ADAM_ET_AL_2024_PROCESSING_ID, DATA_ROOT

# %%
pix.set_openscm_registry_as_default()

# %%
raw_data_path = DATA_ROOT / "global/adam-et-al-2024/data_raw/"
raw_data_path

# %%
adam_et_al_2022_processed_output_file = DATA_ROOT / Path(
    "global",
    "adam-et-al-2024",
    "processed",
    f"adam-et-al-2024_cmip7_global_{ADAM_ET_AL_2024_PROCESSING_ID}.csv",
)
adam_et_al_2022_processed_output_file

# %%
in_file = raw_data_path / "HFC-23_Global_annual_emissions.csv"

# %%
with open(in_file) as fh:
    raw = tuple(fh.readlines())

# %%
for line in raw:
    if "Units:" in line:
        units = line.split("Units: ")[-1].strip()
        break

else:
    msg = "units not found"
    raise AssertionError(msg)

units

# %%
raw_df = pd.read_csv(in_file, comment="#")
raw_df

# %%
raw_df.head()

# %%
out = raw_df[["Year", "Global_annual_emissions"]].rename(
    {"Year": "year", "Global_annual_emissions": "value"}, axis="columns"
)
out["variable"] = "Emissions|HFC|HFC23"
out["unit"] = units.replace("/", " HFC23/")
out["model"] = "Adam et al., 2024"
out["scenario"] = "historical"
out["region"] = "World"
out = out.set_index(list(set(out.columns) - {"value"}))["value"].unstack("year")
out = out.pix.convert_unit("kt HFC23/yr")
out

# %%
adam_et_al_2022_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(adam_et_al_2022_processed_output_file)
adam_et_al_2022_processed_output_file
