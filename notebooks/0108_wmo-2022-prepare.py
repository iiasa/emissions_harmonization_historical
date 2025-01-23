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
# # WMO Ozone Assessment 2022 - process
#
# Process data from [the 2022 WMO Ozone Assessment](https://csl.noaa.gov/assessments/ozone/2022/).

# %%
from pathlib import Path

import pandas as pd
import pandas_indexing as pix

from emissions_harmonization_historical.constants import DATA_ROOT, WMO_2022_PROCESSING_ID

# %%
pix.set_openscm_registry_as_default()

# %%
wmo_2022_processed_output_file = DATA_ROOT / Path(
    "global",
    "wmo-2022",
    "processed",
    f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv",
)
wmo_2022_processed_output_file

# %%
raw_data_path = DATA_ROOT / "global/wmo-2022/data_raw/"
raw_data_path

# %%
var_map = {
    # WMO name: (gas name, variable name)
    "CF4": ("CF4", "Emissions|CF4"),
    "C2F6": ("C2F6", "Emissions|C2F6"),
    "SF6": ("SF6", "Emissions|SF6"),
}

# %%
res_l = []
for gdir in raw_data_path.glob("*"):
    outputs_dir = gdir / "outputs"
    emissions_file = list(outputs_dir.glob("*Global_annual_emissions.csv"))
    if not emissions_file:
        print(f"No data in {outputs_dir=}")
        continue

    if len(emissions_file) != 1:
        raise NotImplementedError(list(outputs_dir.glob("*")))

    emissions_file = emissions_file[0]

    with open(emissions_file) as fh:
        raw = tuple(fh.readlines())

    for line in raw:
        if "Units:" in line:
            units = line.split("Units: ")[-1].strip()
            break

    else:
        msg = "units not found"
        raise AssertionError(msg)

    raw_df = pd.read_csv(emissions_file, comment="#")

    gas = var_map[gdir.name][0]
    variable = var_map[gdir.name][1]
    out = raw_df[["Year", "Global_annual_emissions"]].rename(
        {"Year": "year", "Global_annual_emissions": "value"}, axis="columns"
    )
    out["variable"] = variable
    out["unit"] = units.replace("/", f" {gas}/")
    out["model"] = "WMO 2022"
    out["scenario"] = "historical"
    out["region"] = "World"
    out = out.set_index(list(set(out.columns) - {"value"}))["value"].unstack("year")
    out = out.pix.convert_unit(f"kt {gas}/yr")

    res_l.append(out)

res = pd.concat(res_l)
res

# %%
wmo_2022_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
res.to_csv(wmo_2022_processed_output_file)
wmo_2022_processed_output_file
