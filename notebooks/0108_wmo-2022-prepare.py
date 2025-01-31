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
import seaborn as sns

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
    if str(gdir).endswith("README.md"):
        continue
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
    out["model"] = "WMO 2022 AGAGE inversions"
    out["scenario"] = "historical"
    out["region"] = "World"
    out = out.set_index(list(set(out.columns) - {"value"}))["value"].unstack("year")
    out = out.pix.convert_unit(f"kt {gas}/yr")

    res_l.append(out)

res = pd.concat(res_l)
res

# %%
wmo_input_file_emissions = DATA_ROOT / "global" / "wmo-2022" / "data_raw" / "Emissions_fromWMO2022.xlsx"
wmo_input_file_emissions

# %%
var_map = {
    # WMO name: (gas name, variable name)
    "CFC-11": ("CFC11", "Emissions|Montreal Gases|CFC|CFC11"),
    "CFC-12": ("CFC12", "Emissions|Montreal Gases|CFC|CFC12"),
    "CFC-113": ("CFC113", "Emissions|Montreal Gases|CFC|CFC113"),
    "CFC-114": ("CFC114", "Emissions|Montreal Gases|CFC|CFC114"),
    "CFC-115": ("CFC115", "Emissions|Montreal Gases|CFC|CFC115"),
    "CCl4": ("CCl4", "Emissions|Montreal Gases|CCl4"),
    "CH3CCl3": ("CH3CCl3", "Emissions|Montreal Gases|CH3CCl3"),
    "HCFC-22": ("HCFC22", "Emissions|Montreal Gases|HCFC22"),
    "HCFC-141b": ("HCFC141b", "Emissions|Montreal Gases|HCFC141b"),
    "HCFC-142b": ("HCFC142b", "Emissions|Montreal Gases|HCFC142b"),
    "halon-1211": ("Halon1211", "Emissions|Montreal Gases|Halon1211"),
    "halon-1202": ("Halon1202", "Emissions|Montreal Gases|Halon1202"),
    "halon-1301": ("Halon1301", "Emissions|Montreal Gases|Halon1301"),
    "halon-2402": ("Halon2402", "Emissions|Montreal Gases|Halon2402"),
}

# %%
wmo_raw = pd.read_excel(wmo_input_file_emissions).rename({"Unnamed: 0": "year"}, axis="columns")
wmo_raw

# %%
# Hand woven unit conversion
wmo_clean = (wmo_raw.set_index("year") / 1000).melt(ignore_index=False)
wmo_clean["unit"] = wmo_clean["variable"].map({k: v[0] for k, v in var_map.items()})
wmo_clean["unit"] = "kt " + wmo_clean["unit"] + "/yr"
wmo_clean["variable"] = wmo_clean["variable"].map({k: v[1] for k, v in var_map.items()})
wmo_clean["model"] = "WMO 2022"
wmo_clean["scenario"] = "WMO 2022 projections v20250129"
wmo_clean["region"] = "World"
wmo_clean = wmo_clean.set_index(["variable", "unit", "model", "scenario", "region"], append=True)["value"].unstack(
    "year"
)
# HACK: remove spurious last year
wmo_clean[2100] = wmo_clean[2099]
# HACK: remove negative values
wmo_clean = wmo_clean.where(wmo_clean >= 0, 0.0)
# TODO: apply some smoother

wmo_clean

# %%
pdf = wmo_clean.melt(ignore_index=False).reset_index()
fg = sns.relplot(
    data=pdf,
    x="year",
    y="value",
    col="variable",
    col_wrap=3,
    hue="model",
    kind="line",
    facet_kws=dict(sharey=False),
)
for ax in fg.figure.axes:
    ax.set_ylim(ymin=0)
    ax.grid()

# %%
out = pix.concat([res, wmo_clean])
out

# %%
wmo_2022_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(wmo_2022_processed_output_file)
wmo_2022_processed_output_file
