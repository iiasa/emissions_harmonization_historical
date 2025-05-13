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
# # Process WMO 2022 data
#
# Process data from [the 2022 WMO Ozone Assessment](https://csl.noaa.gov/assessments/ozone/2022/).

# %% [markdown]
# ## Imports

# %%
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_indexing as pix
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func
from scipy.interpolate import make_smoothing_spline

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_SCENARIO_NAME,
    WMO_2022_PROCESSED_DB,
    WMO_2022_RAW_PATH,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Setup

# %%
pix.set_openscm_registry_as_default()


# %% [markdown]
# ## Helper functions


# %%
def load_inversion_file(fp: Path) -> pd.DataFrame:
    """
    Load one of the inversions file
    """
    var_map = {
        # WMO name: (gas name, variable name)
        "CF4": ("CF4", "Emissions|CF4"),
        "C2F6": ("C2F6", "Emissions|C2F6"),
        "SF6": ("SF6", "Emissions|SF6"),
    }

    with open(fp) as fh:
        raw = tuple(fh.readlines())

    for line in raw:
        if "Units:" in line:
            units = line.split("Units: ")[-1].strip()
            break

    else:
        msg = "units not found"
        raise AssertionError(msg)

    raw_df = pd.read_csv(fp, comment="#")

    gas = emissions_file.name.split("_")[0]
    variable = var_map[gas][1]
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

    return out


# %% [markdown]
# ## Load data

# %% [markdown]
# ### Inversions

# %%
inversions_l = []
for output_dir in WMO_2022_RAW_PATH.rglob("outputs"):
    emissions_file_l = list(output_dir.glob("*Global_annual_emissions.csv"))
    if not emissions_file_l:
        msg = f"No data in {output_dir=}"
        raise FileNotFoundError(msg)

    if len(emissions_file_l) != 1:
        raise NotImplementedError(list(output_dir.glob("*")))

    emissions_file = emissions_file_l[0]
    ts_gas = load_inversion_file(emissions_file)
    inversions_l.append(ts_gas)
    inversions = pix.concat(inversions_l)

inversions

# %% [markdown]
# ### Raw emissions

# %%
wmo_raw = pd.read_excel(WMO_2022_RAW_PATH / "Emissions_fromWMO2022.xlsx").rename({"Unnamed: 0": "year"}, axis="columns")
wmo_raw

# %% [markdown]
# ## Extrapolate the inversions
#
# Very basic, but looks fine enough.

# %%
inversions_extrapolated = inversions.copy()
for y in range(inversions_extrapolated.columns.max() + 1, HARMONISATION_YEAR + 1):
    # Very stupid extrapolation
    inversions_extrapolated[y] = 2 * inversions_extrapolated[y - 1] - inversions_extrapolated[y - 2]

inversions_extrapolated.pix.project("variable").T.plot()
inversions_extrapolated

# %% [markdown]
# ## Process the raw emissions

# %%
var_map = {
    # WMO name: (gas name, variable name)
    "CFC-11": ("CFC11", "Emissions|CFC11"),
    "CFC-12": ("CFC12", "Emissions|CFC12"),
    "CFC-113": ("CFC113", "Emissions|CFC113"),
    "CFC-114": ("CFC114", "Emissions|CFC114"),
    "CFC-115": ("CFC115", "Emissions|CFC115"),
    "CCl4": ("CCl4", "Emissions|CCl4"),
    "CH3CCl3": ("CH3CCl3", "Emissions|CH3CCl3"),
    "HCFC-22": ("HCFC22", "Emissions|HCFC22"),
    "HCFC-141b": ("HCFC141b", "Emissions|HCFC141b"),
    "HCFC-142b": ("HCFC142b", "Emissions|HCFC142b"),
    "halon-1211": ("Halon1211", "Emissions|Halon1211"),
    "halon-1202": ("Halon1202", "Emissions|Halon1202"),
    "halon-1301": ("Halon1301", "Emissions|Halon1301"),
    "halon-2402": ("Halon2402", "Emissions|Halon2402"),
}

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

# wmo_clean

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

# %% [markdown]
# ### Smooth

# %%
vars_to_smooth = [
    "Emissions|Halon1202",
    "Emissions|Halon2402",
]
wmo_emissions_smooth_l = []
for variable, vdf in wmo_clean.groupby("variable"):
    if variable in vars_to_smooth:
        vdf = vdf.copy()  # noqa: PLW2901
        if variable == "Emissions|Halon2402":
            vdf = vdf.drop(range(2019, 2035 + 1), axis="columns")  # noqa: PLW2901

        vdf.loc[:, :] = make_smoothing_spline(vdf.columns, vdf.values.squeeze(), lam=None)(vdf.columns)
        for y in range(wmo_clean.columns.min(), wmo_clean.columns.max() + 1):
            if y not in vdf:
                vdf[y] = np.nan

        vdf = vdf.T.interpolate(method="index").T  # noqa: PLW2901
        vdf[vdf < 0.0] = 0.0

    wmo_emissions_smooth_l.append(vdf)

wmo_emissions_smooth = pix.concat(wmo_emissions_smooth_l).pix.assign(model="WMO 2022 projections v20250129 smoothed")
# wmo_emissions_smooth

# %%
pdf = (
    pix.concat(
        [
            wmo_clean,
            wmo_emissions_smooth,
        ]
    )
    .melt(ignore_index=False)
    .reset_index()
)
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

# %% [markdown]
# # Save

# %%
res = pix.concat([inversions_extrapolated, wmo_emissions_smooth]).pix.assign(scenario=HISTORY_SCENARIO_NAME)
# res

# %% [markdown]
# Make sure we can convert the variable names

# %%
update_index_levels_func(
    res,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %% [markdown]
# ## Save

# %%
WMO_2022_PROCESSED_DB.save(res, allow_overwrite=True, groupby=["model"])
