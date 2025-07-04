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
# # Format BB4CMIP7 data
#
# Format the processed biomass burning emissions.

# %% [markdown]
# ## Imports

# %%
import sys

import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import seaborn as sns
import tqdm.auto
import xarray as xr
from gcages.index_manipulation import set_new_single_value_levels
from loguru import logger

from emissions_harmonization_historical.constants_5000 import (
    BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_ID,
    BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR,
    BB4CMIP7_FORMATTING_ID,
    BB4CMIP7_PROCESSED_DB,
    BB4CMIP7_PROCESSED_DB_UNSMOOTHED,
    BB4CMIP7_PROCESSED_DIR,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.units import assert_units_match_wishes
from emissions_harmonization_historical.zenodo import upload_to_zenodo

# %% [markdown]
# ## Setup

# %%
# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
pandas_openscm.register_pandas_accessor()

# %%
# mapping from GFED variables to longer IAMC-style naming
sector_mapping = {
    "AGRI": "Agricultural Waste Burning",
    "BORF": "Forest Burning",
    "DEFO": "Forest Burning",
    "PEAT": "Peat Burning",
    "SAVA": "Grassland Burning",
    "TEMF": "Forest Burning",
}
species_mapping = {
    "BC": "BC",
    "CH4": "CH4",
    "CO": "CO",
    "CO2": "CO2",
    "N2O": "N2O",
    "NH3": "NH3",
    "NMVOC": "VOC",
    "NOx": "NOx",
    "OC": "OC",
    "SO2": "Sulfur",
}

# %% [markdown]
# ## Load data

# %%
species_units = {
    # You can check these using metadata in netCDF files,
    # but we haven't automated that.
    "BC": "BC",
    "CH4": "CH4",
    "CO": "CO",
    "CO2": "CO2",
    "N2O": "N2O",
    "NH3": "NH3",
    "NMVOC": "VOC",
    "NOx": "NO",
    "OC": "OC",
    "SO2": "SO2",
}

# %%
raw_files = tuple(BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR.glob("*.nc"))
res_l = []
for f in tqdm.auto.tqdm(raw_files):
    species, sector = f.stem.split("_")

    sector_iamc = sector_mapping[sector]
    species_iamc = species_mapping[species]
    unit = f"kg {species_units[species]}/yr"

    tmp = xr.open_dataarray(f)
    tmp = tmp.to_dataframe(name="value").set_index("sector", append=True)["value"].unstack("year")
    tmp.index = tmp.index.rename({"iso": "region"})
    #     display(tmp)
    tmp = set_new_single_value_levels(
        tmp, {"scenario": HISTORY_SCENARIO_NAME, "model": "BB4CMIP7", "unit": unit, "species": species}
    )

    res_l.append(tmp)

res = pix.concat(res_l)
# res

# %% [markdown]
# ## Check aggregates

# %%
res_aggregate = res.openscm.groupby_except(["sector", "region"]).sum()
res_aggregate = res_aggregate.pix.convert_unit(lambda u: u.replace("kg", "Tg"))
res_aggregate

# %%
# Created this by looking at the expected outputs
# in netCDF files/the notebooks we've run
exp_sums = pd.Series(
    [
        1.91,
        16.45,
        362.89,
        7059.48,
        1.04,
        4.92,
        52.87,
        13.11,
        20.38,
        2.72,
    ],
    pd.MultiIndex.from_tuples(
        [
            ("historical", "BB4CMIP7", "Tg BC/yr", "BC"),
            ("historical", "BB4CMIP7", "Tg CH4/yr", "CH4"),
            ("historical", "BB4CMIP7", "Tg CO/yr", "CO"),
            ("historical", "BB4CMIP7", "Tg CO2/yr", "CO2"),
            ("historical", "BB4CMIP7", "Tg N2O/yr", "N2O"),
            ("historical", "BB4CMIP7", "Tg NH3/yr", "NH3"),
            ("historical", "BB4CMIP7", "Tg VOC/yr", "NMVOC"),
            ("historical", "BB4CMIP7", "Tg NO/yr", "NOx"),
            ("historical", "BB4CMIP7", "Tg OC/yr", "OC"),
            ("historical", "BB4CMIP7", "Tg SO2/yr", "SO2"),
        ],
        names=["scenario", "model", "unit", "species"],
    ),
    name=2021,
)

pd.testing.assert_series_equal(res_aggregate[2021].round(2), exp_sums, check_like=True)

# %% [markdown]
# ## Create output

# %%
res_formatted = res.openscm.update_index_levels({"sector": sector_mapping, "species": species_mapping}).pix.format(
    variable="Emissions|{species}|{sector}", drop=True
)
res_formatted = res_formatted.groupby(res_formatted.index.names).sum(min_count=1)

out_l = []
for (variable, unit), vdf in res_formatted.groupby(["variable", "unit"]):
    if "NOx" in variable:
        target_unit = "Mt NO2/yr"

    elif "N2O" in variable:
        target_unit = "kt N2O/yr"

    elif "NMVOC" in variable:
        target_unit = "Mt VOC/yr"

    else:
        target_unit = unit.replace("kg", "Mt")

    with pint.get_application_registry().context("NOx_conversions"):
        converted = vdf.pix.convert_unit({unit: target_unit})

    out_l.append(converted)

out = pix.concat(out_l).sort_index(axis="columns")
out

# %% [markdown]
# ## 5 year smoothing
#
# as recommended

# %%
smoothed = out.T.rolling(window=5, min_periods=1, center=True).mean().T
res_smoothed = res.T.rolling(window=5, min_periods=1, center=True).mean().T

# %% [markdown]
# ### plot to visually assess smoothing

# %%
check = pd.concat(
    [
        out.loc[pix.ismatch(region="chn", unit="Mt CO2/yr")],
        smoothed.loc[pix.ismatch(region="chn", unit="Mt CO2/yr")].pix.assign(scenario="historical-smoothed"),
    ]
)

# %%
data = check
df_reset = data.reset_index()

df_long = df_reset.melt(id_vars=data.index.names, var_name="time", value_name="values")

g = sns.relplot(
    data=df_long, x="time", y="values", col="variable", hue="scenario", col_wrap=2, kind="line", height=2.5, aspect=2
)

g._legend.remove()

ax = g.axes[0]

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower left", fontsize="small", frameon=True)

# %% [markdown]
# ## Process

# %%
assert_units_match_wishes(smoothed)

# %% [markdown]
# ## Save formatted BB4CMIP7 data

# %%
BB4CMIP7_PROCESSED_DB_UNSMOOTHED.save(out.pix.assign(stage="iso3c"), allow_overwrite=True)
BB4CMIP7_PROCESSED_DB.save(smoothed.pix.assign(stage="iso3c"), allow_overwrite=True)

# %%
BB4CMIP7_PROCESSED_DB

# %% [markdown]
# ## Upload to zenodo

# %%
# Rewrite as single file
out_file_res = (
    BB4CMIP7_PROCESSED_DIR
    / f"bb4cmip7-country-sector_{BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_ID}-{BB4CMIP7_FORMATTING_ID}.csv"
)
res_smoothed.to_csv(out_file_res)
out_file_res

out_file_res_smoothed = (
    BB4CMIP7_PROCESSED_DIR
    / f"bb4cmip7-country-sector_{BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_ID}-{BB4CMIP7_FORMATTING_ID}_unsmoothed.csv"
)
res.to_csv(out_file_res_smoothed)

# %%
logger.configure(handlers=[dict(sink=sys.stderr, level="INFO")])
logger.enable("openscm_zenodo")

# %%
upload_to_zenodo([out_file_res], remove_existing=False, update_metadata=True)
