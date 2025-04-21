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
from emissions_harmonization_historical.ceds import add_global
from emissions_harmonization_historical.units import assert_units_match_wishes

# %%
pix.set_openscm_registry_as_default()

# %%
raw_data_path = DATA_ROOT / "national/gains/data_raw/"
raw_data_path

# %%
gains_processed_output_file_national = DATA_ROOT / Path(
    "national", "gains", "processed", f"gains_national_{GAINS_PROCESSING_ID}.csv"
)
gains_processed_output_file_global = DATA_ROOT / Path(
    "national", "gains", "processed", f"gains_global_{GAINS_PROCESSING_ID}.csv"
)

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
df["country"] = df["region"].str.lower()
df["model"] = "GAINS"
df["scenario"] = "historical"

# Drop PM2.5 here to avoid later unit registry issues
df = df[df["variable"]!="PM_2_5"]

# df = df.set_index(["model", "scenario", "region", "variable", "unit"])
df = df.pivot_table(index=["model", "scenario", "country", "variable", "unit"], columns="year", values="value", aggfunc="sum") # `aggfunc='sum'` is just a guess how to deal with the duplicates in the data 



df

# %%
# change units to align with IAM scenario data
gains = df

# change name(s) of emissions species
# use 'NOx' instead of 'NOX'
gains.index = pd.MultiIndex.from_tuples(
    [
        (model, scenario, region, "NOx" if variable == "NOX" else variable, unit)
        for model, scenario, region, variable, unit in gains.index
    ],
    names=gains.index.names,
)
# use 'BC' instead of 'PM_BC'
gains.index = pd.MultiIndex.from_tuples(
    [
        (model, scenario, region, "BC" if variable == "PM_BC" else variable, unit)
        for model, scenario, region, variable, unit in gains.index
    ],
    names=gains.index.names,
)
# use 'OC' instead of 'PM_OC'
gains.index = pd.MultiIndex.from_tuples(
    [
        (model, scenario, region, "OC" if variable == "PM_OC" else variable, unit)
        for model, scenario, region, variable, unit in gains.index
    ],
    names=gains.index.names,
)

# change unit(s) of emissions
# adjust units; change column 'units' to 'unit' and add '/yr'
gains = gains.pix.dropna(subset=["unit"]).pix.format(unit="{unit} {variable}/yr", drop=False)
# adjust units; change all to values to Mt instead of kt
gains = pix.units.convert_unit(gains, lambda x: "Mt " + x.removeprefix("kt").strip())
# exception for N2O/yr, which should remain in kt following https://github.com/IAMconsortium/common-definitions/
gains = pix.units.convert_unit(gains, lambda x: "kt " + x.removeprefix("Mt").strip() if x == "Mt N2O/yr" else x)

# change name(s) of emissions species that couldn't be changed before because Pint would have thrown an error in `convert_unit`
# use 'Sulfur' instead of 'SO2'
gains.index = pd.MultiIndex.from_tuples(
    [
        (model, scenario, region, "Sulfur" if variable == "SO2" else variable, unit)
        for model, scenario, region, variable, unit in gains.index
    ],
    names=gains.index.names,
)
# also for units 
# unit of NOx from NOx to NO2
gains.index = pd.MultiIndex.from_tuples(
    [
        (model, scenario, region, variable, "Mt NO2/yr" if unit == "Mt NOx/yr" and variable == "NOx" else unit)
        for model, scenario, region, variable, unit in gains.index
    ],
    names=gains.index.names,
)

# Show the data
gains.index.to_frame(index=False)

# %%
gains = gains.groupby(["model", "scenario", "country", "variable", "unit"]).sum().pix.fixna()  # group and fix NAs

# %%
# aggregate countries where this is necessary, e.g. because of specific other data (like SSP socioeconomic driver data)
# based on the new SSP data, we only need to aggregate Serbia and Kosovo
country_combinations = {
    # "isr_pse": ["isr", "pse"], "sdn_ssd": ["ssd", "sdn"],
    "srb_ksv": ["srb", "srb (kosovo)"]
}
gains = gains.pix.aggregate(country=country_combinations)

# %%
# add global
gains = add_global(gains, groups=["model", "scenario", "variable", "unit"])

# %%
gains_reformatted = gains.rename_axis(index={"country": "region"})
gains_reformatted

# %%
# rename to IAMC-style variable names including standard index order
gains_reformatted_iamc = (
    gains_reformatted.pix.format(variable="Emissions|{variable}", drop=True)
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
).sort_values(by=["region", "variable"])
gains_reformatted_iamc

# %%
gains_reformatted_iamc.pix.unique(['variable'])

# %%
assert_units_match_wishes(gains_reformatted_iamc)

# %% [markdown]
# Save formatted GAINS data

# %%
out_global = gains_reformatted_iamc.loc[pix.isin(region="World")]  # only the added "World" region
out_national = gains_reformatted_iamc.loc[
    ~pix.isin(region="World")
] 

# %% [markdown]
# Check that national sums equal global total.

# %%
# Check that `out_national_with_global` totals (all countries in iso3c + CEDS 'global' region)
# ... are the same as `out_global` totals ("World")
national_sums_checker = (
    pix.assignlevel(out_national.groupby(["model", "scenario", "variable", "unit"]).sum(), region="World")
    .reset_index()
    .set_index(out_global.index.names)
)
national_sums_checker.columns = national_sums_checker.columns.astype(int)
national_sums_checker
pd.testing.assert_frame_equal(out_global, national_sums_checker, check_like=True)

# %%
# national GAINS data
gains_processed_output_file_national.parent.mkdir(exist_ok=True, parents=True)
out_national.to_csv(gains_processed_output_file_national)
gains_processed_output_file_national

# %%
# globally aggregated data (all emissions)
gains_processed_output_file_global.parent.mkdir(exist_ok=True, parents=True)
out_global.to_csv(gains_processed_output_file_global)
gains_processed_output_file_global
