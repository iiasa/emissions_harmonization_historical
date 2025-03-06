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
# # Process GCB
#
# Process data from the global carbon budget (GCB).
# We use the version from [10.5281/zenodo.14106218](https://zenodo.org/records/14106218),
# since the Excel sheet of fossil fuel production by country may have
# errors (in any case the sum of country emissions and bunkers does not equal the global total in the Excel sheets).
# For details on this, see https://bsky.app/profile/cjsmith.be/post/3lbhxt4chqc2x.

# %%
from pathlib import Path

import openscm_units
import pandas as pd
import scmdata

from emissions_harmonization_historical.constants import DATA_ROOT, GCB_PROCESSING_ID

# %%
raw_data_path = DATA_ROOT / "global/gcb/data_raw/"
raw_data_path

# %%
gcb_processed_output_file_fossil = DATA_ROOT / Path(
    "global", "gcb", "processed", f"gcb-fossil_cmip7_global_{GCB_PROCESSING_ID}.csv"
)
gcb_processed_output_file_fossil

# %%
gcb_processed_output_file_afolu = DATA_ROOT / Path(
    "global", "gcb", "processed", f"gcb-afolu_cmip7_global_{GCB_PROCESSING_ID}.csv"
)
gcb_processed_output_file_afolu


# %%
# Fossil stuff commented out as not used right now

# %%
# df = pd.read_csv(
#     raw_data_path / "GCB2024v18_MtCO2_flat.csv"
# )  # , sheet_name='Historical Budget', skiprows=15, index_col="Year")

# %%
# # rename Global to World
# df.loc[df["Country"] == "Global", "Country"] = "World"
# df

# %%
# for country in df.Country.unique():
#     # find NaNs in ISO3 and set these empty values equal to the country name
#     if type(df[(df['Country']==country) & (df.Year==1750)]['ISO 3166-1 alpha-3'].values[0]) is float:
#         #print(country)
#         df.loc[(df['Country']==country), 'ISO 3166-1 alpha-3'] = country

# %%
# times = df.Year.unique()
# nt = len(times)
# nt

# %%
# # iso_list = df['ISO 3166-1 alpha-3'].unique()
# # niso = len(iso_list)
# # niso
# countries = df.Country.unique()
# nc = len(countries)

# %%
# data = np.zeros((nt, nc))
# for i, country in enumerate(countries):
#     data[:, i] = df.loc[(df["Country"] == country)].Total

# totals = pd.DataFrame(
#     data.T,
#     columns=times,
#     index=countries,
# )
# totals

# %%
# # convert nan to zero
# totals[np.isnan(totals)] = 0
# totals

# %%
# df_out = (
#     scmdata.ScmRun(
#         data,
#         index=times,
#         columns={
#             "variable": [
#                 "CMIP7 History|Emissions|CO2|Fossil Fuel and Industrial",
#             ],
#             "unit": ["Mt CO2 / yr"],
#             "region": countries,
#             "model": "History",
#             "scenario": "Global Carbon Budget",
#         },
#     )
#     .interpolate(target_times=np.arange(1750, 2024, dtype=int))
#     .timeseries(time_axis="year")
# )

# %%
# df_out.to_csv(gcb_processed_output_file)


# %%
def read_lu_sheet(sheet_name: str, raw_units: str) -> pd.DataFrame:
    """Read a land-use sheet from GCB"""
    res = pd.read_excel(
        raw_data_path / "National_LandUseChange_Carbon_Emissions_2024v1.0.xlsx",
        sheet_name=sheet_name,
        skiprows=7,
        index_col=0,
    )
    if res.index.name != f"unit: {raw_units}":
        msg = f"Check units. Got units={res.index.name!r}"
        raise AssertionError(msg)

    res.index.name = "year"

    return res


# %%
raw_units = "Tg C/year"
df_lu = (
    pd.concat(
        [
            read_lu_sheet("BLUE", raw_units=raw_units),
            read_lu_sheet("H&C2023", raw_units=raw_units),
            read_lu_sheet("OSCAR", raw_units=raw_units),
            read_lu_sheet("LUCE", raw_units=raw_units),
        ]
    )
    .groupby("year")
    .mean()
)
df_lu

# %%
df_lu_global = df_lu.rename(columns={"Global": "World"})[["World"]]
# # TODO: ask Chris what was going on with this, emissions aren't 600 in 1850...
# df_lu_extended = df_lu_global.copy()
# df_lu_extended.loc[1750:1849, "World"] = np.linspace(3, 597, 100)
df_lu_global

# %%
out_units = "Mt CO2/yr"
conversion_factor = openscm_units.unit_registry.Quantity(1, raw_units).to(out_units).m
conversion_factor

# %%
df_lu_global_converted = df_lu_global * conversion_factor
df_lu_global_converted

# %%
df_lu_out = (
    scmdata.ScmRun(
        df_lu_global_converted,
        index=df_lu_global_converted.index.values,
        columns={
            "variable": [
                "Emissions|CO2|AFOLU",
            ],
            "unit": [out_units],
            "region": df_lu_global_converted.columns,
            "model": "Global Carbon Budget",
            "scenario": "historical",
        },
    )
    # Only need for country data I think.
    # .interpolate(target_times=np.arange(1750, 2024, dtype=int))
    .timeseries(time_axis="year")
)
df_lu_out

# %%
gcb_processed_output_file_afolu.parent.mkdir(exist_ok=True, parents=True)
df_lu_out.to_csv(gcb_processed_output_file_afolu)
gcb_processed_output_file_afolu
