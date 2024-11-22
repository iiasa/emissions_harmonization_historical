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
# Process Global Carbon Budget data
#
# We use the version from 10.5281/zenodo.14106218, since the Excel sheet of fossil fuel production by country may have
# errors (in any case the sum of country emissions and bunkers does not equal the global total in the Excel sheets).
# See https://bsky.app/profile/cjsmith.be/post/3lbhxt4chqc2x.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import scmdata

from emissions_harmonization_historical.constants import DATA_ROOT

# %%
raw_data_path = DATA_ROOT / "national/gcb/data_raw/"
raw_data_path

# %%
gcb_processed_output_file = DATA_ROOT / Path("national", "gcb", "processed", "gcb_cmip7_national_fossil_alpha.csv")

# %%
df = pd.read_csv(
    raw_data_path / "GCB2024v18_MtCO2_flat.csv"
)  # , sheet_name='Historical Budget', skiprows=15, index_col="Year")

# %%
# rename Global to World
df.loc[df["Country"] == "Global", "Country"] = "World"
df

# %%
# for country in df.Country.unique():
#     # find NaNs in ISO3 and set these empty values equal to the country name
#     if type(df[(df['Country']==country) & (df.Year==1750)]['ISO 3166-1 alpha-3'].values[0]) is float:
#         #print(country)
#         df.loc[(df['Country']==country), 'ISO 3166-1 alpha-3'] = country

# %%
times = df.Year.unique()
nt = len(times)
nt

# %%
# iso_list = df['ISO 3166-1 alpha-3'].unique()
# niso = len(iso_list)
# niso
countries = df.Country.unique()
nc = len(countries)

# %%
# iso_list

# %%
data = np.zeros((nt, nc))
for i, country in enumerate(countries):
    data[:, i] = df.loc[(df["Country"] == country)].Total

# %%
# World on top
countries = np.concatenate((countries[-1:], countries[:-1]))
data = np.concatenate((data[:, -1:], data[:, :-1]), axis=1)
data

# %%
# convert nan to zero
data[np.isnan(data)] = 0

# %%
countries

# %%
data

# %%
df_out = (
    scmdata.ScmRun(
        data,
        index=times,
        columns={
            "variable": [
                "CMIP7 History|Emissions|CO2|Fossil Fuel and Industrial",
            ],
            "unit": ["Mt CO2 / yr"],
            "region": countries,
            "model": "History",
            "scenario": "Global Carbon Budget",
        },
    )
    .interpolate(target_times=np.arange(1750, 2023, dtype=int))
    .timeseries(time_axis="year")
)

# %%
df_out.to_csv(gcb_processed_output_file)

# %%
df_lu_blue = pd.read_excel(
    raw_data_path / "National_LandUseChange_Carbon_Emissions_2024v1.0.xlsx", sheet_name="BLUE", skiprows=7, index_col=0
)
df_lu_hc2023 = pd.read_excel(
    raw_data_path / "National_LandUseChange_Carbon_Emissions_2024v1.0.xlsx",
    sheet_name="H&C2023",
    skiprows=7,
    index_col=0,
)
df_lu_oscar = pd.read_excel(
    raw_data_path / "National_LandUseChange_Carbon_Emissions_2024v1.0.xlsx", sheet_name="OSCAR", skiprows=7, index_col=0
)
df_lu_luce = pd.read_excel(
    raw_data_path / "National_LandUseChange_Carbon_Emissions_2024v1.0.xlsx", sheet_name="LUCE", skiprows=7, index_col=0
)

# %%
df_lu4 = pd.concat([df_lu_blue, df_lu_hc2023, df_lu_oscar, df_lu_luce])
df_lu = df_lu4.groupby(level=0).mean()

df_lu.drop(columns={"EU27"}, inplace=True)
df_lu.rename(columns={"Global": "World"}, inplace=True)
# df_lu4 = pd.Panel([df_lu_blue, df_lu_hc2023, df_lu_oscar, df_lu_luce])
# print('Mean of stacked DFs:\n{df}'.format(df=panel.mean(axis=0)))
# # df_lu.head()
df_lu

# %%
# World on top
cols = list(df_lu.columns)
cols = cols[-1:] + cols[:-1]
df_lu_ordered = df_lu[cols]
df_lu_ordered

# %%
# these ones are in fossil data but not in country LUC data. At least it looks like there are no naming conflicts.
print([i for i in df.Country.unique() if i not in df_lu.columns])

# %%
# df_lu_extended.loc[1750:1849] = [np.nan] * 200
# df_lu_extended = pd.concat(pd.DataFrame(np.nan,index=np.arange(1750, 1850, dtype="int"),columns=df.columns))
df_lu_extended = pd.concat([df_lu_ordered, pd.DataFrame(index=np.arange(1750, 1850))])
df_lu_extended.loc[1750:1849, "World"] = np.linspace(3, 597, 100)
df_lu_extended

# %%
df_lu_extended.sort_index(inplace=True)
df_lu_extended

# %%
df_lu_extended

# %%
for i in range(100):
    df_lu_extended.iloc[i, 1:] = df_lu_extended.iloc[100, 1:] * df_lu_extended.iloc[i, 0] / df_lu_extended.iloc[100, 0]

# %%
df_lu_out = (
    scmdata.ScmRun(
        df_lu_extended,
        index=times,
        columns={
            "variable": [
                "CMIP7 History|Emissions|CO2|Land Use Change",
            ],
            "unit": ["Mt CO2 / yr"],
            "region": df_lu_ordered.columns,
            "model": "History",
            "scenario": "Global Carbon Budget",
        },
    )
    .interpolate(target_times=np.arange(1750, 2024, dtype=int))
    .timeseries(time_axis="year")
)

# %%
df_lu_out

# %%
df_lu_out.to_csv(DATA_ROOT / Path("national", "gcb", "processed", "gcb_cmip7_national_luc_alpha.csv"))
