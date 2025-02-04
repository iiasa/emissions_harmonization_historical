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
# # Post-process prepared GFED BB4CMIP data
#
# The 0103 script dumped the files into separate CSVs; here we combine and make consistenct with other emissions files
# and IAMC format

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import pint

from emissions_harmonization_historical.constants import DATA_ROOT, GFED_PROCESSING_ID
from emissions_harmonization_historical.units import assert_units_match_wishes

# %%
# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
data_path = DATA_ROOT / "national/gfed-bb4cmip/processed"

# %%
out_path_global = data_path / f"gfed-bb4cmip_cmip7_global_{GFED_PROCESSING_ID}.csv"
out_path_national = data_path / f"gfed-bb4cmip_cmip7_national_{GFED_PROCESSING_ID}.csv"

# %%
species = [
    "BC",
    "NMVOC",
    "CO",
    "CO2",
    "CH4",
    "N2O",
    "OC",
    "NH3",
    "NOx",
    "SO2",
]

# %%
df_list = []
# Rename variable in place
for s in species:
    for suffix in [f"_world_{GFED_PROCESSING_ID}", f"_national_{GFED_PROCESSING_ID}"]:
        df_in = pd.read_csv(data_path / f"{s}{suffix}.csv")
        df_in.variable = f"Emissions|{s}|Biomass Burning"
        df_list.append(df_in)
        # display(df_in)

# %%
df = pd.concat(df_list)

# %%
df["model"] = "BB4CMIP"
df

# %%
# sort order: region, variable
df_sorted = df.sort_values(["region", "variable"])

# %%
df_sorted

# %%
# fix column order
df_reordered = df_sorted.set_index(["model", "scenario", "region", "variable", "unit"])

# %%
df_renamed = df_reordered.rename(
    index={"Emissions|SO2|Biomass Burning": "Emissions|Sulfur|Biomass Burning"}, level="variable"
)
df_renamed

# %%
with pint.get_application_registry().context("NOx_conversions"):
    df_renamed_desired_units = pix.units.convert_unit(
        df_renamed,
        {"Mt NO / yr": "Mt NO2/yr"},
    )

# %%
df_renamed_desired_units = pix.units.convert_unit(
    df_renamed_desired_units,
    lambda x: x.replace(" / yr", "/yr"),
)
df_renamed_desired_units = pix.units.convert_unit(
    df_renamed_desired_units,
    {"Mt N2O/yr": "kt N2O/yr"},
)

# %%
assert_units_match_wishes(df_renamed_desired_units)
df_renamed_desired_units.columns = df_renamed_desired_units.columns.astype(int)

# %%
out_global = df_renamed_desired_units.loc[pix.ismatch(region="World")]
out_global

# %%
out_national = df_renamed_desired_units.loc[~pix.ismatch(region="World")]
out_national

# %%
assert out_national.shape[0] + out_global.shape[0] == df_renamed_desired_units.shape[0]

# %% [markdown]
# Check that national sums equal global total.

# %%
national_sums_checker = (
    pix.assignlevel(out_national.groupby(["model", "scenario", "variable", "unit"]).sum(), region="World")
    .reset_index()
    .set_index(out_global.index.names)
)
national_sums_checker.columns = national_sums_checker.columns.astype(int)
national_sums_checker

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_PATH


# %%
def transform_rcmip_to_iamc_variable(v):
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (
        ("F-Gases|", ""),
        ("PFC|", ""),
        ("HFC4310mee", "HFC43-10"),
        ("MAGICC AFOLU", "AFOLU"),
        ("MAGICC Fossil and Industrial", "Energy and Industrial Processes"),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
ar6_history = rcmip_clean.loc[pix.isin(mip_era=["CMIP6"], scenario=["ssp245"], region=["World"])]
# ar6_history = (
#     ar6_history.loc[
#         ~pix.ismatch(
#             variable=[
#                 f"Emissions|{stub}|**" for stub in ["BC", "CH4", "CO", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC"]
#             ]
#         )
#         & ~pix.ismatch(variable=["Emissions|CO2|*|**"])
#         & ~pix.isin(variable=["Emissions|CO2"])
#     ]
#     .T.interpolate("index")
#     .T
# )
full_var_set = ar6_history.pix.unique("variable")
n_variables_in_full_scenario = 52
if len(full_var_set) != n_variables_in_full_scenario:
    raise AssertionError

# %%
variable

# %%
ar6_history

# %%

# %%
for variable, vdf in out_global.groupby("variable"):
    tmp = ar6_history.loc[
        pix.ismatch(variable=variable.replace("|Biomass Burning", "|AFOLU|*").replace("NMVOC", "VOC"))
    ]
    tmp = (
        tmp.loc[pix.ismatch(variable="**Burning")]
        .groupby(tmp.index.names.difference(["variable"]))
        .sum()
        .pix.assign(variable="CMIP6 burning sum")
    )

    pix.concat(
        [
            tmp.reset_index(["activity_id", "mip_era"], drop=True),
            vdf,
        ]
    ).loc[:, :2023].T.plot()
    plt.show()
    # break

# %%

vdf

# %%
out_global

# %%
pd.testing.assert_frame_equal(out_global, national_sums_checker, check_like=True)

# %%
out_global.to_csv(out_path_global)
out_path_global

# %%
out_national.to_csv(out_path_national)
out_path_national
