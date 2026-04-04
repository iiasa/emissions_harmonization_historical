# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Export CMIP7 historical GHG equivalent emissions

# %% [markdown]
# ## Imports

# %%
from functools import partial

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pandas_openscm.indexing
import pint
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    HISTORY_ZENODO_RECORD_ID,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pix.set_openscm_registry_as_default()

# %% editable=true slideshow={"slide_type": ""}
out_path = f"cmip7-historical-ghg-eq_zenodo-{HISTORY_ZENODO_RECORD_ID}.csv"

# %% [markdown]
# ## Load data

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions"))
# history

# %% [markdown]
# ### Helper functions

# %%
KYOTO_GHGS = [
    # 'Emissions|CO2|AFOLU',
    # 'Emissions|CO2|Energy and Industrial Processes',
    "Emissions|CO2",
    "Emissions|CH4",
    "Emissions|N2O",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC152a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC236fa",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC365mfc",
    "Emissions|HFC4310mee",
    "Emissions|NF3",
    "Emissions|SF6",
    "Emissions|C2F6",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C6F14",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|CF4",
    "Emissions|cC4F8",
]

ALL_GHGS = [
    *KYOTO_GHGS,
    "Emissions|CCl4",
    "Emissions|CFC11",
    "Emissions|CFC113",
    "Emissions|CFC114",
    "Emissions|CFC115",
    "Emissions|CFC12",
    "Emissions|CH2Cl2",
    "Emissions|CH3Br",
    "Emissions|CH3CCl3",
    "Emissions|CH3Cl",
    "Emissions|CHCl3",
    "Emissions|HCFC141b",
    "Emissions|HCFC142b",
    "Emissions|HCFC22",
    "Emissions|Halon1202",
    "Emissions|Halon1211",
    "Emissions|Halon1301",
    "Emissions|Halon2402",
    "Emissions|SO2F2",
]


def calculate_co2_total(indf: pd.DataFrame) -> pd.DataFrame:  # noqa: D103
    res = (
        indf.loc[
            pix.isin(
                variable=[
                    "Emissions|CO2|Biosphere",
                    "Emissions|CO2|Fossil",
                ]
            )
        ]
        .openscm.groupby_except("variable")
        .sum(min_count=2)
        .pix.assign(variable="Emissions|CO2")
    )

    return res


def interpolate_to_annual(indf: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Interpolate dataframe to annual values."""
    if copy:
        indf = indf.copy()

    out_years = np.arange(indf.columns.min(), indf.columns.max() + 1)
    for y in out_years:
        if y not in indf:
            indf[y] = np.nan

    indf = indf.sort_index(axis="columns")
    indf = indf.T.interpolate(method="index").T

    return indf


def calculate_cumulative_co2s(indf: pd.DataFrame) -> pd.DataFrame:  # noqa: D103
    exp_cols = np.arange(indf.columns.min(), indf.columns.max() + 1)
    np.testing.assert_equal(indf.columns, exp_cols)

    res_l = []
    for v in [v for v in indf.pix.unique("variable") if v.startswith("Emissions|CO2")]:
        co2_df = indf.loc[pix.isin(variable=v)]

        co2_cumulative_df = update_index_levels_func(
            co2_df.cumsum(axis="columns"),
            {
                "unit": lambda x: x.replace("/yr", ""),
                "variable": lambda x: f"Cumulative {x}",
            },
        ).pix.convert_unit("Gt CO2")

        res_l.append(co2_cumulative_df)

    res = pix.concat(res_l)

    return res


def calculate_kyoto_ghgs(indf: pd.DataFrame, gwp: str = "AR6GWP100"):  # noqa: D103
    if "Emissions|CO2" not in indf.pix.unique("variable"):
        raise AssertionError(indf.pix.unique("variable"))

    not_handled = set(indf.pix.unique("variable")) - set(KYOTO_GHGS)
    not_handled_problematic = (
        not_handled
        - {
            "Emissions|OC",
            "Emissions|SOx",
            "Emissions|CO2|Biosphere",
            "Emissions|CO",
            "Emissions|NMVOC",
            "Emissions|BC",
            "Emissions|CO2|Fossil",
            "Emissions|NOx",
            "Emissions|NH3",
        }
        - set(ALL_GHGS)
    )
    if not_handled_problematic:
        raise AssertionError(not_handled_problematic)

    with pint.get_application_registry().context(gwp):
        res = (
            indf.loc[pix.isin(variable=KYOTO_GHGS)]
            .pix.convert_unit("MtCO2 / yr")
            .openscm.groupby_except("variable")
            .sum(min_count=2)
            .pix.assign(variable=f"Emissions|Kyoto GHG {gwp}")
        )

    return res


def calculate_ghgs(indf: pd.DataFrame, gwp: str = "AR6GWP100"):  # noqa: D103
    if "Emissions|CO2" not in indf.pix.unique("variable"):
        raise AssertionError(indf.pix.unique("variable"))

    not_handled = set(indf.pix.unique("variable")) - set(ALL_GHGS)
    not_handled_problematic = not_handled - {
        "Emissions|OC",
        "Emissions|SOx",
        "Emissions|CO2|Biosphere",
        "Emissions|CO",
        "Emissions|NMVOC",
        "Emissions|BC",
        "Emissions|CO2|Fossil",
        "Emissions|NOx",
        "Emissions|NH3",
    }
    if not_handled_problematic:
        raise AssertionError(not_handled_problematic)

    with pint.get_application_registry().context(gwp):
        res = (
            indf.loc[pix.isin(variable=ALL_GHGS)]
            .pix.convert_unit("MtCO2 / yr")
            .openscm.groupby_except("variable")
            .sum(min_count=2)
            .pix.assign(variable=f"Emissions|GHG {gwp}")
        )

    return res


# %%
to_gcages = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    to_convention=SupportedNamingConventions.GCAGES,
)
from_gcages = partial(
    convert_variable_name,
    to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    from_convention=SupportedNamingConventions.GCAGES,
)

# %% [markdown]
# ## Calculate

# %%
history_gcages = update_index_levels_func(history, {"variable": to_gcages})

history_gcages_annual = interpolate_to_annual(history_gcages)

history_gcages_annual_incl_co2_total = pix.concat(
    [
        history_gcages_annual,
        calculate_co2_total(history_gcages_annual.pix.assign(model="CEDS-and-GCB")),
    ]
)
history_gcages_annual_incl_co2_total

history_annual_incl_co2_total = update_index_levels_func(
    history_gcages_annual_incl_co2_total, {"variable": from_gcages}
)


history_out = pix.concat(
    [
        history_annual_incl_co2_total,
        calculate_cumulative_co2s(history_annual_incl_co2_total),
        calculate_kyoto_ghgs(history_gcages_annual_incl_co2_total.pix.assign(model="multiple")),
        calculate_ghgs(history_gcages_annual_incl_co2_total.pix.assign(model="multiple")),
    ]
)
history_out

# %%
history_out.loc[pix.ismatch(variable="**GHG**")].to_csv(out_path)
out_path
