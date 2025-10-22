import numpy as np
import pandas as pd
import pandas_indexing as pix
import pint
from pandas_openscm.index_manipulation import update_index_levels_func

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


def interpolate_to_annual(indf: pd.DataFrame, copy: bool = True) -> pd.DataFrame:  # noqa: D103
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
            {"unit": lambda x: x.replace("/yr", ""), "variable": lambda x: f"Cumulative {x}"},
        ).pix.convert_unit("Gt CO2")

        res_l.append(co2_cumulative_df)

    res = pix.concat(res_l)

    return res


def calculate_kyoto_ghgs_gwp(indf: pd.DataFrame, gwp: str = "AR6GWP100"):  # noqa: D103
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


def calculate_ghgs_gwp(indf: pd.DataFrame, gwp: str = "AR6GWP100"):  # noqa: D103
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
