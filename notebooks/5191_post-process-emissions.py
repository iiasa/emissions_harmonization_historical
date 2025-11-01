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
# # Post process emissions
#
# Here we post process emissions arfter harmonization and infilling.

# %% [markdown]
# ## Imports

# %%
from functools import partial

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_SCENARIO_DB,
    INFILLED_SCENARIOS_DB,
    POST_PROCESSED_TIMESERIES_DB,
    PRE_PROCESSED_SCENARIO_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pix.set_openscm_registry_as_default()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "MESSAGE"
output_to_pdf: bool = False

# %% [markdown]
# ## Load data

# %%
pre_processed_emms_scms_gcages = PRE_PROCESSED_SCENARIO_DB.load(
    pix.ismatch(variable="Emissions**") & pix.isin(stage="global_workflow_emissions") & pix.ismatch(model=f"*{model}*")
)
# pre_processed_emms_scms_gcages

# %%
harmonised_emms = HARMONISED_SCENARIO_DB.load(
    pix.ismatch(variable="Emissions**") & pix.isin(workflow=["for_scms", "gridding"]) & pix.ismatch(model=f"*{model}*"),
    progress=True,
)
# harmonised_emms

# %%
infilled_emms = INFILLED_SCENARIOS_DB.load(
    pix.ismatch(variable="Emissions**") & pix.ismatch(model=f"*{model}*"), progress=True
)
# infilled_emms

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
# ### Pre-processed emissions

# %%
pre_processed_emms_scms_gcages_annual = interpolate_to_annual(pre_processed_emms_scms_gcages)
pre_processed_emms_scms_gcages_annual_incl_co2_total = pix.concat(
    [
        pre_processed_emms_scms_gcages_annual,
        calculate_co2_total(pre_processed_emms_scms_gcages_annual),
    ]
)

pre_processed_emms_scms_annual_incl_co2_total = update_index_levels_func(
    pre_processed_emms_scms_gcages_annual_incl_co2_total, {"variable": from_gcages}
)

pre_processed_emms_scms_out = pix.concat(
    [
        pre_processed_emms_scms_annual_incl_co2_total,
        calculate_cumulative_co2s(pre_processed_emms_scms_annual_incl_co2_total),
        calculate_kyoto_ghgs(pre_processed_emms_scms_gcages_annual_incl_co2_total),
        calculate_ghgs(pre_processed_emms_scms_gcages_annual_incl_co2_total),
    ]
)
# pre_processed_emms_scms_out

# %%
ax = sns.lineplot(
    data=pre_processed_emms_scms_out.loc[
        pix.ismatch(variable="Emissions|CO2**", scenario="SSP2*")
    ].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="tab:gray")

# %%
ax = sns.lineplot(
    data=pre_processed_emms_scms_out.loc[pix.ismatch(variable="Cumulative**", scenario="SSP2*")].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="tab:gray")

# %%
ax = sns.lineplot(
    data=pre_processed_emms_scms_out.loc[pix.ismatch(variable="**GHG**", scenario="SSP2*")].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="tab:gray")

# %% [markdown]
# ### Harmonised emissions

# %%
harmonised_emms_gridding = harmonised_emms.loc[pix.ismatch(workflow="gridding")].reset_index("workflow", drop=True)
# harmonised_emms_gridding

# %%
harmonised_emms_scms = harmonised_emms.loc[pix.ismatch(workflow="for_scms")].reset_index("workflow", drop=True)

harmonised_emms_scms_annual = interpolate_to_annual(harmonised_emms_scms)
harmonised_emms_scms_annual_gcages = update_index_levels_func(harmonised_emms_scms_annual, {"variable": to_gcages})
harmonised_emms_scms_annual_gcages_incl_co2_total = pix.concat(
    [
        harmonised_emms_scms_annual_gcages,
        calculate_co2_total(harmonised_emms_scms_annual_gcages),
    ]
)

harmonised_emms_scms_annual_incl_co2_total = update_index_levels_func(
    harmonised_emms_scms_annual_gcages_incl_co2_total, {"variable": from_gcages}
)

harmonised_emms_scms_out = pix.concat(
    [
        harmonised_emms_scms_annual_incl_co2_total,
        calculate_cumulative_co2s(harmonised_emms_scms_annual_incl_co2_total),
        calculate_kyoto_ghgs(harmonised_emms_scms_annual_gcages_incl_co2_total),
        calculate_ghgs(harmonised_emms_scms_annual_gcages_incl_co2_total),
    ]
)
# harmonised_emms_scms_gcages_out

# %%
ax = sns.lineplot(
    data=harmonised_emms_scms_out.loc[pix.ismatch(variable="Cumulative**", scenario="SSP2*")].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="tab:gray")

# %%
ax = sns.lineplot(
    data=harmonised_emms_scms_out.loc[pix.ismatch(variable="**GHG**", scenario="SSP2*")].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="tab:gray")

# %% [markdown]
# ### Complete

# %%
complete_emissions = infilled_emms.loc[pix.isin(stage="complete")].reset_index("stage", drop=True)

complete_emissions_annual = interpolate_to_annual(complete_emissions)
complete_emissions_annual_gcages = update_index_levels_func(complete_emissions_annual, {"variable": to_gcages})
complete_emissions_annual_gcages_incl_co2_total = pix.concat(
    [
        complete_emissions_annual_gcages,
        calculate_co2_total(complete_emissions_annual_gcages),
    ]
)

complete_emissions_annual_incl_co2_total = update_index_levels_func(
    complete_emissions_annual_gcages_incl_co2_total, {"variable": from_gcages}
)

complete_emissions_out = pix.concat(
    [
        complete_emissions_annual_incl_co2_total,
        calculate_cumulative_co2s(complete_emissions_annual_incl_co2_total),
        calculate_kyoto_ghgs(complete_emissions_annual_gcages_incl_co2_total),
        calculate_ghgs(complete_emissions_annual_gcages_incl_co2_total),
    ]
)
# complete_emissions_out

# %%
ax = sns.lineplot(
    data=complete_emissions_out.loc[pix.ismatch(variable="**GHG**", scenario="SSP2*")].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="gray")

# %%
pdf = (
    pix.concat(
        [
            pre_processed_emms_scms_out.pix.assign(stage="pre-processed"),
            harmonised_emms_scms_out.pix.assign(stage="harmonised"),
            complete_emissions_out.pix.assign(stage="complete"),
        ]
    )
    .loc[
        pix.ismatch(
            variable=[
                "Emissions|GHG AR6GWP100",
                "Emissions|Kyoto GHG AR6GWP100",
                "Emissions|CO2|Energy and Industrial Processes",
                "Emissions|CO2|AFOLU",
            ],
            # scenario="SSP2*",
        )
    ]
    .openscm.to_long_data()
)

fg = sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="scenario",
    style="stage",
    col="variable",
    col_wrap=min(2, len(pdf["variable"].unique())),
    kind="line",
    # facet_kws=dict(sharey=False),
)
for ax in fg.axes.flatten():
    # sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.axhline(0.0, linestyle="--", color="gray")

# %% [markdown]
# ## Save

# %%
for df, db in (
    (pre_processed_emms_scms_out.pix.assign(stage="pre-processed-scms"), POST_PROCESSED_TIMESERIES_DB),
    (harmonised_emms_gridding.pix.assign(stage="harmonised-gridding"), POST_PROCESSED_TIMESERIES_DB),
    (harmonised_emms_scms_out.pix.assign(stage="harmonised-scms"), POST_PROCESSED_TIMESERIES_DB),
    (complete_emissions_out.pix.assign(stage="complete"), POST_PROCESSED_TIMESERIES_DB),
):
    db.save(df, allow_overwrite=True)
