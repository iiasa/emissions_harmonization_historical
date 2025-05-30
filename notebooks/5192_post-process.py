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
# # Post process
#
# Here we post process the results.

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
from gcages.ar6.post_processing import (
    categorise_scenarios,
    get_exceedance_probabilities,
    get_exceedance_probabilities_over_time,
    get_temperatures_in_line_with_assessment,
)
from gcages.index_manipulation import set_new_single_value_levels
from gcages.post_processing import PostProcessingResult
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_SCENARIO_DB,
    INFILLED_SCENARIOS_DB,
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_EXCEEDANCE_PROBABILITIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSED_METADATA_RUN_ID_DB,
    POST_PROCESSED_TIMESERIES_DB,
    POST_PROCESSED_TIMESERIES_EXCEEDANCE_PROBABILITIES_DB,
    POST_PROCESSED_TIMESERIES_QUANTILE_DB,
    POST_PROCESSED_TIMESERIES_RUN_ID_DB,
    PRE_PROCESSED_SCENARIO_DB,
    SCM_OUTPUT_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pix.set_openscm_registry_as_default()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "COFFEE"
scm: str = "MAGICCv7.6.0a3"
output_to_pdf: bool = False

# %%
assessed_gsat_variable = "Surface Temperature (GSAT)"
gsat_assessment_median = 0.85
gsat_assessment_time_period = range(1995, 2014 + 1)
gsat_assessment_pre_industrial_period = range(1850, 1900 + 1)
quantiles_of_interest = (
    0.05,
    0.10,
    1.0 / 6.0,
    0.33,
    0.50,
    0.67,
    5.0 / 6.0,
    0.90,
    0.95,
)
exceedance_thresholds_of_interest = np.arange(1.0, 4.01, 0.5)

# %% [markdown]
# ## Load data

# %% [markdown]
# ### SCM output

# %%
raw_gsat_variable_in = "Surface Air Temperature Change"

# %%
in_df = SCM_OUTPUT_DB.load(pix.ismatch(variable=raw_gsat_variable_in, model=f"*{model}*", climate_model=f"*{scm}*"))
# in_df

# %% [markdown]
# ## Emissions

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
# ## Post process

# %% [markdown]
# ### Temperatures in line with the assessment

# %%
temperatures_in_line_with_assessment = update_index_levels_func(
    get_temperatures_in_line_with_assessment(
        in_df.loc[in_df.index.get_level_values("variable").isin([raw_gsat_variable_in])],
        assessment_median=gsat_assessment_median,
        assessment_time_period=gsat_assessment_time_period,
        assessment_pre_industrial_period=gsat_assessment_pre_industrial_period,
        group_cols=["climate_model", "model", "scenario"],
    ),
    {"variable": lambda x: assessed_gsat_variable},
)
# temperatures_in_line_with_assessment

# %%
ax = temperatures_in_line_with_assessment.loc[:, 2010:2100].openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id", hue_var="scenario", style_var="climate_model"
)
ax.grid()

# %%
temperatures_in_line_with_assessment.loc[:, 2010:2030].openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id", hue_var="scenario", style_var="climate_model"
)

# %% [markdown]
# #### Quantiles

# %%
temperatures_in_line_with_assessment_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(
        temperatures_in_line_with_assessment,
        "run_id",
    ).quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
    new_name="quantile",
)
# temperatures_in_line_with_assessment_quantiles

# %% [markdown]
# ### Exceedance probabilities, peak warming and categorisation

# %%
exceedance_probabilities_over_time = get_exceedance_probabilities_over_time(
    temperatures_in_line_with_assessment,
    exceedance_thresholds_of_interest=exceedance_thresholds_of_interest,
    group_cols=["model", "scenario", "climate_model"],
    unit_col="unit",
    groupby_except_levels="run_id",
)

# %%
peak_warming = set_new_single_value_levels(temperatures_in_line_with_assessment.max(axis="columns"), {"metric": "max"})
peak_warming_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(peak_warming, "run_id").quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
    new_name="quantile",
)

eoc_warming = set_new_single_value_levels(temperatures_in_line_with_assessment[2100], {"metric": 2100})
eoc_warming_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(eoc_warming, "run_id").quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
    new_name="quantile",
)
peak_warming_year = set_new_single_value_levels(
    update_index_levels_func(
        temperatures_in_line_with_assessment.idxmax(axis="columns"),  # type: ignore # error in pandas-openscm
        {"unit": lambda x: "yr"},
    ),
    {"metric": "max_year"},
)
peak_warming_year_quantiles = fix_index_name_after_groupby_quantile(
    groupby_except(peak_warming_year, "run_id").quantile(
        quantiles_of_interest  # type: ignore # pandas-stubs out of date
    ),
    new_name="quantile",
)

exceedance_probabilities = get_exceedance_probabilities(
    temperatures_in_line_with_assessment,
    exceedance_thresholds_of_interest=exceedance_thresholds_of_interest,
    group_cols=["model", "scenario", "climate_model"],
    unit_col="unit",
    groupby_except_levels="run_id",
)

categories = categorise_scenarios(
    peak_warming_quantiles=peak_warming_quantiles,
    eoc_warming_quantiles=eoc_warming_quantiles,
    group_levels=["climate_model", "model", "scenario"],
    quantile_level="quantile",
)
categories.sort_values()

# %% [markdown]
# ### Compile climate output result

# %%
timeseries_run_id = pd.concat([temperatures_in_line_with_assessment])
timeseries_quantile = pd.concat([temperatures_in_line_with_assessment_quantiles])
timeseries_exceedance_probabilities = pd.concat([exceedance_probabilities_over_time])

metadata_run_id = pd.concat(  # type: ignore # pandas-stubs out of date
    [peak_warming, eoc_warming, peak_warming_year]
)
metadata_quantile = pd.concat(  # type: ignore # pandas-stubs out of date
    [
        peak_warming_quantiles,
        eoc_warming_quantiles,
        peak_warming_year_quantiles,
    ]
)
metadata_exceedance_probabilities = exceedance_probabilities
metadata_categories = categories

res = PostProcessingResult(
    timeseries_run_id=timeseries_run_id,
    timeseries_quantile=timeseries_quantile,
    timeseries_exceedance_probabilities=timeseries_exceedance_probabilities,
    metadata_run_id=metadata_run_id,
    metadata_quantile=metadata_quantile,
    metadata_exceedance_probabilities=metadata_exceedance_probabilities,
    metadata_categories=metadata_categories,
)

# %%
res.metadata_categories.unstack(["metric"]).sort_values

# %%
res.metadata_quantile.unstack(["metric", "unit", "quantile"]).loc[
    :,
    [
        (metric, "K", p)
        for (metric, p) in [
            ("max", 0.33),
            ("max", 0.5),
            ("max", 0.67),
            (2100, 0.5),
        ]
    ],
].sort_values(by=[("max", "K", 0.33)]).round(3)

# %% [markdown]
# ## Emissions

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


def calculate_co2_total(indf: pd.DataFrame) -> pd.DataFrame:
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
    if copy:
        indf = indf.copy()

    out_years = np.arange(indf.columns.min(), indf.columns.max() + 1)
    for y in out_years:
        if y not in indf:
            indf[y] = np.nan

    indf = indf.sort_index(axis="columns")
    indf = indf.T.interpolate(method="index").T

    return indf


def calculate_cumulative_co2s(indf: pd.DataFrame) -> pd.DataFrame:
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


def calculate_kyoto_ghgs(indf: pd.DataFrame, gwp: str = "AR6GWP100"):
    if "Emissions|CO2" not in indf.pix.unique("variable"):
        raise AssertionError(indf.pix.unique("variable"))

    with pint.get_application_registry().context(gwp):
        res = (
            indf.loc[pix.isin(variable=KYOTO_GHGS)]
            .pix.convert_unit("MtCO2 / yr")
            .openscm.groupby_except("variable")
            .sum(min_count=2)
            .pix.assign(variable=f"Emissions|Kyoto GHG {gwp}")
        )

    return res


def calculate_ghgs(indf: pd.DataFrame, gwp: str = "AR6GWP100"):
    if "Emissions|CO2" not in indf.pix.unique("variable"):
        raise AssertionError(indf.pix.unique("variable"))

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
        calculate_kyoto_ghgs(pre_processed_emms_scms_annual_incl_co2_total),
        calculate_ghgs(pre_processed_emms_scms_annual_incl_co2_total),
    ]
)
# pre_processed_emms_scms_out

# %%
ax = sns.lineplot(
    data=pre_processed_emms_scms_out.loc[
        pix.ismatch(variable="Cumulative**", scenario="SSP2*Low*")
    ].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
ax = sns.lineplot(
    data=pre_processed_emms_scms_out.loc[pix.ismatch(variable="**GHG**", scenario="SSP2*Low*")].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

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
    data=harmonised_emms_scms_out.loc[
        pix.ismatch(variable="Cumulative**", scenario="SSP2*Low*")
    ].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
ax = sns.lineplot(
    data=harmonised_emms_scms_out.loc[pix.ismatch(variable="**GHG**", scenario="SSP2*Low*")].openscm.to_long_data(),
    x="time",
    y="value",
    hue="scenario",
    style="variable",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

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
        calculate_kyoto_ghgs(complete_emissions_annual_incl_co2_total),
        calculate_ghgs(complete_emissions_annual_incl_co2_total),
    ]
)
# complete_emissions_out

# %%
ax = sns.lineplot(
    data=complete_emissions_out.loc[pix.ismatch(variable="**GHG**", scenario="SSP2*Low*")].openscm.to_long_data(),
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
    .loc[pix.ismatch(variable="Emissions|Kyoto GHG AR6GWP100", scenario="SSP2*Low*")]
    .openscm.to_long_data()
)

ax = sns.lineplot(
    data=pdf,
    x="time",
    y="value",
    hue="scenario",
    style="stage",
)
sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="gray")

# %% [markdown]
# ## Save

# %%
# Annoying hack, one to fix in future
tmp = res.metadata_quantile.to_frame(name="value").reset_index("metric")
tmp["metric"] = tmp["metric"].astype(str)
tmp = tmp.set_index("metric", append=True)
res.metadata_quantile = tmp

# %%
tmp = res.metadata_run_id.to_frame(name="value").reset_index("metric")
tmp["metric"] = tmp["metric"].astype(str)
tmp = tmp.set_index("metric", append=True)
res.metadata_run_id = tmp

# %%
for df, db in (
    (res.metadata_categories.to_frame(name="value"), POST_PROCESSED_METADATA_CATEGORIES_DB),
    (res.metadata_exceedance_probabilities.to_frame(name="value"), POST_PROCESSED_METADATA_EXCEEDANCE_PROBABILITIES_DB),
    (res.metadata_quantile, POST_PROCESSED_METADATA_QUANTILE_DB),
    (res.metadata_run_id, POST_PROCESSED_METADATA_RUN_ID_DB),
    (res.timeseries_exceedance_probabilities, POST_PROCESSED_TIMESERIES_EXCEEDANCE_PROBABILITIES_DB),
    (res.timeseries_quantile, POST_PROCESSED_TIMESERIES_QUANTILE_DB),
    (res.timeseries_run_id, POST_PROCESSED_TIMESERIES_RUN_ID_DB),
    (pre_processed_emms_scms_out.pix.assign(stage="pre-processed-scms"), POST_PROCESSED_TIMESERIES_DB),
    (harmonised_emms_gridding.pix.assign(stage="harmonised-gridding"), POST_PROCESSED_TIMESERIES_DB),
    (harmonised_emms_scms_out.pix.assign(stage="harmonised-scms"), POST_PROCESSED_TIMESERIES_DB),
    (complete_emissions_out.pix.assign(stage="complete"), POST_PROCESSED_TIMESERIES_DB),
):
    db.save(df, allow_overwrite=True)
