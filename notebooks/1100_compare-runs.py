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
# # Compare runs

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import seaborn as sns
from gcages.io import load_timeseries_csv
from gcages.pandas_helpers import multi_index_lookup
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

from emissions_harmonization_historical.constants import COMBINED_HISTORY_ID, DATA_ROOT, SCENARIO_TIME_ID, WORKFLOW_ID

# %% [markdown]
# ## Define some constants

# %%
ar6_workflow_out_dir = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_ar6-workflow"
)
ar6_workflow_out_dir_magicc = ar6_workflow_out_dir / "magicc-ar6"
ar6_workflow_out_dir_magicc

# %%
updated_workflow_dir = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"
)
updated_workflow_dir

# %%
updated_workflow_dir_magicc_v753 = updated_workflow_dir / "magicc-v7-5-3_600-member"
updated_workflow_dir_magicc_v753

# %%
updated_workflow_dir_magicc_v760 = updated_workflow_dir / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0"
updated_workflow_dir_magicc_v760

# %% [markdown]
# ## Look at metadata

# %%
metadata_l = []
for label, out_dir in (
    ("ar6-workflow_magiccv7.5.3", ar6_workflow_out_dir_magicc),
    ("updated-workflow_magiccv7.5.3", updated_workflow_dir_magicc_v753),
    ("updated-workflow_magiccv7.6.0", updated_workflow_dir_magicc_v760),
):
    tmp = pd.read_csv(out_dir / "metadata.csv").set_index(["model", "scenario"]).pix.assign(workflow=label)
    metadata_l.append(tmp)

metadata = pix.concat(metadata_l)
metadata


# %%
def get_scenario_group(scenario: str) -> str:
    """Get the scenario group"""
    return scenario.split(" - ")[-1].split("_")[0].strip()


# %%
scenario_group_order = [
    "Very Low Emissions",
    "Low Overshoot",
    "Low Emissions",
    "Medium-Low Emissions",
    "Medium Emissions",
    "High Emissions",
]

# %%
tmp = metadata.pix.assign(scenario_group=metadata.index.get_level_values("scenario").map(get_scenario_group))
disp = (
    tmp.groupby(["scenario_group", "workflow"])["category"]
    .value_counts()
    .to_frame("n_scenarios")
    .reset_index()
    .pivot_table(
        aggfunc="sum", values=["n_scenarios"], columns=["workflow"], index=["category", "scenario_group"], margins=True
    )
    .drop(("n_scenarios", "All"), axis="columns")
)
# Surely there is an easier way to do this
disp_l = []
for category, cdf in disp.groupby("category"):
    if category == "All":
        all_df = cdf

    else:
        disp_l.append(
            cdf.loc[
                (
                    slice(None),
                    [v for v in scenario_group_order if v in cdf.index.get_level_values("scenario_group").values],
                ),
                :,
            ]
        )

disp = pd.concat([*disp_l, all_df])
disp

# %%
metadata_by_workflow = metadata.stack().unstack("workflow").unstack()
metadata_by_workflow

# %%
deltas_l = []
for label, workflow_new, workflow_base in (
    ("delta_total", "updated-workflow_magiccv7.6.0", "ar6-workflow_magiccv7.5.3"),
    ("delta_magicc_update", "updated-workflow_magiccv7.6.0", "updated-workflow_magiccv7.5.3"),
    ("delta_harmonisation_infilling_update", "updated-workflow_magiccv7.5.3", "ar6-workflow_magiccv7.5.3"),
):
    tmp = (
        metadata_by_workflow[workflow_new]["Peak warming 50.0"]
        - metadata_by_workflow[workflow_base]["Peak warming 50.0"].sort_values()
    )
    tmp.name = label

    deltas_l.append(tmp)

deltas = pd.concat(deltas_l, axis="columns")
deltas

# %%
box_kwargs = dict(saturation=0.3, legend=False)
swarm_kwargs = dict(dodge=True)

# %%
pdf = deltas.melt(ignore_index=False).reset_index()
pdf["scenario_group"] = pdf["scenario"].apply(get_scenario_group)

fig, ax = plt.subplots(figsize=(12, 8))

pkwargs = dict(
    data=pdf,
    y="value",
    x="scenario_group",
    order=scenario_group_order,
    hue="variable",
    ax=ax,
)
sns.boxplot(**pkwargs, **box_kwargs)
sns.swarmplot(**pkwargs, **swarm_kwargs)

ax.axhline(0.0, color="tab:gray", zorder=1.2)

# %%
for start, title in (
    (deltas["delta_total"], "Change in peak warming"),
    (deltas["delta_magicc_update"], "Change in peak warming due to MAGICC update"),
    (deltas["delta_harmonisation_infilling_update"], "Change in peak warming due to infilling and harmonisation"),
):
    pdf = start.to_frame().reset_index()
    pdf["scenario_group"] = pdf["scenario"].apply(get_scenario_group)

    fig, ax = plt.subplots(figsize=(20, 8))

    pkwargs = dict(
        data=pdf,
        y=start.name,
        # x="scenario_group",
        # order=scenario_group_order,
        # hue="model",
        # hue_order=sorted(pdf["model"].unique()),
        x="model",
        order=sorted(pdf["model"].unique()),
        hue="scenario_group",
        hue_order=scenario_group_order,
        ax=ax,
    )
    sns.boxplot(**pkwargs, **box_kwargs)
    sns.swarmplot(**pkwargs, **swarm_kwargs)

    ax.axhline(0.0, color="tab:gray")

    plt.xticks(rotation=90)

    # ax.grid()
    fig.suptitle(title)


# %% [markdown]
# ## Helpers for the rest


# %%
def transform_ar6_workflow_output_to_iamc_variable(v):
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (("PFC|", ""),)
    for old, new in replacements:
        res = res.replace(old, new)

    direct_replacements = (
        ("HFC245ca", "HFC245fa"),
        ("CCl4", "Montreal Gases|CCl4"),
        ("CFC11", "Montreal Gases|CFC|CFC11"),
        ("CFC12", "Montreal Gases|CFC|CFC12"),
        ("CFC113", "Montreal Gases|CFC|CFC113"),
        ("CFC114", "Montreal Gases|CFC|CFC114"),
        ("CFC115", "Montreal Gases|CFC|CFC115"),
        ("CH2Cl2", "Montreal Gases|CH2Cl2"),
        ("CH3Br", "Montreal Gases|CH3Br"),
        ("CH3CCl3", "Montreal Gases|CH3CCl3"),
        ("CH3Cl", "Montreal Gases|CH3Cl"),
        ("CHCl3", "Montreal Gases|CHCl3"),
        ("HCFC141b", "Montreal Gases|HCFC141b"),
        ("HCFC142b", "Montreal Gases|HCFC142b"),
        ("HCFC22", "Montreal Gases|HCFC22"),
        ("Halon1202", "Montreal Gases|Halon1202"),
        ("Halon1211", "Montreal Gases|Halon1211"),
        ("Halon1301", "Montreal Gases|Halon1301"),
        ("Halon2402", "Montreal Gases|Halon2402"),
    )
    for old, new in direct_replacements:
        if res.endswith(old):
            res = res.replace(old, new)

    return res


# %%
def load_stage(
    out_file_name: str,
    ar6_prefix: str,
    index_columns: tuple[str, ...] = ("model", "scenario", "region", "variable", "unit"),
    magicc_output: bool = False,
) -> pd.DataFrame:
    """
    Load data for a given stage
    """
    stage_l = []
    if magicc_output:
        to_load = (
            ("ar6-workflow_magicc-v7.5.3", ar6_workflow_out_dir_magicc, True, True),
            ("updated-workflow_magicc-v7.5.3", updated_workflow_dir_magicc_v753, False, False),
            ("updated-workflow_magicc-v7.6.0", updated_workflow_dir_magicc_v760, False, False),
        )

    else:
        to_load = (
            ("ar6-workflow", ar6_workflow_out_dir, True, True),
            ("updated-workflow", updated_workflow_dir, False, False),
        )

    for label, out_dir, strip_out_ar6_prefix, transform_variables in to_load:
        tmp = load_timeseries_csv(
            out_dir / out_file_name,
            index_columns=list(index_columns),
            out_column_type=int,
        ).pix.assign(workflow=label)

        if strip_out_ar6_prefix:
            tmp = pix.assignlevel(
                tmp,
                variable=tmp.index.get_level_values("variable").map(lambda x: x.replace(ar6_prefix, "")),
            )

        # Hmmm, need to work out what to do upstream to remove the need for this
        if transform_variables:
            tmp = pix.assignlevel(
                tmp,
                variable=tmp.index.get_level_values("variable").map(transform_ar6_workflow_output_to_iamc_variable),
            )

        stage_l.append(tmp)

    stage = pix.concat(stage_l)

    # Interpolate to make later life easier
    stage = stage.T.interpolate("index").T
    stage = stage.sort_index()
    stage = stage.pix.assign(scenario_group=stage.index.get_level_values("scenario").map(get_scenario_group))

    return stage


# %%
def get_sns_df(indf):
    """
    Get data frame to use with seaborn's plotting
    """
    out = indf.copy()
    out.columns.name = "year"
    out = out.stack().to_frame("value").reset_index()

    return out


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
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"

# %%
history = strip_pint_incompatible_characters_from_units(
    load_timeseries_csv(
        HISTORICAL_GLOBAL_COMPOSITE_PATH,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
)
history_cut = history.loc[:, 1990:2025]
# Interpolate
history_cut = history_cut.T.interpolate("index", limit_area="inside").T
history_cut

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"

# %%
rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
ar6_history = (
    rcmip_clean.loc[pix.ismatch(mip_era="CMIP6") & pix.ismatch(scenario="historical") & pix.ismatch(region="World")]
    .reset_index(["mip_era", "activity_id"], drop=True)
    .dropna(how="all", axis="columns")
)
ar6_history.loc[:, 1990:]

# %% [markdown]
# ## Harmonisation

# %%
pre_processed = load_stage(
    out_file_name="pre-processed.csv",
    ar6_prefix="",
)
pre_processed

# %%
harmonised = load_stage(
    out_file_name="harmonised.csv",
    ar6_prefix="AR6 climate diagnostics|Harmonized|",
)
harmonised

# %%
variables_to_plot = [
    # 'Emissions|BC',
    # 'Emissions|C2F6',
    # # 'Emissions|C3F8',
    # # 'Emissions|C4F10',
    # # 'Emissions|C5F12',
    # 'Emissions|C6F14',
    # # 'Emissions|C7F16',
    # # 'Emissions|C8F18',
    # 'Emissions|CF4',
    # 'Emissions|CH4',
    # 'Emissions|CO',
    # 'Emissions|CO2|AFOLU',
    "Emissions|CO2|Energy and Industrial Processes",
    # # 'Emissions|HFC|HFC125',
    # 'Emissions|HFC|HFC134a',
    # # 'Emissions|HFC|HFC143a',
    # # 'Emissions|HFC|HFC152a',
    # # 'Emissions|HFC|HFC227ea',
    # # 'Emissions|HFC|HFC23',
    # # 'Emissions|HFC|HFC236fa',
    # # 'Emissions|HFC|HFC245fa',
    # # 'Emissions|HFC|HFC32',
    # # 'Emissions|HFC|HFC365mfc',
    # # 'Emissions|HFC|HFC43-10',
    # # 'Emissions|Montreal Gases|CCl4',
    # 'Emissions|Montreal Gases|CFC|CFC11',
    # # 'Emissions|Montreal Gases|CFC|CFC113',
    # # 'Emissions|Montreal Gases|CFC|CFC114',
    # # 'Emissions|Montreal Gases|CFC|CFC115',
    # 'Emissions|Montreal Gases|CFC|CFC12',
    # # 'Emissions|Montreal Gases|CH2Cl2',
    # # 'Emissions|Montreal Gases|CH3Br',
    # # 'Emissions|Montreal Gases|CH3CCl3',
    # # 'Emissions|Montreal Gases|CH3Cl',
    # # 'Emissions|Montreal Gases|CHCl3',
    # # 'Emissions|Montreal Gases|HCFC141b',
    # # 'Emissions|Montreal Gases|HCFC142b',
    # # 'Emissions|Montreal Gases|HCFC22',
    # # 'Emissions|Montreal Gases|Halon1202',
    # # 'Emissions|Montreal Gases|Halon1211',
    # # 'Emissions|Montreal Gases|Halon1301',
    # # 'Emissions|Montreal Gases|Halon2402',
    # 'Emissions|N2O',
    # 'Emissions|NF3',
    # 'Emissions|NH3',
    # 'Emissions|NOx',
    "Emissions|OC",
    # 'Emissions|SF6',
    # # 'Emissions|SO2F2',
    # 'Emissions|Sulfur',
    # 'Emissions|VOC',
    # # 'Emissions|cC4F8',
]

# %%
for variable, vdf in (
    harmonised.pix.assign(stage="harmonised").loc[pix.isin(variable=variables_to_plot)].groupby("variable")
):
    scenario_group_history_l = []
    for scenario_group in vdf.pix.unique("scenario_group"):
        scenario_group_history_l.extend(
            [
                ar6_history.loc[pix.isin(variable=variable)]
                .pix.assign(workflow="history-ar6", scenario_group=scenario_group)
                .loc[:, 1990:],
                history_cut.loc[pix.isin(variable=variable)]
                .pix.assign(workflow="history", scenario_group=scenario_group)
                .loc[:, 1990:],
            ]
        )
        scenario_group_history = get_sns_df(pd.concat(scenario_group_history_l))

    pdf = pd.concat(
        [
            scenario_group_history,
            get_sns_df(vdf),
        ]
    )

    fg = sns.relplot(
        data=pdf,
        x="year",
        y="value",
        hue="scenario",
        units="model",
        estimator=None,
        style="workflow",
        dashes={
            "history": "",
            "updated-workflow": "",
            "history-ar6": (3, 3),
            "ar6-workflow": (3, 3),
        },
        col="scenario_group",
        col_wrap=3,
        col_order=["Very Low Emissions", "Low Emissions", "Low Overshoot", "Medium Emissions", "High Emissions"],
        facet_kws=dict(sharey=True),
        kind="line",
        alpha=0.7,
    )
    if "CO2" not in variable:
        for ax in fg.axes.flatten():
            ax.set_ylim(ymin=0)

    fg.figure.suptitle(f"{variable}", y=1.02)
    plt.show()

    # break

# %%
for variable, vdf in (
    pix.concat([pre_processed.pix.assign(stage="pre_processed"), harmonised.pix.assign(stage="harmonised")])
    .loc[pix.isin(variable=variables_to_plot)]
    .groupby("variable")
):
    for workflow, wdf in vdf.groupby("workflow"):
        scenario_group_history_l = []
        for scenario_group in wdf.pix.unique("scenario_group"):
            scenario_group_history_l.extend(
                [
                    ar6_history.loc[pix.isin(variable=variable)]
                    .pix.assign(stage="history-ar6", scenario_group=scenario_group)
                    .loc[:, 1990:],
                    history_cut.loc[pix.isin(variable=variable)]
                    .pix.assign(stage="history", scenario_group=scenario_group)
                    .loc[:, 1990:],
                ]
            )
            scenario_group_history = get_sns_df(pd.concat(scenario_group_history_l))

        pdf = pd.concat(
            [
                scenario_group_history,
                get_sns_df(wdf),
            ]
        )

        fg = sns.relplot(
            data=pdf,
            x="year",
            y="value",
            hue="scenario",
            units="model",
            estimator=None,
            style="stage",
            dashes={
                "harmonised": "",
                "history": "",
                "history-ar6": "",
                "pre_processed": (3, 3),
            },
            col="scenario_group",
            col_wrap=3,
            col_order=["Very Low Emissions", "Low Emissions", "Low Overshoot", "Medium Emissions", "High Emissions"],
            facet_kws=dict(sharey=True),
            kind="line",
        )
        if "CO2" not in variable:
            for ax in fg.axes.flatten():
                ax.set_ylim(ymin=0)

        fg.figure.suptitle(f"{variable} | {workflow}", y=1.02)
        plt.show()

    #     break
    # break

# %% [markdown]
# ## Infilled emissions

# %%
infilled = load_stage(
    out_file_name="infilled.csv",
    ar6_prefix="AR6 climate diagnostics|Infilled|",
)
exp_n_variables = 52
if len(sorted(infilled.pix.unique("variable"))) != exp_n_variables:
    raise AssertionError

infilled

# %%
all_variables = sorted(infilled.pix.unique("variable"))

# %%
zn_custom_variables = [
    "Emissions|BC",
    "Emissions|C2F6",
    # 'Emissions|C3F8',
    # 'Emissions|C4F10',
    # 'Emissions|C5F12',
    "Emissions|C6F14",
    # 'Emissions|C7F16',
    # 'Emissions|C8F18',
    "Emissions|CF4",
    "Emissions|CH4",
    "Emissions|CO",
    "Emissions|CO2|AFOLU",
    "Emissions|CO2|Energy and Industrial Processes",
    # 'Emissions|HFC|HFC125',
    "Emissions|HFC|HFC134a",
    # 'Emissions|HFC|HFC143a',
    # 'Emissions|HFC|HFC152a',
    # 'Emissions|HFC|HFC227ea',
    # 'Emissions|HFC|HFC23',
    # 'Emissions|HFC|HFC236fa',
    # 'Emissions|HFC|HFC245fa',
    # 'Emissions|HFC|HFC32',
    # 'Emissions|HFC|HFC365mfc',
    # 'Emissions|HFC|HFC43-10',
    # 'Emissions|Montreal Gases|CCl4',
    "Emissions|Montreal Gases|CFC|CFC11",
    # 'Emissions|Montreal Gases|CFC|CFC113',
    # 'Emissions|Montreal Gases|CFC|CFC114',
    # 'Emissions|Montreal Gases|CFC|CFC115',
    "Emissions|Montreal Gases|CFC|CFC12",
    # 'Emissions|Montreal Gases|CH2Cl2',
    # 'Emissions|Montreal Gases|CH3Br',
    # 'Emissions|Montreal Gases|CH3CCl3',
    # 'Emissions|Montreal Gases|CH3Cl',
    # 'Emissions|Montreal Gases|CHCl3',
    # 'Emissions|Montreal Gases|HCFC141b',
    # 'Emissions|Montreal Gases|HCFC142b',
    # 'Emissions|Montreal Gases|HCFC22',
    # 'Emissions|Montreal Gases|Halon1202',
    # 'Emissions|Montreal Gases|Halon1211',
    # 'Emissions|Montreal Gases|Halon1301',
    # 'Emissions|Montreal Gases|Halon2402',
    "Emissions|N2O",
    # 'Emissions|NF3',
    "Emissions|NH3",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|SF6",
    # 'Emissions|SO2F2',
    "Emissions|Sulfur",
    "Emissions|VOC",
    # 'Emissions|cC4F8',
]

# %%
variables_to_plot = all_variables
variables_to_plot = zn_custom_variables

# %% [markdown]
# Plot the infilled emissions for the scenarios that have the biggest change in peak warming
# due to harmonisation and infilling.

# %%
workflow_new = "updated-workflow"
workflow_base = "ar6-workflow"

for model, pwd in deltas["delta_harmonisation_infilling_update"].groupby("model"):
    print(f"{model}: {workflow_new} - {workflow_base}")
    display(pwd.sort_values())  # noqa: F821
    plot_mod_scen = pd.concat([pwd.iloc[:3], pwd.iloc[-3:]]).index
    pdf = get_sns_df(
        pix.concat(
            [
                multi_index_lookup(infilled, plot_mod_scen).loc[pix.isin(workflow=[workflow_base, workflow_new])],
                history_cut.pix.assign(
                    scenario="updated-historical", scenario_group="history", workflow=workflow_new, model="history"
                ),
                ar6_history.pix.assign(
                    scenario="ar6-historical", scenario_group="history", workflow=workflow_base, model="history"
                ).loc[:, 1990:],
            ]
        ).loc[pix.isin(variable=variables_to_plot)]
    )
    pdf["model-scenario"] = pdf["model"] + " - " + pdf["scenario"]
    fg = sns.relplot(
        data=pdf,
        x="year",
        y="value",
        hue="model-scenario",
        style="workflow",
        dashes={
            workflow_new: "",
            workflow_base: (3, 3),
            "updated-workflow": "",
        },
        col="variable",
        col_wrap=3,
        col_order=sorted(variables_to_plot),
        facet_kws=dict(sharey=False),
        kind="line",
    )
    for ax in fg.axes.flatten():
        if "CO2" not in ax.get_title():
            ax.set_ylim(ymin=0)

        ax.grid()

    fg.figure.suptitle(model, y=1.01)

    plt.show()
    # break

# %% [markdown]
# ## SCM-effective emissions

# %%
scm_effective = load_stage(
    out_file_name="scm-effective-emissions.csv",
    ar6_prefix="AR6 climate diagnostics|Infilled|",
    magicc_output=True,
)
exp_n_variables = 52
if len(sorted(scm_effective.pix.unique("variable"))) != exp_n_variables:
    raise AssertionError

scm_effective

# %%
variables_to_plot = all_variables
variables_to_plot = zn_custom_variables
variables_to_plot = ["Emissions|OC", "Emissions|Sulfur"]

# %% [markdown]
# Compare the emissions across workflows.

# %%
for model, mdf in scm_effective.groupby("model"):
    tmp = (
        pix.concat(
            [
                mdf,
                history_cut.pix.assign(
                    scenario="updated-historical",
                    scenario_group="history",
                    workflow="updated-workflow_magicc-v7.6.0",
                    model="history",
                ),
                ar6_history.pix.assign(
                    scenario="ar6-historical",
                    scenario_group="history",
                    workflow="ar6-workflow_magicc-v7.5.3",
                    model="history",
                ).loc[:, 1990:],
            ]
        )
        .loc[pix.isin(variable=variables_to_plot)]
        .sort_index(axis="columns")
        .loc[:, 2010:2030]
    )
    pdf = get_sns_df(tmp)
    pdf["model-scenario"] = pdf["model"] + " - " + pdf["scenario"]
    fg = sns.relplot(
        data=pdf,
        x="year",
        y="value",
        hue="model-scenario",
        style="workflow",
        dashes={
            "ar6-workflow_magicc-v7.5.3": "",
            "updated-workflow_magicc-v7.5.3": (1, 1),
            "updated-workflow_magicc-v7.6.0": (3, 3),
        },
        col="variable",
        col_wrap=3,
        col_order=sorted(variables_to_plot),
        facet_kws=dict(sharey=False),
        kind="line",
    )
    for ax in fg.axes.flatten():
        if "CO2" not in ax.get_title():
            ax.set_ylim(ymin=0)

        ax.grid()

    fg.figure.suptitle(model, y=1.01)

    plt.show()
    # break

# %% [markdown]
# Plot the scm-effective emissions for the scenarios that have the biggest change in peak warming
# due to harmonisation and infilling.

# %%
workflow_base = "ar6-workflow_magicc-v7.5.3"
workflow_new = "updated-workflow_magicc-v7.5.3"

# %%
for model, pwd in deltas["delta_harmonisation_infilling_update"].groupby("model"):
    print(f"{model}: {workflow_new} - {workflow_base}")
    # display(pwd.sort_values())
    plot_mod_scen = pd.concat([pwd.iloc[:3], pwd.iloc[-3:]]).index
    pdf = get_sns_df(
        pix.concat(
            [
                multi_index_lookup(scm_effective, plot_mod_scen).loc[pix.isin(workflow=[workflow_base, workflow_new])],
                history_cut.pix.assign(
                    scenario="updated-historical", scenario_group="history", workflow=workflow_new, model="history"
                ),
                ar6_history.pix.assign(
                    scenario="ar6-historical", scenario_group="history", workflow=workflow_base, model="history"
                ).loc[:, 1990:],
            ]
        ).loc[pix.isin(variable=variables_to_plot)]
    )
    pdf["model-scenario"] = pdf["model"] + " - " + pdf["scenario"]
    fg = sns.relplot(
        data=pdf,
        x="year",
        y="value",
        hue="model-scenario",
        style="workflow",
        dashes={
            workflow_new: "",
            workflow_base: (3, 3),
            "updated-workflow": "",
        },
        col="variable",
        col_wrap=3,
        col_order=sorted(variables_to_plot),
        facet_kws=dict(sharey=False),
        kind="line",
    )
    for ax in fg.axes.flatten():
        if "CO2" not in ax.get_title():
            ax.set_ylim(ymin=0)

        ax.grid()

    fg.figure.suptitle(model, y=1.01)

    plt.show()
    # break

# %% [markdown]
# ## Climate variables

# %%
climate_percentiles = load_stage(
    out_file_name="timeseries-percentiles.csv",
    ar6_prefix="AR6 climate diagnostics|",
    index_columns=("climate_model", "model", "scenario", "region", "variable", "percentile", "unit"),
    magicc_output=True,
)
climate_percentiles.pix.unique("workflow")

# %%
variables_to_plot = climate_percentiles.pix.unique("variable")
percentile_to_plot = 50.0

# %%
for model, pwd in deltas["delta_harmonisation_infilling_update"].groupby("model"):
    print(f"{model}: {workflow_new} - {workflow_base}")
    # display(pwd.sort_values())
    plot_mod_scen = pd.concat([pwd.iloc[:3], pwd.iloc[-3:]]).index
    pdf = get_sns_df(
        multi_index_lookup(climate_percentiles, plot_mod_scen)
        # .loc[pix.isin(workflow=[workflow_base, workflow_new])]
        .loc[pix.isin(percentile=percentile_to_plot)]
        .loc[:, 2015:2050]
    )
    pdf["model-scenario"] = pdf["model"] + " - " + pdf["scenario"]
    fg = sns.relplot(
        data=pdf,
        x="year",
        y="value",
        hue="model-scenario",
        style="workflow",
        dashes={
            "ar6-workflow_magicc-v7.5.3": "",
            "updated-workflow_magicc-v7.5.3": (1, 1),
            "updated-workflow_magicc-v7.6.0": (3, 3),
        },
        # dashes={
        #     workflow_new: "",
        #     workflow_base: (3, 3),
        #     "updated-workflow": "",
        # },
        col="variable",
        col_wrap=3,
        col_order=sorted(variables_to_plot),
        facet_kws=dict(sharey=False),
        kind="line",
    )
    for ax in fg.axes.flatten():
        # if "CO2" not in ax.get_title():
        #     ax.set_ylim(ymin=0)

        ax.grid()

    fg.figure.suptitle(model, y=1.01)

    plt.show()
    # break
