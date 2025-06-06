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
# # Compare runs - metadata

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import seaborn as sns
from gcages.pandas_helpers import multi_index_lookup

from emissions_harmonization_historical.constants import DATA_ROOT, SCENARIO_TIME_ID, WORKFLOW_ID

# %% [markdown]
# ## Define some constants

# %%
ar6_workflow_output_dir = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_ar6-workflow"
)
ar6_workflow_output_dir

# %%
updated_workflow_output_dir = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"
)
updated_workflow_output_dir

# %%
ar6_workflow_magicc_v753_output_dir = ar6_workflow_output_dir / "magicc-ar6"
ar6_workflow_magicc_v76_output_dir = ar6_workflow_output_dir / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0"

# %%
updated_workflow_magicc_v753_output_dir = updated_workflow_output_dir / "magicc-v7-5-3_600-member"
updated_workflow_magicc_v76_output_dir = (
    updated_workflow_output_dir / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0"
)

# %%
magicc_v76_version_label = "magicc-v7.6.0a3"

# %% [markdown]
# ## Look at metadata

# %%
scenario_group_order = [
    "VLLO",
    "VLHO",
    "L",
    "ML",
    "M",
    "H",
]


def get_scenario_group(scenario: str) -> str:
    """Get the scenario group"""
    scenario_map_mapping = {
        "Very Low Emissions": "VLLO",
        "Low Overshoot": "VLHO",
        "Low Emissions": "L",
        "Medium-Low Emissions": "ML",
        "Medium Emissions": "M",
        "High Emissions": "H",
    }
    key = scenario.split(" - ")[-1].split("_")[0].strip()

    return scenario_map_mapping[key]


# %%
def load_labelled_metadata(
    ar6_workflow_magicc_v753_output_dir: Path,
    ar6_workflow_magicc_v76_output_dir: Path,
    updated_workflow_magicc_v753_output_dir: Path,
    updated_workflow_magicc_v76_output_dir: Path,
    magicc_v76_version_label: str,
) -> pd.DataFrame:
    """
    Load metadata with labels
    """
    metadata_l = []
    for label, out_dir in (
        ("ar6-workflow_magiccv7.5.3", ar6_workflow_magicc_v753_output_dir),
        (f"ar6-workflow_{magicc_v76_version_label}", ar6_workflow_magicc_v76_output_dir),
        ("updated-workflow_magiccv7.5.3", updated_workflow_magicc_v753_output_dir),
        (f"updated-workflow_{magicc_v76_version_label}", updated_workflow_magicc_v76_output_dir),
    ):
        fto_load = out_dir / "metadata.csv"
        if not fto_load.exists():
            print(f"Does not exist: {fto_load=}")
            continue

        tmp = pd.read_csv(fto_load).set_index(["model", "scenario"]).pix.assign(workflow=label)
        metadata_l.append(tmp)

    metadata = pix.concat(metadata_l)
    return metadata


# %% [markdown]
# ## Load

# %%
metadata = load_labelled_metadata(
    ar6_workflow_magicc_v753_output_dir=ar6_workflow_magicc_v753_output_dir,
    ar6_workflow_magicc_v76_output_dir=ar6_workflow_magicc_v76_output_dir,
    updated_workflow_magicc_v753_output_dir=updated_workflow_magicc_v753_output_dir,
    updated_workflow_magicc_v76_output_dir=updated_workflow_magicc_v76_output_dir,
    magicc_v76_version_label=magicc_v76_version_label,
)

# %% [markdown]
# ## Compare with what is in the database

# %%
db_meta_file = DATA_ROOT / "scenarios" / "data_raw" / f"{SCENARIO_TIME_ID}_all-meta.csv"
db_meta = pd.read_csv(db_meta_file).set_index(["model", "scenario"])
# db_meta

# %%
for our_column, db_column in (
    ("Peak warming 50.0", "median peak warming (MAGICCv7.5.3)"),
    ("Peak warming 33.0", "p33 peak warming (MAGICCv7.5.3)"),
):
    print(f"{our_column=}")
    our_res = metadata.loc[pix.isin(workflow=["ar6-workflow_magiccv7.5.3"])][our_column].droplevel("workflow")
    display(  # noqa: F821
        (our_res - multi_index_lookup(db_meta[db_column], our_res.index)).abs().sort_values(ascending=False).iloc[:30]
    )

# %% [markdown]
# ## Number of scenarios in each category

# %%
workflow_map = {
    "ar6-workflow_magiccv7.5.3": "AR6",
    "ar6-workflow_magicc-v7.6.0a3": "AR6_updated-MAGICC",
    # 'updated-workflow_magiccv7.5.3': "AR7FT_AR6-MAGICC",
    "updated-workflow_magicc-v7.6.0a3": "AR7FT-MAGICCv7.6.0a3",
    # "updated-workflow_fair-v2.2.2": "AR7FT-FaIRv2.2.2",
}
col_show_order = ["AR6", "AR6_updated-MAGICC", "AR7FT-MAGICCv7.6.0a3"]

tmp = metadata
# # If Chris does separate runs, can use this
# col_show_order = ["AR6", "AR6_updated-MAGICC", "AR7FT-FaIRv2.2.2", "AR7FT-MAGICCv7.6.0a3"]
# fair_v2_2_2_2021_harm_calib_metadata = pd.read_csv(DATA_ROOT / "20250204_categorization-chris.csv").drop(
#     "Unnamed: 0", axis="columns"
# )
# fair_v2_2_2_2021_harm_calib_metadata["workflow"] = "updated-workflow_fair-v2.2.2"
# fair_v2_2_2_2021_harm_calib_metadata = fair_v2_2_2_2021_harm_calib_metadata.set_index(
#    ["model", "scenario", "workflow"]
# )
# fair_v2_2_2_2021_harm_calib_metadata = fair_v2_2_2_2021_harm_calib_metadata.rename(
#     {
#         "peak_warming_p33": "Peak warming 33.0",
#         "peak_warming_p50": "Peak warming 50.0",
#         "peak_warming_p67": "Peak warming 67.0",
#         "2100_warming_p50": "EOC warming 50.0",
#     },
#     axis="columns",
# )
# fair_v2_2_2_2021_harm_calib_metadata
# tmp = pix.concat([tmp, fair_v2_2_2_2021_harm_calib_metadata])

tmp = tmp.pix.assign(scenario_group=tmp.index.get_level_values("scenario").map(get_scenario_group))
tmp = tmp.pix.assign(workflow=tmp.index.get_level_values("workflow").map(workflow_map))


disp = (
    tmp.groupby(["scenario_group", "workflow"])["category"]
    .value_counts()
    .to_frame("n_scenarios")
    .reset_index()
    .pivot_table(
        aggfunc="sum", values=["n_scenarios"], columns=["workflow"], index=["scenario_group", "category"], margins=True
    )
    .drop(("n_scenarios", "All"), axis="columns")
)
disp.groupby("category").sum().drop("").loc[:, (["n_scenarios"], col_show_order)]

# %%
disp.sort_index().loc[
    [
        *[v for v in scenario_group_order if v in disp.index.get_level_values("scenario_group")],
        "All",
    ],
    (["n_scenarios"], list(workflow_map.values())),
    # [("n_scenarios",), tuple(workflow_map.values())],
].loc[:, (["n_scenarios"], col_show_order)]

# %% [markdown]
# ## Plots of warming

# %%
palette = {
    "AR6": "tab:orange",
    "AR6_updated-MAGICC": "tab:purple",
    "AR7FT_AR6-MAGICC": "tab:olive",
    "AR7FT-MAGICCv7.6.0a3": "tab:blue",
    "AR7FT-FaIRv2.2.2": "tab:green",
}
variables = ["Peak warming 33.0", "Peak warming 50.0", "EOC warming 50.0"]
box_kwargs = dict(saturation=0.6)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 9))
for variable, axes_row in zip(variables, axes):
    tmp = metadata
    try:
        tmp = pix.concat([tmp, fair_v2_2_2_2021_harm_calib_metadata])
    except NameError:
        # no FaIR data
        pass

    sns_df = tmp[[variable]].melt(ignore_index=False).reset_index()
    sns_df["scenario_group"] = sns_df["scenario"].apply(get_scenario_group)
    sns_df["workflow"] = sns_df["workflow"].map(workflow_map)

    for i, (ax, sgs) in enumerate(zip(axes_row, [["VLLO", "VLHO", "L"], ["ML", "M", "H"]])):
        pkwargs = dict(
            data=sns_df[sns_df["scenario_group"].isin(sgs)],
            y="value",
            x="scenario_group",
            order=[s for s in scenario_group_order if s in sgs],
            hue="workflow",
            palette=palette,
            ax=ax,
        )

        sns.boxplot(**pkwargs, **box_kwargs)

        # sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

        ax.set_yticks(np.arange(1.0, ax.get_ylim()[1], 0.5))

        ax.grid()
        ax.set_title(variable)

fig.tight_layout()

# %%
metadata_by_workflow = metadata.copy()
try:
    metadata_by_workflow = pix.concat([metadata_by_workflow, fair_v2_2_2_2021_harm_calib_metadata]).copy()
except NameError:
    # No FaIR data
    pass

metadata_by_workflow = metadata_by_workflow.pix.assign(
    workflow=metadata_by_workflow.index.get_level_values("workflow").map(workflow_map)
)
metadata_by_workflow = metadata_by_workflow[~metadata_by_workflow.index.get_level_values("workflow").isnull()]
metadata_by_workflow = metadata_by_workflow.stack().unstack("workflow").unstack()
metadata_by_workflow.loc[:, (slice(None), "Peak warming 33.0")].sort_values(
    ("AR7FT-MAGICCv7.6.0a3", "Peak warming 33.0")
).iloc[:13,].loc[:, col_show_order].astype(float).round(2)

# %%
metadata_by_workflow = metadata.stack().unstack("workflow").unstack()
metadata_by_workflow

# %%
deltas_l = []
for compare_col, prefix in (
    ("Peak warming 50.0", "peak_warming"),
    ("EOC warming 50.0", "2100_warming"),
):
    for label, workflow_new, workflow_base in (
        ("delta_total", f"updated-workflow_{magicc_v76_version_label}", "ar6-workflow_magiccv7.5.3"),
        ("delta_magicc_update_ar6_workflow", f"ar6-workflow_{magicc_v76_version_label}", "ar6-workflow_magiccv7.5.3"),
        (
            "delta_magicc_update",
            f"updated-workflow_{magicc_v76_version_label}",
            "updated-workflow_magiccv7.5.3",
        ),
        (
            "delta_other_updates",
            f"updated-workflow_{magicc_v76_version_label}",
            f"ar6-workflow_{magicc_v76_version_label}",
        ),
        ("delta_other_updates_magiccv7.5.3", "updated-workflow_magiccv7.5.3", "ar6-workflow_magiccv7.5.3"),
    ):
        tmp = (
            metadata_by_workflow[workflow_new][compare_col]
            - metadata_by_workflow[workflow_base][compare_col].sort_values()
        )
        tmp.name = f"{prefix}_{label}"

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
    data=pdf[
        pdf["variable"].isin(
            [
                "peak_warming_delta_magicc_update",
                "2100_warming_delta_magicc_update",
            ]
        )
    ],
    y="value",
    x="scenario_group",
    order=scenario_group_order,
    hue="variable",
    ax=ax,
)
sns.boxplot(**pkwargs, **box_kwargs)
sns.swarmplot(**pkwargs, **swarm_kwargs)

sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.axhline(0.0, color="tab:gray", zorder=1.2)
ax.grid()

# %%
pdf = deltas.melt(ignore_index=False).reset_index()
pdf["scenario_group"] = pdf["scenario"].apply(get_scenario_group)
tmp = (
    metadata["Peak warming 50.0"].loc[pix.isin(workflow="ar6-workflow_magiccv7.5.3")].reset_index("workflow", drop=True)
)
tmp.name = "peak_warming_ar6"
pdf = pd.concat([pdf.set_index(["model", "scenario"]), tmp], axis="columns").reset_index()

fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(
    data=pdf[
        pdf["variable"].isin(
            [
                "peak_warming_delta_magicc_update",
                "2100_warming_delta_magicc_update",
            ]
        )
    ],
    x="peak_warming_ar6",
    y="value",
    hue="variable",
    alpha=0.5,
    ax=ax,
)

sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.axhline(0.0, color="tab:gray", zorder=1.2)
ax.grid()

# %%
pdf = deltas.melt(ignore_index=False).reset_index()
pdf["scenario_group"] = pdf["scenario"].apply(get_scenario_group)

fig, ax = plt.subplots(figsize=(12, 8))

pkwargs = dict(
    data=pdf[pdf["variable"].isin(["peak_warming_delta_total", "2100_warming_delta_total"])],
    y="value",
    x="scenario_group",
    order=scenario_group_order,
    hue="variable",
    ax=ax,
)
sns.boxplot(**pkwargs, **box_kwargs)
sns.swarmplot(**pkwargs, **swarm_kwargs)

sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.axhline(0.0, color="tab:gray", zorder=1.2)

# %%
pdf = deltas.melt(ignore_index=False).reset_index()
pdf["scenario_group"] = pdf["scenario"].apply(get_scenario_group)

fig, ax = plt.subplots(figsize=(12, 8))

pkwargs = dict(
    data=pdf[
        pdf["variable"].isin(
            [
                "peak_warming_delta_total",
                # "peak_warming_delta_magicc_update",
                "peak_warming_delta_other_updates",
                "2100_warming_delta_total",
                # "2100_warming_delta_magicc_update",
                "2100_warming_delta_other_updates",
            ]
        )
    ],
    y="value",
    x="scenario_group",
    order=scenario_group_order,
    hue="variable",
    ax=ax,
)
sns.boxplot(**pkwargs, **box_kwargs)
sns.swarmplot(**pkwargs, **swarm_kwargs)

sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.axhline(0.0, color="tab:gray", zorder=1.2)

# %%
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

sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.axhline(0.0, color="tab:gray", zorder=1.2)

# %%
for start, title in (
    (deltas["peak_warming_delta_total"], "Change in peak warming"),
    (deltas["peak_warming_delta_magicc_update"], "Change in peak warming due to MAGICC update"),
    (
        deltas["peak_warming_delta_other_updates"],
        "Change in peak warming due to other updates (historical emissions, infilling and harmonisation)",
    ),
    (deltas["2100_warming_delta_total"], "Change in 2100 warming"),
    (deltas["2100_warming_delta_magicc_update"], "Change in 2100 warming due to MAGICC update"),
    (
        deltas["2100_warming_delta_other_updates"],
        "Change in 2100 warming due to other updates (historical emissions, infilling and harmonisation)",
    ),
):
    pdf = start.to_frame().reset_index()
    pdf["scenario_group"] = pdf["scenario"].apply(get_scenario_group)

    fig, ax = plt.subplots(figsize=(16, 6))

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

    plt.xticks(rotation=60)

    # ax.grid()
    fig.suptitle(title)
