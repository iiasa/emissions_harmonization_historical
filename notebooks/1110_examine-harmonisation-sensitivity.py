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
# # Examine harmonisation sensitivity

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import seaborn as sns
from gcages.io import load_timeseries_csv
from tqdm.auto import tqdm

from emissions_harmonization_historical.constants import DATA_ROOT, SCENARIO_TIME_ID, WORKFLOW_ID

# %%
out_paths_metadata = {
    "default": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"
    / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0",
    "co2_afolu_2050": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow-harmonise-co2-afolu-2050",
    "all_2050": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow-harmonise-all-2050",
}
out_paths_emissions = {
    "default": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow",
    "co2_afolu_2050": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow-harmonise-co2-afolu-2050",
    "all_2050": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow-harmonise-all-2050",
}

# %%
metadata_l = []
for key, out_path in out_paths_metadata.items():
    key_metadata = pd.read_csv(out_path / "metadata.csv").set_index(["model", "scenario"])
    metadata_l.append(key_metadata.pix.assign(workflow=key))

metadata = pix.concat(metadata_l)
metadata

# %%
box_kwargs = dict(saturation=0.3, legend=False)
swarm_kwargs = dict(dodge=True)

variables = ["Peak warming 33.0", "Peak warming 50.0", "EOC warming 50.0"]
sns_df = (
    metadata[variables].unstack("workflow").dropna().stack(future_stack=True).melt(ignore_index=False).reset_index()
)

fig, ax = plt.subplots(figsize=(12, 8))

pkwargs = dict(
    data=sns_df,
    y="value",
    x="variable",
    order=variables,
    hue="workflow",
    ax=ax,
)
sns.boxplot(**pkwargs, **box_kwargs)
sns.swarmplot(**pkwargs, **swarm_kwargs)

sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.axhline(0.0, color="tab:gray", zorder=1.2)
ax.grid()


# %%
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
sns_df["scenario_group"] = sns_df["scenario"].apply(get_scenario_group)
mosaic = [
    ["VLLO", "VLHO"],
    # ["L",
    # "ML"],
    # ["M",
    # "H"],
]

for model, mdf in sns_df[sns_df["scenario_group"].isin([vv for v in mosaic for vv in v])].groupby("model"):
    for variables in [["Peak warming 33.0", "Peak warming 50.0"], ["EOC warming 50.0"]]:
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 4), sharey=False)

        for sg, sgdf in mdf.groupby("scenario_group"):
            pkwargs = dict(
                data=sgdf[sgdf["variable"].isin(variables)],
                y="value",
                x="variable",
                order=variables,
                hue="workflow",
                hue_order=["default", "co2_afolu_2050", "all_2050"],
                ax=axes[sg],
            )
            ax = sns.boxplot(**pkwargs, **box_kwargs)
            ax = sns.swarmplot(**pkwargs, **swarm_kwargs)

            ax.set_title(sg)

        fig.suptitle(model)
        fig.tight_layout()
        plt.show()

# %%
box_kwargs = dict(saturation=0.3, legend=False)
swarm_kwargs = dict(dodge=True)

variables = ["Peak warming 33.0", "Peak warming 50.0", "EOC warming 50.0"]
tmp = metadata[variables].unstack("workflow").dropna()
tmp = tmp.swaplevel(0, 1, axis="columns")
tmp.columns.names = [*tmp.columns.names[:-1], "variable"]
tmp = tmp.stack(future_stack=True)
deltas_l = []
for other in ["co2_afolu_2050", "all_2050"]:
    tmp_h = tmp[other] - tmp["default"]
    tmp_h.name = f"switch_to_{other}"
    deltas_l.append(tmp_h)

sns_df = pix.concat(deltas_l, axis="columns").melt(ignore_index=False, var_name="harmonisation_change").reset_index()

fig, ax = plt.subplots(figsize=(12, 8))

pkwargs = dict(
    data=sns_df,
    y="value",
    x="variable",
    order=variables,
    hue="harmonisation_change",
    ax=ax,
)
sns.boxplot(**pkwargs, **box_kwargs)
sns.swarmplot(**pkwargs, **swarm_kwargs)

sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.axhline(0.0, color="tab:gray", zorder=1.2)
ax.grid()

# %% [markdown]
# ## Timeseries by model

# %%
tmp_l = []
for key, out_path in out_paths_emissions.items():
    for stage, file in (
        ("pre-processed", "pre-processed.csv"),
        ("harmonised", "harmonised.csv"),
        ("infilled", "infilled.csv"),
    ):
        loaded = load_timeseries_csv(
            out_path / file,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_column_type=int,
        ).pix.assign(stage=stage, workflow=key)
        tmp_l.append(loaded)

tmp = pix.concat(tmp_l)

# %%
sns_df = tmp.melt(ignore_index=False, var_name="year").reset_index()
sns_df["ms"] = sns_df["model"] + sns_df["scenario"]
sns_df

# %%
for model, mdf in tqdm(sns_df.groupby("model")):
    # if "GCAM" not in model:
    #     continue

    fg = sns.relplot(
        data=mdf,
        x="year",
        y="value",
        col="variable",
        col_wrap=3,
        col_order=sorted(mdf["variable"].unique()),
        hue="workflow",
        style="stage",
        dashes={
            "pre-processed": (3, 3),
            "infilled": (1, 1),
            "harmonised": "",
        },
        units="ms",
        estimator=None,
        facet_kws=dict(sharey=False),
        kind="line",
        alpha=0.7,
    )
    for ax in fg.figure.axes:
        ax.grid()
        if "CO2" not in ax.get_title():
            ax.set_ylim(ymin=0)

    fg.figure.suptitle(model, y=1.01)
    plt.show()
    # break
