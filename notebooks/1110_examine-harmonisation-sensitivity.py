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

from emissions_harmonization_historical.constants import DATA_ROOT, SCENARIO_TIME_ID, WORKFLOW_ID

# %%
out_paths = {
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

# %%
metadata_l = []
for key, out_path in out_paths.items():
    key_metadata = pd.read_csv(out_path / "metadata.csv").set_index(["model", "scenario"])
    metadata_l.append(key_metadata.pix.assign(workflow=key))

metadata = pix.concat(metadata_l)
metadata

# %%
box_kwargs = dict(saturation=0.3, legend=False)
swarm_kwargs = dict(dodge=True)

variables = ["Peak warming 33.0", "Peak warming 50.0", "EOC warming 50.0"]
sns_df = metadata[variables].melt(ignore_index=False).reset_index()

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
                ax=axes[sg],
            )
            ax = sns.boxplot(**pkwargs, **box_kwargs)
            ax = sns.swarmplot(**pkwargs, **swarm_kwargs)

            ax.set_title(sg)

        fig.suptitle(model)
        fig.tight_layout()
        plt.show()
