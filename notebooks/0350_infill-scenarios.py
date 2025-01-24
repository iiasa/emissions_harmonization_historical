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
# # Infill - silicone
#
# Here we infill the variables we can using [silicone]().

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import pyam
import silicone.database_crunchers
import tqdm.autonotebook as tqdman

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    INFILLING_SILICONE_ID,
)
from emissions_harmonization_historical.io import load_csv

# %%
SCENARIO_TIME_ID = "20250122-140031"

# %%
harmonised_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "harmonised"
    / f"harmonised-scenarios_{SCENARIO_TIME_ID}_{HARMONISATION_ID}.csv"
)
harmonised_file

# %%
out_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "interim"
    / f"infilled-silicone_{SCENARIO_TIME_ID}_{INFILLING_SILICONE_ID}.csv"
)
out_file

# %%
harmonised = load_csv(harmonised_file)

# %%
# TODO: undo this once we have WITCH data that makes sense
harmonised = harmonised.loc[:, :2100]

# %%
all_iam_variables = harmonised.pix.unique("variable")
all_iam_variables

# %%
variables_to_infill_l = []
for (model, scenario), msdf in harmonised.groupby(["model", "scenario"]):
    to_infill = all_iam_variables.difference(msdf.pix.unique("variable"))
    variables_to_infill_l.extend(to_infill.tolist())

variables_to_infill = set(variables_to_infill_l)
variables_to_infill

# %%
# TODO: think some of these through a bit more
lead_vars_crunchers = {
    "Emissions|BC": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|C2F6": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|C6F14": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|CF4": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|CO": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|HFC|HFC125": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC134a": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC143a": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC227ea": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC23": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC245fa": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC32": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC43-10": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|NH3": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|NOx": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|OC": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|SF6": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|Sulfur": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|VOC": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
}

# %%
infillers = {}
for v_infill in tqdman.tqdm(variables_to_infill):
    leader, cruncher = lead_vars_crunchers[v_infill]

    v_infill_db = harmonised.loc[pix.isin(variable=[v_infill, leader])]
    infillers[v_infill] = cruncher(pyam.IamDataFrame(v_infill_db)).derive_relationship(
        variable_follower=v_infill,
        variable_leaders=[leader],
    )
    # break

# %%
infilled_l = []
for (model, scenario), msdf in tqdman.tqdm(harmonised.groupby(["model", "scenario"])):
    to_infill = all_iam_variables.difference(msdf.pix.unique("variable"))
    if to_infill.empty:
        continue

    msdf_infilled_l = []
    for v_infill in to_infill:
        infillers[v_infill](pyam.IamDataFrame(msdf))
        msdf_infilled_l.append(infillers[v_infill](pyam.IamDataFrame(msdf)).timeseries())
        # break

    msdf_infilled = pix.concat(msdf_infilled_l)
    infilled_l.append(msdf_infilled)
    # break

infilled = pix.concat(infilled_l)
infilled


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
for v in variables_to_infill:
    # if not v.endswith("HFC134a"):
    #     continue

    leader, cruncher = lead_vars_crunchers[v]

    fig, axes = plt.subplot_mosaic(
        [
            ["lead_all", "follow_all"],
            ["lead_infilled", "follow_infilled"],
        ],
        figsize=(12, 8),
    )

    background_pkwargs = dict(
        legend=False,
        color="tab:gray",
        linewidth=0.5,
    )
    foreground_pkwargs = dict(
        linewidth=3,
        zorder=3,
    )

    def prep_for_plot(indf: pd.DataFrame) -> pd.DataFrame:
        """
        Prep for plot
        """
        return indf.dropna(axis="columns", how="all")

    prep_for_plot(harmonised.loc[pix.isin(variable=[leader])]).reset_index(drop=True).T.interpolate(
        method="index"
    ).plot(ax=axes["lead_all"], **background_pkwargs)

    harmonised.loc[pix.isin(variable=[v])].reset_index(drop=True).T.interpolate(method="index").plot(
        ax=axes["follow_all"], **background_pkwargs
    )

    infilled_lines = (
        prep_for_plot(infilled.loc[pix.isin(variable=[v])])
        .reset_index(infilled.index.names.difference(["model", "scenario"]), drop=True)
        .T.interpolate(method="index")
    )
    infilled_lines.plot(ax=axes["follow_all"], legend=False, **foreground_pkwargs)

    harmonised_infilled_scenarios = (
        prep_for_plot(harmonised[harmonised.index.isin(infilled_lines.T.index)].loc[pix.isin(variable=[leader])])
        .reset_index(infilled.index.names.difference(["model", "scenario"]), drop=True)
        .T.interpolate(method="index")
    )
    harmonised_infilled_scenarios.plot(ax=axes["lead_all"], legend=False, **foreground_pkwargs)

    harmonised_infilled_scenarios.plot(ax=axes["lead_infilled"], **foreground_pkwargs)

    infilled_lines.plot(ax=axes["follow_infilled"], legend=False, alpha=0.7, **foreground_pkwargs)
    axes["lead_infilled"].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2))

    axes["lead_all"].set_title(leader)
    axes["follow_all"].set_title(v)
    plt.show()

    # break

# %%
out_file.parent.mkdir(exist_ok=True, parents=True)
infilled.to_csv(out_file)
out_file
