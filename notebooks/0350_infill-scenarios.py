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
# - infill using silicone
# - should get us so that we have the same emissions from all IAMs
# - in follow up notebooks, infill the rest with:
#     - Velders stuff
#     - WMO stuff
#     - whatever else we need to use

# %%
import matplotlib.pyplot as plt
import pandas_indexing as pix
import pyam
import seaborn as sns
import silicone.database_crunchers
import tqdm.autonotebook as tqdman

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
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
lead_vars = {
    "Emissions|BC": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|C2F6": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|C6F14": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CF4": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CO": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC125": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC134a": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC143a": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC227ea": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC23": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC245fa": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC32": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|HFC|HFC43-10": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|NH3": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|NOx": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|OC": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|SF6": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|Sulfur": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|VOC": "Emissions|CO2|Energy and Industrial Processes",
}

# %%
infillers = {}
for v_infill in variables_to_infill:
    # if not v_infill.endswith("HFC23"):
    #     continue

    leader = lead_vars[v_infill]
    v_infill_db = harmonised.loc[pix.isin(variable=[v_infill, leader])]
    infillers[v_infill] = silicone.database_crunchers.QuantileRollingWindows(
        pyam.IamDataFrame(v_infill_db)
    ).derive_relationship(
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
pdf_all = pix.concat(
    [
        harmonised.pix.assign(stage="harmonised"),
        infilled.pix.assign(stage="infilled"),
    ]
)

for v in variables_to_infill:
    pdf = pdf_all.loc[pix.isin(variable=v)]

    sns_df = get_sns_df(pdf)
    sns_df["scenario_group"] = sns_df["scenario"].apply(lambda x: x.split(" - ")[-1].strip())
    sns_df = sns_df[sns_df["scenario_group"].isin(sns_df[sns_df["stage"] == "infilled"]["scenario_group"].unique())]

    fg = sns.relplot(
        sns_df,
        x="year",
        y="value",
        col="scenario_group",
        hue="model",
        style="stage",
        units="scenario",
        estimator=None,
        col_wrap=3,
        facet_kws=dict(sharey=False),
        kind="line",
    )
    fg.fig.suptitle(v, y=1.1)

    plt.show()
