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

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants_5000 import REPO_ROOT

# %%
pandas_openscm.register_pandas_accessor()

# %%
db_l = []
for src_folder, name in (
    (
        REPO_ROOT
        / "emissions-for-sharepoint"
        / "20250818_0004_0003_0002_0003_0002_0003_0002_0002_0002_0002_dc4de51f613de5e8f2f16b686106720316cfb8e1_0003_0003_0002_0002",  # noqa: E501
        "20250818",
    ),
    (
        REPO_ROOT / "emissions-for-sharepoint" / "0012-All-mz",
        "June",
    ),
):
    for stage in ["pre-processed", "harmonised"]:
        loaded = pd.concat(
            [
                load_timeseries_csv(
                    f,
                    lower_column_names=True,
                    index_columns=["model", "scenario", "region", "variable", "unit"],
                    out_columns_type=int,
                )
                for f in src_folder.rglob(f"{stage}*.csv")
            ]
        ).pix.assign(stage=stage, processing_date=name)

        db_l.append(loaded)

db = pd.concat(db_l)
db


# %%
def add_model_scenario_column(indf: pd.DataFrame, ms_separator: str, ms_level: str, copy: bool = True) -> pd.DataFrame:
    """
    Add a model-scenario column

    TODO: push this to pandas-openscm as something like
    `update_index_levels_multi_input`
    that allows users to updated index levels
    based on the value of multiple other index columns.
    """
    out = indf
    if copy:
        out = out.copy()

    # Push ability to create a new level from multiple other levels into pandas-openscm
    new_name = ms_level
    new_level = (
        indf.index.droplevel(out.index.names.difference(["model", "scenario"]))
        .drop_duplicates()
        .map(lambda x: ms_separator.join(x))
    )

    if new_level.shape[0] != indf.shape[0]:
        dup_level = out.index.get_level_values("model") + ms_separator + out.index.get_level_values("scenario")
        new_level = dup_level.unique()
        new_codes = new_level.get_indexer(dup_level)
    else:
        new_codes = np.arange(new_level.shape[0])

    out.index = pd.MultiIndex(
        levels=[*out.index.levels, new_level],
        codes=[*out.index.codes, new_codes],
        names=[*out.index.names, new_name],
    )

    return out


# %%
ms_separator = " || "
ms_level = "model || scenario"
# palette = {ms_separator.join(v[1]): v[0] for v in scratch_selection_l}
hue = "processing_date"
style = "stage"

# %%
emissions_to_plot = [
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|GHG AR6GWP100",
    "Emissions|CO2|AFOLU",
    "Cumulative Emissions|CO2",
    "Emissions|CH4",
    "Emissions|CFC12",
    "Emissions|N2O",
    "Emissions|Sulfur",
    "Emissions|CO",
    "Emissions|BC",
    "Emissions|OC",
    "Emissions|NOx",
    "Emissions|NH3",
    "Emissions|VOC",
]
pdf_emissions = add_model_scenario_column(
    db.loc[
        pix.isin(variable=emissions_to_plot)
        & pix.ismatch(model="MESSAGE*")
        & pix.ismatch(scenario="SSP2 - Low Emissions")
    ],
    ms_separator=ms_separator,
    ms_level=ms_level,
)
# pdf_emissions

# %%
ncols = 2
nrows = len(emissions_to_plot) // ncols + len(emissions_to_plot) % ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 5))
axes_flat = axes.flatten()

for i, variable_to_plot in enumerate(emissions_to_plot):
    ax = axes_flat[i]

    vdf = pdf_emissions.loc[pix.isin(variable=variable_to_plot)].openscm.to_long_data().dropna()
    if vdf.empty:
        print(f"No emissions for {variable_to_plot}")
        continue

    sns.lineplot(
        ax=ax,
        data=vdf,
        x="time",
        y="value",
        hue=hue,
        # palette=palette,
        style=style,
    )
    ax.set_title(variable_to_plot, fontdict=dict(fontsize="medium"))
    ax.set_xticks(np.arange(2010.0, 2101.0, 10.0))
    unit_l = vdf["unit"].unique().tolist()
    if len(unit_l) > 1:
        raise AssertionError(unit_l)

    ax.set_ylabel(unit_l[0])

    if i % 2:
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.legend().remove()

    ax.grid()
    # break

# %%
