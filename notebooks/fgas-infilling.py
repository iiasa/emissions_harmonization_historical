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

# %% editable=true slideshow={"slide_type": ""}
import matplotlib.pyplot as plt
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns

from emissions_harmonization_historical.constants_5000 import (
    INFILLED_SCENARIOS_DB,
    INFILLING_DB,
    MARKERS_BY_SCENARIOMIP_NAME,
)

# %%
pandas_openscm.register_pandas_accessor()

# %%
complete_scenarios = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete"))
# complete_scenarios

# %%
plotting_df_l = []
markers_of_interest = ["vl", "ln"]
for marker, info in MARKERS_BY_SCENARIOMIP_NAME.items():
    if markers_of_interest is not None and marker not in markers_of_interest:
        continue

    locator = pix.isin(model=info["model"], scenario=info["scenario"])
    marker_df = complete_scenarios.loc[locator]
    if marker_df.empty:
        continue

    plotting_df_l.append(marker_df.pix.assign(source=marker).reset_index("stage", drop=True))

# %% editable=true slideshow={"slide_type": ""}
infilling_db = INFILLING_DB.load()
infilling_db

# %%
velders_locator = pix.ismatch(model="Velders*")
plotting_df_l.append(
    infilling_db.loc[~velders_locator].openscm.update_index_levels_from_other({"source": ("model", lambda x: x)})
)
velders_scenario = infilling_db.loc[velders_locator & pix.ismatch(scenario="Kigali*lower")].pix.assign(
    source="infilling_db_velders"
)

# %% editable=true slideshow={"slide_type": ""}
plotting_df = pix.concat(plotting_df_l)
plotting_df

# %%
import pandas as pd


# TODO: put something like this in openscm
def get_variable_relation_df(
    df: pd.DataFrame,
    variable_level: str = "variable",
    unit_level: str = "unit",
    variable_unit_level: str = "variable_unit",
) -> tuple[pd.DataFrame, str, str]:
    variable_unit = df.index.droplevel(df.index.names.difference([variable_level, unit_level])).drop_duplicates()
    if variable_unit.shape[0] != 2:
        msg = "Unit conversion required"
        display(variable_unit)
        raise AssertionError(msg)

    res = df.pix.format(variable_unit=f"{{{variable_level}}} ({{{unit_level}}})", drop=True)
    x_l = [v for v in res.pix.unique(variable_unit_level) if v.startswith(lead)]
    if len(x_l) != 1:
        raise AssertionError(x_l)
    x = x_l[0]
    y_l = [v for v in res.pix.unique(variable_unit_level) if v.startswith(g)]
    if len(y_l) != 1:
        raise AssertionError(y_l)
    y = y_l[0]

    res = res.stack().unstack(variable_unit_level)

    return res, x, y


# %%
plotting_df.pix.unique("variable")

# %%
marker_colours = {
    "vl": "#499edb",
    "ln": "#4b3d89",
    "l": "#f7a84f",
    "ml": "#e1ad01",
    "m": "#2e9e68",
    "hl": "#800080",
    "h": "#7f3e3e",
}

model_colours = {
    "GCAM 8s": "tab:blue",
    "AIM 3.0": "tab:red",
    "WITCH 6.0": "tab:pink",
    "IMAGE 3.4": "tab:brown",
    "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12": "tab:blue",
}

palette = {**marker_colours, **model_colours}

# %%
gases_to_plot = [
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC125",
    "Emissions|HFC|HFC23",
    "Emissions|SF6",
    "Emissions|CF4",
    "Emissions|HFC|HFC32",
    "Emissions|C2F6",
]

# %%
pdf = plotting_df.loc[pix.isin(variable=gases_to_plot)]
pdf = pdf.openscm.set_index_levels(
    {
        "ms": [
            f"{model} -- {scenario}"
            for model, scenario in pdf.index.droplevel(pdf.index.names.difference(["model", "scenario"]))
        ]
    }
).openscm.to_long_data()

sizes = {v: 0.25 if v not in MARKERS_BY_SCENARIOMIP_NAME else 4 for v in pdf["source"].unique()}
sns.relplot(
    data=pdf,
    y="value",
    x="time",
    col="variable",
    col_order=gases_to_plot,
    col_wrap=3,
    hue="source",
    size="source",
    sizes=sizes,
    palette=palette,
    kind="line",
    units="ms",
    estimator=None,
    facet_kws=dict(sharey=False),
)

# %% editable=true slideshow={"slide_type": ""}
lead = "Emissions|CO2|Energy and Industrial Processes"
years_to_plot = [2023, 2025, 2030, 2050, 2100]

for g in gases_to_plot:
    print(g)
    pdf, x, y = get_variable_relation_df(plotting_df.loc[pix.isin(variable=[lead, g]), years_to_plot])
    # Drop anything which can't be used for infilling this gas
    pdf = pdf.dropna()

    sizes = {v: 45 if v not in MARKERS_BY_SCENARIOMIP_NAME else 200 for v in pdf.pix.unique("source")}

    fg = sns.relplot(
        data=pdf,
        x=x,
        y=y,
        hue="source",
        palette=palette,
        size="source",
        sizes=sizes,
        col="year",
        col_wrap=3,
        alpha=0.7,
        facet_kws=dict(sharex=False, sharey=False),
    )
    for ax in fg.axes.flatten():
        ax.set_ylim(ymin=0)

    plt.show()
