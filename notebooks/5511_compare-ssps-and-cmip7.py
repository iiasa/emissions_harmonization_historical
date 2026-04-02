# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare SSPs and CMIP7 scenarios

# %% [markdown]
# ## Imports

# %%
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name

from emissions_harmonization_historical.constants_5000 import (
    AR6_LIKE_SCM_OUTPUT_DB,
    MARKERS,
    POST_PROCESSED_TIMESERIES_RUN_ID_DB,
    SCM_OUTPUT_DB,
)

# %%
pandas_openscm.register_pandas_accessor()

# %% [markdown]
# ## General set up

# %%
model_scenario_cmip_name = [
    ("IMAGE", "ssp119", "ssp119"),
    ("IMAGE", "ssp126", "ssp126"),
    ("MESSAGE-GLOBIOM", "ssp245", "ssp245"),
    ("GCAM4", "ssp434", "ssp434"),
    ("GCAM4", "ssp460", "ssp460"),
    ("AIM/CGE", "ssp370", "ssp370"),
    ("REMIND-MAGPIE", "ssp534-over", "ssp534-over"),
    ("REMIND-MAGPIE", "ssp585", "ssp585"),
    *[
        (v[0], v[1], v[2])
        for v in MARKERS
        # if "MESSAGE" in v[0]
    ],
]
model_scenario_cmip_name

# %% [markdown]
# ## Load data

# %%
scenarios_to_load = pd.MultiIndex.from_tuples(
    [(v[0], v[1]) for v in model_scenario_cmip_name], names=["model", "scenario"]
)
# scenarios_to_load

# %%
cmip7_output = pix.concat(
    [
        SCM_OUTPUT_DB.load(scenarios_to_load, progress=True),
        POST_PROCESSED_TIMESERIES_RUN_ID_DB.load(scenarios_to_load, progress=True),
    ]
)
ssps_output = AR6_LIKE_SCM_OUTPUT_DB.load(
    scenarios_to_load[scenarios_to_load.get_level_values("scenario").str.startswith("ssp")],
    progress=True,
)

# %%
cmip6_historical_warming = AR6_LIKE_SCM_OUTPUT_DB.load(
    pix.isin(variable="Surface Air Temperature Change", scenario="ssp585")
)
cmip6_historical_warming

# %%
cmip6_historical_warming_rel_pi = cmip6_historical_warming.subtract(
    cmip6_historical_warming.loc[:, 1850:1900].mean(axis="columns"), axis="rows"
)
shift = (
    cmip6_historical_warming_rel_pi.loc[:, 1995:2014]
    .mean(axis="columns")
    .groupby(cmip6_historical_warming_rel_pi.index.names.difference(["run_id"]))
    .median()
    - 0.85
)
cmip6_historical_warming_adjusted = cmip6_historical_warming_rel_pi.subtract(shift, axis="rows")
cmip6_historical_warming_adjusted.loc[:, 1995:2014].mean(axis="columns").groupby(
    cmip6_historical_warming_rel_pi.index.names.difference(["run_id"])
).median()

# %%
emms_locator = pix.ismatch(variable="Emissions**")

cmip7_emissions = cmip7_output.loc[emms_locator & pix.isin(climate_model="MAGICCv7.6.0a3")].reset_index(
    ["climate_model", "run_id"], drop=True
)
cmip7_emissions = cmip7_emissions.openscm.update_index_levels(
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    },
)
ssps_emissions = ssps_output.loc[emms_locator].reset_index(["climate_model", "run_id"], drop=True)
emissions = pix.concat([cmip7_emissions, ssps_emissions])

cmip7_scm_output = cmip7_output.loc[~emms_locator]
ssps_scm_output = ssps_output.loc[~emms_locator]
scm_output = pix.concat([ssps_scm_output, cmip7_scm_output])

# %% [markdown]
# ## Add relevant metadata

# %%
peak_temp_median = (
    scm_output.loc[pix.isin(variable="Surface Temperature (GSAT)", climate_model="MAGICCv7.6.0a3"),]
    .max(axis="columns")
    .groupby(scm_output.index.names.difference(["run_id"]))
    .median()
)


def peak_temp_median_to_group(pt: float) -> str:
    if pt < 2.0:
        return "peak-temp<2"

    if pt < 3.0:
        return "peak-temp<3"

    return "peak-temp>=3"


group_series = peak_temp_median.map(peak_temp_median_to_group)
group_series = group_series.droplevel(group_series.index.names.difference(["model", "scenario"]))
group_series.sort_index()

# %%
cmip_name_map = {(v[0], v[1]): v[2] for v in MARKERS}


def get_cmip_name(x):
    model, scenario = x
    if scenario.startswith("ssp"):
        return scenario

    return cmip_name_map[(model, scenario)]


cmip_scenario_name_series = pd.Series(
    group_series.index.map(get_cmip_name),
    index=group_series.index,
)

# %%
# # Only works with newer pandas-openscm
# emissions.openscm.update_index_levels_from_other(
#     {
#         "group": (tuple(group_series.index.names), group_series)
#     }
# )
emissions = emissions.reorder_levels(["model", "scenario", "variable", "region", "unit"])
emissions["group"] = group_series
emissions = emissions.set_index("group", append=True)
emissions["cmip_name"] = cmip_scenario_name_series
emissions = emissions.set_index("cmip_name", append=True)
emissions = emissions.openscm.update_index_levels_from_other(
    {"cmip_era": ("scenario", lambda x: "CMIP6" if x.startswith("ssp") else "CMIP7")}
)
emissions

scm_output = scm_output.reorder_levels(["model", "scenario", "variable", "region", "unit", "climate_model", "run_id"])
scm_output["group"] = group_series
scm_output = scm_output.set_index("group", append=True)
scm_output["cmip_name"] = cmip_scenario_name_series
scm_output = scm_output.set_index("cmip_name", append=True)
scm_output = scm_output.openscm.update_index_levels_from_other(
    {"cmip_era": ("scenario", lambda x: "CMIP6" if x.startswith("ssp") else "CMIP7")}
)
scm_output

# %% [markdown]
# ## Plot emissions

# %%
palette = {
    # "history": "k",
    # "history-cmip6": "tab:grey",
    "historical": "k",
    "historical-cmip6": "tab:grey",
    "vl": "#24a4ff",
    "ln": "#4a0daf",
    "l": "#00cc69",
    "ml": "#f5ac00",
    "m": "#ffa9dc",
    "h": "#700000",
    "hl": "#8f003b",
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274ae",
    "ssp460": "#b0724e",
    "ssp534-over": "#92397a",
    "ssp585": "#980002",
}

scenario_order = [
    "h",
    "hl",
    "m",
    "ml",
    "l",
    "ln",
    "vl",
    "ssp585",
    "ssp370",
    "ssp460",
    "ssp434",
    "ssp534-over",
    "ssp245",
    "ssp126",
    "ssp119",
    "historical",
    "historical-cmip6",
]

# %%
variables_to_plot = [
    "Emissions|CO2|Fossil",
    "Emissions|CO2|Biosphere",
    "Emissions|CH4",
    "Emissions|N2O",
    "Emissions|CFC12",
    "Emissions|SOx",
    "Emissions|OC",
    "Emissions|BC",
]
pdf = emissions.loc[pix.isin(variable=variables_to_plot), 2015:2100]
fg = sns.relplot(
    data=pdf.openscm.to_long_data(),
    x="time",
    y="value",
    hue="cmip_name",
    palette=palette,
    hue_order=scenario_order,
    style="cmip_era",
    dashes={"CMIP6": (3, 3), "CMIP7": ""},
    col="group",
    col_order=["peak-temp<2", "peak-temp<3", "peak-temp>=3"],
    row="variable",
    kind="line",
    # facet_kws=dict(sharey="row"),
    facet_kws=dict(sharey=False),
)

for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, color="gray", linestyle="--")
    else:
        ax.set_ylim(ymin=0.0)

# %%
variables_to_plot = [
    "Surface Temperature (GSAT)",
    "Effective Radiative Forcing",
]
pdf = scm_output.loc[pix.isin(variable=variables_to_plot, climate_model="MAGICCv7.6.0a3"), 2015:2100]
pdf = pdf.openscm.groupby_except("run_id").median()
fg = sns.relplot(
    data=pdf.openscm.to_long_data(),
    x="time",
    y="value",
    hue="cmip_name",
    palette=palette,
    hue_order=scenario_order,
    style="cmip_era",
    dashes={"CMIP6": (3, 3), "CMIP7": ""},
    col="group",
    col_order=["peak-temp<2", "peak-temp<3", "peak-temp>=3"],
    row="variable",
    kind="line",
    # facet_kws=dict(sharey="row"),
    facet_kws=dict(sharey=False),
)

# %%
scm_output.columns = scm_output.columns.astype(int)

fig, axes_d = plt.subplot_mosaic(
    [
        [
            "Effective Radiative Forcing--peak-temp<2",
            "Effective Radiative Forcing--peak-temp<3",
            "Effective Radiative Forcing--peak-temp>=3",
        ],
        [
            "Surface Temperature (GSAT)--peak-temp<2",
            "Surface Temperature (GSAT)--peak-temp<3",
            "Surface Temperature (GSAT)--peak-temp>=3",
        ],
    ],
    figsize=(12, 8),
)

for i, (ax_name, ax) in enumerate(axes_d.items()):
    variable, group = ax_name.split("--")
    scm_output.loc[
        pix.isin(variable=variable, group=group, climate_model="MAGICCv7.6.0a3"), 2015:2100
    ].openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.3)),
        hue_var="cmip_name",
        palette=palette,
        style_var="cmip_era",
        ax=ax,
    )
    # openscm facet based plotting would be handy right now to do the legend better
    # if i < len(axes_d) - 1:
    ax.get_legend().remove()

    ax.set_title(f"{variable} | {group}", fontsize="small")
    # break

plt.tight_layout()

# %%
scm_output = scm_output.sort_index(axis="columns")
scm_output.head(1)

# %%
cmip6_historical_warming_adjusted.head(1)


# %%
def create_legend(ax, lhs):
    lhs_d = {lh.get_label(): lh for lh in lhs}
    ax.legend(
        handles=[
            lhs_d["Quantile"],
            lhs_d["0.5"],
            lhs_d["0.05 - 0.95"],
            lhs_d["Scenario"],
            *[lhs_d[scen] for scen in scenario_order if scen in lhs_d],
        ],
        loc="upper left",
    )


climate_model = "MAGICCv7.6.0a3"

fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

for ax, (xlim, ylim) in zip(
    axes,
    (
        ((1950, 2100), (0, 6)),
        ((1850, 2500), (-0.2, 12)),
    ),
):
    pdf = pix.concat(
        [
            scm_output.loc[
                pix.isin(
                    variable="Surface Temperature (GSAT)",
                    climate_model=climate_model,
                )
                & pix.ismatch(scenario="ssp*"),
                2015:2500,
            ].reset_index(["group", "cmip_era"], drop=True),
            cmip6_historical_warming_adjusted.pix.assign(
                scenario="historical", variable="Surface Temperature (GSAT)", cmip_name="historical-cmip6"
            ).loc[
                pix.isin(
                    variable="Surface Temperature (GSAT)",
                    # group=group,
                    climate_model=climate_model,
                ),
                1750:2014,
            ],
        ]
    )
    ax = pdf.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.3)),
        hue_var="cmip_name",
        hue_var_label="Scenario",
        palette=palette,
        # style_var="cmip_era",
        # ax=ax,
        create_legend=create_legend,
        ax=ax,
    )
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


# %%
def create_legend(ax, lhs):
    lhs_d = {lh.get_label(): lh for lh in lhs}
    ax.legend(
        handles=[
            lhs_d["Quantile"],
            lhs_d["0.5"],
            lhs_d["0.05 - 0.95"],
            lhs_d["Scenario"],
            *[lhs_d[scen] for scen in scenario_order if scen in lhs_d],
        ],
        loc="upper left",
    )


climate_model = "MAGICCv7.6.0a3"

fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

for ax, (xlim, ylim) in zip(
    axes,
    (
        ((1950, 2100), (0, 6)),
        ((1850, 2500), (-2.0, 10)),
    ),
):
    pdf = pix.concat(
        [
            scm_output.loc[
                pix.isin(
                    variable="Surface Temperature (GSAT)",
                    climate_model=climate_model,
                )
                & ~pix.ismatch(scenario="ssp*"),
                2022:2500,
            ].reset_index(["group", "cmip_era"], drop=True),
            scm_output.loc[
                pix.isin(variable="Surface Temperature (GSAT)", climate_model=climate_model, cmip_name="h"),
                1750:2021,
            ]
            .reset_index(["group", "cmip_era"], drop=True)
            .pix.assign(cmip_name="historical"),
        ]
    )
    ax = pdf.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        quantiles_plumes=((0.5, 0.9), ((0.05, 0.95), 0.3)),
        hue_var="cmip_name",
        hue_var_label="Scenario",
        palette=palette,
        # style_var="cmip_era",
        # ax=ax,
        create_legend=create_legend,
        ax=ax,
    )
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

# %%
