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
# # Explore results
#
# Here we explore the results.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from pandas_openscm.indexing import multi_index_lookup

from emissions_harmonization_historical.constants_5000 import (
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSED_TIMESERIES_RUN_ID_DB,
    SCM_OUTPUT_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pd.set_option("display.max_rows", 100)

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Categories

# %%
categories = POST_PROCESSED_METADATA_CATEGORIES_DB.load()["value"]
# categories

# %%
metadata_quantile = POST_PROCESSED_METADATA_QUANTILE_DB.load()["value"]
# metadata_quantile

# %%
temperatures_in_line_with_assessment = POST_PROCESSED_TIMESERIES_RUN_ID_DB.load(
    pix.isin(variable="Surface Temperature (GSAT)")
)
# temperatures_in_line_with_assessment

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Are any scenarios close to C1?

# %%
categories.unstack(["metric", "climate_model"])["category_name"].sort_values(by=["MAGICCv7.6.0a3"])

# %%
tmp = (
    metadata_quantile.unstack(["metric", "unit", "quantile"])
    .loc[
        :,
        [
            (metric, "K", p)
            for (metric, p) in [
                ("max", 0.33),
                ("max", 0.5),
                ("max", 0.67),
                ("2100", 0.5),
            ]
        ],
    ]
    .reset_index("region", drop=True)
    .reorder_levels(["model", "scenario", "variable", "climate_model"])
    .sort_values(by=[("max", "K", 0.33)])
    .round(3)
)

tmp.loc[
    # (tmp[("max", "K", 0.33)] > 0.0) & (tmp[("max", "K", 0.33)] < 1.6)
    (tmp[("max", "K", 0.67)] > 1.8) & (tmp[("max", "K", 0.67)] < 2.05)  # noqa: PLR2004
    # (tmp[("2100", "K", 0.5)] > 1.7)
    # & (tmp[("2100", "K", 0.5)] < 2.0)
    # (tmp[("2100", "K", 0.5)] > 2.1)
    # & (tmp[("2100", "K", 0.5)] < 2.5)
    # # (tmp[("2100", "K", 0.5)] > 2.5)
    # # & (tmp[("2100", "K", 0.5)] < 3.0)
    # # (tmp[("2100", "K", 0.5)] > 3.0) & (tmp[("2100", "K", 0.5)] < 3.6)
]

# %%
tmp = (
    metadata_quantile.unstack(["metric", "unit", "quantile"])
    .loc[
        :,
        [
            ("max", "K", 0.5),
            ("max_year", "yr", 0.5),
        ],
    ]
    .reset_index("region", drop=True)
    .reorder_levels(["model", "scenario", "variable", "climate_model"])
    .sort_values(by=[("max", "K", 0.5)])
    .round(3)
)
tmp

# %% [markdown]
# ## Exploratory marker selection
#
# You'd probably want to look at emissions too
# before making such a decision,
# but as an exploration here is something.

# %%
scratch_selection_l = [
    # ("#7f3e3e", ("REMIND-MAgPIE 3.5-4.10", "SSP3 - High Emissions")),
    # ("#7f3e3e", ("GCAM 7.1 scenarioMIP", "SSP5 - High Emissions")),
    # ("#7f3e3e", ("IMAGE 3.4", "SSP5 - High Emissions")),
    ("#7f3e3e", ("AIM 3.0", "SSP5 - High Emissions")),
    # ("#f7a84f", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Medium Emissions")),
    # ("#f7a84f", ("GCAM 7.1 scenarioMIP", "SSP2 - Medium Emissions")),
    # ("#f7a84f", ("IMAGE 3.4", "SSP2 - Medium Emissions")),
    # ("#f7a84f", ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Medium Emissions")),
    ("#f7a84f", ("WITCH 6.0", "SSP2 - Medium Emissions")),
    # ("#e1ad01", ("REMIND-MAgPIE 3.5-4.10", "SSP3 - Medium-Low Emissions")),
    # ("#e1ad01", ("COFFEE 1.6", "SSP2 - Medium-Low Emissions")),
    ("#e1ad01", ("GCAM 7.1 scenarioMIP", "SSP2 - Medium-Low Emissions")),
    # ("#586643", ("REMIND-MAgPIE 3.5-4.10", "SSP1 - Medium-Low Emissions")),
    # ("#2e9e68", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Low Emissions")),
    # ("#2e9e68", ("COFFEE 1.6", "SSP2 - Low Emissions")),
    ("#2e9e68", ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Emissions")),
    ("#4b3d89", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Low Overshoot_d")),
    # ("#4b3d89", ("GCAM 7.1 scenarioMIP", "SSP2 - Low Overshoot")),
    # ("#499edb", ("REMIND-MAgPIE 3.5-4.10", "SSP1 - Very Low Emissions_c")),
    ("#499edb", ("COFFEE 1.6", "SSP2 - Very Low Emissions")),
    # ("#499edb", ("WITCH 6.0", "SSP2 - Low Overshoot")),
]

# %%
ms_separator = " || "
palette = {ms_separator.join(v[1]): v[0] for v in scratch_selection_l}
scratch_selection = pd.MultiIndex.from_tuples([v[1] for v in scratch_selection_l], names=["model", "scenario"])
pdf = temperatures_in_line_with_assessment.loc[pix.isin(climate_model="MAGICCv7.6.0a3"), 2000:].openscm.mi_loc(
    scratch_selection
)

ms_level = "model || scenario"

# Push ability to create a new level from multiple other levels into pandas-openscm
new_name = ms_level
new_level = (
    pdf.index.droplevel(pdf.index.names.difference(["model", "scenario"]))
    .drop_duplicates()
    .map(lambda x: ms_separator.join(x))
)

if new_level.shape[0] != pdf.shape[0]:
    dup_level = pdf.index.get_level_values("model") + " || " + pdf.index.get_level_values("scenario")
    new_level = dup_level.unique()
    new_codes = new_level.get_indexer(dup_level)
else:
    new_codes = np.arange(new_level.shape[0])

pdf.index = pd.MultiIndex(
    levels=[*pdf.index.levels, new_level],
    codes=[*pdf.index.codes, new_codes],
    names=[*pdf.index.names, new_name],
)


# %%
def create_legend(ax, handles) -> None:
    """Create legend helper"""
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.0))


fig, axes = plt.subplots(nrows=2, figsize=(4, 8), sharex=True)
for i, (ax, yticks) in enumerate(zip(axes, [np.arange(0.5, 4.01, 0.5), np.arange(0.7, 2.21, 0.1)])):
    pdf.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        hue_var=ms_level,
        style_var="climate_model",
        palette=palette,
        quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.3), ((0.05, 0.95), 0.0)),
        ax=ax,
        create_legend=(lambda x, y: None) if i > 0 else create_legend,
    )
    ax.set_xlim([2000, 2100])
    ax.set_yticks(yticks)
    ax.set_ylim(ymin=yticks.min(), ymax=yticks.max())
    # ax.set_ylim(ymax=ymax)
    ax.grid()

# %%
scm_emissions = SCM_OUTPUT_DB.load(pix.ismatch(variable="Emissions|**"))
# scm_emissions
pdf = scm_emissions.openscm.mi_loc(scratch_selection)

# Push ability to create a new level from multiple other levels into pandas-openscm
new_name = ms_level
new_level = (
    pdf.index.droplevel(pdf.index.names.difference(["model", "scenario"]))
    .drop_duplicates()
    .map(lambda x: ms_separator.join(x))
)

if new_level.shape[0] != pdf.shape[0]:
    dup_level = pdf.index.get_level_values("model") + " || " + pdf.index.get_level_values("scenario")
    new_level = dup_level.unique()
    new_codes = new_level.get_indexer(dup_level)
else:
    new_codes = np.arange(new_level.shape[0])

pdf.index = pd.MultiIndex(
    levels=[*pdf.index.levels, new_level],
    codes=[*pdf.index.codes, new_codes],
    names=[*pdf.index.names, new_name],
)
pdf = pdf.loc[pix.ismatch(variable=["**CO2**", "**CH4", "**BC", "**Sulfur", "**OC", "**VOC"])]

pdf = pdf.pix.format(variable="{variable} ({unit})", drop=True).openscm.to_long_data()
# pdf

# %%
fg = sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="model || scenario",
    palette=palette,
    col="variable",
    col_wrap=3,
    col_order=sorted(pdf["variable"].unique()),
    facet_kws=dict(sharey=False),
    kind="line",
)
for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="gray")
    else:
        ax.set_ylim(ymin=0.0)

    ax.grid()

# %%
multi_index_lookup(categories, scratch_selection).unstack(["metric", "climate_model"]).sort_values(
    ("category", "MAGICCv7.6.0a3")
).sort_index(axis="columns")

# %%
multi_index_lookup(metadata_quantile, scratch_selection).unstack(["metric", "unit", "quantile"]).loc[
    :,
    [
        (metric, "K", p)
        for (metric, p) in [
            ("max", 0.33),
            ("max", 0.5),
            ("max", 0.67),
            ("2100", 0.5),
        ]
    ],
].reset_index("region", drop=True).reorder_levels(["model", "scenario", "variable", "climate_model"]).sort_values(
    by=[("max", "K", 0.33)]
).round(3)

# %%
multi_index_lookup(metadata_quantile, scratch_selection).unstack(["metric", "unit", "quantile"])["max_year"]["yr"][0.5]

# %% [markdown]
# ### How much difference is the MAGICC update making?

# %%
iam = "AIM"
tmp = temperatures_in_line_with_assessment.loc[pix.ismatch(model=f"{iam}**"), 2000:]

fig, axes = plt.subplots(ncols=2, figsize=(16, 4))


def create_legend(ax, lh):
    """Create legend for our plots"""
    ax.legend(handles=lh, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)


for i, (ax, quantile_plumes) in enumerate(
    zip(axes, [[(0.33, 0.9), ((0.05, 0.95), 0.4)], [(0.5, 0.9), ((0.33, 0.67), 0.4)]])
):
    tmp.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        style_var="climate_model",
        quantiles_plumes=quantile_plumes,
        ax=ax,
        create_legend=create_legend,
    )
    ax.set_yticks(np.arange(0.5, 5.01, 0.5))
    ax.axhline(1.5, linestyle="--", color="k")
    ax.axhline(2.0, linestyle="--", color="k")
    ax.grid()

plt.show()

magicc_diff = (
    tmp.stack().unstack("climate_model")["MAGICCv7.6.0a3"] - tmp.stack().unstack("climate_model")["MAGICCv7.5.3"]
).unstack("time")
ax = magicc_diff.openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id",
)
ax.grid()
plt.show()

# %%
tmp = metadata_quantile.unstack("climate_model")
magicc_diff = tmp["MAGICCv7.6.0a3"] - tmp["MAGICCv7.5.3"]
magicc_diff.unstack(["metric", "unit", "quantile"])[
    [(metric, "K", percentile) for metric in ["max", "2100"] for percentile in [0.33, 0.5, 0.67, 0.95]]
].sort_values(by=("max", "K", 0.5)).describe().round(3)
