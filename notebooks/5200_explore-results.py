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

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["papermill-error-cell-tag"]
# # Explore results
#
# Here we explore the results.

# %% [markdown]
# ## Imports

# %%
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import seaborn as sns
import tqdm.auto
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
pix.set_openscm_registry_as_default()

# %%
pd.set_option("display.max_rows", 100)

# %% [markdown]
# ## Load data

# %%
categories = POST_PROCESSED_METADATA_CATEGORIES_DB.load()["value"]
categories

# %%
metadata_quantile = POST_PROCESSED_METADATA_QUANTILE_DB.load()["value"]
metadata_quantile.unstack(["metric", "quantile", "unit"])[("max", 0.5)]

# %%
temperatures_in_line_with_assessment = POST_PROCESSED_TIMESERIES_RUN_ID_DB.load(
    pix.isin(variable="Surface Temperature (GSAT)")
)
# temperatures_in_line_with_assessment

# %%
# slr = SCM_OUTPUT_DB.load(
#     pix.ismatch(
#         variable=[
#             "Sea Level Rise",
#         ],
#         climate_model="MAGICCv7.6*"
#     ),
#     progress=True,
#     max_workers=multiprocessing.cpu_count(),
# )
# pdf = slr.openscm.groupby_except("run_id").median().loc[:, 2000:]
# ax = sns.lineplot(
#     data=pdf.openscm.to_long_data(),
#     x="time",
#     y="value",
#     hue="scenario",
#     style="model"
# )
# sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5), ncols=3)

# %%
raw_scm_output = SCM_OUTPUT_DB.load(
    pix.ismatch(
        variable=[
            "Heat Uptake",
            "Effective Radiative Forcing**",
            "Atmospheric Concentrations|CO2",
            "Atmospheric Concentrations|CH4",
            "Sea Level Rise",
        ]
    ),
    progress=True,
    max_workers=multiprocessing.cpu_count(),
)

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

vllo_peak = 1.66
l_min_diff = 0.1
l_max_diff = 0.4
tmp.loc[
    # :, :
    (tmp[("max", "K", 0.33)] > 0.0) & (tmp[("max", "K", 0.33)] < 6.6)  # noqa: PLR2004
    # ((tmp[("max", "K", 0.5)] > vllo_peak - 0.01) & (tmp[("max", "K", 0.5)] < vllo_peak + 0.01))
    # | ((tmp[("max", "K", 0.5)] > vllo_peak + l_min_diff) & (tmp[("max", "K", 0.5)] < vllo_peak + l_max_diff))
    # (tmp[("max", "K", 0.67)] > 1.8) & (tmp[("max", "K", 0.67)] < 2.05)
    # (tmp[("2100", "K", 0.5)] > 1.7)
    # & (tmp[("2100", "K", 0.5)] < 2.0)
    # (tmp[("2100", "K", 0.5)] > 2.2) & (tmp[("2100", "K", 0.5)] < 2.7)
    # # (tmp[("2100", "K", 0.5)] > 2.5)
    # # & (tmp[("2100", "K", 0.5)] < 3.0)
    # (tmp[("2100", "K", 0.5)] > 3.0) & (tmp[("2100", "K", 0.5)] < 30.6)
].loc[pix.ismatch(model=["REMIND*", "AIM*", "IMAGE*"], climate_model="MAGICCv7.6*")]

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
    # # # HL
    ("#7f3e3e", ("WITCH 6.0", "SSP5 - Medium-Low Emissions_a")),
    # # # H
    # # # ("#7f3e3e", ("REMIND-MAgPIE 3.5-4.10", "SSP3 - High Emissions")),
    ("#7f3e3e", ("GCAM 8s", "SSP3 - High Emissions")),
    # # ("#7f3e3e", ("IMAGE 3.4", "SSP3 - High Emissions")),
    # # ("#7f3e3e", ("WITCH 6.0", "SSP5 - High Emissions")),
    # # ("#7f3e3e", ("AIM 3.0", "SSP5 - High Emissions")),
    # # # M
    # # # ("#f7a84f", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Medium Emissions")),
    # # ("#f7a84f", ("GCAM 7.1 scenarioMIP", "SSP2 - Medium Emissions")),
    # # # ("#f7a84f", ("IMAGE 3.4", "SSP2 - Medium Emissions")),
    ("#f7a84f", ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Emissions")),
    # # ("#f7a84f", ("WITCH 6.0", "SSP2 - Medium Emissions")),
    # # # ML
    # # # ("#e1ad01", ("REMIND-MAgPIE 3.5-4.10", "SSP3 - Medium-Low Emissions")),
    # # # # Very high sulfur emissions, not ideal
    ("#e1ad01", ("COFFEE 1.6", "SSP2 - Medium-Low Emissions")),
    # # ("#e1ad01", ("GCAM 7.1 scenarioMIP", "SSP2 - Medium-Low Emissions")),
    # # # ("#586643", ("REMIND-MAgPIE 3.5-4.10", "SSP1 - Medium-Low Emissions")),
    # # L
    # # ("#2e9e68", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Low Emissions")),
    # # ("#2e9e68", ("COFFEE 1.6", "SSP2 - Low Emissions")),
    # # ("#2e9e68", ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Emissions")),
    ("#2e9e68", ("IMAGE 3.4", "SSP2 - Medium Emissions")),
    # # VLHO
    # # ("#4b3d89", ("REMIND-MAgPIE 3.5-4.10", "SSP2 - Low Overshoot_d")),
    ("#4b3d89", ("AIM 3.0", "SSP2 - Low Overshoot_a")),
    # # ("#4b3d89", ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Overshoot_a")),
    # # ("#4b3d89", ("GCAM 7.1 scenarioMIP", "SSP1 - Low Overshoot")),
    # # ("#4b3d89", ("GCAM 7.1 scenarioMIP", "SSP2 - Low Overshoot")),
    # # VLLO
    # # Not sure if any of models can produce a 1.5C stable rather than overshoot
    # # ("#899edb", ("COFFEE 1.6", "SSP2 - Very Low Emissions")),
    # # Something weird happening with 2023 emissions,
    # # probably from interpolation I would guess.
    # # Hopefully they can fix
    # # ("#899edb", ("WITCH 6.0", "SSP1 - Low Overshoot")),
    # # ("#899edb", ("WITCH 6.0", "SSP2 - Low Overshoot")),
    # # ("#899edb", ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP4 - Very Low Emissions")),
    ("#499edb", ("REMIND-MAgPIE 3.5-4.11", "SSP1 - Very Low Emissions")),
    # # VLLOD
    # # ("#499edb", ("AIM 3.0", "SSP1 - Very Low Emissions_a")),
    # # ("#499edb", ("AIM 3.0", "SSP1 - Very Low Emissions")),
    # # ("#499edb", ("GCAM 7.1 scenarioMIP", "SSP1 - Very Low Emissions")),
    # # ("#499edb", ("REMIND-MAgPIE 3.5-4.10", "SSP1 - Very Low Emissions")),
    # ## Scratch
    # ("#4b3d89", ("WITCH 6.0", "SSP1 - Low Overshoot")),
    # ("#899edb", ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions")),
    # ("#499edb", ("REMIND-MAgPIE 3.5-4.10", "SSP1 - Very Low Emissions_c")),
]


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
palette = {ms_separator.join(v[1]): v[0] for v in scratch_selection_l}
palette_model = {k.split(ms_separator)[0]: v for k, v in palette.items()}
scratch_selection = pd.MultiIndex.from_tuples([v[1] for v in scratch_selection_l], names=["model", "scenario"])
pdf_temperature = temperatures_in_line_with_assessment.loc[
    pix.isin(climate_model="MAGICCv7.6.0a3"), 2000:
].openscm.mi_loc(scratch_selection)

pdf_temperature = add_model_scenario_column(pdf_temperature, ms_separator=ms_separator, ms_level=ms_level)
# pdf_temperature


# %%
def create_legend(ax, handles) -> None:
    """Create legend helper"""
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.0))


hue = ms_level
palette_h = palette
# hue = "model"
# palette_h = palette_model

fig, axes = plt.subplots(nrows=2, figsize=(4, 8), sharex=True)
for i, (ax, yticks) in enumerate(zip(axes, [np.arange(0.5, 4.01, 0.5), np.arange(0.7, 2.21, 0.1)])):
    pdf_temperature.openscm.plot_plume_after_calculating_quantiles(
        quantile_over="run_id",
        hue_var=hue,
        style_var="climate_model",
        palette=palette_h,
        quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.75), ((0.05, 0.95), 0.0)),
        # quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.0), ((0.05, 0.95), 0.0)),
        ax=ax,
        create_legend=(lambda x, y: None) if i > 0 else create_legend,
    )
    ax.set_xlim([2000, 2100])
    ax.set_yticks(yticks)
    ax.set_ylim(ymin=yticks.min(), ymax=yticks.max())
    # ax.set_ylim(ymax=ymax)
    ax.grid()

# %%
pdf_emissions = add_model_scenario_column(
    SCM_OUTPUT_DB.load(pix.ismatch(variable="Emissions|**", climate_model="MAGICCv7.6.0a3"), progress=True).reset_index(
        "climate_model", drop=True
    ),
    ms_separator=ms_separator,
    ms_level=ms_level,
).sort_index(axis="columns")

gwp = "AR6GWP100"
with pint.get_application_registry().context(gwp):
    ghg_eq = pdf_emissions.loc[
        ~pix.ismatch(variable=[f"**|{v}" for v in ["BC", "OC", "Sulfur", "NOx", "NH3", "VOC", "CO"]])
    ].pix.convert_unit("MtCO2 / yr")

pdf_emissions = pix.concat(
    [
        pdf_emissions,
        ghg_eq.openscm.groupby_except("variable").sum().pix.assign(variable=f"Emissions|GHG {gwp}"),
        (
            pdf_emissions.loc[pix.ismatch(variable="**CO2|*")]
            .groupby(pdf_emissions.index.names.difference(["variable"]))
            .sum(min_count=2)
            .cumsum(axis=1)
            / 1e3
        ).pix.assign(variable="Cumulative Emissions|CO2", unit="GtCO2"),
    ]
)
pdf_emissions

# %%
pdf_emissions = pdf_emissions.pix.format(variable="{variable} ({unit})", drop=True).openscm.mi_loc(scratch_selection)

pdf = pdf_emissions.loc[pix.ismatch(variable=["**CO2**", "**CH4**", "**BC**", "**Sulfur**", "**OC**", "**VOC**"])]
pdf = pdf.openscm.to_long_data()
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
        if "Energy" in ax.get_title():
            ax.set_yticks(np.arange(-2e4, 7e4 + 1, 1e4))
        elif "GHG" in ax.get_title():
            ax.set_yticks(np.arange(-2e4, 7e4 + 1, 1e4))
        elif "Cumulative" in ax.get_title():
            ax.set_yticks(np.arange(0, 4e3 + 1, 1e3))
        else:
            ax.set_yticks(np.arange(-7e3, 10e3 + 1, 1e3))

    else:
        ax.set_ylim(ymin=0.0)

    ax.grid()

# %%
pdf_raw_scm_output = add_model_scenario_column(
    raw_scm_output.loc[pix.isin(climate_model="MAGICCv7.6.0a3"), 2000:].openscm.mi_loc(scratch_selection),
    ms_separator=ms_separator,
    ms_level=ms_level,
)

# pdf_raw_scm_output

# %%
pdf_emissions.loc[pix.ismatch(variable="Cumulative**")].max(axis=1)

# %%
variables_src = [
    ("Emissions|CO2|Energy and Industrial Processes", pdf_emissions, True, False),
    ("Emissions|GHG", pdf_emissions, True, False),
    ("Emissions|CO2|AFOLU", pdf_emissions, True, True),
    ("Cumulative Emissions|CO2", pdf_emissions, True, True),
    ("Emissions|CH4", pdf_emissions, True, False),
    ("Emissions|CFC12", pdf_emissions, True, False),
    ("Emissions|N2O", pdf_emissions, True, False),
    ("Emissions|Sulfur", pdf_emissions, True, False),
    ("Emissions|CO", pdf_emissions, True, False),
    ("Emissions|BC", pdf_emissions, True, False),
    ("Emissions|OC", pdf_emissions, True, False),
    ("Emissions|NOx", pdf_emissions, True, False),
    ("Emissions|NH3", pdf_emissions, True, False),
    ("Emissions|VOC", pdf_emissions, True, False),
    ("Atmospheric Concentrations|CO2", pdf_raw_scm_output, False, False),
    ("Atmospheric Concentrations|CH4", pdf_raw_scm_output, False, False),
    ("Effective Radiative Forcing", pdf_raw_scm_output, False, False),
    ("Effective Radiative Forcing|Greenhouse Gases", pdf_raw_scm_output, False, False),
    ("Effective Radiative Forcing|Aerosols", pdf_raw_scm_output, False, False),
    ("Effective Radiative Forcing|CO2", pdf_raw_scm_output, False, False),
    ("Effective Radiative Forcing|CH4", pdf_raw_scm_output, False, False),
    ("Heat Uptake", pdf_raw_scm_output, False, False),
    ("Surface Temperature (GSAT)", pdf_temperature, False, True),
]

nrows = len(variables_src) // 2 + len(variables_src) % 2
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, nrows * 5))
axes_flat = axes.flatten()
hue = ms_level
palette_h = palette
hue = "model"
palette_h = palette_model

for i, (variable, src, emissions, show_legend) in tqdm.auto.tqdm(enumerate(variables_src)):
    ax = axes_flat[i]

    if emissions:
        vdf = src.loc[pix.ismatch(variable=f"{variable}*")].openscm.to_long_data()

        sns.lineplot(
            ax=ax,
            data=vdf,
            x="time",
            y="value",
            hue=hue,
            palette=palette_h,
        )
        ax.set_title(vdf["variable"].unique()[0], fontdict=dict(fontsize="medium"))

    else:
        vdf = src.loc[pix.isin(variable=variable)]
        vdf.openscm.plot_plume_after_calculating_quantiles(
            quantile_over="run_id",
            hue_var=hue,
            style_var="climate_model",
            palette=palette_h,
            quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.3), ((0.05, 0.95), 0.0)),
            ax=ax,
        )
        ax.set_title(vdf.pix.unique("variable")[0], fontdict=dict(fontsize="medium"))

    # if show_legend:
    if i % 2:
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.05, 0.5))
    else:
        ax.legend().remove()

    if "Emissions" in variable and ("CO2" in variable or "GHG" in variable):
        ax.axhline(0.0, linestyle="--", color="gray")
        if "Energy" in variable:
            ax.set_yticks(np.arange(-2e4, 6e4 + 1, 1e4))
        elif "GHG" in variable:
            ax.set_yticks(np.arange(-2e4, 9e4 + 1, 1e4))
        elif "Cumulative" in variable:
            ax.set_yticks(np.arange(-1e3, 5e3 + 1, 5e2))
        else:
            ax.set_yticks(np.arange(-7.5e3, 10e3 + 1, 2.5e3))

    elif "Effective Radiative Forcing" in variable:
        pass

    else:
        ax.set_ylim(ymin=0.0)

    ax.grid()
    # break

plt.savefig("full-dive.pdf", bbox_inches="tight")

# %%
erf_nat = pdf_raw_scm_output.loc[pix.isin(variable="Effective Radiative Forcing")].reset_index(
    "variable", drop=True
) - pdf_raw_scm_output.loc[pix.isin(variable="Effective Radiative Forcing|Anthropogenic")].reset_index(
    "variable", drop=True
)
erf_nat.openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id",
    hue_var="scenario",
    style_var="climate_model",
    # palette=palette_h,
    quantiles_plumes=((0.5, 1.0), ((0.33, 0.67), 0.3), ((0.05, 0.95), 0.0)),
    # ax=ax,
)

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
iam = "REMIND"
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

# %% editable=true slideshow={"slide_type": ""}
tmp = metadata_quantile.unstack("climate_model")
magicc_diff = tmp["MAGICCv7.6.0a3"] - tmp["MAGICCv7.5.3"]
magicc_diff.unstack(["metric", "unit", "quantile"])[
    [(metric, "K", percentile) for metric in ["max", "2100"] for percentile in [0.33, 0.5, 0.67, 0.95]]
].sort_values(by=("max", "K", 0.5)).describe().round(3)

# %% editable=true slideshow={"slide_type": ""}
iam = "REMIND"
pdf = raw_scm_output.loc[pix.isin(variable="Atmospheric Concentrations|CH4") & pix.ismatch(model=f"*{iam}*"), :]

ax = pdf.loc[:, 2000:].openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id",
    style_var="climate_model",
    quantiles_plumes=quantile_plumes,
    create_legend=create_legend,
)
ax.axhline(pdf[1750].unique(), linestyle="--", color="gray", label="pre-industrial levels")
ax.annotate("pre-industrial concentration", (2040, pdf[1750].unique()))

ax.set_ylim(ymin=500.0)
