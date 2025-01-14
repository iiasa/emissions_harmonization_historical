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
# # Harmonise scenarios

# %%
import logging
import multiprocessing
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import scipy.stats
import seaborn as sns
import tqdm.autonotebook as tqdman
from gcages.harmonisation import Harmoniser
from gcages.pre_processing import PreProcessor

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HISTORICAL_COMPOSITE_PROCESSING_ID,
)
from emissions_harmonization_historical.io import load_csv

# %%
# Disable all logging to avoid a million messages
logging.disable()

# %%
SCENARIO_TIME_ID = "20250113-200523"

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = (
    DATA_ROOT / "global-composite" / f"historical-global-composite_{HISTORICAL_COMPOSITE_PROCESSING_ID}.csv"
)


# %%
history = load_csv(HISTORICAL_GLOBAL_COMPOSITE_PATH)
history_cut = history.loc[:, 1990:]
history_cut

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
scenario_files = tuple(SCENARIO_PATH.glob(f"{SCENARIO_TIME_ID}__scenarios-scenariomip__*.csv"))
if not scenario_files:
    msg = f"Check your scenario ID. {list(SCENARIO_PATH.glob('*.csv'))=}"
    raise AssertionError(msg)

scenario_files[:5]

# %%
scenarios_raw = pix.concat([load_csv(f) for f in tqdman.tqdm(scenario_files)]).sort_index(axis="columns")
scenarios_raw_global = scenarios_raw.loc[
    pix.ismatch(region="World") & pix.isin(variable=history_cut.pix.unique("variable"))
]
scenarios_raw_global

# %%
# pandas-indexing is so well done
# scenarios_raw_global.pix.extract(
#     variable="Emissions|{species}|{sector}|{subsector}", dropna=False, keep=True
# ).index.to_frame(index=False)

# %%
scenarios_raw_global.pix.unique(["model", "scenario"]).to_frame(index=False)


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
make_all_var_plot = partial(
    sns.relplot,
    x="year",
    y="value",
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
)

# %%
# make_all_var_plot(
#     data=get_sns_df(history_cut),
#     kind="line",
#     hue="scenario",
#     style="model",
# )

# %%
# make_all_var_plot(
#     data=get_sns_df(scenarios_raw_global),
#     kind="scatter",
#     hue="scenario",
#     style="model",
# )

# %%
# for model, mdf in scenarios_raw_global.groupby("model"):
#     pdf = pd.concat([get_sns_df(mdf), get_sns_df(history_cut)])
#     make_all_var_plot(
#         data=pdf,
#         kind="line",
#         hue="scenario",
#         style="model",
#     )
#     plt.show()

# %%
pre_processor = PreProcessor(emissions_out=tuple(history.pix.unique("variable")))

# %%
# As at 2024-01-13, just the list from AR6.
# We can tweak from here.
aneris_overrides = pd.DataFrame(
    [
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|BC'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C2F6",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C6F14",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|CF4",
        },  # high historical variance (cov=11.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|CO",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2080",
            "variable": "Emissions|CO2",
        },  # always ratio method by choice
        {
            # high historical variance,
            # but using offset method to prevent diff from increasing
            # when going negative rapidly (cov=23.2)
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'},
        {
            "method": "constant_ratio",
            "variable": "Emissions|F-Gases",
        },  # basket not used in infilling (sum of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC",
        },  # basket not used in infilling (sum of subset of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC125",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC134a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC143a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC227ea",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC23",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC32",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC43-10",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|OC",
        },  # high historical variance (cov=18.5)
        {
            "method": "constant_ratio",
            "variable": "Emissions|SF6",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|VOC",
        },  # high historical variance (cov=12.0)
    ]
)

# %%
# TODO: discuss and think through better
harmonisation_year = 2021
calc_scaling_year = 2015

# %%
history_values = history_cut.loc[:, calc_scaling_year:harmonisation_year].copy()

# TODO: decide which variables exactly to use averaging with
high_variability_variables = ("Emissions|BC", "Emissions|CO", "Emissions|OC")
n_years_for_regress = 10
for high_variability_variable in high_variability_variables:
    regress_vals = history_cut.loc[
        pix.ismatch(variable=high_variability_variable),
        harmonisation_year - n_years_for_regress + 1 : harmonisation_year,
    ]
    regress_res = scipy.stats.linregress(x=regress_vals.columns, y=regress_vals.values)
    regressed_value = regress_res.slope * harmonisation_year + regress_res.intercept

    # Should somehow keep track that we've done this
    history_values.loc[pix.ismatch(variable=high_variability_variable), harmonisation_year] = regressed_value

history_values

# %%
harmoniser = Harmoniser(
    historical_emissions=history_values,
    harmonisation_year=harmonisation_year,
    calc_scaling_year=calc_scaling_year,
    aneris_overrides=aneris_overrides,
    n_processes=multiprocessing.cpu_count(),
    # TODO: implement and enable
    run_checks=False,
)

# %%
pre_processed = pre_processor(scenarios_raw_global)
harmonised = harmoniser(pre_processed)

# %%
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colours

# %%
for model, mdf in pix.concat(
    [
        scenarios_raw_global.pix.assign(stage="raw"),
        harmonised.pix.assign(stage="harmonised"),
    ]
).groupby("model"):
    variables_to_show = mdf.pix.unique("variable").intersection(history_cut.pix.unique("variable"))
    pdf_wide = pix.concat([mdf, history_cut.pix.assign(stage="history")]).loc[pix.isin(variable=variables_to_show)]
    # pdf_wide = pdf_wide.loc[:, 2010: 2028]
    palette = {k: colours[i % len(colours)] for i, k in enumerate(pdf_wide.pix.unique("scenario"))}
    fg = make_all_var_plot(
        data=get_sns_df(pdf_wide),
        kind="line",
        hue="scenario",
        hue_order=sorted(pdf_wide.pix.unique("scenario")),
        palette={**palette, "history": "black"},
        style="stage",
    )
    fg.fig.suptitle(model, y=1.01)
    plt.show()
    # break
