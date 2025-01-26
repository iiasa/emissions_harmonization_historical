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
# # Check a single model against historical data

# %%
import logging
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import seaborn as sns
import tqdm.autonotebook as tqdman
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
)
from emissions_harmonization_historical.io import load_csv

# %%
# Disable all logging to avoid a million messages
logging.disable()

# %%
SCENARIO_TIME_ID = "20250122-140031"

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"


# %%
history = strip_pint_incompatible_characters_from_units(load_csv(HISTORICAL_GLOBAL_COMPOSITE_PATH))
history_cut = history.loc[:, 1990:2025]
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
model_to_find = "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12".replace("/", "_").replace(" ", "-")
files_to_load = [f for f in scenario_files if model_to_find in str(f)]
files_to_load[:3]

# %%
scenarios_raw = pix.concat([load_csv(f) for f in tqdman.tqdm(files_to_load)]).sort_index(axis="columns")
scenarios_raw_global = scenarios_raw.loc[
    pix.ismatch(region="World")
    # & pix.isin(variable=history_cut.pix.unique("variable"))
]
scenarios_raw_global

# %%
scenarios_raw_global.pix.unique(["model", "scenario"]).to_frame(index=False).sort_values(by="scenario")


# %%
def get_sns_df(indf):
    """
    Get data frame to use with seaborn's plotting
    """
    out = indf.copy()
    out.columns.name = "year"
    out = out.stack().dropna().to_frame("value").reset_index()

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
variables_of_interest = ["**CH4", "**BC", "**CO", "**NOx", "**NH3", "**N2O", "**OC", "**Sulfur"]
pdf = pd.concat(
    [
        get_sns_df(scenarios_raw_global.loc[pix.ismatch(variable=variables_of_interest)]),
        get_sns_df(history_cut.loc[pix.ismatch(variable=variables_of_interest)]),
    ]
)
pdf["model_helper"] = pdf["model"]
pdf.loc[pdf["scenario"].str.endswith("_a"), "model_helper"] = "GAINS data"

fg = make_all_var_plot(
    data=pdf,  # [pdf["year"] <= 2050],
    kind="line",
    hue="scenario",
    style="model_helper",
    hue_order=sorted(pdf["scenario"].unique()),
)

for ax in fg.axes.flatten():
    ax.set_ylim(ymin=0)

# plt.savefig("tmp.pdf", transparent=True)
plt.show()
