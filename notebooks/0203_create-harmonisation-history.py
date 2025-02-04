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
# # Create harmonisation history
#
# Here we create our harmonisation history timeseries.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas_indexing as pix
import scipy.stats
from gcages.io import load_timeseries_csv
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    HARMONISATION_VALUES_ID,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Load composite history

# %%
out_file = (
    DATA_ROOT
    / "global-composite"
    / f"cmip7-harmonisation-history_world_{COMBINED_HISTORY_ID}_{HARMONISATION_VALUES_ID}.csv"
)

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"
history = strip_pint_incompatible_characters_from_units(
    load_timeseries_csv(
        HISTORICAL_GLOBAL_COMPOSITE_PATH,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
)


# %% [markdown]
# ## Use regressions for high variability variables

# %%
# TODO: decide which variables exactly to use averaging with
high_variability_variables = (
    "Emissions|BC",
    "Emissions|CO",
    "Emissions|CH4",
    # # Having looked at the data, I'm not sure I would do this for CO2 AFOLU
    # "Emissions|CO2|AFOLU",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|VOC",
)
n_years_for_regress = 10

harmonisation_values_l = []
for variable, vdf in history.groupby("variable"):
    if variable in high_variability_variables:
        regress_vals = vdf.loc[
            :,
            HARMONISATION_YEAR - n_years_for_regress + 1 : HARMONISATION_YEAR,
        ]
        regress_res = scipy.stats.linregress(x=regress_vals.columns, y=regress_vals.values)
        regressed_value = regress_res.slope * HARMONISATION_YEAR + regress_res.intercept

        tmp = vdf[HARMONISATION_YEAR].copy()
        tmp.loc[:] = regressed_value

        ax = vdf.loc[:, 1990:].T.plot()
        ax.scatter(HARMONISATION_YEAR, regressed_value, marker="x", color="tab:orange")
        ax.grid(which="major")
        ax.set_xticks(regress_vals.columns, minor=True)
        ax.grid(which="minor")
        plt.show()

    else:
        tmp = vdf[HARMONISATION_YEAR]

    harmonisation_values_l.append(tmp)

harmonisation_values = pix.concat(harmonisation_values_l)

# %%
history_harmonisation = pix.concat(
    [history.loc[:, 2000 : HARMONISATION_YEAR - 1], harmonisation_values.to_frame()], axis="columns"
)
history_harmonisation

# %%
out_file.parent.mkdir(exist_ok=True, parents=True)
history_harmonisation.to_csv(out_file)
out_file
