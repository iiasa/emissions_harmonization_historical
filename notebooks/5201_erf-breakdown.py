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
# # ERF breakdown
#
# An ERF breakdown over time.
# Very custom.
# If you want to edit, probably best to duplicate
# then edit rather than trying to make this do multiple things.

# %% [markdown]
# ## Imports

# %%
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm

from emissions_harmonization_historical.constants_5000 import (
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

# %% [markdown]
# ### Categories

# %%
model = "AIM**"
scenario = "SSP2**Low Overshoot"

# %%
raw_scm_output = SCM_OUTPUT_DB.load(
    pix.ismatch(
        variable=[
            "Effective Radiative Forcing**",
        ],
        model=model,
        scenario=scenario,
        climate_model="**MAGICCv7.6**",
    ),
    progress=True,
    max_workers=multiprocessing.cpu_count(),
)
raw_scm_output

# %%
median = raw_scm_output.openscm.groupby_except("run_id").median()
median.sort_index()

# %%
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

# %%
pdf = (
    median.loc[
        pix.ismatch(
            variable=[
                f"**{v}" for v in ["Greenhouse Gases", "Aerosols", "CO2", "CH4", "N2O", "Forcing", "Solar", "Volcanic"]
            ]
        ),
        2000:,
    ]
    .pix.project("variable")
    .T
)
# pdf

fig, ax = plt.subplots()

for i, variable in enumerate(pdf):
    colour = colors[i]
    pdf[variable].plot(color=colour, label=variable)

    for loc, name in [(pdf[variable].idxmax(), "max"), (pdf[variable].idxmin(), "min")]:
        if loc == pdf.index.max() or (loc == pdf.index.min()):
            continue

        ax.axvline(loc, linestyle="--", label=f"{variable} {name}", color=colour)

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="gray")
ax.set_xticks(np.arange(2000, 2101, 10))
ax.set_yticks(np.arange(-1.5, 4.01, 0.5))
ax.grid()

# %%
assert False

# %%
pdf = (
    median.loc[
        pix.ismatch(
            variable=[
                f"**{v}" for v in ["Greenhouse Gases", "Aerosols", "CO2", "CH4", "N2O", "Forcing", "Solar", "Volcanic"]
            ]
        ),
        2000:,
    ]
    .pix.project("variable")
    .T
)
# pdf

fig, ax = plt.subplots()

for i, variable in enumerate(pdf):
    colour = colors[i]
    pdf[variable].plot(color=colour, label=variable)

    for loc, name in [(pdf[variable].idxmax(), "max"), (pdf[variable].idxmin(), "min")]:
        if loc == pdf.index.max() or (loc == pdf.index.min()):
            continue

        ax.axvline(loc, linestyle="--", label=f"{variable} {name}", color=colour)

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.axhline(0.0, linestyle="--", color="gray")
ax.set_xticks(np.arange(2000, 2101, 10))
ax.set_yticks(np.arange(-1.5, 4.01, 0.5))
ax.grid()
