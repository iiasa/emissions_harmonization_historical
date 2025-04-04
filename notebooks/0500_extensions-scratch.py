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
# # Extensions - scratch
#
# A place for experimenting with extensions.
# Once we know what we are doing better,
# we will split this into something more organised.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas_indexing as pix
import seaborn as sns

from emissions_harmonization_historical.constants import DATA_ROOT
from emissions_harmonization_historical.io import load_global_scenario_data

# %% [markdown]
# ## Set up

# %%
SCENARIO_TIME_ID = "20250313-140552"

# %% [markdown]
# ## Load scenario data

# %%
scenarios_raw_global = load_global_scenario_data(
    scenario_path=DATA_ROOT / "scenarios" / "data_raw",
    scenario_time_id=SCENARIO_TIME_ID,
    progress=True,
).loc[:, :2100]  # TODO: drop 2100 end once we have usable scenario data post-2100

# %%
scenarios_raw_global.loc[
    pix.ismatch(variable="**CO2", model="REMIND*", scenario="SSP1 - Very Low Emissions_a")
].T.plot()

# %%
import pandas as pd


def interpolate_to_annual(idf: pd.DataFrame, max_supplement: float = 1e-5) -> pd.DataFrame:
    # TODO: push into pandas-openscm
    missing_cols = np.setdiff1d(np.arange(idf.columns.min(), idf.columns.max() + max_supplement), idf.columns)

    out = idf.copy()
    out.loc[:, missing_cols] = np.nan
    out = out.sort_index(axis="columns").T.interpolate("index").T

    return out


# %%
selected_scenarios_idx = pd.MultiIndex.from_tuples(
    (
        # A randomly drawn set, but not a terrible start for experimenting
        ("REMIND-MAgPIE 3.4-4.8", "SSP1 - Very Low Emissions"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Overshoot"),
        ("IMAGE 3.4", "SSP1 - Low Emissions"),
        ("AIM 3.0", "SSP1 - Very Low Emissions"),
        ("MESSAGEix-GLOBIOM 2.1-M-R12", "SSP5 - High Emissions"),
        ("IMAGE 3.4", "SSP5 - High Emissions"),
        ("AIM 3.0", "SSP2 - Medium-Low Emissions"),
        ("WITCH 6.0", "SSP2 - Low Emissions"),
        ("REMIND-MAgPIE 3.4-4.8", "SSP2 - Low Overshoot_b"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP5 - Low Overshoot"),
        ("COFFEE 1.5", "SSP2 - Medium Emissions"),
        ("GCAM 7.1 scenarioMIP", "SSP2 - Medium Emissions"),
        ("IMAGE 3.4", "SSP2 - Very Low Emissions"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions"),
    ),
    name=["model", "scenario"],
)
start = scenarios_raw_global[scenarios_raw_global.index.isin(selected_scenarios_idx)]
start

# %%
lazy_linear = start.loc[pix.ismatch(variable="**CO2")].copy()
lazy_linear[2150] = 0.0
lazy_linear[2500] = 0.0
interpolate_to_annual(lazy_linear)
lazy_linear = interpolate_to_annual(lazy_linear)
ax = lazy_linear.T.plot()
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
import scipy.interpolate

# %%
new_time_axis = np.arange(2101, 2500 + 1)
spline = scipy.interpolate.make_smoothing_spline(start.columns, start.values.squeeze(), lam=None)
spline.derivative()(2150)
interp_ext = pd.DataFrame(
    spline(new_time_axis)[:, np.newaxis].T,
    columns=new_time_axis,
    index=start.index,
)
ax = start.T.plot()
interp_ext.T.plot(ax=ax)


# %%
def extend_linear(df: pd.DataFrame, points_to_hit: tuple[tuple[int, float]]) -> pd.DataFrame:
    out = df.copy()
    for year, point in points_to_hit:
        out[year] = point

    return interpolate_to_annual(out)


# %%
from functools import partial

# %%
# linear (if only to show that it's really not an ideal solution)
# extend constant thing (up until some time point, if only to show that it's really not an ideal solution)
# gradient preservation
# asymptoting stuff
# exponential declines (including preserving gradients)
# spline extensions
# # copy gradient from last few years

# probably we want to be able to define the approach in a piecewise way:
# - e.g. asymptote until a year based on something, then linear decline from there
# - e.g. asymptote until a year based on something, then exponential decline from there

extension_functions = (
    (
        pix.ismatch(variable="**CO2") & ~pix.ismatch(scenario="*Very Low*"),
        partial(extend_linear, points_to_hit=((2200, 0.0), (2500, 0.0))),
    ),
    (
        pix.ismatch(variable="**CO2", scenario="*Very Low*"),
        partial(extend_linear, points_to_hit=((2150, -2000.0), (2500, 0.0))),
    ),
)

# %% [markdown]
# - compare options for extension (within one scenario)
# - comparing scenarios with same extension
# - comparing both at once (harder to visualise)

# %%
# timeseries metadata -> function which can extend that timeseries
in_df = start.loc[pix.ismatch(variable="**CO2")].copy()

res_l = []
for metadata_lookup, extension_func in extension_functions:
    extended_ts = extension_func(in_df.loc[metadata_lookup])
    res_l.append(extended_ts)

res = pd.concat(res_l)

ax = res.pix.project(["model", "scenario"]).T.plot()
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
plt.show()

res

# %%
sdf = res.melt(ignore_index=False, var_name="year").reset_index().dropna()
sdf

# %%
sns.relplot(
    data=sdf,
    x="year",
    y="value",
    col="model",
    row="scenario",
    hue="scenario",
    style="model",
    kind="line",
)

# %%
