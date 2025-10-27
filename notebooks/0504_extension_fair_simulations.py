# %% [markdown]
# # FaIR Climate Model Simulations with Extended Emissions Scenarios
#
# This notebook runs the FaIR v1.4.1 climate model with extended emissions scenarios
# (1750-2501) to generate climate projections. It processes emissions through
# CO2-equivalent calculations, applies blending for smooth transitions, and produces
# temperature and concentration projections across seven scenarios ranging from very
# low (VL) to very high (HL) emissions.

# %%
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from fair import FAIR
from fair.interface import initialise
from fair.io import read_properties

# %%
f = FAIR()

# %% [markdown]
#

# %% [markdown]
# Need to pull the calibrated_constrained_parameters_1.4.1.csv from https://zenodo.org/records/8399112
# and place here:
# ../data/fair-inputs/calibrated_constrained_parameters_1.4.1.csv
#

# %%
snames = ["VL", "LN", "L", "ML", "M", "H", "HL"]
snames_short = ["VL", "LN", "L", "ML", "M", "H", "HL"]
sname21_short = ["VL", "LN", "L", "ML", "M", "H", "HL"]

f.define_time(1750, 2501, 1)
f.define_scenarios(snames)
species, properties = read_properties("../data/fair-inputs/species_configs_properties_1.4.1.csv")
f.define_species(species, properties)
f.ch4_method = "Thornhill2021"
df_configs = pd.read_csv("../data/fair-inputs/calibrated_constrained_parameters_1.4.1.csv", index_col=0)
f.define_configs(df_configs.index)
f.allocate()

# %%
scens = f.emissions.scenario.values

# %%
ldict = {}
ldict21 = {}
for i, s in enumerate(snames):
    ldict[s] = snames_short[i]
    ldict21[s] = sname21_short[i]


# %%
colors = {
    snames[6]: "#800000",
    snames[5]: "#ff0000",
    snames[4]: "#fc7b03",
    snames[3]: "#d3a640",
    snames[2]: "#098740",
    snames[1]: "#0080d0",
    snames[0]: "#100060",
}

# %%
os.makedirs("../plots", exist_ok=True)

# %% [markdown]
# ../data/fair-inputs/emissions_1750-2500.csv
# is generated from 0503_extension_functioality_as_notebook.py

# %%
df_emis = pd.read_csv("../data/fair-inputs/emissions_1750-2500.csv")
df_emis.head()

# %% [markdown]
# ## Setup and Configuration
#
# **Scenarios**: Seven emissions scenarios (VL, LN, L, ML, M, H, HL) representing very
# low to very high emissions pathways
# **Time range**: 1750-2501 (752 years)
# **Species**: CO2 (FFI & AFOLU), CH4, N2O, plus 37 other GHGs and aerosols
# **FaIR configuration**: Using calibrated parameters from Smith et al. with legacy CH4 lifetime method

# %%
gwpmat = pd.read_csv("../data/fair-inputs/gwp_mass_adjusted_100y.csv", index_col=0)

# %%
f.fill_from_csv(
    forcing_file="../data/fair-inputs/volcanic_solar.csv",
    emissions_file="../data/fair-inputs/emissions_1750-2500.csv",
)

# %%
gwp_nonco2 = gwpmat.copy()
gwp_nonco2.loc["CO2 AFOLU"] = np.nan
gwp_nonco2.loc["CO2 FFI"] = np.nan


# %%
nonco2 = f.emissions.sel(specie="CO2 FFI")[:, :, 0].copy()
for specie in f.emissions.specie.values:
    try:
        gwp = gwp_nonco2[specie]
    except KeyError:
        gwp = np.nan
    if ~np.isnan(gwp):
        nonco2 = nonco2 + f.emissions.sel(specie=specie)[:, :, 0] * gwp
    else:
        0


# %%
ncflr = np.ones(len(scens))
for i in range(len(scens)):
    ncflr[i] = nonco2.sel(scenario=scens[i])[-1] / 1e6
ncflr

# %%
scens_shrt = [ldict[s] for s in scens]

# %% [markdown]
# ## CO2-Equivalent Emissions Calculation
#
# Convert all GHG emissions to CO2-equivalents using 100-year Global Warming Potentials
# (GWP100). This aggregates the climate forcing from all greenhouse gases into a single
# metric for comparison across scenarios.
#
# **Method**: Multiply each species' emissions by its GWP (e.g., CH4 = 29.8, N2O = 273)
# and sum to get total CO2e.

# %% [markdown]
#
# - Solar forcing set to zero (natural forcing handled separately by FaIR)

# %%
for s in f.scenarios:
    f.forcing.loc[dict(scenario=s, specie="Solar")] = 0


# %% [markdown]
# Plot emissions before running

# %%
fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(14, 5))
for scenario in f.scenarios:
    ax[0].plot(
        f.timepoints,
        (
            f.emissions.sel(scenario=scenario, specie="CO2 FFI", config=f.configs[0])
            + f.emissions.sel(scenario=scenario, specie="CO2 AFOLU", config=f.configs[0])
        ),
        label=scenario,
        color=colors[scenario],
    )
ax[0].set_ylabel("CO$_2$ emissions, GtCO$_2$ yr$^{-1}$")
ax[0].axhline(ls=":", color="k", lw=0.5)
ax[0].legend()
ax[0].grid()
for scenario in f.scenarios:
    ax[1].plot(
        f.timepoints,
        (
            f.emissions.sel(scenario=scenario, specie="CO2 FFI", config=f.configs[0])
            + f.emissions.sel(scenario=scenario, specie="CO2 AFOLU", config=f.configs[0])
        ).cumsum(),
        label=scenario,
        color=colors[scenario],
    )
ax[1].set_ylabel("Cumulative CO$_2$ emissions, GtCO$_2$ yr$^{-1}$")
ax[1].axhline(ls=":", color="k", lw=0.5)
ax[1].legend()
ax[1].grid()
pl.savefig("../plots/co2_emissions.png")

# %%
scens_out = []
for s in scens:
    df_scen = f.emissions.sel(scenario=s, config=1234).to_pandas().T
    df_scen.insert(loc=0, column="Scenario", value=s)
    df_scen.dropna(inplace=True)
    scens_out.append(df_scen)
scens_out = pd.concat(scens_out)


# %% [markdown]
# Calculate CO2e

# %%
co2eo = f.emissions.sel(specie="CO2 FFI")[:, :, 0].copy()
for specie in f.emissions.specie.values:
    try:
        gwp = gwpmat[specie]
    except KeyError:
        gwp = np.nan
    if ~np.isnan(gwp):
        co2eo = co2eo + f.emissions.sel(specie=specie)[:, :, 0] * gwp
    else:
        0
co2e = co2eo * 1e6  # -co2eo.loc[dict(timepoints=2019.5)].values+53.e6

# %%
fig, ax = pl.subplots(1, 2, figsize=(14, 5))
for scenario in f.scenarios:
    ax[0].plot(
        f.timepoints,
        (
            f.emissions.sel(scenario=scenario, specie="CO2 FFI", config=f.configs[0])
            + f.emissions.sel(scenario=scenario, specie="CO2 AFOLU", config=f.configs[0])
        ),
        label=ldict21[scenario],
        color=colors[scenario],
    )
ax[0].set_ylabel("CO$_2$ emissions, GtCO$_2$ yr$^{-1}$")
ax[0].axhline(ls=":", color="k", lw=0.5)
ax[0].set_xlim(2015, 2300)
ax[0].set_ylim(-40, 100)

ax[0].legend()
ax[0].grid()

for scenario in f.scenarios:
    ax[1].plot(
        f.timepoints,
        co2e.sel(scenario=scenario) / 1e6,
        label=ldict21[scenario],
        color=colors[scenario],
    )
ax[1].set_ylabel("GHG emissions, GtCO$_2$eq yr$^{-1}$")
ax[1].axhline(ls=":", color="k", lw=0.5)
# ax[1].legend()
ax[1].set_xlim(2015, 2300)
ax[1].set_ylim(-50, 100)

ax[1].grid()
pl.savefig("../plots/ghg_emissions.png")

# %% [markdown]
# ## Run FaIR

# %%
f.fill_species_configs("../data/fair-inputs/species_configs_properties_1.4.1.csv")
f.override_defaults("../data/fair-inputs/calibrated_constrained_parameters_1.4.1.csv")
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)
initialise(f.ocean_heat_content_change, 0)
f.run()

# %% [markdown]
# ## Results Visualization
#
# Generate comprehensive plots showing:
# 1. **GHG emissions** (CO2e) trajectories with uncertainty bands (33rd-66th percentiles
#    until 2100, extended projections 2100-2150)
# 2. **Temperature anomalies** relative to 1850-1900 baseline with 5th-95th percentile
#    ranges
# 3. **Multi-panel diagnostics**: CO2 emissions, cumulative CO2, CO2e, radiative forcing,
#    CO2 concentrations, and temperature
# 4. **Probability distributions**: Temperature outcomes at 2100, 2300, and peak warming
#    across scenarios

# %% [markdown]
#

# %%
# nohos=[x for x in f.scenarios if x != "high-overshoot"]
nohos = [x for x in f.scenarios]

# %%
fig, ax = pl.subplots(1, 2, figsize=(12, 5))

unc = np.tanh((co2e.sel(scenario=nohos[0]) - co2e.sel(scenario=nohos[-2])) / 1e6 / 10) * 8
for scenario in nohos:
    ax[0].fill_between(
        f.timebounds[:351],
        co2e.sel(scenario=scenario)[:351] / 1e6 - unc[:351],
        co2e.sel(scenario=scenario)[:351] / 1e6 + unc[:351],
        color=colors[scenario],
        lw=0,
        alpha=0.3,
    )
    ax[0].fill_between(
        f.timepoints[350:],
        co2e.sel(scenario=scenario)[350:] / 1e6 - unc[350:],
        co2e.sel(scenario=scenario)[350:] / 1e6 + unc[350:],
        color=colors[scenario],
        hatch="XXX",
        lw=0,
        alpha=0.1,
    )
    ax[0].plot(
        f.timepoints[:275],
        co2e.sel(scenario=scenario)[:275] / 1e6,
        label=ldict21[scenario],
        color=colors[scenario],
    )
    # ax[0].plot(
    #     f.timepoints[350:],
    #     co2e.sel(scenario=scenario)[350:]/ 1e6,
    #     label=ldict21[scenario],
    #     color=colors[scenario],linestyle='--'
    # )

ax[0].set_ylabel("GHG emissions, GtCO$_2$eq yr$^{-1}$")
ax[0].axhline(ls=":", color="k", lw=0.5)
# ax[1].legend()
ax[0].set_xlim(2000, 2150)
ax[0].set_ylim(-50, 80)

ax[0].grid()
ax[0].set_title("(a)")

for i, scenario in enumerate(nohos):
    ax[1].fill_between(
        f.timebounds[:351],
        (
            f.temperature.sel(scenario=scenario, layer=0)[:351]
            - f.temperature.sel(scenario=scenario, layer=0, timebounds=np.arange(1850, 1902)).mean(dim="timebounds")
        ).quantile(0.33, dim="config"),
        (
            f.temperature.sel(scenario=scenario, layer=0)[:351]
            - f.temperature.sel(scenario=scenario, layer=0, timebounds=np.arange(1850, 1902)).mean(dim="timebounds")
        ).quantile(0.66, dim="config"),
        color=colors[scenario],
        lw=0,
        alpha=0.3,
    )
    ax[1].fill_between(
        f.timebounds[350:],
        (
            f.temperature.sel(scenario=scenario, layer=0)[350:]
            - f.temperature.sel(scenario=scenario, layer=0, timebounds=np.arange(1850, 1902)).mean(dim="timebounds")
        ).quantile(0.33, dim="config"),
        (
            f.temperature.sel(scenario=scenario, layer=0)[350:]
            - f.temperature.sel(scenario=scenario, layer=0, timebounds=np.arange(1850, 1902)).mean(dim="timebounds")
        ).quantile(0.66, dim="config"),
        color=colors[scenario],
        hatch="XXX",
        lw=0,
        alpha=0.1,
        label=snames_short[i],
    )

ax[1].axhline(0, ls=":", color="k", lw=0.5)
ax[1].set_ylabel("temperature above 1850-1900, K")
ax[1].set_ylim(0, 5)
ax[1].set_xlim(2000, 2150)

ax[1].grid()
ax[1].legend()

ax[1].set_title("(b)")
pl.savefig("../plots/temperature_emis.png")

# %%
fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(14, 15))
ax = ax.flatten()
for scenario in f.scenarios:
    ax[0].plot(
        f.timepoints,
        (
            f.emissions.sel(scenario=scenario, specie="CO2 FFI", config=f.configs[0])
            + f.emissions.sel(scenario=scenario, specie="CO2 AFOLU", config=f.configs[0])
        ),
        label=ldict[scenario],
        color=colors[scenario],
    )
ax[0].set_ylabel("CO$_2$ emissions, GtCO$_2$ yr$^{-1}$")
ax[0].axhline(ls=":", color="k", lw=0.5)
ax[0].legend()
ax[0].grid()
for scenario in f.scenarios:
    ax[1].plot(
        f.timepoints,
        (
            f.emissions.sel(scenario=scenario, specie="CO2 FFI", config=f.configs[0])
            + f.emissions.sel(scenario=scenario, specie="CO2 AFOLU", config=f.configs[0])
        ).cumsum(),
        label=scenario,
        color=colors[scenario],
    )
ax[1].set_ylabel("Cumulative CO$_2$ emissions, GtCO$_2$ yr")
ax[1].axhline(ls=":", color="k", lw=0.5)
# ax[1].legend()
ax[1].grid()

for scenario in f.scenarios:
    ax[2].plot(
        f.timepoints,
        co2e.sel(scenario=scenario) / 1e6,
        label=scenario,
        color=colors[scenario],
    )
ax[2].set_ylabel("GHG emissions, GtCO$_2$eq yr$^{-1}$")
ax[2].axhline(ls=":", color="k", lw=0.5)
# ax[2].legend()
ax[2].grid()

for scenario in f.scenarios:
    ax[3].plot(
        f.timebounds,
        f.forcing_sum.sel(scenario=scenario).median(dim="config"),
        label=scenario,
        color=colors[scenario],
    )
ax[3].set_ylabel("Effective radiative forcing, W m$^{-2}$")
# pl.legend();
ax[3].grid()

for scenario in f.scenarios:
    ax[4].fill_between(
        f.timebounds,
        (
            f.concentration.sel(specie="CO2").sel(
                scenario=scenario,
            )
        ).quantile(0.05, dim="config"),
        (f.concentration.sel(specie="CO2").sel(scenario=scenario)).quantile(0.95, dim="config"),
        color=colors[scenario],
        lw=0,
        alpha=0.1,
    )
    ax[4].plot(
        f.timebounds,
        (f.concentration.sel(specie="CO2").sel(scenario=scenario)).median(dim="config"),
        label=scenario,
        color=colors[scenario],
    )
ax[4].axhline(0, ls=":", color="k", lw=0.5)
ax[4].set_ylabel("Atmospheric CO2 concentration, ppm")
ax[4].set_ylim(0, 1500)
ax[4].grid()
for scenario in f.scenarios:
    ax[5].fill_between(
        f.timebounds,
        (
            f.temperature.sel(scenario=scenario, layer=0)
            - f.temperature.sel(scenario=scenario, layer=0, timebounds=np.arange(1850, 1902)).mean(dim="timebounds")
        ).quantile(0.05, dim="config"),
        (
            f.temperature.sel(scenario=scenario, layer=0)
            - f.temperature.sel(scenario=scenario, layer=0, timebounds=np.arange(1850, 1902)).mean(dim="timebounds")
        ).quantile(0.95, dim="config"),
        color=colors[scenario],
        lw=0,
        alpha=0.1,
    )
    ax[5].plot(
        f.timebounds,
        (
            f.temperature.sel(scenario=scenario, layer=0)
            - f.temperature.sel(scenario=scenario, layer=0, timebounds=np.arange(1850, 1902)).mean(dim="timebounds")
        ).median(dim="config"),
        label=ldict[scenario],
        color=colors[scenario],
    )
ax[5].axhline(0, ls=":", color="k", lw=0.5)
ax[5].set_ylabel("temperature above 1850-1900, K")
ax[5].set_ylim(-3, 8)
ax[5].legend()

ax[5].grid()

pl.savefig("../plots/extensions.png")

# %%
f21c = scens

# %%
fig, ax = pl.subplots(3, 1, figsize=(12, 8))
ax = ax.flatten()

for scenario in f21c:
    a = ax[0].ecdf(
        f.temperature.sel(scenario=scenario, layer=0, timebounds=2100)
        - f.temperature.sel(scenario=scenario, layer=0, timebounds=1850),
        color=colors[scenario],
        label=ldict21[scenario],
    )
ax[0].set_title("Temperature anomaly in 2100 relative to 1850, K")
ax[0].set_xlabel("K")
ax[0].set_ylabel("Cumulative probability")
ax[0].set_yticks([0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9])
ax[0].set_xticks(np.array([-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]) * 2)
ax[0].set_xlim([-1, 10])
ax[0].legend()
ax[0].grid()


for scenario in f21c:
    a = ax[1].ecdf(
        f.temperature.sel(scenario=scenario, layer=0, timebounds=2300)
        - f.temperature.sel(scenario=scenario, layer=0, timebounds=1850),
        color=colors[scenario],
        label=ldict21[scenario],
    )
ax[1].set_title("Temperature anomaly in 2300 relative to 1850, K")
ax[1].set_ylabel("Cumulative probability")
ax[1].set_xlabel("K")

ax[1].set_yticks([0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9])
ax[1].set_xticks(np.array([-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]) * 2)
ax[1].set_xlim([-1, 10])
ax[1].legend()
ax[1].grid()


for scenario in f21c:
    a = ax[2].ecdf(
        f.temperature.sel(scenario=scenario, layer=0).max(dim="timebounds")
        - f.temperature.sel(scenario=scenario, layer=0, timebounds=1850),
        color=colors[scenario],
        label=ldict21[scenario],
    )
ax[2].set_title("Maximum temperature anomaly relative to 1850, K")
ax[2].set_xlabel("K")
ax[2].set_ylabel("Cumulative probability")
ax[2].set_yticks([0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9])
ax[2].set_xticks(np.array([-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]) * 2)
ax[2].set_xlim([-1, 10])
ax[2].legend(loc="upper right")
ax[2].grid()
pl.tight_layout()
