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
# # Process CMIP7 GHG data
#
# Invert greenhouse gas (GHG) concentrations from CMIP7:
# https://github.com/climate-resource/CMIP-GHG-Concentration-Generation.
# We do this for a very limited number of species
# where we have no other obvious option.
#
# Note on N<sub>2</sub>O: The global N<sub>2</sub>O budget is reported in units
# of TgN / yr, but the unit is really Tg N2 / yr.
# The clearest evidence of this is the Supplement to
# Global N<sub>2</sub>O budget Table 3, where 44 / 28 = 1.57
# (https://essd.copernicus.org/articles/16/2543/2024/essd-16-2543-2024-supplement.pdf).

# %% [markdown]
# ## Imports

# %%
from functools import partial

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import scipy.interpolate
import seaborn as sns
import tqdm.auto
import xarray as xr
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    CMIP7_GHG_PROCESSED_DB,
    CMIP7_GHG_RAW_PATH,
    CMIP7_GHG_VERSION_ID,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Setup

# %%
pandas_openscm.register_pandas_accessor()

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %%
CMIP7_TO_HERE_VARIABLE_MAP = {
    "hfc23": "HFC23",
    "hfc32": "HFC32",
    "hfc125": "HFC125",
    "hfc134a": "HFC134a",
    "hfc143a": "HFC143a",
    "hfc152a": "HFC152a",
    "hfc227ea": "HFC227ea",
    "hfc236fa": "HFC236fa",
    "hfc245fa": "HFC245fa",
    "hfc365mfc": "HFC365mfc",
    "hfc4310mee": "HFC4310mee",
    "halon1211": "Halon1211",
    "halon1301": "Halon1301",
    "halon2402": "Halon2402",
    "n2o": "N2O",
    "hcfc22": "HCFC22",
    "hcfc141b": "HCFC141b",
    "hcfc142b": "HCFC142b",
    "cfc11": "CFC11",
    "cfc12": "CFC12",
    "cfc113": "CFC113",
    "cfc114": "CFC114",
    "cfc115": "CFC115",
    "cf4": "CF4",
    "c2f6": "C2F6",
    "c3f8": "C3F8",
    "c4f10": "C4F10",
    "c5f12": "C5F12",
    "c6f14": "C6F14",
    "c7f16": "C7F16",
    "c8f18": "C8F18",
    "cc4f8": "cC4F8",
    "ch2cl2": "CH2Cl2",
    "ch3br": "CH3Br",
    "ch3cl": "CH3Cl",
    "chcl3": "CHCl3",
    "nf3": "NF3",
    "sf6": "SF6",
    "so2f2": "SO2F2",
    "ch3ccl3": "CH3CCl3",
    "ccl4": "CCl4",
}
here_to_cmip7_variable_map = {v: k for k, v in CMIP7_TO_HERE_VARIABLE_MAP.items()}

# %%
# Table A-5 of WMO 2022
# https://csl.noaa.gov/assessments/ozone/2022/downloads/Annex_2022OzoneAssessment.pdf
wmo_lifetimes = {
    # Gas: lifetime years
    "CF4": Q(50000.0, "yr"),
    "C2F6": Q(10000.0, "yr"),
    "C3F8": Q(2600.0, "yr"),
    "C4F10": Q(2600.0, "yr"),
    "C5F12": Q(4100.0, "yr"),
    "C6F14": Q(3100.0, "yr"),
    "C7F16": Q(3000.0, "yr"),
    "C8F18": Q(3000.0, "yr"),
    "CH2Cl2": Q(176.0 / 365.0, "yr"),
    "CH3Br": Q(0.8, "yr"),
    "CH3Cl": Q(0.9, "yr"),
    "CHCl3": Q(178.0 / 365.0, "yr"),
    "HFC23": Q(228, "yr"),
    "HFC32": Q(5.27, "yr"),
    "HFC125": Q(30.7, "yr"),
    "HFC134a": Q(13.5, "yr"),
    "HFC143a": Q(51.8, "yr"),
    "HFC152a": Q(1.5, "yr"),
    "HFC227ea": Q(35.8, "yr"),
    "HFC236fa": Q(213.0, "yr"),
    "HFC245fa": Q(6.61, "yr"),
    "HFC365mfc": Q(8.86, "yr"),
    "HFC4310mee": Q(17.0, "yr"),
    "NF3": Q(569.0, "yr"),
    "SF6": Q((850 + 1280) / 2.0, "yr"),
    "SO2F2": Q(36.0, "yr"),
    "cC4F8": Q(3200.0, "yr"),
    "N2O": Q(109.0, "yr"),
    "Halon1211": Q(16, "yr"),
    "Halon1301": Q(72, "yr"),
    "Halon2402": Q(28, "yr"),
    "HCFC22": Q(11.6, "yr"),
    "HCFC141b": Q(8.81, "yr"),
    "HCFC142b": Q(17.1, "yr"),
    "CFC11": Q(52, "yr"),
    "CFC12": Q(102, "yr"),
    "CFC113": Q(93, "yr"),
    "CFC114": Q(189, "yr"),
    "CFC115": Q(540, "yr"),
    "CH3CCl3": Q(5, "yr"),
    "CCl4": Q(30, "yr"),
}
molecular_masses = {
    # Gas: molecular mass
    "CF4": Q(12.01 + 4 * 19.0, "gCF4 / mole"),
    "C2F6": Q(2 * 12.01 + 6 * 19.0, "gC2F6 / mole"),
    "C3F8": Q(3 * 12.01 + 8 * 19.0, "gC3F8 / mole"),
    "C4F10": Q(4 * 12.01 + 10 * 19.0, "gC4F10 / mole"),
    "C5F12": Q(5 * 12.01 + 12 * 19.0, "gC5F12 / mole"),
    "C6F14": Q(6 * 12.01 + 14 * 19.0, "gC6F14 / mole"),
    "C7F16": Q(7 * 12.01 + 16 * 19.0, "gC7F16 / mole"),
    "C8F18": Q(8 * 12.01 + 18 * 19.0, "gC8F18 / mole"),
    "CH2Cl2": Q(12.01 + 2 * 1.0 + 2 * 35.45, "gCH2Cl2 / mole"),
    "CH3Br": Q(12.01 + 3 * 1.0 + 79.90, "gCH3Br / mole"),
    "CH3Cl": Q(12.01 + 3 * 1.0 + 35.45, "gCH3Cl / mole"),
    "CHCl3": Q(12.01 + 1.0 + 3 * 35.45, "gCHCl3 / mole"),
    "HFC23": Q(12.01 + 1.0 + 19.0, "gHFC23 / mole"),  # CHF3
    "HFC32": Q(12.01 + 2 * 1.0 + 2 * 19.0, "gHFC32 / mole"),  # CH2F2
    "HFC125": Q(12.01 + 2 * 1.0 + 2 * 19.0 + 12.01 + 3 * 19.0, "gHFC125 / mole"),  # CHF2CF3
    "HFC134a": Q(12.01 + 2 * 1.0 + 19.0 + 12.01 + 3 * 19.0, "gHFC134a / mole"),  # CH2FCF3
    "HFC143a": Q(12.01 + 3 * 1.0 + 12.01 + 3 * 19.0, "gHFC143a / mole"),  # CH3CF3
    "HFC152a": Q(12.01 + 3 * 1.0 + 12.01 + 1.0 + 2 * 19.0, "gHFC152a / mole"),  # CH3CHF2
    "HFC227ea": Q(12.01 + 3 * 19.0 + 12.01 + 1.0 + 19.0 + 12.01 + 3 * 19.0, "gHFC227ea / mole"),  # CF3CHFCF3
    "HFC236fa": Q(12.01 + 3 * 19.0 + 12.01 + 2 * 1.0 + 12.01 + 3 * 19.0, "gHFC236fa / mole"),  # CF3CH2CF3
    "HFC245fa": Q(
        12.01 + 2 * 1.0 + 19.0 + 12.01 + 2 * 19.0 + 12.01 + 1.0 + 2 * 19.0, "gHFC245fa / mole"
    ),  # CH2FCF2CHF2
    "HFC365mfc": Q(
        12.01 + 3 * 1.0 + 12.01 + 2 * 19.0 + 12.01 + 2 * 1.0 + 12.01 + 3 * 19.0, "gHFC365mfc / mole"
    ),  # CH3CF2CH2CF3
    "HFC4310mee": Q(
        12.01 + 3 * 19.0 + 2 * (12.01 + 1.0 + 19.0) + 12.01 + 2 * 19.0 + 12.01 + 3 * 19.0, "gHFC4310 / mole"
    ),  # CF3CHFCHFCF2CF3
    "NF3": Q(14.01 + 3 * 19.0, "gNF3 / mole"),
    "SF6": Q(32.07 + 6 * 19.0, "gSF6 / mole"),
    "SO2F2": Q(32.07 + 2 * 16.0 + 2 * 19.0, "gSO2F2 / mole"),
    "cC4F8": Q(4 * 12.01 + 8 * 19.0, "gcC4F8 / mole"),
    "N2O": Q(2 * 14.01 + 16.0, "gN2O / mole"),
    "Halon1211": Q(12.01 + 79.9 + 35.45 + 2 * 19.0, "gHalon1211 / mole"),  # CBrClF2
    "Halon1301": Q(12.01 + 79.9 + 3 * 19.0, "gHalon1301 / mole"),  # CBrF3
    "Halon2402": Q(12.01 + 79.9 + 2 * 19.0 + 12.01 + 79.9 + 2 * 19.0, "gHalon2402 / mole"),  # CBrF2CBrF2
    "HCFC22": Q(12.01 + 1.0 + 2 * 19.0 + 35.45, "gHCFC22 / mole"),  # CHF2Cl
    "HCFC141b": Q(12.01 + 3 * 1.0 + 12.01 + 2 * 35.45 + 19.0, "gHCFC141b / mole"),  # CH3CCl2F
    "HCFC142b": Q(12.01 + 3 * 1.0 + 12.01 + 35.45 + 2 * 19.0, "gHCFC142b / mole"),  # CH3CClF2
    "CFC11": Q(12.01 + 3 * 35.45 + 19.0, "gCFC11 / mole"),  # CCl3F
    "CFC12": Q(12.01 + 2 * 35.45 + 2 * 19.0, "gCFC12 / mole"),  # CCl2F2
    "CFC113": Q(12.01 + 2 * 35.45 + 19.0 + 12.01 + 35.45 + 2 * 19.0, "gCFC113 / mole"),  # CCl2FCClF2
    "CFC114": Q(12.01 + 35.45 + 2 * 19.0 + 12.01 + 35.45 + 2 * 19.0, "gCFC114 / mole"),  # CClF2CClF2
    "CFC115": Q(12.01 + 35.45 + 2 * 19.0 + 12.01 + 3 * 19.0, "gCFC115 / mole"),  # CClF2CF3
    "CH3CCl3": Q(12.01 + 3 * 1.0 + 12.01 + 3 * 35.45, "gCH3CCl3 / mole"),
    "CCl4": Q(12.01 + 4 * 35.45, "gCCl4 / mole"),
}

# %%
# CDIAC https://web.archive.org/web/20170118004650/http://cdiac.ornl.gov/pns/convert.html
ATMOSPHERE_MASS = Q(5.137 * 10**18, "kg")
# https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
MOLAR_MASS_DRY_AIR = Q(28.9, "g / mol")
atm_moles = (ATMOSPHERE_MASS / MOLAR_MASS_DRY_AIR).to("mole")
# Lines up with CDIAC: https://web.archive.org/web/20170118004650/http://cdiac.ornl.gov/pns/convert.html
fraction_factor = Q(1e-6, "1 / ppm")
mass_one_ppm_co2 = atm_moles * fraction_factor * Q(12.01, "gC / mole")
cdiac_expected = 2.13
if np.round(mass_one_ppm_co2.to("GtC / ppm").m, 2) != cdiac_expected:
    raise AssertionError

# %%
do_not_assume_flat_first_year = ["N2O"]
out_ts_vars = [
    "Emissions|CF4",
    "Emissions|C2F6",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C6F14",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|cC4F8",
    "Emissions|HFC23",
    "Emissions|HFC32",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC152a",
    "Emissions|HFC227ea",
    "Emissions|HFC236fa",
    "Emissions|HFC245fa",
    "Emissions|HFC365mfc",
    "Emissions|HFC4310mee",
    "Emissions|NF3",
    "Emissions|SF6",
    "Emissions|SO2F2",
    "Emissions|N2O",
    "Emissions|CH3CCl3",
    "Emissions|CH2Cl2",
    "Emissions|CHCl3",
    "Emissions|CH3Br",
    "Emissions|CH3Cl",
    "Emissions|CCl4",
    "Emissions|CFC11",
    "Emissions|CFC12",
    "Emissions|CFC113",
    "Emissions|CFC114",
    "Emissions|CFC115",
    "Emissions|Halon1211",
    "Emissions|Halon1301",
    "Emissions|Halon2402",
    "Emissions|HCFC22",
    "Emissions|HCFC141b",
    "Emissions|HCFC142b",
]

# %%
PI_YEAR = 1750

# %% [markdown]
# ## Process

# %%
inverse_info_l = []
for emissions_var in tqdm.auto.tqdm(sorted(out_ts_vars)):
    # converting to set then back to list removes duplicates

    gas = emissions_var.split("|")[-1]
    variable_id_cmip = here_to_cmip7_variable_map[gas]
    target_unit = f"kt {gas}/yr".replace("-", "")

    lifetime = wmo_lifetimes[gas]
    molecular_mass = molecular_masses[gas]

    to_load = list(CMIP7_GHG_RAW_PATH.glob(f"*{variable_id_cmip}_*.nc"))
    if len(to_load) != 1:
        raise AssertionError

    ds = xr.open_dataset(to_load[0])
    units = ds[variable_id_cmip].attrs["units"]
    concs = Q(ds[variable_id_cmip].values, units)

    if units == "ppt":
        fraction_factor = Q(1e-12, "1 / ppt")

    elif units == "ppb":
        fraction_factor = Q(1e-9, "1 / ppb")

    else:
        raise NotImplementedError(units)

    emm_factor = 1 / (atm_moles * fraction_factor * molecular_mass)
    print(gas, emm_factor)

    # Implicitly assume that first year values are flat,
    # unless overridden
    if gas in do_not_assume_flat_first_year:
        pass

    elif concs[1] != concs[0]:
        raise AssertionError

    # Probably a better way to do this, whatever.
    concs_start_year = Q(np.zeros(concs.size + 1), concs.u)
    concs_start_year[0] = concs[0]
    for i in range(1, concs_start_year.size):
        concs_start_year[i] = 2 * concs[i - 1] - concs_start_year[i - 1]
    # plt.plot(np.arange(concs_start_year.size), concs_start_year)
    # plt.step(np.arange(concs.size) + 1.0, concs)
    # a = 140
    # plt.xlim([a, a + 10])
    # plt.ylim(ymax=0.0001, ymin=0.0)
    # plt.show()

    time_step = Q(1, "yr")
    inverse_emissions = (
        1 / emm_factor * (np.diff(concs_start_year) / time_step + concs_start_year[:-1] / lifetime)
    ).to(target_unit)

    # Double check with a forward model
    concs_check = Q(np.zeros(concs.size + 1), concs.u)
    concs_check[0] = concs[list(ds["time"].dt.year.values).index(PI_YEAR)]
    for i in range(1, concs_check.size):
        concs_check[i] = (
            emm_factor * inverse_emissions[i - 1] * time_step
            - concs_check[i - 1] / lifetime * time_step
            + concs_check[i - 1]
        )

    if lifetime.to("yr").m > 1.0:
        np.testing.assert_allclose(
            concs.to(units).m,
            ((concs_check[1:] + concs_check[:-1]) / 2.0).to(units).m,
        )
    else:
        pass
        # The check doesn't work for very short lifetimes

    x = ds[variable_id_cmip]["time"].dt.year.values.squeeze()
    y = inverse_emissions.to(target_unit).m.squeeze()

    y_smooth = scipy.interpolate.make_smoothing_spline(x=x, y=y, lam=100.0)(x)
    y_smooth[np.where(y_smooth < 0.0)] = 0.0
    y_smooth[np.where(y_smooth < y_smooth.max() * 1e-6)] = 0.0

    df = pd.DataFrame(
        np.vstack([y, y_smooth]),
        columns=pd.Index(x, name="year"),
        index=pd.MultiIndex.from_tuples(
            (
                (f"{CMIP7_GHG_VERSION_ID}-inverse", HISTORY_SCENARIO_NAME, "World", emissions_var, target_unit),
                (f"{CMIP7_GHG_VERSION_ID}-inverse-smooth", HISTORY_SCENARIO_NAME, "World", emissions_var, target_unit),
            ),
            names=["model", "scenario", "region", "variable", "unit"],
        ),
    )
    inverse_info_l.append(df)

inverse_info = pix.concat(inverse_info_l)
inverse_info

# %% [markdown]
# We calculate 1750 total N2O emissions to be in the region of 19 Tg N2O / yr from the inversion,
# which is close to the middle of the natural range from GNB (natural sources 11.8 TgN2 / yr = 18.5 Tg N2O / yr).
# Source: fig. 1 in https://essd.copernicus.org/articles/16/2543/2024/

# %%
inverse_info.loc[inverse_info.index.get_level_values("variable") == "Emissions|N2O", 1750]

# %% [markdown]
# ## Extrapolate

# %%
pdf = inverse_info.loc[pix.ismatch(model="**smooth")].openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    col="variable",
    col_order=sorted(pdf["variable"].unique()),
    col_wrap=3,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %%
inverse_extrapolated = (
    inverse_info.loc[pix.ismatch(model="**smooth")].copy().pix.assign(model="CR-CMIP-1-0-0-inverse-smooth-extrapolated")
)
for y in range(inverse_extrapolated.columns.max(), HARMONISATION_YEAR + 1):
    # Very basic linear extrapolation
    inverse_extrapolated[y] = 2 * inverse_extrapolated[y - 1] - inverse_extrapolated[y - 2]

# Just in case
inverse_extrapolated[inverse_extrapolated < 0.0] = 0.0

# inverse_extrapolated

# %%
pdf = inverse_extrapolated.openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    col="variable",
    col_order=sorted(pdf["variable"].unique()),
    col_wrap=3,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# ## Save

# %% [markdown]
# ### Pre-industrial emissions

# %% [markdown]
# Make sure we can convert the variable names

# %%
update_index_levels_func(
    inverse_extrapolated,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
CMIP7_GHG_PROCESSED_DB.save(inverse_extrapolated, groupby=["model"], allow_overwrite=True)
