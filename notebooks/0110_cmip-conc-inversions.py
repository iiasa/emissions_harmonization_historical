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
# # CMIP concentration inversions
#
# Invert concentrations from CMIP7.
# We do this for a very limited number of species
# where we have no other obvious option.

# %%
import json

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pooch
import scipy.interpolate
import tqdm.autonotebook as tqdman
import xarray as xr

from emissions_harmonization_historical.constants import (
    CMIP_CONCENTRATION_INVERSION_ID,
    DATA_ROOT,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.infilling_followers import FOLLOW_LEADERS

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %%
source_id = "CR-CMIP-0-4-0"
pub_date = "v20241205"

# %%
DOWNLOAD_PATH = DATA_ROOT / "global" / "esgf" / source_id
DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)

# %%
OUT_PATH = DATA_ROOT / "global" / "esgf" / source_id / f"inverse_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.csv"
OUT_PATH_PI_EMMS = (
    DATA_ROOT / "global" / "esgf" / source_id / f"pre-industrial_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.json"
)

# %%
known_hashes = {
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc218/gm/v20241205/pfc218_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "ec3268b69d93dc3e8a145779a8147c3081ba296d16fd1921787c0ecd047115ff",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc3110/gm/v20241205/pfc3110_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "90cfb99191fd9400fb39090099efb8c9de1f20a361312be5412d2059b771a27b",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc4112/gm/v20241205/pfc4112_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "9386660172c2a14922b65e024471418dae4e37305632012462bef1d974f46be3",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc5114/gm/v20241205/pfc5114_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "3192923ebef94e4aec1e54d21e03407208c3fde5492fd2ae6f0bded6fc342056",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc6116/gm/v20241205/pfc6116_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "9f615f3c2acea2044a1f29d603a0f6c2d3d846f20f09efff8dae7d37b205c2cc",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc7118/gm/v20241205/pfc7118_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "a9263d21124f289cc37d81f0823ca49fb3c0338b3636a4c924719b622d06a063",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/ch2cl2/gm/v20241205/ch2cl2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "0071728ce3eb88bb096a778987b84225b5ff63bd71300e1d7c2f5df27bf37096",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/ch3br/gm/v20241205/ch3br_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "c24f330cfef24e853d641b3fcafbd664aede3f527df3fc635d040df75fd47bae",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/ch3cl/gm/v20241205/ch3cl_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "071f4e35366730b39510aa43e6f1bb9096588050554f78ca20cade73cf5b5256",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/chcl3/gm/v20241205/chcl3_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "7ce10d9deab218f45ce225636ef2e650a1a0fb68f931c2e0d89e6347e7f2e192",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/nf3/gm/v20241205/nf3_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "63f8040cf80b7ca759a223a685b6551535b136f795ea1ed4ff7b2e0a7da380b9",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/so2f2/gm/v20241205/so2f2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "591ca47be1f6ec0c097815927e81ed39dbc0d1a95122e092df08282877edf5c3",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc318/gm/v20241205/pfc318_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "3e79c42aa78249229f6754e48b045ff7a42872b81012d5b9b2d6ed56630bb4ac",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc116/gm/v20241205/pfc116_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "335977b1cabaf777ed8636b4d8eebbcf4fe15c655fe1c1399b7fd305637d8bc4",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/cf4/gm/v20241205/cf4_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "98403b75d8b80bcaf8f11948e2abe50152f73b8ed9077acab338c24ca8d917a4",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/sf6/gm/v20241205/sf6_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "10b86d98513b1478ebe66ba0f0a389b71f7f4ff89bf5501608d0bef0fa78de2e",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc134a/gm/v20241205/hfc134a_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "694c4bbc920c349464b0f30b4bfcd95d505c3de73f49d3325bcfb83b60f9120b",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc245fa/gm/v20241205/hfc245fa_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "0cb771733abdb0ca5153485f49a437a7a99ca67fea8238d074f93e87f934b532",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc4310mee/gm/v20241205/hfc4310mee_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "f27b811d06b5464f384be3c16fa09060569b54a4b2094a91999c1432d0129a2d",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc152a/gm/v20241205/hfc152a_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "591199d4d6581f04d5cf705289fe7cbaab782e8f9507e9a176f8a1b5b87034ed",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc236fa/gm/v20241205/hfc236fa_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "0930150c238e26553dbd34cd9160e2b9f8db42ddcad21e7bf354c446456f74dc",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc365mfc/gm/v20241205/hfc365mfc_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "5c4883532bf08ea45c936bfa3ff7a66a91f71fe82b6d090c3f4966749299f3b0",  # noqa: E501
}

# %%
CMIP7_TO_HERE_VARIABLE_MAP = {
    "hfc134a": "HFC134a",
    "hfc152a": "HFC152a",
    "hfc236fa": "HFC236fa",
    "hfc245fa": "HFC245fa",
    "hfc365mfc": "HFC365mfc",
    "hfc4310mee": "HFC43-10",
    "cf4": "CF4",
    "pfc116": "C2F6",
    "pfc218": "C3F8",
    "pfc3110": "C4F10",
    "pfc4112": "C5F12",
    "pfc5114": "C6F14",
    "pfc6116": "C7F16",
    "pfc7118": "C8F18",
    "pfc318": "cC4F8",
    "ch2cl2": "CH2Cl2",
    "ch3br": "CH3Br",
    "ch3cl": "CH3Cl",
    "chcl3": "CHCl3",
    "nf3": "NF3",
    "sf6": "SF6",
    "so2f2": "SO2F2",
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
    "HFC134a": Q(13.5, "yr"),
    "HFC152a": Q(1.5, "yr"),
    "HFC236fa": Q(213.0, "yr"),
    "HFC245fa": Q(6.61, "yr"),
    "HFC365mfc": Q(8.86, "yr"),
    "HFC43-10": Q(17.0, "yr"),
    "NF3": Q(569.0, "yr"),
    "SF6": Q((850 + 1280) / 2.0, "yr"),
    "SO2F2": Q(36.0, "yr"),
    "cC4F8": Q(3200.0, "yr"),
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
    "HFC134a": Q(12.01 + 2 * 1.0 + 19.0 + 12.01 + 3 * 19.0, "gHFC134a / mole"),  # CH2FCF3
    "HFC152a": Q(12.01 + 3 * 1.0 + 12.01 + 1.0 + 2 * 19.0, "gHFC152a / mole"),  # CH3CHF2
    "HFC236fa": Q(12.01 + 3 * 19.0 + 12.01 + 2 * 1.0 + 12.01 + 3 * 19.0, "gHFC236fa / mole"),  # CF3CH2CF3
    "HFC245fa": Q(
        12.01 + 2 * 1.0 + 19.0 + 12.01 + 2 * 19.0 + 12.01 + 1.0 + 2 * 19.0, "gHFC245fa / mole"
    ),  # CH2FCF2CHF2
    "HFC365mfc": Q(
        12.01 + 3 * 1.0 + 12.01 + 2 * 19.0 + 12.01 + 2 * 1.0 + 12.01 + 3 * 19.0, "gHFC365mfc / mole"
    ),  # CH3CF2CH2CF3
    "HFC43-10": Q(
        12.01 + 3 * 19.0 + 2 * (12.01 + 1.0 + 19.0) + 12.01 + 2 * 19.0 + 12.01 + 3 * 19.0, "gHFC4310 / mole"
    ),  # CF3CHFCHFCF2CF3
    "NF3": Q(14.01 + 3 * 19.0, "gNF3 / mole"),
    "SF6": Q(32.07 + 6 * 19.0, "gSF6 / mole"),
    "SO2F2": Q(32.07 + 2 * 16.0 + 2 * 19.0, "gSO2F2 / mole"),
    "cC4F8": Q(4 * 12.01 + 8 * 19.0, "gcC4F8 / mole"),
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
assumed_unit = "ppt"
background_emissions_d = {}
inverse_info_l = []
out_ts_vars = [*FOLLOW_LEADERS.keys(), "Emissions|C6F14"]
out_pi_vars = [
    "Emissions|CF4",
    "Emissions|C2F6",
    "Emissions|HFC|HFC152a",
    "Emissions|HFC|HFC236fa",
    "Emissions|HFC|HFC365mfc",
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC43-10",
    "Emissions|SF6",
]
pi_year = 1750
for emissions_var in tqdman.tqdm(sorted({*out_ts_vars, *out_pi_vars})):
    if "HFC" in emissions_var and emissions_var not in out_pi_vars:
        # Use Guus' data instead
        continue

    gas = emissions_var.split("|")[-1]
    variable_id_cmip = here_to_cmip7_variable_map[gas]
    download_url = f"https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/{source_id}/atmos/yr/{variable_id_cmip}/gm/{pub_date}/{variable_id_cmip}_input4MIPs_GHGConcentrations_CMIP_{source_id}_gm_1750-2022.nc"

    if download_url not in known_hashes:
        pooch.retrieve(
            download_url,
            known_hash=None,
            path=DOWNLOAD_PATH,
        )

    else:
        pooch.retrieve(
            download_url,
            known_hash=known_hashes[download_url],
            path=DOWNLOAD_PATH,
        )

    gas = emissions_var.split("|")[-1]
    variable_id_cmip = here_to_cmip7_variable_map[gas]
    target_unit = f"kt {gas}/yr".replace("-", "")

    lifetime = wmo_lifetimes[gas]
    molecular_mass = molecular_masses[gas]

    to_load = list(DOWNLOAD_PATH.glob(f"*{variable_id_cmip}*.nc"))
    if len(to_load) != 1:
        raise AssertionError

    ds = xr.open_dataset(to_load[0])
    units = ds[variable_id_cmip].attrs["units"]
    concs = Q(ds[variable_id_cmip].values, units)

    if units == "ppt":
        fraction_factor = Q(1e-12, "1 / ppt")
    else:
        raise NotImplementedError(units)

    emm_factor = 1 / (atm_moles * fraction_factor * molecular_mass)

    if emissions_var in out_pi_vars:
        background_emissions = concs[list(ds["time"].dt.year.values).index(pi_year)] / lifetime / emm_factor
        background_emissions_d[emissions_var] = (float(background_emissions.to(target_unit).m), target_unit)
        if emissions_var not in out_ts_vars:
            continue

    elif emissions_var in out_ts_vars:
        pass

    else:
        raise NotImplementedError(emissions_var)

    # Implicitly assume that first year values are flat
    if concs[1] != concs[0]:
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
    # The check doesn't work otherwise

    x = ds[variable_id_cmip]["time"].dt.year.values.squeeze()
    y = inverse_emissions.to(target_unit).m.squeeze()

    y_smooth = scipy.interpolate.make_smoothing_spline(x=x, y=y, lam=100.0)(x)
    y_smooth[np.where(y_smooth < 0.0)] = 0.0
    y_smooth[np.where(y_smooth < y_smooth.max() * 1e-6)] = 0.0

    df = pd.DataFrame(
        np.vstack([y, y_smooth, concs.to(units).m]),
        columns=pd.Index(x, name="year"),
        index=pd.MultiIndex.from_tuples(
            (
                (f"{source_id}-inverse", HISTORY_SCENARIO_NAME, "World", emissions_var, target_unit),
                (f"{source_id}-inverse-smooth", HISTORY_SCENARIO_NAME, "World", emissions_var, target_unit),
                ("CMIP-concs", HISTORY_SCENARIO_NAME, "World", f"Atmospheric Concentrations|{gas}", units),
            ),
            names=["model", "scenario", "region", "variable", "unit"],
        ),
    )
    inverse_info_l.append(df)

# %%
background_emissions_d

# %%
inverse_info = pix.concat(inverse_info_l)
inverse_info

# %%
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
inverse_info.loc[pix.ismatch(variable="Emissions**", model=f"{source_id}-inverse-smooth")].to_csv(OUT_PATH)
OUT_PATH

# %%
OUT_PATH_PI_EMMS

# %%
OUT_PATH_PI_EMMS.parent.mkdir(exist_ok=True, parents=True)
with open(OUT_PATH_PI_EMMS, "w") as fh:
    json.dump(background_emissions_d, fh, indent=2)

OUT_PATH_PI_EMMS
