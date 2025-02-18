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
#
# Note on N2O: Global N2O budget is reported in units
# of TgN / yr, but the unit is really Tg N2 / yr. The
# clearest evidence of this is the Supplement to
# Global N2O budget Table 3, where
# 44 / 28 = 1.57 (https://essd.copernicus.org/articles/16/2543/2024/essd-16-2543-2024-supplement.pdf)
#
# I calculate 1750 total N2O emissions to be in the region of 19 Tg N2O / yr from the inversion,
# which is close to the middle of the natural range from GNB (natural sources 11.8 TgN2 / yr = 18.5 Tg N2O / yr).

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
FOLLOW_LEADERS

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
OUT_PATH_EMMS = DATA_ROOT / "global" / "esgf" / source_id / f"inverse_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.csv"
OUT_PATH_CONCS = DATA_ROOT / "global" / "esgf" / source_id / f"concentrations_{CMIP_CONCENTRATION_INVERSION_ID}.csv"
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
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/cfc115/gm/v20241205/cfc115_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "c420e55a979267274b455867c8c89d5dee255e80a568e32a70f657a7c7b82725",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/cfc11/gm/v20241205/cfc11_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "41fdc3026ed80b1204fda052848480dce94b80f7f39f8bc0af589daaf1753354",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/cfc113/gm/v20241205/cfc113_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "54464a39d0cf2dad8c7bc5c2931187d6f6d580c3b971e4a9b12546b2135fed23",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/cfc114/gm/v20241205/cfc114_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "f2b8d39b29902f639cda82b769e19d00d49e684ea34a795b8d0bc6d329b19c6d",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/cfc12/gm/v20241205/cfc12_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "502b33315c880129a1bf0c70153db1ee88c064d3981663d690060dd52fd0b2fb",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hcfc142b/gr1z/v20241205/hcfc142b_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gr1z_1750-2022.nc": "70732b36a361cafe216c352eb08f3c9af0ed3a3cdf05789f042a7e93ca869a04",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hcfc22/gm/v20241205/hcfc22_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "4cfd71cfc715dcf3a144e0ae3f8253cf3d2a340c0838fa0afcd254f15f5d9b26",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc125/gm/v20241205/hfc125_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "9fa213defc99c984b980756e3ee441bd072c303d724e308f001d5ceb444e8b3f",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc32/gm/v20241205/hfc32_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "71669dc1cfb7731aee38be8d376e923d2f4569a40205c313ed8cd5512930bfea",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/n2o/gm/v20241205/n2o_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "49d8edd9000283479892dc4e6e978c399fefe777bbc839f0036a76393f36cee6",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc23/gm/v20241205/hfc23_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "99d4389cb6a115dde7e7cc5f06e6c6f3c54a1e82eedae676ea3e833b58307c8a",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/halon2402/gm/v20241205/halon2402_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "eb46c0145ea7a19e2c36e74ae6dd759de95bec47adf5113591622db9696d1b73",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hcfc141b/gm/v20241205/hcfc141b_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "7a965274951b229ab3e407db35ed29da845282ce01a6dfffffb48e07c9250e2a",  # noqa: E501,
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hcfc142b/gm/v20241205/hcfc142b_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "1af3fa2918f5550d1b722e22579a7649ffb3fd6c10a3eebbff333844503f81b4",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc143a/gm/v20241205/hfc143a_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "ed4fca4e3d3d56e27f6223af75f82deddad54f7a323b4b0e26f4d78915c1337d",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hfc227ea/gm/v20241205/hfc227ea_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "9d4376171f8146ea6fe23cca1568d108036d34815a056a815de97901d251940f",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/halon1211/gm/v20241205/halon1211_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "01fabc09dd6a1a3f895798ad5276242bcf0d306c97594b837040e1053d45cc25",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/halon1301/gm/v20241205/halon1301_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "0d5426374e30cc0f73bb33db2999a28d8c0f82905365ebd6feb0052afec74756",  # noqa: E501
    "https://esgf-data2.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/hcc140a/gm/v20241205/hcc140a_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "c4dd32571455aaf749abe8bf2c6ac81bcbcc1b59020c76c0960cccb9ed59038b",  # noqa: E501
}

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
    "hfc4310mee": "HFC43-10",
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
    "hcc140a": "CH3CCl3",
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
    "HFC43-10": Q(17.0, "yr"),
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
    "HCFC142b": Q(1.58, "yr"),
    "CFC11": Q(52, "yr"),
    "CFC12": Q(102, "yr"),
    "CFC113": Q(93, "yr"),
    "CFC114": Q(189, "yr"),
    "CFC115": Q(540, "yr"),
    "CH3CCl3": Q(5, "yr"),
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
    "HFC43-10": Q(
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
background_emissions_d = {}
inverse_info_l = []
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
    "Emissions|HFC|HFC23",
    "Emissions|HFC|HFC32",
    "Emissions|HFC|HFC125",
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC152a",
    "Emissions|HFC|HFC227ea",
    "Emissions|HFC|HFC236fa",
    "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC365mfc",
    "Emissions|HFC|HFC43-10",
    "Emissions|NF3",
    "Emissions|SF6",
    "Emissions|SO2F2",
    "Emissions|N2O",  # we'll not do N2O for now, scale PRIMAP instead
    "Emissions|HFC|HFC43-10",
    "Emissions|Montreal Gases|CH3CCl3",  # not in CR data
    "Emissions|Montreal Gases|CH2Cl2",
    "Emissions|Montreal Gases|CHCl3",
    "Emissions|Montreal Gases|CH3Br",
    "Emissions|Montreal Gases|CH3Cl",
    "Emissions|Montreal Gases|CFC|CFC11",
    "Emissions|Montreal Gases|CFC|CFC12",
    "Emissions|Montreal Gases|CFC|CFC113",
    "Emissions|Montreal Gases|CFC|CFC114",
    "Emissions|Montreal Gases|CFC|CFC115",
    "Emissions|Montreal Gases|Halon1211",
    "Emissions|Montreal Gases|Halon1301",
    "Emissions|Montreal Gases|Halon2402",
]

# conc_unit = {"variable: ppt" for variable in out_ts_vars}
# conc_unit["Emissions|N2O"] = "ppb"

pi_year = 1750
for emissions_var in tqdman.tqdm(sorted(out_ts_vars)):
    # converting to set then back to list removes duplicates

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

    to_load = list(DOWNLOAD_PATH.glob(f"*{variable_id_cmip}_*.nc"))
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

    if emissions_var in out_ts_vars:
        background_emissions = concs[list(ds["time"].dt.year.values).index(pi_year)] / lifetime / emm_factor
        background_emissions_d[emissions_var] = (float(background_emissions.to(target_unit).m), target_unit)
        if emissions_var not in out_ts_vars:
            continue

    else:
        raise NotImplementedError(emissions_var)

    # Implicitly assume that first year values are flat, override for N2O
    if gas == "N2O":
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
    concs_check[0] = concs[list(ds["time"].dt.year.values).index(pi_year)]
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
OUT_PATH_EMMS.parent.mkdir(exist_ok=True, parents=True)
inverse_info.loc[pix.ismatch(variable="Emissions**", model=f"{source_id}-inverse-smooth")].to_csv(OUT_PATH_EMMS)
OUT_PATH_EMMS

# %%
OUT_PATH_PI_EMMS

# %%
OUT_PATH_PI_EMMS.parent.mkdir(exist_ok=True, parents=True)
with open(OUT_PATH_PI_EMMS, "w") as fh:
    json.dump(background_emissions_d, fh, indent=2)

OUT_PATH_PI_EMMS

# %%
inverse_info.loc[pix.ismatch(variable="Atmospheric Concentrations**", model="CMIP-concs")].to_csv(OUT_PATH_CONCS)

# %%
OUT_PATH_CONCS

# %%
