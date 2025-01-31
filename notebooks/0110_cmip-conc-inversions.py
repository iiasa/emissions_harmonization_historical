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
# # CMIP concentration inversions
#
# Invert concentrations from CMIP7.
# We do this for a very limited number of species
# where we have no other obvious option.

# %%
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pooch
import seaborn as sns

from emissions_harmonization_historical.constants import DATA_ROOT, CMIP_CONCENTRATION_INVERSION_ID, WMO_2022_PROCESSING_ID
from emissions_harmonization_historical.infilling_followers import FOLLOW_LEADERS

# %%
source_id = "CR-CMIP-0-4-0"
pub_date = "v20241205"

# %%
DOWNLOAD_PATH = DATA_ROOT / "global" / "esgf" / source_id
DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)

# %%
known_hashes = {}

# %%
CMIP7_TO_HERE_VARIABLE_MAP = {
    'pfc218': 'C3F8',
    'pfc3110': 'C4F10',
    'pfc4112': 'C5F12',
    # 'pfc5114': 'c6f14',
    'pfc6116': 'C7F16',
    'pfc7118': 'C8F18',
    # 'pfc318': 'cc4f8',
    # 'ccl4': 'ccl4',
    # 'cf4': 'cf4',
    # 'cfc11': 'cfc11',
    # 'cfc113': 'cfc113',
    # 'cfc114': 'cfc114',
    # 'cfc115': 'cfc115',
    # 'cfc12': 'cfc12',
    'ch2cl2': 'CH2Cl2',
    'ch3br': 'CH3Br',
    # 'hcc140a': 'ch3ccl3',
    'ch3cl': 'CH3Cl',
    'chcl3': 'CHCl3',
    # 'halon1211': 'halon1211',
    # 'halon1301': 'halon1301',
    # 'halon2402': 'halon2402',
    # 'hcfc141b': 'hcfc141b',
    # 'hcfc142b': 'hcfc142b',
    # 'hcfc22': 'hcfc22',
    # 'hfc125': 'hfc125',
    # 'hfc134a': 'hfc134a',
    # 'hfc143a': 'hfc143a',
    # 'hfc152a': 'hfc152a',
    # 'hfc227ea': 'hfc227ea',
    # 'hfc23': 'hfc23',
    # 'hfc236fa': 'hfc236fa',
    # 'hfc245fa': 'hfc245fa',
    # 'hfc32': 'hfc32',
    # 'hfc365mfc': 'hfc365mfc',
    # 'hfc4310mee': 'hfc4310mee',
    'nf3': 'NF3',
    # 'sf6': 'sf6',
    'so2f2': 'SO2F2',
    # 'cfc11eq': 'cfc11eq',
    # 'cfc12eq': 'cfc12eq',
    # 'hfc134aeq': 'hfc134aeq',
}
here_to_cmip7_variable_map = {v: k for k, v in CMIP7_TO_HERE_VARIABLE_MAP.items()}

# %%
known_hashes = {
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc218/gm/v20241205/pfc218_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "ec3268b69d93dc3e8a145779a8147c3081ba296d16fd1921787c0ecd047115ff",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc3110/gm/v20241205/pfc3110_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "90cfb99191fd9400fb39090099efb8c9de1f20a361312be5412d2059b771a27b",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc4112/gm/v20241205/pfc4112_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "9386660172c2a14922b65e024471418dae4e37305632012462bef1d974f46be3",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc6116/gm/v20241205/pfc6116_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "9f615f3c2acea2044a1f29d603a0f6c2d3d846f20f09efff8dae7d37b205c2cc",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/pfc7118/gm/v20241205/pfc7118_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "a9263d21124f289cc37d81f0823ca49fb3c0338b3636a4c924719b622d06a063",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/ch2cl2/gm/v20241205/ch2cl2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "0071728ce3eb88bb096a778987b84225b5ff63bd71300e1d7c2f5df27bf37096",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/ch3br/gm/v20241205/ch3br_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "c24f330cfef24e853d641b3fcafbd664aede3f527df3fc635d040df75fd47bae",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/ch3cl/gm/v20241205/ch3cl_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "071f4e35366730b39510aa43e6f1bb9096588050554f78ca20cade73cf5b5256",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/chcl3/gm/v20241205/chcl3_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "7ce10d9deab218f45ce225636ef2e650a1a0fb68f931c2e0d89e6347e7f2e192",
    "https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-4-0/atmos/yr/nf3/gm/v20241205/nf3_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-0-4-0_gm_1750-2022.nc": "63f8040cf80b7ca759a223a685b6551535b136f795ea1ed4ff7b2e0a7da380b9",
}

# %%
for emissions_var in sorted(FOLLOW_LEADERS.keys()):
    if "HFC" in emissions_var or "Halon1202" in emissions_var:
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

# %%

# %%
here_to_cmip7_variable_map[gas]

# %%
pix.set_openscm_registry_as_default()

# %%
wmo_2022_processed_output_file = DATA_ROOT / Path(
    "global",
    "wmo-2022",
    "processed",
    f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv",
)
wmo_2022_processed_output_file

# %%
raw_data_path = DATA_ROOT / "global/wmo-2022/data_raw/"
raw_data_path

# %%
var_map = {
    # WMO name: (gas name, variable name)
    "CF4": ("CF4", "Emissions|CF4"),
    "C2F6": ("C2F6", "Emissions|C2F6"),
    "SF6": ("SF6", "Emissions|SF6"),
}

# %%
res_l = []
for gdir in raw_data_path.glob("*"):
    outputs_dir = gdir / "outputs"
    emissions_file = list(outputs_dir.glob("*Global_annual_emissions.csv"))
    if not emissions_file:
        print(f"No data in {outputs_dir=}")
        continue

    if len(emissions_file) != 1:
        raise NotImplementedError(list(outputs_dir.glob("*")))

    emissions_file = emissions_file[0]

    with open(emissions_file) as fh:
        raw = tuple(fh.readlines())

    for line in raw:
        if "Units:" in line:
            units = line.split("Units: ")[-1].strip()
            break

    else:
        msg = "units not found"
        raise AssertionError(msg)

    raw_df = pd.read_csv(emissions_file, comment="#")

    gas = var_map[gdir.name][0]
    variable = var_map[gdir.name][1]
    out = raw_df[["Year", "Global_annual_emissions"]].rename(
        {"Year": "year", "Global_annual_emissions": "value"}, axis="columns"
    )
    out["variable"] = variable
    out["unit"] = units.replace("/", f" {gas}/")
    out["model"] = "WMO 2022 AGAGE inversions"
    out["scenario"] = "historical"
    out["region"] = "World"
    out = out.set_index(list(set(out.columns) - {"value"}))["value"].unstack("year")
    out = out.pix.convert_unit(f"kt {gas}/yr")

    res_l.append(out)

res = pd.concat(res_l)
res

# %%
wmo_input_file_emissions = DATA_ROOT / "global" / "wmo-2022" / "data_raw" / "Emissions_fromWMO2022.xlsx"
wmo_input_file_emissions

# %%
var_map = {
    # WMO name: (gas name, variable name)
    "CFC-11": ("CFC11", "Emissions|Montreal Gases|CFC|CFC11"),
    "CFC-12": ("CFC12", "Emissions|Montreal Gases|CFC|CFC12"),
    "CFC-113": ("CFC113", "Emissions|Montreal Gases|CFC|CFC113"),
    "CFC-114": ("CFC114", "Emissions|Montreal Gases|CFC|CFC114"),
    "CFC-115": ("CFC115", "Emissions|Montreal Gases|CFC|CFC115"),
    "CCl4": ("CCl4", "Emissions|Montreal Gases|CCl4"),
    "CH3CCl3": ("CH3CCl3", "Emissions|Montreal Gases|CH3CCl3"),
    "HCFC-22": ("HCFC22", "Emissions|Montreal Gases|HCFC22"),
    "HCFC-141b": ("HCFC141b", "Emissions|Montreal Gases|HCFC141b"),
    "HCFC-142b": ("HCFC142b", "Emissions|Montreal Gases|HCFC142b"),
    "halon-1211": ("Halon1211", "Emissions|Montreal Gases|Halon1211"),
    "halon-1202": ("Halon1202", "Emissions|Montreal Gases|Halon1202"),
    "halon-1301": ("Halon1301", "Emissions|Montreal Gases|Halon1301"),
    "halon-2402": ("Halon2402", "Emissions|Montreal Gases|Halon2402"),
}

# %%
wmo_raw = pd.read_excel(wmo_input_file_emissions).rename({"Unnamed: 0": "year"}, axis="columns")
wmo_raw

# %%
# Hand woven unit conversion
wmo_clean = (wmo_raw.set_index("year") / 1000).melt(ignore_index=False)
wmo_clean["unit"] = wmo_clean["variable"].map({k: v[0] for k, v in var_map.items()})
wmo_clean["unit"] = "kt " + wmo_clean["unit"] + "/yr"
wmo_clean["variable"] = wmo_clean["variable"].map({k: v[1] for k, v in var_map.items()})
wmo_clean["model"] = "WMO 2022"
wmo_clean["scenario"] = "WMO 2022 projections v20250129"
wmo_clean["region"] = "World"
wmo_clean = wmo_clean.set_index(["variable", "unit", "model", "scenario", "region"], append=True)["value"].unstack(
    "year"
)
# HACK: remove spurious last year
wmo_clean[2100] = wmo_clean[2099]
# HACK: remove negative values
wmo_clean = wmo_clean.where(wmo_clean >= 0, 0.0)
# TODO: apply some smoother

wmo_clean

# %%
pdf = wmo_clean.melt(ignore_index=False).reset_index()
fg = sns.relplot(
    data=pdf,
    x="year",
    y="value",
    col="variable",
    col_wrap=3,
    hue="model",
    kind="line",
    facet_kws=dict(sharey=False),
)
for ax in fg.figure.axes:
    ax.set_ylim(ymin=0)
    ax.grid()

# %%
out = pix.concat([res, wmo_clean])
out

# %%
wmo_2022_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(wmo_2022_processed_output_file)
wmo_2022_processed_output_file
