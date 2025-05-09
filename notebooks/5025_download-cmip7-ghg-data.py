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
# # Download CMIP7 GHG data
#
# Download the greenhouse gas (GHG) data
# used for CMIP7: https://github.com/climate-resource/CMIP-GHG-Concentration-Generation.

# %% [markdown]
# ## Imports

# %%

import pandas_indexing  # noqa: F401
import pooch
import tqdm.auto

from emissions_harmonization_historical.constants_5000 import (
    CMIP7_GHG_PUB_DATE,
    CMIP7_GHG_RAW_PATH,
    CMIP7_GHG_VERSION_ID,
)

# %% [markdown]
# ## Download

# %%
cmip_variable_ids = [
    ("c2f6", "9cab68e4ca3bf4cdbe2fd70c48e00decce0058716492a933e85f609a89540064"),
    ("c3f8", "7dd9099c46585181ffdc8b0c358433296e4445d12378006acc31d70a8f16f396"),
    ("c4f10", "604f4689a571b95eb93f8227bddc85469bc16c5c6c219c37c75cd6409ac32e8a"),
    ("c5f12", "27ce618c16b302089c64ae58c50c69b4d0e8d96ab1ba86080b1ca0d0a274810b"),
    ("c6f14", "987437737329a5150ff0eb3d03bad0ad145a8fa788931249992e6dea98dc59b2"),
    ("c7f16", "c8a9fd81a48e65fbc81f1dc431a9a16d5361ce225a6d7f223e25dfe9b921e02d"),
    ("c8f18", "b341f82a37f1bec2f657214473d674adeb4487c1ab35a8330683e5632485ab7c"),
    ("cc4f8", "1fef3d48b0b98d728b30fa2c273fbe0c53f5c3a4a940177888acc4d8746fb245"),
    ("ccl4", "e43a9d2bc9c4147deb93b9de83a9c1a1dfa706f5db73c851d3c98b55885aeff3"),
    ("cf4", "54dde0126a45b790e062bd906cd4790634bd907ad82578b0abd7f8eb94df31e0"),
    ("cfc11", "cb464b65f7c9eb2b410cd2a224881ad7de6b263c7ecf5f4fc338eceb5ba55f05"),
    ("cfc113", "8e61e09f5d7f8b4447745b6c535f76276ab5177edb1b3fa8b851f027b922237c"),
    ("cfc114", "1cd5e65a8475da672e44323b2a6d1612f8caab67f9a8f7d0667b65d3ee06120b"),
    ("cfc115", "77555f82995cd2da98a1c4a0357f912ee2ac26118a55920daeb106b2fddc68f3"),
    ("cfc12", "24e26469b3890a75b69e6e8bef58912212b0e1b817323112683f94ba7bb34d0c"),
    ("ch2cl2", "556177a4623500f0c162b4b37fc1356cfe9a2e68a7e212854471965caf97900e"),
    ("ch3br", "a3e7cfac547e5116b007ca3ba3068bfbe08b1e51149ba1e6f794fc752f21db43"),
    ("ch3ccl3", "6ba68935768b9fce0abf836911d87926d7fa7a7c4206ab224414c64ced25a31e"),
    ("ch3cl", "ca47bb4b1e6eabd1e29b4e086d3fffa460f7ea35a765c3489162b14bfd31599d"),
    ("chcl3", "538d9abf2ec711fa1fa5a802dece642e98dfedf76596f4d0c32b6d9a57eb4d60"),
    ("halon1211", "634e34cc2312e443885b279f34c4f95e1086b4ef651cfd7754cf6aa1bb8ba6b3"),
    ("halon1301", "42ba5f6cb52abd1f558a252e0ddb73a4ff60c677c0f488340aa21014314049b4"),
    ("halon2402", "593a0916ec58775d965c92687f37ffe8ddeef3bee94cafa16a8a433777b4e4f2"),
    ("hcfc141b", "abdc922ddd71de3c9280cdc498a84ae7759173557d1c57d0e7e98747715ca260"),
    ("hcfc142b", "cd61d397c2d2342038456892b3296458cbb433f01807deeaba27c9f05a5f8ab8"),
    ("hcfc22", "f38e5bb2ce57f89220dd00a3e0b1e43287f1fad02a46a8ca9725dbfc2885c1c8"),
    ("hfc125", "3b8e27ec5ff392de1f3756676891bcf9efb9738a2036793c418a70d829b1131c"),
    ("hfc134a", "1ead1cead68454ce95c22699be53dd918b9ccb315dfe2cd0991798be62ce038c"),
    ("hfc143a", "6d915470a98ee2a53ea829a3772047f7f58a73af0e0b4778644e1d7ee4adda27"),
    ("hfc152a", "5a98e80c924caabe881dbf44d1eb270902f9b87dba329ee403652a71312d11b5"),
    ("hfc227ea", "3fd3ae71ab5a83f2b0a33dbaedb222d2168a13cc81fc61e8f78db5b4fba436c2"),
    ("hfc23", "721f9f993c0a2ab9a2977ec940f7e9c4425a11ce78824cfd871e3b5569ed7b91"),
    ("hfc236fa", "c73102013e646beebc180957d8de22934610b24678a929612b145c8a96e0f942"),
    ("hfc245fa", "80925cc15a7a7cee27502c9702b11535ae98a85dc05ad609391c28ac32e4eebd"),
    ("hfc32", "77d6994158d390c2c7f44be7dabbec501d806ef0fc76ca114916df8826567e4c"),
    ("hfc365mfc", "e5d03fdb61a5536fcad277e1969d914001340d0545f5e2534931bafb2bd5687a"),
    ("hfc4310mee", "4dad51cf6d07af0ebdceb2b8c7893abe9aed74a34e1ce2c4dc69dcd9ee3fb6e5"),
    ("n2o", "49db1fbe14bdfad9c9b5a78e0b37b50007394f6fb252ff39b622b4c01ef934c4"),
    ("nf3", "013c723eede5a757d70763bee9313b7c340e20e8f1e1c6e493ef87119e5b48ff"),
    ("sf6", "731d94af578e832e6b1e2e512f7e2bed926f4f4d9d0075409edc0b24ad5ed20e"),
    ("so2f2", "aed6f09e098fb27472aeafaad6ebf1bceafdeca8dd22a3f5e01cb26ca6aa4b9f"),
]

# %%
for cmip_variable_id, known_hash in tqdm.auto.tqdm(sorted(cmip_variable_ids)):
    # converting to set then back to list removes duplicates

    download_url = f"https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/CR/{CMIP7_GHG_VERSION_ID}/atmos/yr/{cmip_variable_id}/gm/{CMIP7_GHG_PUB_DATE}/{cmip_variable_id}_input4MIPs_GHGConcentrations_CMIP_{CMIP7_GHG_VERSION_ID}_gm_1750-2022.nc"

    downloaded_file = pooch.retrieve(
        download_url,
        known_hash=known_hash,
        path=CMIP7_GHG_RAW_PATH,
    )
