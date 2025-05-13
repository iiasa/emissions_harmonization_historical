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
# # Download CEDS
#
# Download data from [CEDS](https://github.com/JGCRI/CEDS).

# %% [markdown]
# ## Imports

# %%
import pooch

from emissions_harmonization_historical.constants_5000 import CEDS_RAW_PATH, CEDS_VERSION_ID

# %% [markdown]
# ## Download

# %%
LINK_TO_HIT = "https://zenodo.org/records/15059443/files/CEDS_v_2025_03_18_aggregate.zip?download=1"
if CEDS_VERSION_ID not in LINK_TO_HIT:
    msg = "Please update CEDS_VERSION_ID (and CEDS_PROCESSING_ID)"
    raise AssertionError(msg)

# %%
unzipped_files = pooch.retrieve(
    url=LINK_TO_HIT,
    fname="CEDS_v_2025_03_18_aggregate.zip",
    path=CEDS_RAW_PATH,
    known_hash="b55f0dddb4eb8e14435b13d835f99497b789cb3378eee391d1c78af58c5e8095",
    processor=pooch.Unzip(extract_dir=CEDS_RAW_PATH),
    progressbar=True,
)
len(unzipped_files)

# %%
LINK_TO_HIT = "https://zenodo.org/records/15059443/files/CEDS_v_2025_03_18_supplementary_extension.zip?download=1"
if CEDS_VERSION_ID not in LINK_TO_HIT:
    msg = "Please update CEDS_VERSION_ID (and CEDS_PROCESSING_ID)"
    raise AssertionError(msg)

# %%
unzipped_files = pooch.retrieve(
    url=LINK_TO_HIT,
    fname="CEDS_v_2025_03_18_supplementary_extension.zip",
    path=CEDS_RAW_PATH,
    known_hash="c62973a8745346e5a0df3d381cdc01529aa76877013dfd02b559745571a58813",
    processor=pooch.Unzip(extract_dir=CEDS_RAW_PATH),
    progressbar=True,
)
len(unzipped_files)
