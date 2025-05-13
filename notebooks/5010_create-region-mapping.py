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
# # Create region mapping

# %% [markdown]
# ## Imports

# %%
from git import Repo

from emissions_harmonization_historical.constants_5000 import (
    COMMON_DEFINITIONS_COMMIT,
    COMMON_DEFINITIONS_PATH,
    REGION_MAPPING_FILE,
)
from emissions_harmonization_historical.region_mapping import create_region_mapping

# %% [markdown]
# ## Set common definitions to right commit

# %%
if COMMON_DEFINITIONS_PATH.exists():
    repo = Repo(COMMON_DEFINITIONS_PATH)

else:
    print("Cloning common-definitions")
    repo = Repo.clone_from("https://github.com/IAMconsortium/common-definitions", COMMON_DEFINITIONS_PATH)

repo.git.checkout(COMMON_DEFINITIONS_COMMIT)

# %% [markdown]
# ## Create the region mapping

# %%
REGION_MAPPING_FILE.parent.mkdir(exist_ok=True, parents=True)

# %%
create_region_mapping(
    out_file=REGION_MAPPING_FILE,
    common_definitions_path=COMMON_DEFINITIONS_PATH,
)
