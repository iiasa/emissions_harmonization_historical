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
import ast

import nomenclature
import pandas as pd
from git import Repo

from emissions_harmonization_historical.constants_5000 import (
    COMMON_DEFINITIONS_COMMIT,
    COMMON_DEFINITIONS_PATH,
    REGION_MAPPING_FILE,
    REGION_MAPPING_PATH,
)
from emissions_harmonization_historical.region_mapping import create_region_mapping

# %%
nomenclature.__version__

# %% [markdown]
# ## Set common definitions to right commit

# %%
print(f"{COMMON_DEFINITIONS_COMMIT=}")

# %%
if COMMON_DEFINITIONS_PATH.exists():
    repo = Repo(COMMON_DEFINITIONS_PATH)
    print("Fetching latest updates")
    repo.remotes.origin.fetch()

else:
    print("Cloning common-definitions")
    repo = Repo.clone_from("https://github.com/IAMconsortium/common-definitions", COMMON_DEFINITIONS_PATH)

repo.remotes["origin"].fetch()
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

# %% [markdown]
# ## Create the region mapping for concordia

# %%
regionmapping = pd.read_csv(REGION_MAPPING_FILE, index_col=0)

# %%
models = [
    "GCAM 8s",
    "IMAGE 3.4",
    "WITCH 6.0",
    "COFFEE 1.6",
    "REMIND-MAgPIE 3.5-4.11",
    "MESSAGEix-GLOBIOM-GAINS 2.1-R12",
    "AIM 3.0",
]

# %%
for model in models:
    data = regionmapping[regionmapping["hierarchy"] == model]
    data = data.drop(columns=["countries", "hierarchy"])
    data["iso3"] = data["iso3"].apply(ast.literal_eval)
    output = data.explode("iso3").reset_index(drop=True)

    output.to_csv(REGION_MAPPING_PATH / f"region_definitions_{model}.csv", index=False)
