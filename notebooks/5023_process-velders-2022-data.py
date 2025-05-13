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
# # Process Velders et al., 2022 data
#
# Process data from [Velders et al., 2022](https://doi.org/10.5194/acp-22-6087-2022).

# %% [markdown]
# ## Imports

# %%
import io
from functools import partial

import pandas as pd
import pandas_indexing as pix
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_SCENARIO_NAME,
    VELDERS_ET_AL_2022_PROCESSED_DB,
    VELDERS_ET_AL_2022_RAW_PATH,
)

# %% [markdown]
# ## Setup

# %%
pix.set_openscm_registry_as_default()


# %% [markdown]
# ## Load data

# %%
with open(VELDERS_ET_AL_2022_RAW_PATH / "KGL2021_constrProdEmis_ObsAgage_2500_OECD-SSP5.dat") as fh:
    raw = tuple(fh.readlines())

# %% [markdown]
# ## Reformat

# %%
expected_species = [
    "HFC-32",
    "HFC-125",
    "HFC-134a",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-236fa",
    "HFC-245fa",
    "HFC-365mfc",
    "HFC-43-10mee",
]

# %%
start_idx = 74
block_length = 512
expected_n_blocks = 10

# %%
clean_l = []
for i in range(expected_n_blocks):
    start = start_idx + i * (block_length + 1)

    block = io.StringIO("\n".join(raw[start : start + block_length]))
    species_df = pd.read_csv(block, sep=r"\s+")

    gas = species_df["Species"].unique()
    if len(gas) != 1:
        raise AssertionError
    gas = gas[0]

    keep = species_df[["Year", "Emis_tot"]].rename({"Year": "year", "Emis_tot": str(gas)}, axis="columns")
    keep = keep.set_index("year")
    # display(keep)

    clean_l.append(keep)

clean = pd.concat(clean_l, axis="columns")
if set(clean.columns) != set(expected_species):
    raise AssertionError

clean

# %%
velders_variable_normalisation_map = {
    "HFC-32": "HFC32",
    "HFC-125": "HFC125",
    "HFC-134a": "HFC134a",
    "HFC-143a": "HFC143a",
    "HFC-152a": "HFC152a",
    "HFC-227ea": "HFC227ea",
    "HFC-236fa": "HFC236fa",
    "HFC-245fa": "HFC245fa",
    "HFC-365mfc": "HFC365mfc",
    "HFC-43-10mee": "HFC4310mee",
}
clean = clean.rename(velders_variable_normalisation_map, axis="columns").T
clean.columns = clean.columns.astype(int)
clean.index.name = "variable"
# Assuming that input data unit is t / yr
clean = clean / 1000  # convert to kt / yr
clean = clean.pix.format(unit="kt {variable}/yr")
clean = clean.pix.format(variable="Emissions|{variable}")
clean

# %%
out = clean.pix.assign(model="Velders et al., 2022", scenario=HISTORY_SCENARIO_NAME, region="World")
out

# %% [markdown]
# ## Save

# %% [markdown]
# Make sure we can convert the variable names

# %%
update_index_levels_func(
    out,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
VELDERS_ET_AL_2022_PROCESSED_DB.save(out, allow_overwrite=True)
