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
# # Download scenarios
#
# Here we download scenarios from a given integrated assessment model (IAM).
#
# Scenario data can be downloaded from IIASA's resources here:
# https://data.ece.iiasa.ac.at/ssp-submission.
#
# Before running the notebook,
# you will need to run the following in a new terminal
#
# ```sh
# pixi run ixmp4 login <your-username>
# ```
#
# where `<your-username` is the username you use to login to
# https://data.ece.iiasa.ac.at/ssp-submission.

# %% [markdown]
# ## Imports

# %%
import hashlib
import json
import tempfile
from pathlib import Path

import pyam
import tqdm.auto
from pandas.util import hash_pandas_object

from emissions_harmonization_historical.constants_5000 import (
    DATA_ROOT,
    DOWNLOAD_SCENARIOS_ID,
    RAW_SCENARIO_DB,
)

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model_search: str = "AIM"

# %%
output_dir_model = DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / model_search
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %%
known_hashes_file = DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / "known_hashes.json"
known_hashes_file.parent.mkdir(exist_ok=True, parents=True)
known_hashes_file

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Load the known hashes

# %%
if known_hashes_file.exists():
    with open(known_hashes_file) as fh:
        known_hashes = json.load(fh)

else:
    known_hashes = {}

known_hashes

# %% [markdown]
# ## Download data

# %% [markdown]
# ### Connect to the database

# %%
pyam.iiasa.Connection("ssp_submission")
conn_ssp = pyam.iiasa.Connection("ssp_submission")
props = conn_ssp.properties().reset_index()

# %% [markdown]
# ### Find scenarios to download

# %%
to_download = props[props["model"].str.contains(model_search)]
to_download.shape[0]

# %% [markdown]
# ### Download the scenarios
#
# The logic here is the following:
#
# 1. get the scenario
# 2. get its hash
# 3. if the hash doesn't match what we expect, raise an error
#    because we need to update to use the new data
# 4. if the hash matches or it's a new scenario, save as CSV to the raw directory
#    and save to the database
# 5. also save the properties from the database we can get hold of these later
#    for our own records if needed

# %%
tmpdir = Path(tempfile.mkdtemp(prefix="ssp-submission-db"))

# %%
for _, row in tqdm.auto.tqdm(to_download.iterrows(), total=to_download.shape[0]):
    model = row.model
    scenario = row.scenario
    scenario_clean = scenario.replace(" ", "_")

    df = pyam.read_iiasa("ssp_submission", model=model, scenario=scenario, variable="Emissions|*")
    if df.empty:
        msg = f"No data for {model=} {scenario=}"
        raise AssertionError(msg)

    df_ts = df.timeseries()
    to_hash = df_ts.reorder_levels(sorted(df_ts.index.names)).sort_index().sort_index(axis="columns").reset_index()
    df_ts_hash = hashlib.sha256(hash_pandas_object(to_hash).values).hexdigest()

    if model not in known_hashes:
        known_hashes[model] = {}

    if model in known_hashes and scenario in known_hashes[model]:
        if df_ts_hash != known_hashes[model][scenario]:
            msg = "Scenario data has changed. Please update the value of `DOWNLOAD_SCENARIOS_ID`."
            raise AssertionError(msg)
        # else:
        # Already know about this scenario and data matches the expected hash
    else:
        known_hashes[model][scenario] = df_ts_hash

    with open(known_hashes_file, "w") as fh:
        json.dump(known_hashes, fh, indent=2, sort_keys=True)
        # add new line to end of file
        fh.write("\n")

    # If we got to this stage, we are fine with an overwrite
    RAW_SCENARIO_DB.save(df_ts, allow_overwrite=True)
    row.to_frame().T.to_csv(output_dir_model / f"{scenario_clean}_properties.csv", index=False)
    # break


# %%
db_metadata = RAW_SCENARIO_DB.load_metadata().to_frame(index=False)
db_metadata[["model", "scenario"]][db_metadata["model"].str.contains(model_search)].drop_duplicates().reset_index(
    drop=True
)
