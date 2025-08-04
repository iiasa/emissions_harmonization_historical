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
import json
import tempfile
from pathlib import Path

import pyam
import tqdm.auto

from emissions_harmonization_historical.constants_5000 import (
    DATA_ROOT,
    DOWNLOAD_SCENARIOS_ID,
    RAW_SCENARIO_DB,
)

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model_search: str = "IMAGE"

# %%
output_dir_model = DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / model_search
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %%
versions_file = DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / "versions.json"
versions_file.parent.mkdir(exist_ok=True, parents=True)
versions_file

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Load the known hashes

# %%
if versions_file.exists():
    with open(versions_file) as fh:
        known_versions = json.load(fh)

else:
    known_versions = {}

known_versions

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

# if model_search == "REMIND":
#     to_download = to_download[to_download["scenario"].str.endswith("- Very Low Emissions")]
# if model_search == "AIM":
#     # to_download = to_download[to_download["scenario"].str.endswith("- Low Overshoot")]
#     to_download = to_download[to_download["scenario"].str.contains("- Low Overshoot")]
# if model_search == "MESSAGE":
#     to_download = to_download[to_download["scenario"].str.endswith("- Low Emissions")]
# if model_search == "IMAGE":
#     to_download = to_download[to_download["scenario"].str.endswith("- Medium Emissions")]
# if model_search == "COFFEE":
#     to_download = to_download[to_download["scenario"].str.endswith("- Medium-Low Emissions")]
if model_search == "GCAM":
    # to_download = to_download[to_download["scenario"].str.endswith("- High Emissions")]
    to_download = to_download[to_download["scenario"].str.contains("SSP3 - High Emissions")]
# if model_search == "WITCH":
#     to_download = to_download[to_download["scenario"].str.contains("- Medium-Low Emissions")]

to_download.shape[0]

# %%
to_download  # .head(2)

# %% [markdown]
# ### Check the versions
#
# If the scenario doesn't match what we expect, raise an error
# because we need to update to use the new data.

# %%
for _, row in tqdm.auto.tqdm(to_download.iterrows(), total=to_download.shape[0]):
    model = row.model
    scenario = row.scenario
    scenario_clean = scenario.replace(" ", "_")
    version = row.version

    if model not in known_versions:
        known_versions[model] = {}

    if model in known_versions and scenario in known_versions[model]:
        if version != known_versions[model][scenario]:
            msg = (
                f"Scenario data has changed for {model} {scenario}. "
                f"Current version: {version}. "
                f"Known version: {known_versions[model][scenario]}. "
                "Please update the value of `DOWNLOAD_SCENARIOS_ID`."
            )
            raise AssertionError(msg)
        # else:
        # Already know about this scenario and data matches the expected hash
    else:
        known_versions[model][scenario] = version

    with open(versions_file, "w") as fh:
        json.dump(known_versions, fh, indent=2, sort_keys=True)
        # add new line to end of file
        fh.write("\n")

    # Save the properties
    row.to_frame().T.to_csv(output_dir_model / f"{scenario_clean}_properties.csv", index=False)

# %% [markdown]
# ### Download the scenarios

# %%
tmpdir = Path(tempfile.mkdtemp(prefix="ssp-submission-db"))

# %%
for _, row in tqdm.auto.tqdm(to_download.iterrows(), total=to_download.shape[0]):
    model = row.model
    scenario = row.scenario

    df = pyam.read_iiasa("ssp_submission", model=model, scenario=scenario, variable=["Emissions|*", "Carbon Removal|*"])
    if df.empty:
        msg = f"No data for {model=} {scenario=}"
        raise AssertionError(msg)

    df_ts = df.timeseries()

    # If we got to this stage, we are fine with an overwrite
    RAW_SCENARIO_DB.save(df_ts, allow_overwrite=True)


# %%
db_metadata = RAW_SCENARIO_DB.load_metadata().to_frame(index=False)
db_metadata[["model", "scenario"]][db_metadata["model"].str.contains(model_search)].drop_duplicates().reset_index(
    drop=True
)
