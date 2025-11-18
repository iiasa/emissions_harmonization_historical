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
from functools import partial
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pooch
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    VELDERS_ET_AL_2022_PROCESSED_DB,
    VELDERS_ET_AL_2022_RAW_PATH,
)

# %% [markdown]
# ## Setup

# %%
pix.set_openscm_registry_as_default()


# %%
pandas_openscm.register_pandas_accessor()

# %% [markdown]
# ## Download data

# %%
LINK_TO_HIT = "https://zenodo.org/records/6520707/files/veldersguus/HFC-scenarios-2022-v1.0.zip?download=1"

# %%
unzipped_files = pooch.retrieve(
    url=LINK_TO_HIT,
    fname="HFC-scenarios-2022-v1.0.zip",
    path=VELDERS_ET_AL_2022_RAW_PATH,
    known_hash="74fe066fac06b742ba4fec6ad3af52a595f81a2a1c69d53a8eaf9ca846b3a7cd",
    processor=pooch.Unzip(extract_dir=VELDERS_ET_AL_2022_RAW_PATH),
    progressbar=True,
)

# %%
scenario_files = [
    v
    for v in (VELDERS_ET_AL_2022_RAW_PATH / "veldersguus-HFC-scenarios-2022-859d44c").glob("*.xlsx")
    if "Consumption" not in v.name
]
scenario_files


# %% [markdown]
# ## Load data


# %%
def load_velders_scenario(sf: Path, sheet: str) -> pd.DataFrame:
    scenario_name_base = sf.name.split("HFC_")[1].split("_Scenario")[0]

    scenario_name = f"{scenario_name_base}-{sheet.lower()}"

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

    raw = pd.read_excel(sf, sheet_name=sheet, header=4)

    processed = raw[["Species", "Year", "Emis_tot"]].dropna()
    processed = (
        processed[processed["Species"].str.startswith("HFC")].set_index(["Species", "Year"])["Emis_tot"].unstack("Year")
    )

    if set(processed.index.get_level_values("Species")) != set(expected_species):
        raise AssertionError

    if processed.isnull().any().any():
        raise AssertionError

    clean = processed.T.rename(velders_variable_normalisation_map, axis="columns").T
    clean.columns = clean.columns.astype(int)
    clean.columns.name = "year"
    clean.index.name = "variable"
    # Assuming that input data unit is t / yr
    clean = clean / 1000  # convert to kt / yr
    clean = clean.pix.format(unit="kt {variable}/yr")
    clean = clean.pix.format(variable="Emissions|{variable}")
    clean = clean.pix.assign(model="Velders et al., 2022", scenario=scenario_name, region="World")

    return clean


# %%
velders_scenarios = pix.concat(
    [load_velders_scenario(sf, sheet) for sf in scenario_files for sheet in ["Upper", "Lower"]]
)
velders_scenarios

# %%
# Comm from Guus
history = (
    velders_scenarios.loc[pix.isin(scenario="Current_Policy_2022-upper")]
    .copy()
    .pix.assign(scenario="history")
    .loc[:, :2023]
)
velders_scenarios = pix.concat([velders_scenarios, history])

# %%
pdf = velders_scenarios.openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    col="variable",
    col_wrap=3,
    hue="scenario",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# ## Save

# %% [markdown]
# Make sure we can convert the variable names

# %%
update_index_levels_func(
    velders_scenarios,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
VELDERS_ET_AL_2022_PROCESSED_DB.save(velders_scenarios, allow_overwrite=True)
