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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Create infilling database
#
# Here we create the infilling database,
# including uploading it to Zenodo.
# Note that the result depends on the scenarios
# you have processed and harmonised, be careful.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import datetime as dt
import sys
from functools import partial

import git
import numpy as np
import pandas as pd
import pandas.io.excel
import pandas_indexing as pix
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from loguru import logger
from markdown_it import MarkdownIt
from pandas_openscm.comparison import compare_close
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    DATA_ROOT,
    DOWNLOAD_SCENARIOS_ID,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLING_DB_INTERIM_DIR,
    REPO_ROOT,
    VELDERS_ET_AL_2022_PROCESSED_DB,
    WMO_2022_PROCESSED_DB,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
    harmonise,
)
from emissions_harmonization_historical.zenodo import upload_to_zenodo

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""}
pd.set_option("display.max_colwidth", None)

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
# No parameters

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
# Only World level data for specific variables
# is used for infilling
# (we deliberately don't include all variables
# so people don't think that the infilling database
# is the full scenario set,
# if people want that they have to go to the scenario explorer).
variables_not_for_infilling = [
    "Emissions|BC",
    "Emissions|CH4",
    "Emissions|CO",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|Sulfur",
    "Emissions|VOC",
    "Emissions|CO2|AFOLU",
]
scenarios_for_infilling_db = HARMONISED_SCENARIO_DB.load(
    pix.isin(region="World", workflow="for_scms") & ~pix.isin(variable=variables_not_for_infilling), progress=True
).reset_index("workflow", drop=True)
if scenarios_for_infilling_db.empty:
    raise AssertionError

if scenarios_for_infilling_db.isnull().any().any():
    raise AssertionError

scenarios_for_infilling_db.index.droplevel(
    scenarios_for_infilling_db.index.names.difference(["model", "scenario"])
).drop_duplicates()

# %% [markdown]
# ### WMO 2022

# %%
wmo_2022_scenarios = WMO_2022_PROCESSED_DB.load(pix.ismatch(model="WMO-2022-CMIP7-concentration-inversions")).loc[
    :, HARMONISATION_YEAR:2100
]
if wmo_2022_scenarios.empty:
    raise AssertionError

# wmo_2022_scenarios

# %% [markdown]
# ### Velders et al., 2022

# %%
velders_2022_scenarios = VELDERS_ET_AL_2022_PROCESSED_DB.load(~pix.ismatch(scenario="history"))
if velders_2022_scenarios.empty:
    raise AssertionError

velders_2022_scenarios = update_index_levels_func(
    velders_2022_scenarios,
    updates={
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)
velders_2022_scenarios = update_index_levels_func(
    velders_2022_scenarios, {"unit": lambda x: x.replace("HFC4310mee", "HFC4310")}
)

# velders_2022_scenarios

# %% [markdown]
# ### History

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)
# Drop out years with NaNs as they break aneris
history = history.dropna(axis="columns")
# history

# %% [markdown]
# ## Harmonise

# %%
infilling_db_raw = pix.concat([scenarios_for_infilling_db, wmo_2022_scenarios, velders_2022_scenarios]).sort_index(
    axis="columns"
)
# infilling_db_raw

# %%
# Could load in user overrides from elsewhere here.
# They need to be a series with name "method".
# TODO: make sure these are consistent with what we do in 5094
user_overrides = None

# %%
harmonise_res = harmonise(
    scenarios=infilling_db_raw,
    history=history,
    harmonisation_year=HARMONISATION_YEAR,
    user_overrides=user_overrides,
)
if harmonise_res.timeseries.isnull().any().any():
    raise AssertionError(harmonise_res.timeseries.isnull().any(axis=1))

# %%
compare_infilling_harmonisation = compare_close(
    harmonise_res.timeseries,
    infilling_db_raw.loc[:, harmonise_res.timeseries.columns],
    "harmonised",
    "input",
    isclose=partial(np.isclose, rtol=1e-4),
)
models_reharmonised = compare_infilling_harmonisation.index.get_level_values("model").unique().tolist()
if models_reharmonised != ["Velders et al., 2022"]:
    msg = f"Data from {models_reharmonised} was re-harmonised. This shouldn't happen?"
    raise AssertionError(msg)

# %%
tmp = compare_infilling_harmonisation
tmp.columns.name = "stage"
tmp = tmp.stack().to_frame("value").reset_index()
sns.relplot(
    data=tmp,
    x="year",
    y="value",
    col="variable",
    col_wrap=3,
    hue="scenario",
    style="stage",
    facet_kws=dict(sharey=False),
    kind="line",
)

# %% [markdown]
# ## Save in various formats

# %%
# Linearly interpolate the output so we get sensible infilling

# %%
out = harmonise_res.timeseries
for y in range(out.columns.min(), out.columns.max() + 1):
    if y not in out:
        out[y] = np.nan

out = out.sort_index(axis="columns")
out = out.T.interpolate(method="index").T
out.index.droplevel(out.index.names.difference(["model", "scenario"])).drop_duplicates().to_frame()["model"].unique()

# %% editable=true slideshow={"slide_type": ""}
INFILLING_DB_INTERIM_DIR.mkdir(exist_ok=True, parents=True)

files_for_zenodo = []
for suffix, method, kwargs in (
    (".csv", "to_csv", {}),
    (".feather", "to_feather", {}),
    (".parquet.gzip", "to_parquet", dict(compression="gzip")),
):
    out_file = INFILLING_DB_INTERIM_DIR / f"infiling-db_{INFILLING_DB_INTERIM_DIR.stem}{suffix}"

    getattr(out, method)(out_file, **kwargs)
    files_for_zenodo.append(out_file)
    print(f"Wrote {out_file.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## Write README

# %% editable=true slideshow={"slide_type": ""}
readme_txt = """# Infilling database for CMIP7 ScenarioMIP simple climate model assessment

The files here are infilling databases.
They are used for the 'infilling' step
of simple climate model assessment.
This involves inferring emissions of one species (e.g. HFC32)
based on emissions of another species (e.g. CO2)
and relationships between these two species seen in other scenarios.
For more details about infilling, see
[Lamboll et al., 2020](https://doi.org/10.5194/gmd-13-5259-2020).

The database is provided in three different formats.
The versions of the scenarios used to compile this database
is provided in the `versions.json` file.

The database was derived using the code in this repository:
https://github.com/iiasa/emissions_harmonization_historical.
The filenames are composed of identifiers related to the processing
of each of the different input data sources.
To identify the exact meaning of these identifiers,
please see the processing code in
https://github.com/iiasa/emissions_harmonization_historical.
"""

# %%
repo = git.Repo(REPO_ROOT)
if not repo.is_dirty():
    readme_txt = f"""{readme_txt}
The files were produced with the following commit:
[{repo.head.object.hexsha}](https://github.com/iiasa/emissions_harmonization_historical/tree/{repo.head.object.hexsha})"""

# %%
readme_file = INFILLING_DB_INTERIM_DIR / "README.md"
with open(readme_file, "w") as fh:
    fh.write(readme_txt)

files_for_zenodo.append(readme_file)

# %%
# # !tail {readme_file}

# %% editable=true slideshow={"slide_type": ""}
versions_json = DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / "versions.json"
if not versions_json.exists():
    raise AssertionError

files_for_zenodo.append(versions_json)

# %% [markdown]
# ## Set metadata
#
# We can be a bit more relaxed about this
# because it can be updated after publication.
# To save ourselves some typing and clicking,
# we automate the initial values.

# %%
metadata = {
    "metadata": {
        "version": dt.datetime.utcnow().strftime("%Y.%m.%d"),
        "title": "CMIP7 ScenarioMIP infilling database for simple climate model workflow",
        "description": MarkdownIt().render(readme_txt),
        "upload_type": "dataset",
        # Sometime in the future.
        # We can make it open manually sooner,
        # but using embargo here means it will be open
        # eventually, even if we forget.
        "access_right": "embargoed",
        "embargo_date": "2026-06-30",
        # TODO: check
        # Note: you can't set None, so you have to go in
        # and manually remove the license before publishing.
        # "license": "cc-by-4.0",
        "creators": [
            {
                "name": "Nicholls, Zebedee",
                "affiliation": ";".join(
                    [
                        "Climate Resource S GmbH",
                        "International Institute for Applied Systems Analysis",
                        "University of Melbourne",
                    ]
                ),
                "orcid": "0000-0002-4767-2723",
            },
            {
                "name": "Kikstra, Jarmo",
                "affiliation": "International Institute for Applied Systems Analysis",
                "orcid": "0000-0001-9405-1228",
            },
            {
                "name": "Zecchetto, Marco",
                "affiliation": "International Institute for Applied Systems Analysis",
                "orcid": "0000-0002-7506-2631",
            },
            {
                "name": "Hoegner, Annika",
                "affiliation": "International Institute for Applied Systems Analysis",
                "orcid": "0000-0002-4178-9664",
            },
        ],
        "related_identifiers": [
            # TODO: add these.
            # e.g. WMO 2022, Velders
            # (I can get these from the concentration references sometime)
            # and scenario references.
        ],
        "custom": {
            "code:codeRepository": "https://github.com/iiasa/emissions_harmonization_historical",
            "code:developmentStatus": {"id": "active", "title": {"en": "Active"}},
            "code:programmingLanguage": [{"id": "python", "title": {"en": "Python"}}],
        },
    }
}

# %% [markdown]
# ## Upload to Zenodo


# %% editable=true slideshow={"slide_type": ""}
logger.configure(handlers=[dict(sink=sys.stderr, level="INFO")])
logger.enable("openscm_zenodo")

# %% editable=true slideshow={"slide_type": ""}
# # Useful if you're trying to figure out metadata fields
# from emissions_harmonization_historical.zenodo import get_zenodo_interactor
#
# zenodo_interactor = get_zenodo_interactor()
# zenodo_interactor.get_metadata(
#     zenodo_interactor.get_draft_deposition_id("17514979")
# )

# %%
upload_to_zenodo(
    files_for_zenodo,
    any_deposition_id="17514979",
    remove_existing=True,
    metadata=metadata,
)
