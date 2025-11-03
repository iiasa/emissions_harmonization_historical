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
# # Upload history files to Zenodo
#
# Upload the history files to Zenodo.
# Forcing the rest of the workflow
# to retrieve the files from Zenodo ensures
# that we use a consistent set of history files,
# regardless of who/where/how we have run
# the rest of the workflow.

# %% [markdown]
# ## Imports

# %%
import datetime as dt
import sys
from functools import partial

import git
import pandas as pd
from loguru import logger
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants_5000 import (
    COUNTRY_LEVEL_HISTORY,
    CREATE_HISTORY_FOR_GLOBAL_WORKFLOW_ID,
    HISTORY_HARMONISATION_INTERIM_DIR,
    REPO_ROOT,
)
from emissions_harmonization_historical.zenodo import upload_to_zenodo

# %% [markdown]
# ## Setup

# %% [markdown]
# ## Save in varous formats

# %%
files_for_zenodo = []
for in_file, loader in (
    (
        HISTORY_HARMONISATION_INTERIM_DIR / f"global-workflow-history_{CREATE_HISTORY_FOR_GLOBAL_WORKFLOW_ID}.feather",
        pd.read_feather,
    ),
    (
        COUNTRY_LEVEL_HISTORY,
        partial(
            load_timeseries_csv,
            index_columns=[
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
            ],
            out_columns_name="year",
            out_columns_type=int,
        ),
    ),
    # (HISTORY_HARMONISATION_INTERIM_DIR / f"gridding-history_{CREATE_HISTORY_FOR_GRIDDING_ID}.feather",
    # pd.read_feather),
):
    df = loader(in_file)
    display(df.head(2))  # noqa: F821
    files_for_zenodo.append(in_file)
    for suffix, method, kwargs in (
        (".csv", "to_csv", {}),
        (".feather", "to_feather", {}),
        (".parquet.gzip", "to_parquet", dict(compression="gzip")),
    ):
        out_file = HISTORY_HARMONISATION_INTERIM_DIR / in_file.with_suffix(suffix)
        if out_file == in_file:
            continue

        getattr(df, method)(out_file, **kwargs)
        files_for_zenodo.append(out_file)
        print(f"Wrote {out_file.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## Write README

# %%
readme_txt = """# History for ScenarioMIP emissions harmonisation

The files here are compiled historical experiment timeseries.
They were compiled for use as part of the CMIP7 ScenarioMIP exercise
and are primarily used for supporting emissions harmonisation.
Here, 'harmonisation' means alignment of modelled emissions from IAMs
with the emissions used for the CMIP7 historical experiment.

As a result, they are a key input for the process of
'gridding' emissions (i.e. taking raw emissions from IAMs
and assigning them to a spatial grid, ready for use by Earth System Models (ESMs))
and for running the simple climate model based assessment of the scenarios
to derive a first-order estimate of the warming associated with these scenarios
(the ESMs will quantify the warming and other climate change associated with these scenarios
as part of ScenarioMIP, and this quantification is underpinned by a deeper,
more physically-based set of modelling assumptions.)

There are three different files,
provided in three different formats each.
The three files are:

1. `gridding-history*`: the history used for harmonisation at the 'gridding' level.
   The gridding requires emissions with regional and sectoral detail.
   It also has to support every IAMs' native regions.
   As a result, there are lots (of order 30 000) timeseries.
1. `country-history*`: same as above, but at the country level
   rather than in native IAM regions.
1. `global-workflow-history*`: the history used for harmonisation at the 'global' level.
   This only has global total emissions,
   except for CO2 which is split into fossil-based
   and land-based (i.e. originating from the land carbon pool) emissions.
   It includes a number of species that are not used in the gridding workflow
   but are relevant for climate projections
   e.g. all of the greenhouse gases covered by the Montreal Protocol.
   As a result, there are only 52 timeseries.

The files were derived using the code in this repository:
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
readme_file = HISTORY_HARMONISATION_INTERIM_DIR / "README.md"
with open(readme_file, "w") as fh:
    fh.write(readme_txt)

files_for_zenodo.append(readme_file)

# %%
# !tail {readme_file}

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
        "title": "CMIP7 ScenarioMIP historical timeseries for harmonisation and simple climate model workflow",
        "description": (
            """Harmonisation data set used in creating input for CMIP7's ScenarioMIP

See README for further details.""".replace("\n", "<br>")
        ),
        "upload_type": "dataset",
        # Long in future.
        # We can make it open manually sooner,
        # but using embargo here means it will be open
        # eventually, even if we forget.
        "visibility": "restricted",
        "embargo_date": "2026-07-01",
        # # TODO: check
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
            # CEDS zenodo
            {
                "identifier": "10.5281/zenodo.15059443",
                "relation": "isDerivedFrom",
                "resource_type": "dataset",
                "scheme": "doi",
            },
            # CEDS 2017 paper
            {
                "identifier": "10.5194/gmd-11-369-2018",
                "relation": "isDerivedFrom",
                "resource_type": "publication",
                "scheme": "doi",
            },
            # van Marle 2017 paper
            {
                "identifier": "10.5194/gmd-10-3329-2017",
                "relation": "isDerivedFrom",
                "resource_type": "publication",
                "scheme": "doi",
            },
            # GFED4 paper
            {
                "identifier": "10.5194/essd-9-697-2017",
                "relation": "isDerivedFrom",
                "resource_type": "publication",
                "scheme": "doi",
            },
            # TODO: do the rest of the inputs
            # e.g. WMO 2022, Velders, Adams etc.
            # I can get these from the concentration references sometime.
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

# %%
logger.configure(handlers=[dict(sink=sys.stderr, level="INFO")])
logger.enable("openscm_zenodo")

# %%
# # Useful if you're trying to figure out metadata fields
# from openscm_zenodo.zenodo import ZenodoInteractor
# import os
# zenodo_interactor = ZenodoInteractor(token=os.environ["ZENODO_TOKEN"])
# zenodo_interactor.get_metadata(
#     zenodo_interactor.get_draft_deposition_id("15357373")
# )

# %%
upload_to_zenodo(
    files_for_zenodo,
    any_deposition_id="15357373",
    remove_existing=True,
    metadata=metadata,
)
