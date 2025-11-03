"""
Support uploading to zenodo
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openscm_zenodo.zenodo import ZenodoInteractor

from emissions_harmonization_historical.constants_5000 import INFILLED_OUT_DIR_ID

pd.set_option("display.max_rows", 100)


def create_metadata_zenodo_history() -> dict[str, dict[str, Any]]:
    """
    Create metadata for Zenodo deposit of history data
    """
    return {
        "metadata": {
            "version": INFILLED_OUT_DIR_ID,
            "title": "CMIP7 ScenarioMIP harmonisation and infilling data for simple climate model workflows",
            "description": (
                """Harmonisation data set and infilling database used in creating input for CMIP7's ScenarioMIP

These were used for harmonisation before gridding
as well as for creating 'complete' scenarios for running simple climate models
(the so-called 'global-workflow' files focus on this application).""".replace("\n", "<br>")
            ),
            "upload_type": "dataset",
            # # TODO: check
            # "access_right": "open",
            # "license": "cc-by-4.0",
            "creators": [
                {
                    "name": "Nicholls, Zebedee",
                    "affiliation": "Climate Resource S GmbH; International Institute for Applied Systems Analysis",
                    "orcid": "0000-0002-4767-2723",
                },
                {
                    "name": "Kikstra, Jarmo",
                    "affiliation": "International Institute for Applied Systems Analysis",
                    "orcid": "0000-0001-9405-1228",
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
                # {
                #     "identifier": "https://gml.noaa.gov/hats/",
                #     "relation": "isDerivedFrom",
                #     "resource_type": "dataset",
                #     "scheme": "url",
                # },
            ],
            "custom": {
                "code:codeRepository": "https://github.com/iiasa/emissions_harmonization_historical",
                "code:developmentStatus": {"id": "active", "title": {"en": "Active"}},
            },
        }
    }


def upload_to_zenodo(
    files_to_upload: Iterable[Path],
    any_deposition_id: int,
    metadata: dict[str, Any] | None = None,
    remove_existing: bool = False,
) -> None:
    """
    Upload to zenodo

    Parameters
    ----------
    files_to_upload
        Files to upload

    any_deposition_id
        Any deposition ID in the series of uploads to contribute to

    metadata
        If supplied, used to update the deposition's metadata

    remove_existing
        Should existing files in the deposit be removed?
    """
    load_dotenv()

    if "ZENODO_TOKEN" not in os.environ:
        msg = "Please copy the `.env.sample` file to `.env` " "and ensure you have set your ZENODO_TOKEN."
        raise KeyError(msg)

    zenodo_interactor = ZenodoInteractor(token=os.environ["ZENODO_TOKEN"])

    latest_deposition_id = zenodo_interactor.get_latest_deposition_id(
        any_deposition_id=any_deposition_id,
    )
    draft_deposition_id = zenodo_interactor.get_draft_deposition_id(latest_deposition_id=latest_deposition_id)

    if metadata:
        zenodo_interactor.update_metadata(deposition_id=draft_deposition_id, metadata=metadata)

    if remove_existing:
        # Remove the previous version's files from the new deposition
        zenodo_interactor.remove_all_files(deposition_id=draft_deposition_id)

    # Upload files
    bucket_url = zenodo_interactor.get_bucket_url(deposition_id=draft_deposition_id)
    for file in files_to_upload:
        zenodo_interactor.upload_file_to_bucket_url(
            file,
            bucket_url=bucket_url,
        )

    print(f"You can preview the draft upload at https://zenodo.org/uploads/{draft_deposition_id}")
