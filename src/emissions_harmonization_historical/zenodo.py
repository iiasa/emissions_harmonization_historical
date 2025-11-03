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

pd.set_option("display.max_rows", 100)


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
