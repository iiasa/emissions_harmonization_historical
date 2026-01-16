"""
Support uploading to zenodo
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import httpx
import tqdm.auto
from dotenv import load_dotenv
from openscm_zenodo.zenodo import ZenodoInteractor


def get_zenodo_interactor() -> ZenodoInteractor:
    """
    Get zenodo interactor

    Returns
    -------
    :
        Initialised [openscm_zenodo.zenodo.ZenodoInteractor][]

    Raises
    ------
    KeyError
        The `ZENODO_TOKEN` environment variable is not set
    """
    load_dotenv()

    if "ZENODO_TOKEN" not in os.environ:
        msg = "Please copy the `.env.sample` file to `.env` " "and ensure you have set your ZENODO_TOKEN."
        raise KeyError(msg)

    zenodo_interactor = ZenodoInteractor(token=os.environ["ZENODO_TOKEN"])

    return zenodo_interactor


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
    zenodo_interactor = get_zenodo_interactor()

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


# TODO: move into openscm_zenodo


class Writable(Protocol):
    """A writable object"""

    def write(self, chunk: bytes) -> Any:
        """
        Write chunk
        """
        ...


def download_zenodo_url(
    url: str,
    zenodo_interactor: ZenodoInteractor,
    fh: Writable,
    # zenodo_interactor: ZenodoInteractor,
    max_desc_width: int = 100,
    size: int | None = None,
) -> Writable:
    """
    Download a Zenodo URL to a file

    Parameters
    ----------
    url
        URL to download

    zenodo_interactor
        Zenodo interactor to use for authorisation

    fh
        File handle to write to

    max_desc_width
        Maximum width to show in the progress bar description

    size
        Size of the file being download.

        If not known, the progress bar will not show a total size

    Returns
    -------
    :
        File handle
    """
    if len(url) > max_desc_width:
        desc = f"{url[:30]}...{url[-70:]}"
    else:
        desc = url

    with (
        httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            params={"access_token": zenodo_interactor.token},
        ) as request,
    ):
        with (
            tqdm.auto.tqdm(
                desc=desc,
                total=size,
                miniters=1,
                unit="B",
                unit_scale=True,
                unit_divisor=2**10,
                # Useful things if we ever want to make this parallel
                # position=thread_positions[thread_id],
                # leave=False,
            ) as pbar,
        ):
            for chunk in request.iter_bytes(chunk_size=2**17):
                fh.write(chunk)
                pbar.update(len(chunk))

    return fh
