"""
Helpers for working with nomenclature

See https://github.com/IAMconsortium/nomenclature
"""

from __future__ import annotations

from pathlib import Path

import git


def get_common_definitions(
    out_path: Path,
    repo_url: str = "https://github.com/IAMconsortium/common-definitions",
    commit_id: str = "72707a466882b0ded4a582c27cc4d70f213215a7",
) -> None:
    """
    Get common definitions

    Parameters
    ----------
    out_path
        Path in which to clone the repository

    repo_url
        URL from which to clone the repository

    commit_id
        Commit to check out
    """
    msg = f"Grabbing common-definitions, cloning to {out_path} " f"and checking out commit {commit_id}."
    print(msg)
    repo = git.Repo.clone_from(repo_url, out_path)
    repo.git.checkout(commit_id)
