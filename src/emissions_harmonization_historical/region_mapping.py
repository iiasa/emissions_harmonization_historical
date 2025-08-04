"""Here we specify functions used to generate the 'region_df.csv' from the common-definitions repository."""

from pathlib import Path

import pandas as pd
import requests
from nomenclature import countries
from nomenclature.definition import DataStructureDefinition
from nomenclature.processor import RegionProcessor


def get_latest_commit_hash(
    repo_owner: str, repo_name: str, fallback_commit: str = "cc69ed0a415a63c7ce7372d5a36c088d9cbee055"
):
    """
    Fetch the latest commit hash from a GitHub repository.

    Falls back to a default hash, here for IAMconsortium/common-definitions, if the request fails.

    Parameters
    ----------
    repo_owner
        GitHub user

    repo_name
        Repository name

    fallback_commit
        Commit hash to use if request fails.

    Returns
    -------
    :
        The latest commit hash or the fallback hash.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()[0]["sha"]
    except (requests.RequestException, KeyError, IndexError):
        return fallback_commit


def get_iso3_list(country_list: list[str]) -> list[str]:
    """
    Get list of ISO3 values for a list of countries

    Parameters
    ----------
    country_list
        Countries for which to retrieve ISO3 values

    Returns
    -------
    :
        List of ISO3 values.

        For any country for which an ISO3 could not be created,
        `None` is returned in the output.
    """
    if not country_list:
        return None

    iso3_list = []
    for country in country_list:
        try:
            iso3 = countries.get(name=country)
            if iso3:
                iso3_list.append(iso3.alpha_3)
            else:
                print(f"No iso3 for {country=}")
                iso3_list.append(None)  # If no match found
        except Exception as exc:
            print(f"Exception retrieving iso3 for {country=}. {exc=}")
            iso3_list.append(None)

    return iso3_list


def create_region_mapping(out_file: Path, common_definitions_path: Path) -> Path:
    """
    Create the region mapping file

    Parameters
    ----------
    out_file
        File in which to write the mapping

    common_definitions_path
        Path in which the common-definitions repository can be found.

        This is the repo it should point to:
        https://github.com/IAMconsortium/common-definitions.

    Returns
    -------
    :
        File in which the mapping was written
    """
    dsd = DataStructureDefinition(common_definitions_path / "definitions")

    region_df = pd.DataFrame(
        [(r.name, r.hierarchy, r.countries, r.iso3_codes) for r in dsd.region.values()],
        columns=["name", "hierarchy", "countries", "iso3"],
    )

    region_df["iso3"] = region_df["countries"].apply(get_iso3_list)

    rp = RegionProcessor.from_directory(common_definitions_path / "mappings", dsd)
    rows = []
    for ram in rp.mappings.values():
        rows_model = [
            (
                ram.model,
                common_region.name,
                [ram.rename_mapping[nr] for nr in common_region.constituent_regions],
            )
            for common_region in ram.common_regions
        ]

        rows.extend(rows_model)

    region_df.to_csv(out_file)

    return out_file
