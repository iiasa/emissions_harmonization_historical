"""
Extract reporting results into a single folder that can be put on sharepoint
"""

import shutil
from pathlib import Path

from emissions_harmonization_historical.constants_5000 import (
    DATA_ROOT,
    DOWNLOAD_SCENARIOS_ID,
)


def main():
    """
    Extract the data
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    SEARCH_PATH = DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID
    OUT_PATH = REPO_ROOT / "reporting-checking" / DOWNLOAD_SCENARIOS_ID

    for file in [
        *SEARCH_PATH.rglob("reporting-checking_*.xlsx"),
        *SEARCH_PATH.rglob("internal-consistency-checking_*.xlsx"),
    ]:
        model = file.stem.split("_")[1]

        out_dir = OUT_PATH / model
        out_dir.mkdir(exist_ok=True, parents=True)
        print(f"Extracting to {out_dir}")
        shutil.copyfile(file, out_dir / file.name)


if __name__ == "__main__":
    main()
