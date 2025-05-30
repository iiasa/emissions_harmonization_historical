"""
Extract harmonisation results into a single folder that can be put on sharepoint
"""

import shutil
from pathlib import Path

from emissions_harmonization_historical.constants_5000 import HARMONISED_OUT_DIR


def main():
    """
    Extract the data
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    SEARCH_PATH = HARMONISED_OUT_DIR
    OUT_PATH = REPO_ROOT / "harmonisation-for-sharepoint" / SEARCH_PATH.name

    for file in [
        *SEARCH_PATH.rglob("harmonisation-results*.pdf"),
        *SEARCH_PATH.rglob("harmonisation-results*.txt"),
        *SEARCH_PATH.rglob("harmonisation-methods*.csv"),
    ]:
        if file.name.endswith(".txt"):
            # I am so stupid to have made this issue,
            # apologies to all who have to deal with this.
            model = file.stem.split("_")[1]
        else:
            model = file.stem.split("_")[-1]

        out_dir = OUT_PATH / model
        out_dir.mkdir(exist_ok=True, parents=True)
        print(f"Extracting to {out_dir}")
        shutil.copyfile(file, out_dir / file.name)


if __name__ == "__main__":
    main()
