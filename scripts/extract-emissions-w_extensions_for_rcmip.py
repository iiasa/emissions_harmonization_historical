"""
Extract emissions results into a single folder that can be put on sharepoint
"""

from pathlib import Path

from emissions_harmonization_historical.constants_5000 import EXTENSIONS_OUT_DIR, EXTENSIONS_OUTPUT_DB


def main():
    """
    Extract the data
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    OUT_PATH = REPO_ROOT / "emissions-for-sharepoint" / EXTENSIONS_OUT_DIR.name
    OUT_PATH.mkdir(exist_ok=True, parents=True)
    out = EXTENSIONS_OUTPUT_DB.load()
    out.to_csv(OUT_PATH / "all_extensions_output.csv")


if __name__ == "__main__":
    main()
