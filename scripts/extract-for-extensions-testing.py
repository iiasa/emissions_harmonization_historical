"""
Extract results required for testing the extension algorithm
"""

from pathlib import Path

import pandas_indexing as pix
import pandas_openscm

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_OUT_DIR,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    HISTORY_HARMONISATION_DIR,
    INFILLED_SCENARIOS_DB,
    INFILLING_DB_DIR,
)


def main():
    """
    Extract the data
    """
    pandas_openscm.register_pandas_accessor()
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    OUT_PATH = REPO_ROOT / "for-extension-testing" / INFILLING_DB_DIR.name

    harmonised_for_gridding = HARMONISED_SCENARIO_DB.load(pix.ismatch(variable="Emissions**", workflow="gridding"))
    infilled_for_scms = INFILLED_SCENARIOS_DB.load(pix.ismatch(variable="Emissions**", stage="complete"))
    history_harmonised = HISTORY_HARMONISATION_DB.load()

    out_path_infilled = OUT_PATH / f"{INFILLING_DB_DIR.name}_infilled-emissions.csv"
    out_path_infilled.parent.mkdir(exist_ok=True, parents=True)
    infilled_for_scms.to_csv(out_path_infilled)
    print(f"Wrote {out_path_infilled}")

    out_path_harmonised_for_gridding = OUT_PATH / f"{HARMONISED_OUT_DIR.name}_harmonised-gridding-emissions.csv"
    out_path_harmonised_for_gridding.parent.mkdir(exist_ok=True, parents=True)
    harmonised_for_gridding.to_csv(out_path_harmonised_for_gridding)
    print(f"Wrote {out_path_harmonised_for_gridding}")

    history_harmonised = history_harmonised.loc[:, :2023]
    history_gridding = history_harmonised.loc[pix.isin(purpose="gridding_emissions")]
    history_scms = history_harmonised.loc[pix.isin(purpose="global_workflow_emissions")]

    out_path_history_gridding = OUT_PATH / f"{HISTORY_HARMONISATION_DIR.name}_history-gridding.csv"
    out_path_history_gridding.parent.mkdir(exist_ok=True, parents=True)
    history_gridding.to_csv(out_path_history_gridding)
    print(f"Wrote {out_path_history_gridding}")

    out_path_history_scms = OUT_PATH / f"{HISTORY_HARMONISATION_DIR.name}_history-scms.csv"
    out_path_history_scms.parent.mkdir(exist_ok=True, parents=True)
    history_scms.to_csv(out_path_history_scms)
    print(f"Wrote {out_path_history_scms}")


if __name__ == "__main__":
    main()
