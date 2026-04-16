"""
Extract results into a single folder that can be used for regression testing
"""

from pathlib import Path

import pandas as pd
import pandas_indexing as pix

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_OUT_DIR,
    HARMONISED_SCENARIO_DB,
    INFILLED_SCENARIOS_DB,
    MARKERS,
    POST_PROCESSED_TIMESERIES_DB,
    PRE_PROCESSED_SCENARIO_DB,
)


def main():
    """
    Extract the data
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    ID = "0001"
    OUT_PATH = REPO_ROOT / "regression-testing-outputs" / ID

    OUT_PATH.mkdir(exist_ok=True, parents=True)

    for model, scenario, _, _ in MARKERS:
        pre_processed = PRE_PROCESSED_SCENARIO_DB.load(pix.ismatch(model=model, scenario=scenario))
        pre_processed.to_csv(OUT_PATH / f"{model}_{scenario}_pre-processed.csv")

        harmonised = HARMONISED_SCENARIO_DB.load(pix.ismatch(model=model, scenario=scenario))
        harmonised.to_csv(OUT_PATH / f"{model}_{scenario}_harmonised.csv")

        complete = INFILLED_SCENARIOS_DB.load(
            pix.ismatch(model=model, scenario=scenario, stage="complete")
        ).reset_index("stage", drop=True)
        complete.to_csv(OUT_PATH / f"{model}_{scenario}_complete.csv")

        ghg_aggregates = POST_PROCESSED_TIMESERIES_DB.load(pix.ismatch(model=model, scenario=scenario))
        # breakpoint()
        ghg_aggregates.to_csv(OUT_PATH / f"{model}_{scenario}_ghg-aggregates.csv")

    overrides_global_l = []
    overrides_gridding_l = []
    for model in set(v[0] for v in MARKERS):
        model_short_name = model.split(" ")[0].split("-")[0].replace("ix", "")

        overrides_files = list(HARMONISED_OUT_DIR.rglob(f"**/harmonisation-methods*{model_short_name}*.csv"))
        overrides_files_n_exp = 2
        if len(overrides_files) != overrides_files_n_exp:
            print(f"Wrong overrides files found for {model} {overrides_files=}")
            continue

        overrides_file_global = [v for v in overrides_files if "global" in str(v)]
        if len(overrides_file_global) != 1:
            raise AssertionError
        overrides_model_global = pd.read_csv(overrides_file_global[0])

        overrides_file_gridding = [v for v in overrides_files if "gridding" in str(v)]
        if len(overrides_file_gridding) != 1:
            raise AssertionError
        overrides_model_gridding = pd.read_csv(overrides_file_gridding[0])

        overrides_model_global = overrides_model_global.set_index(["model", "region", "variable"])["method"]
        if overrides_model_global.index.duplicated().any():
            raise AssertionError

        overrides_model_gridding = overrides_model_gridding.set_index(["model", "region", "variable"])["method"]
        if overrides_model_gridding.index.duplicated().any():
            raise AssertionError

        overrides_global_l.append(overrides_model_global)
        overrides_gridding_l.append(overrides_model_gridding)

    overrides_global = pd.concat(overrides_global_l)
    overrides_global.to_csv(OUT_PATH / "aneris-overrides-global.csv")

    overrides_gridding = pd.concat(overrides_gridding_l)
    overrides_gridding.to_csv(OUT_PATH / "aneris-overrides-gridding.csv")


if __name__ == "__main__":
    main()
