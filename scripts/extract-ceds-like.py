"""
Extract emissions in a CEDS-like format for review
"""

from pathlib import Path

import pandas_indexing as pix
import pandas_openscm

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_OUT_DIR,
    HARMONISED_SCENARIO_DB,
    MARKERS,
)


def main():
    """
    Extract the data
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    OUT_PATH = REPO_ROOT / "emissions-ceds-like" / HARMONISED_OUT_DIR.name
    MARKERS_ONLY = True

    pandas_openscm.register_pandas_accessor()

    if MARKERS_ONLY:
        harmonised_emissions_l = []
        for model, scenario, _ in MARKERS:
            harmonised_emissions_l.append(HARMONISED_SCENARIO_DB.load(pix.isin(model=model, scenario=scenario)))
        harmonised_emissions = pix.concat(harmonised_emissions_l)

    else:
        harmonised_emissions = HARMONISED_SCENARIO_DB.load()

    harmonised_emissions_gridding = harmonised_emissions.loc[pix.isin(workflow="gridding")]
    harmonised_emissions_gridding_region_sum = harmonised_emissions_gridding.openscm.groupby_except("region").sum()

    harmonised_emissions_global_workflow = harmonised_emissions.loc[pix.isin(workflow="for_scms")]

    for df, out_file_name in (
        (harmonised_emissions_gridding_region_sum.reset_index("workflow", drop=True), "global-emissions-by-sector"),
        (harmonised_emissions_global_workflow.reset_index("workflow", drop=True), "global-emissions-scm-style"),
    ):
        if MARKERS_ONLY:
            out_file = OUT_PATH / f"markers_{out_file_name}_{OUT_PATH.name}.csv"
        else:
            out_file = OUT_PATH / f"{out_file_name}_{OUT_PATH.name}.csv"

        out_file.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_file)
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
