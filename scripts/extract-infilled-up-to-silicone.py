"""
Extract results of infilling up to the silicone step
"""

from pathlib import Path

import pandas_indexing as pix
import pandas_openscm

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
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
    OUT_PATH = REPO_ROOT / "infilled-emissions-up-to-silicone"

    harmonised = HARMONISED_SCENARIO_DB.load(pix.ismatch(variable="Emissions**", workflow="for_scms"))

    infilled_silicone = INFILLED_SCENARIOS_DB.load(pix.isin(stage="silicone"))

    relevant_emissions = (
        harmonised.index.droplevel(harmonised.index.names.difference(["variable", "unit"]))
        .append(infilled_silicone.index.droplevel(infilled_silicone.index.names.difference(["variable", "unit"])))
        .drop_duplicates()
    )
    # MAGICC's old SCEN number
    exp_n_variables = 23
    if len(relevant_emissions) != exp_n_variables:
        raise AssertionError(len(relevant_emissions))

    history = HISTORY_HARMONISATION_DB.load(pix.isin(purpose="global_workflow_emissions"))
    history_relevant = history.openscm.mi_loc(relevant_emissions)

    res = pix.concat(
        [
            harmonised.reset_index("workflow", drop=True),
            infilled_silicone.reset_index("stage", drop=True),
            history_relevant.reset_index("purpose", drop=True),
        ]
    ).sort_index(axis="columns")

    out_path = OUT_PATH / f"{INFILLING_DB_DIR.name}_harmonised-emissions-up-to-sillicone.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    res.to_csv(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
