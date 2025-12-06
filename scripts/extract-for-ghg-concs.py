"""
Extract results required for GHG concentration projections
"""

from pathlib import Path

import pandas_indexing as pix
import pandas_openscm
from gcages.cmip7_scenariomip.gridding_emissions import CO2_BIOSPHERE_SECTORS_GRIDDING

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
    OUT_PATH = REPO_ROOT / "for-ghg-concs" / INFILLING_DB_DIR.name

    history = HISTORY_HARMONISATION_DB.load(pix.isin(purpose="global_workflow_emissions"))

    complete = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete"))

    for (model, scenario), msdf in complete.groupby(["model", "scenario"]):
        relevant_emissions = msdf.index.droplevel(msdf.index.names.difference(["variable", "unit"])).drop_duplicates()
        exp_n_variables = 52
        if len(relevant_emissions) != exp_n_variables:
            msg = f"{model} {scenario} {len(relevant_emissions)}"
            raise AssertionError(msg)

    history_to_align = history.reset_index(["purpose", "model", "scenario"], drop=True).loc[
        :, : complete.columns.min() - 1
    ]
    complete_a, history_a = complete.reset_index("stage", drop=True).align(history_to_align)
    complete_a = complete_a.dropna(how="all", axis="columns")
    history_a = history_a.dropna(how="all", axis="columns")
    res = pix.concat(
        [
            complete_a,
            history_a,
        ],
        axis=1,
    ).sort_index(axis="columns")
    if res.isnull().any().any():
        raise AssertionError
    if res.shape[0] != complete.shape[0]:
        raise AssertionError

    res = pix.concat([res, history.reset_index("purpose", drop=True)])

    out_path = OUT_PATH / f"{INFILLING_DB_DIR.name}_complete-emissions.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    res.to_csv(out_path)
    print(f"Wrote {out_path}")

    harmonised_sectoral = HARMONISED_SCENARIO_DB.load(pix.ismatch(variable="Emissions**", workflow="gridding"))
    history_sectoral = HISTORY_HARMONISATION_DB.load(pix.isin(purpose="gridding_emissions"))

    raw_sectoral = (
        pix.concat(
            [
                harmonised_sectoral.reset_index("workflow", drop=True),
                history_sectoral.reset_index("purpose", drop=True),
            ]
        )
        .sort_index(axis="columns")
        .pix.extract(variable="{table}|{species}|{sector}", drop=True)
    )
    raw_sectors = raw_sectoral.index.get_level_values("sector")
    sector_fossil_bio = raw_sectors.map(
        # Assuming that CO2 mapping holds for other gases, fine for now
        lambda v: "Biosphere" if v.split("|")[-1] in CO2_BIOSPHERE_SECTORS_GRIDDING else "Fossil"
    )

    tmp = raw_sectoral.pix.assign(sector_fossil_bio=sector_fossil_bio)
    tmp_sum = (
        tmp.openscm.groupby_except("sector")
        .sum(min_count=1)
        .pix.format(variable="{table}|{species}|{sector_fossil_bio}", drop=True)
    )
    out_l = []
    for scenario, sdf in tmp_sum.groupby("scenario"):
        if scenario == "historical":
            # Use R5 regions, as good a choice as any
            historical_sum_r5 = (
                sdf.loc[pix.ismatch(region=["World", "**R5**"])]
                .openscm.groupby_except(["model", "region"])
                .sum(min_count=1)
                .pix.assign(model="historical-sources")
            )
            # historical_sum_r10 = (
            #     sdf.loc[pix.ismatch(region=["World", "**R10**"])]
            #     .openscm.groupby_except(["model", "region"])
            #     .sum(min_count=1)
            #     .pix.assign(model="historical-sources")
            # )
            # historical_sum_rm = (
            #     sdf.loc[pix.ismatch(region=["World", "REMIND-MAgPIE 3.5-4.10|*"])]
            #     .openscm.groupby_except(["model", "region"])
            #     .sum(min_count=1)
            #     .pix.assign(model="historical-sources")
            # )
            out_l.append(historical_sum_r5)

        else:
            for _, mdf in sdf.groupby("model"):
                # Single model hence can sum across regions with confidence
                out_l.append(mdf.openscm.groupby_except("region").sum(min_count=1))

    out = pix.concat(out_l)
    out_path = OUT_PATH / f"{INFILLING_DB_DIR.name}_harmonised-emissions-fossil-biosphere-aggregation.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
