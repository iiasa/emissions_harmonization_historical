"""
Extract results required for GHG concentration projections
"""

from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pandas_openscm
from gcages.cmip7_scenariomip.gridding_emissions import CO2_BIOSPHERE_SECTORS_GRIDDING
from gcages.harmonisation import assert_harmonised

from emissions_harmonization_historical.ceds import get_map
from emissions_harmonization_historical.constants_5000 import (
    CEDS_RAW_PATH,
    CEDS_TOP_LEVEL_RAW_PATH,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_SCENARIOS_DB,
    INFILLING_DB_DIR,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
)


def add_ceds_extension(
    history_global_sum: pd.DataFrame, ceds_only_sum: pd.DataFrame, last_ceds_year: int = 1970
) -> pd.DataFrame:
    """
    Add CEDS extension to our fossil-biosphere history split
    """
    ceds_mapping = pd.read_excel(
        CEDS_TOP_LEVEL_RAW_PATH / "auxilliary" / "sector_mapping.xlsx",
        sheet_name="CEDS Mapping 2024",
    )
    ceds_map = (
        get_map(
            ceds_mapping,
            "59_Sectors_2024",  # note; with 7BC and 2L now added it is actually 61 sectors, not 59 anymore
        )
        .to_frame()["sector"]
        .reset_index("sector", drop=True)
    )

    res = history_global_sum.copy()
    for ext_f in (CEDS_RAW_PATH / "CEDS_v_2025_03_18_supplementary_extension").glob(
        "*Extension_CEDS_global_estimates_by_sector_v*.csv"
    ):
        gas = ext_f.name.split("_")[0]
        if gas not in ["CH4", "N2O"]:
            continue

        tmp = pd.read_csv(ext_f)

        em_l = tmp["em"].unique()
        if len(em_l) > 1:
            raise AssertionError(em_l)

        em = em_l[0]
        species = em.split("_Extension")[0]
        species_loc = pix.ismatch(variable=f"**{species}**")

        tmp["unit"] = tmp["units"] + f" {species}/yr"
        tmp["variable"] = f"Emissions|{species}"
        tmp = tmp.drop(["em", "units"], axis="columns").set_index(["variable", "unit", "sector"])
        tmp.columns = [int(v.lstrip("X")) for v in tmp.columns]
        tmp = tmp.loc[~pix.isin(sector="6B_Other-not-in-total")]

        tmp = tmp.pix.assign(sector=tmp.index.get_level_values("sector").map(ceds_map))
        if tmp.index.get_level_values("sector").isnull().any():
            raise AssertionError

        tmp = tmp.pix.assign(
            sector=tmp.index.get_level_values("sector").map(
                lambda v: "Biosphere" if v.split("|")[-1] in CO2_BIOSPHERE_SECTORS_GRIDDING else "Fossil"
            )
        )
        ext_ts = (
            tmp.groupby(tmp.index.names)
            .sum()
            .pix.convert_unit({"kt CH4/yr": "Mt CH4/yr"})
            .pix.format(variable="{variable}|{sector}", drop=True)
            .rename_axis("year", axis="columns")
        )
        pd.testing.assert_frame_equal(
            ext_ts.loc[:, last_ceds_year:],
            ceds_only_sum.loc[species_loc, last_ceds_year : ext_ts.columns.max()].pix.project(["variable", "unit"]),
            check_like=True,
        )

        history_species_incl_extension = (
            history_global_sum.loc[species_loc]
            .subtract(ceds_only_sum.loc[species_loc].reset_index("model", drop=True), axis="rows")
            .add(ext_ts, axis="rows")
        )

        res = pix.concat([res.loc[~species_loc], history_species_incl_extension])
        pd.testing.assert_frame_equal(
            history_global_sum.loc[species_loc, last_ceds_year:], res.loc[species_loc, last_ceds_year:], check_like=True
        )

    return res


def main():
    """
    Extract the data
    """
    pix.set_openscm_registry_as_default()
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
        # Assuming that CO2 mapping holds for other gases, fine
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
            history_countries = sdf.loc[pix.ismatch(region=["**iso3**"])]
            historical_sum_by_source = history_countries.openscm.groupby_except(["region"]).sum(min_count=1)
            historical_sum = (
                history_countries.openscm.groupby_except(["model", "region"])
                .sum(min_count=1)
                .pix.assign(model="historical-sources")
            )
            historical_sum = add_ceds_extension(
                historical_sum, historical_sum_by_source.loc[pix.ismatch(model="**CEDS**")]
            )
            out_l.append(historical_sum)

        else:
            for _, mdf in sdf.groupby("model"):
                # Single model hence can sum across regions with confidence
                out_l.append(mdf.openscm.groupby_except("region").sum(min_count=1))

    out = pix.concat(out_l)
    # Make sure we didn't use a broken historical region aggregation
    assert_harmonised(
        df=out.loc[~pix.isin(scenario="historical")],
        history=out.loc[pix.isin(scenario="historical")].pix.project(["variable", "unit"]),
        harmonisation_time=HARMONISATION_YEAR,
    )

    out_path = OUT_PATH / f"{INFILLING_DB_DIR.name}_harmonised-emissions-fossil-biosphere-aggregation.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
