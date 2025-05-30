"""
Extract climate assessment results into a single folder that can be put on sharepoint
"""

from pathlib import Path

import pandas_indexing as pix
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)

from emissions_harmonization_historical.constants_5000 import (
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSED_TIMESERIES_QUANTILE_DB,
    POST_PROCESSING_DIR,
    SCM_OUTPUT_DB,
)


def main():
    """
    Extract the data
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    OUT_PATH = REPO_ROOT / "climate-assessment-for-sharepoint" / POST_PROCESSING_DIR.name

    iams = POST_PROCESSED_METADATA_CATEGORIES_DB.load_metadata().get_level_values("model").unique()
    for iam in iams:
        out_dir = OUT_PATH / iam
        out_dir.mkdir(exist_ok=True, parents=True)
        print(f"Extracting to {out_dir}")

        model_locator = pix.isin(model=iam)
        latest_magicc_locator = pix.isin(climate_model="MAGICCv7.6.0a3")

        categories = POST_PROCESSED_METADATA_CATEGORIES_DB.load(model_locator & latest_magicc_locator).unstack("metric")
        categories_file = out_dir / f"categories_{iam}.csv"
        categories.to_csv(categories_file)

        warming_quantiles = POST_PROCESSED_METADATA_QUANTILE_DB.load(model_locator & latest_magicc_locator)
        warming_quantiles = warming_quantiles.loc[pix.isin(metric=["max", "2100"], quantile=[0.33, 0.5, 0.67])].unstack(
            ["metric", "quantile"]
        )
        warming_quantiles_file = out_dir / f"warming-quantiles_{iam}.csv"
        warming_quantiles.to_csv(warming_quantiles_file)

        assessed_warming = POST_PROCESSED_TIMESERIES_QUANTILE_DB.load(model_locator & latest_magicc_locator).loc[
            :, 2000:
        ]
        assessed_warming_file = out_dir / f"assessed-warming-timeseries-quantiles_{iam}.csv"
        assessed_warming.to_csv(assessed_warming_file)

        erfs = SCM_OUTPUT_DB.load(
            model_locator & latest_magicc_locator & pix.ismatch(variable="Effective Radiative Forc**")
        ).loc[:, 2000:]
        quantiles_of_interest = (
            0.05,
            0.10,
            1.0 / 6.0,
            0.33,
            0.50,
            0.67,
            5.0 / 6.0,
            0.90,
            0.95,
        )
        erfs_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(
                erfs,
                "run_id",
            ).quantile(quantiles_of_interest),  # type: ignore # pandas-stubs confused
            new_name="quantile",
        )
        erfs_file = out_dir / f"erf-timeseries-quantiles_{iam}.csv"
        erfs_quantiles.to_csv(erfs_file)


if __name__ == "__main__":
    main()
