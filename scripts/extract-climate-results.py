"""
Extract climate assessment results into a single folder that can be put on sharepoint
"""

from pathlib import Path

import pandas_indexing as pix

from emissions_harmonization_historical.constants_5000 import (
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSING_DIR,
)


def main():
    """
    Run the 500x series of notebooks
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    OUT_PATH = REPO_ROOT / "climate-assessment-for-sharepoint" / POST_PROCESSING_DIR.name

    iams = POST_PROCESSED_METADATA_CATEGORIES_DB.load_metadata().get_level_values("model").unique()
    for iam in iams:
        out_dir = OUT_PATH / iam
        out_dir.mkdir(exist_ok=True, parents=True)

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


if __name__ == "__main__":
    main()
