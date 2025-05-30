"""
Extract emissions results into a single folder that can be put on sharepoint
"""

from pathlib import Path

import pandas_indexing as pix
import tqdm.auto

from emissions_harmonization_historical.constants_5000 import POST_PROCESSED_TIMESERIES_DB, POST_PROCESSING_DIR


def main():
    """
    Extract the data
    """
    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    OUT_PATH = REPO_ROOT / "emissions-for-sharepoint" / POST_PROCESSING_DIR.name

    iams = POST_PROCESSED_TIMESERIES_DB.load_metadata().get_level_values("model").unique()
    for iam in tqdm.auto.tqdm(iams, desc="IAMs"):
        out_dir = OUT_PATH / iam
        out_dir.mkdir(exist_ok=True, parents=True)
        print(f"Extracting to {out_dir}")

        iam_emissions = POST_PROCESSED_TIMESERIES_DB.load(pix.isin(model=iam))
        for stage, out_file in [
            ("pre-processed-scms", f"pre-processed-scms_{iam}.csv"),
            ("harmonised-gridding", f"harmonised-gridding_{iam}.csv"),
            ("harmonised-scms", f"harmonised-scms_{iam}.csv"),
            ("complete", f"infilled_{iam}.csv"),
        ]:
            to_save = iam_emissions.loc[pix.isin(stage=stage)]
            to_save.reset_index("stage", drop=True).to_csv(out_dir / out_file)


if __name__ == "__main__":
    main()
