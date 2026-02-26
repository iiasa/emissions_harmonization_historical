"""
Extract the scm ensemble members to csv
"""

from pathlib import Path

import pandas_indexing as pix
import tqdm.auto

from emissions_harmonization_historical.constants_5000 import (
    DOWNLOAD_SCENARIOS_ID,
    SCM_OUTPUT_DB,
)


def main():
    """
    Extract the data
    """
    scm: str = "MAGICCv7.6.0a3"

    HERE = Path(__file__).parent
    REPO_ROOT = HERE.parent
    OUT_PATH = REPO_ROOT / "ensemble-members" / DOWNLOAD_SCENARIOS_ID

    raw_gsat_variable_in = "Surface Air Temperature Change"
    iams = SCM_OUTPUT_DB.load_metadata().get_level_values("model").unique()

    for model in tqdm.auto.tqdm(iams, desc="IAMs"):
        gmt_df = SCM_OUTPUT_DB.load(
            pix.ismatch(variable=raw_gsat_variable_in, model=f"*{model}*", climate_model=f"*{scm}*")
        )

        out_dir = OUT_PATH / model
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / f"{model}_{scm}_ensemble_members.csv"

        print(f"Extracting ensembles to {out_dir}")
        gmt_df.to_csv(out_dir / out_file)


if __name__ == "__main__":
    main()
