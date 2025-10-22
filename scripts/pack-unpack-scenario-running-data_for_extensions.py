#!.pixi/envs/default/bin/python3.11
"""
Pack and unpack scenario running data
"""

import tempfile
from pathlib import Path

from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

from emissions_harmonization_historical.constants_5000 import (
    HARMONISATION_ID,
    HARMONISED_SCENARIO_DB,
    HISTORY_FOR_HARMONISATION_ID,
    HISTORY_HARMONISATION_DB,
    INFILLED_OUT_DIR_ID,
    INFILLED_SCENARIOS_DB,
)


def main(pack: bool = True) -> None:
    """Unpack or pack data"""
    REPO_ROOT = Path(__file__).parents[1]

    if pack:
        # model_to_grab = "WITCH"

        # raw_scenario_data = RAW_SCENARIO_DB.load(pix.ismatch(model=f"**{model_to_grab}**"))
        harmonisation_history = HISTORY_HARMONISATION_DB.load()
        harmonised_scenarios = HARMONISED_SCENARIO_DB.load()
        infilled_processed = INFILLED_SCENARIOS_DB.load()

        for data, gzip in (
            # (raw_scenario_data, REPO_ROOT / f"raw-scenarios_{DOWNLOAD_SCENARIOS_ID}.tar.gz"),
            (harmonisation_history, REPO_ROOT / f"harmonisation-history_{HISTORY_FOR_HARMONISATION_ID}.tar.gz"),
            (harmonised_scenarios, REPO_ROOT / f"harmonised-scenarios_{HARMONISATION_ID}.tar.gz"),
            (infilled_processed, REPO_ROOT / f"infilled_{INFILLED_OUT_DIR_ID}.tar.gz"),
        ):
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_db_dir = tmp_dir / "db"
            tmp_db = OpenSCMDB(
                db_dir=tmp_db_dir,
                backend_data=FeatherDataBackend(),
                backend_index=FeatherIndexBackend(),
            )
            tmp_db.save(data)

            print(f"Creating {gzip}")
            tmp_db.to_gzipped_tar_archive(gzip)

    else:
        for gzip, dest in (
            # (REPO_ROOT / f"raw-scenarios_{DOWNLOAD_SCENARIOS_ID}.tar.gz", RAW_SCENARIO_DB.db_dir),
            (
                REPO_ROOT / f"harmonisation-history_{HISTORY_FOR_HARMONISATION_ID}.tar.gz",
                HISTORY_HARMONISATION_DB.db_dir,
            ),
            (
                REPO_ROOT / f"harmonised-scenarios_{HARMONISATION_ID}.tar.gz",
                HARMONISED_SCENARIO_DB.db_dir,
            ),
            (REPO_ROOT / f"infilled_{INFILLED_OUT_DIR_ID}.tar.gz", INFILLED_SCENARIOS_DB.db_dir),
        ):
            OpenSCMDB.from_gzipped_tar_archive(
                tar_archive=gzip,
                db_dir=dest,
            )
            print(f"Unpacked {dest}")


if __name__ == "__main__":
    pack = True
    pack = False

    main(pack=pack)
