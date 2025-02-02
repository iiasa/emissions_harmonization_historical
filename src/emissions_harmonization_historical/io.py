"""
Input/output supporting code
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_indexing as pix
from gcages.io import load_timeseries_csv


def load_csv(fp: Path) -> pd.DataFrame:
    """
    Load a pyam-style CSV

    Parameters
    ----------
    fp
        File path to load

    Returns
    -------
    :
        Loaded data
    """
    out = pd.read_csv(fp)
    out.columns = out.columns.str.lower()
    out = out.set_index(["model", "scenario", "variable", "region", "unit"])
    out.columns = out.columns.astype(int)

    return out


def load_global_scenario_data(scenario_path: Path, scenario_time_id: str, progress: bool = False) -> pd.DataFrame:
    """
    Load global scenario data

    Parameters
    ----------
    scenario_path
        Path in which the scenarios were sved

    scenario_time_id
        Time ID used for saving the scenarios

    progress
        Show a progress bar while loading?

    Returns
    -------
    :
        Loaded global scenario data
    """
    scenario_files = tuple(scenario_path.glob(f"{scenario_time_id}__scenarios-scenariomip__*.csv"))
    if not scenario_files:
        msg = f"Check your scenario ID. {list(scenario_path.glob('*.csv'))=}"
        raise AssertionError(msg)

    if progress:
        from tqdm.auto import tqdm

        scenario_files = tqdm(scenario_files, desc="Scenario files")

    scenarios_raw = pix.concat(
        [
            load_timeseries_csv(
                f,
                index_columns=["model", "scenario", "region", "variable", "unit"],
                out_column_type=int,
            )
            for f in scenario_files
        ]
    ).sort_index(axis="columns")

    scenarios_raw_global = scenarios_raw.loc[pix.ismatch(region="World")]

    return scenarios_raw_global
