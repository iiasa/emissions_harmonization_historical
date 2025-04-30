"""
Harmonisation configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from gcages.units_helpers import strip_pint_incompatible_characters_from_units
from pandas_openscm.io import load_timeseries_csv

HARMONISATION_YEAR = 2023

HARMONISATION_YEAR_MISSING_SCALING_YEAR = 2015
"""
Year to scale if the harmonisation year is missing from a submission
"""


def load_default_history(data_root: Path) -> pd.DataFrame:
    """Load default emissions history"""
    from emissions_harmonization_historical.constants import COMBINED_HISTORY_ID, HARMONISATION_VALUES_ID

    history_path = (
        data_root
        / "global-composite"
        / f"cmip7-harmonisation-history_world_{COMBINED_HISTORY_ID}_{HARMONISATION_VALUES_ID}.csv"
    )

    history = strip_pint_incompatible_characters_from_units(
        load_timeseries_csv(
            history_path,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_column_type=int,
        )
    )

    return history
