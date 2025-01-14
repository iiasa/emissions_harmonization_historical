"""
Input/output supporting code
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
