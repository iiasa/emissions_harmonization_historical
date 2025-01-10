"""
Helpers for unit handling
"""

from __future__ import annotations

import pandas as pd
import pandas_indexing as pix


def strip_pint_incompatible_characters_from_units(
    indf: pd.DataFrame, units_index_level: str = "unit"
) -> pd.DataFrame:
    """
    Strip pint-incompatible characters from units

    Parameters
    ----------
    indf
        Input data from which to strip pint-incompatible characters

    units_index_level
        Column in `indf`'s index that holds the units values

    Returns
    -------
    :
        `indf` with pint-incompatible characters
        removed from the `units_index_level` of its index.
    """
    res = pix.assignlevel(
        indf,
        **{
            units_index_level: indf.index.get_level_values(units_index_level).map(
                lambda x: x.replace("-", "")
            ),
        },
    )

    return res
