"""
Reproduction of the AR6 workflow
"""

from __future__ import annotations

import pandas as pd
from attrs import define

AR6_RAW_VARIABLES: tuple[str, ...] = (
    "Emissions|CH4",
    "Emissions|N2O",
)
"""
Raw variables that were used in the AR6 workflow

Many variables were dropped before the workflow was entered.
For example, most sectoral detail.
"""


@define
class AR6Harmoniser:
    """
    Harmoniser that follows the same logic as was used in AR6
    """

    def __call__(self, raw_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise

        Parameters
        ----------
        raw_emissions
            Emissions to harmonise


        Returns
        -------
        :
            Harmonised emissions
        """
        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in raw_emissions
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable
        raise NotImplementedError


def get_ar6_harmoniser() -> AR6Harmoniser:
    """Docstring TBD"""
    return AR6Harmoniser()
