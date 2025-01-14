"""
Tools for pre-processing
"""

from __future__ import annotations

import pandas as pd
import pandas_indexing as pix
from attrs import define

from gcages.units_helpers import strip_pint_incompatible_characters_from_units


@define
class PreProcessor:
    emissions_out: tuple[str, ...]

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        # TODO: add checks:
        # - no rows should be all zero or all nan
        # - data should be available for all required years
        # - no negative values for non-CO2
        res: pd.DataFrame = in_emissions.loc[pix.isin(variable=self.emissions_out)]

        res = strip_pint_incompatible_characters_from_units(
            res, units_index_level="unit"
        )

        return res
