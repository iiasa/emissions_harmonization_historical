"""
Tools for pre-processing
"""

from __future__ import annotations

import pandas as pd
import pandas_indexing as pix  # type: ignore
from attrs import define

from gcages.units_helpers import strip_pint_incompatible_characters_from_units


@define
class PreProcessor:
    """
    Pre-processor
    """

    emissions_out: tuple[str, ...]
    """
    Names of emissions that can be included in the result of pre-processing

    Not all these emissions need to be there,
    but any names which are not in this list will be flagged
    (if `self.run_checks` is `True`).
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process

        Parameters
        ----------
        in_emissions
            Emissions to pre-process

        Returns
        -------
        :
            Pre-processed emissions
        """
        # TODO: add checks:
        # - no rows should be all zero or all nan
        # - data should be available for all required years
        # - no negative values for non-CO2
        res: pd.DataFrame = in_emissions.loc[pix.isin(variable=self.emissions_out)]

        res = strip_pint_incompatible_characters_from_units(
            res, units_index_level="unit"
        )

        return res
