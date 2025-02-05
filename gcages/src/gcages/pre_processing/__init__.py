"""
Tools for pre-processing
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pandas_indexing as pix  # type: ignore
from attrs import define

from gcages.parallelisation import assert_only_working_on_variable_unit_variations
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


def reclassify_variables(
    indf: pd.DataFrame,
    reclassifications: Mapping[str, tuple[str, ...]],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Reclassify variables

    Parameters
    ----------
    indf
        Data to add sums to

    reclassifications
        Definition of the reclassifications.

        For each variable (key) in `reclassifications`, the variables in its value
        will be reclassified as part of its total.

        For example, if `reclassifications` is

        ```python
        {"var_a": ("var_b", "var_c")}
        ```

        then if "var_b" or "var_c" (or both) is in `indf`,
        they will be removed and their contents will be added to the total of `var_a`.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf`, reclassified as needed.
    """
    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_target, v_sources in reclassifications.items():
        locator_sources = pix.isin(variable=v_sources)
        to_add = out.loc[locator_sources]
        if not to_add.empty:
            out.loc[pix.isin(variable=v_target)] += to_add.sum()
            out = out.loc[~locator_sources]

    return out


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
