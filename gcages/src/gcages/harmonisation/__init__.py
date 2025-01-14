"""
Tools for harmonisation
"""

from __future__ import annotations

import multiprocessing

import pandas as pd
import pandas_indexing as pix  # type: ignore
from attrs import define

from gcages.aneris_helpers import harmonise_all
from gcages.harmonisation.helpers import add_harmonisation_year_if_needed
from gcages.parallelisation import (
    assert_only_working_on_variable_unit_variations,
    run_parallel,
)


def harmonise_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    harmonisation_year: int,
    overrides: pd.DataFrame | None,
    calc_scaling_year: int,
) -> pd.DataFrame:
    """
    Harmonise a scenario

    This is quite a basic function,
    intended to have as few bells and whistles as possible.

    Parameters
    ----------
    indf
        Input data to harmonise

    history
        History to use for harmonisation

    harmonisation_year
        Year in which to harmonise `indf` and `history`

    overrides
        Overrides to pass to `aneris`

    calc_scaling_year
        Year to use for calculating a scaling based on history

        Only used if `indf` does not have data for `harmonisation_year`
        in all rows.


    Returns
    -------
    :
        Harmonised `indf`
    """
    assert_only_working_on_variable_unit_variations(indf)

    # Make sure columns are sorted, things go weird if they're not
    indf = indf.sort_index(axis="columns")

    emissions_to_harmonise = add_harmonisation_year_if_needed(
        indf,
        harmonisation_year=harmonisation_year,
        calc_scaling_year=calc_scaling_year,
        emissions_history=history,
    )

    harmonised = harmonise_all(
        emissions_to_harmonise,
        history=history,
        year=harmonisation_year,
        overrides=overrides,
    )

    return harmonised


@define
class Harmoniser:
    """
    Harmoniser
    """

    historical_emissions: pd.DataFrame
    """
    Historical emissions to use for harmonisation
    """

    harmonisation_year: int
    """
    Year in which to harmonise
    """

    calc_scaling_year: int
    """
    Year to use for calculating a scaling factor from historical

    This is only needed if `self.harmonisation_year`
    is not in the emissions to be harmonised.

    For example, if `self.harmonisation_year` is 2015
    and `self.calc_scaling_year` is 2010
    and we have a scenario without 2015 data,
    then we will use the difference from historical in 2010
    to infer a value for 2015.

    This logic was perculiar to AR6, it may not be repeated.
    """

    aneris_overrides: pd.DataFrame | None
    """
    Overrides to supply to `aneris.convenience.harmonise_all`

    For source code and docs,
    see e.g. https://github.com/iiasa/aneris/blob/v0.4.2/src/aneris/convenience.py.
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    n_processes: int = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to 1 to process in serial.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise

        Parameters
        ----------
        in_emissions
            Emissions to harmonise

        Returns
        -------
        :
            Harmonised emissions
        """
        if self.run_checks:
            # TODO: add checks back in
            raise NotImplementedError

        harmonised_df: pd.DataFrame = pix.concat(
            run_parallel(
                func_to_call=harmonise_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                ),
                input_desc="model-scenario combinations to harmonise",
                n_processes=self.n_processes,
                history=self.historical_emissions,
                harmonisation_year=self.harmonisation_year,
                overrides=self.aneris_overrides,
                calc_scaling_year=self.calc_scaling_year,
            )
        )

        # Not sure why this is happening, anyway
        harmonised_df.columns = harmonised_df.columns.astype(int)
        harmonised_df = harmonised_df.sort_index(axis="columns")

        return harmonised_df
