"""
Tools for infilling
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping
from typing import Callable

import pandas as pd
import pandas_indexing as pix  # type: ignore
import pyam
from attrs import define

from gcages.parallelisation import (
    assert_only_working_on_variable_unit_variations,
    run_parallel,
)


def infill_scenario(
    indf: pd.DataFrame,
    infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    progress: bool = False,
) -> pd.DataFrame | None:
    """
    Infill a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    infillers
        Functions to use for infilling each variable.

        The keys define the variable that can be infilled.
        The variables define the function which,
        given inputs with the expected lead variables,
        returns the infilled time series.

    progress
        Should the progress of infilling variables be shown with a progress bar?

    Returns
    -------
    :
        Infilled scenario. `None` if nothing needs to be infilled.
    """
    assert_only_working_on_variable_unit_variations(indf)

    indf_variables = indf.pix.unique("variable")

    to_infill = [v for v in infillers if v not in indf_variables]
    if not to_infill:
        return None

    if progress:
        import tqdm.autonotebook as tqdman

        to_infill = tqdman.tqdm(to_infill)

    infilled_l = []
    for v_to_infill in to_infill:
        infiller = infillers[v_to_infill]
        infilled_l.append(infiller(indf))

    infilled = pix.concat(infilled_l).sort_index(axis="columns")

    return infilled


@define
class Infiller:
    """
    Infiller
    """

    infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
    """
    Functions to use for infilling each variable.

    The keys define the variable that can be infilled.
    The variables define the function which,
    given inputs with the expected lead variables,
    returns the infilled time series.
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
        Infill

        Parameters
        ----------
        in_emissions
            Emissions to infill

        Returns
        -------
        :
            Infilled emissions
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        # The most efficient parallelisation is not always obvious.
        # We have the 'most obvious' solution below.
        # If we needed, we could open things up to better injection
        # to better support other patterns.
        infilled_df = pix.concat(
            [
                v
                for v in run_parallel(
                    func_to_call=infill_scenario,
                    iterable_input=(
                        gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                    ),
                    input_desc="model-scenario combinations to infill",
                    n_processes=self.n_processes,
                    infillers=self.infillers,
                )
                if v is not None
            ]
        )

        # TODO:
        #   - enable optional checks for:
        #       - input and output metadata is identical
        #         (except maybe a stage indicator)
        #           - no mangled variable names
        #           - no mangled units
        #           - output timesteps are from harmonisation year onwards,
        #             otherwise identical to input
        #       - output scenarios all have common starting point

        return infilled_df
