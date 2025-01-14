"""
Tools for harmonisation
"""

from __future__ import annotations

import multiprocessing

import pandas as pd
import pandas_indexing as pix
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
    historical_emissions: pd.DataFrame
    harmonisation_year: int
    calc_scaling_year: int
    aneris_overrides: pd.DataFrame | None
    n_processes: int = multiprocessing.cpu_count()

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        # TODO: add checks back in
        harmonised_df = pix.concat(
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
