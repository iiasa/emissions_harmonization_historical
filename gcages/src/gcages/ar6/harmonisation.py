"""
Harmonisation part of the AR6 workflow
"""

from __future__ import annotations

import importlib
import multiprocessing
from pathlib import Path

import pandas as pd
import pandas_indexing as pix  # type: ignore
from attrs import define

from gcages.aneris_helpers import harmonise_all
from gcages.harmonisation.helpers import add_historical_year_based_on_scaling
from gcages.parallelisation import (
    assert_only_working_on_variable_unit_variations,
    run_parallel,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


def load_ar6_historical_emissions() -> pd.DataFrame:
    """
    Load the historical emissions that were used in AR6

    The data is massaged to what is expected by our harmonisation,
    it isn't the raw data (at least not raw metadata).

    Returns
    -------
    :
        Historical emissions used in AR6
    """
    filepath: Path = Path(  # type: ignore
        importlib.resources.files("gcages") / "ar6" / "history_ar6.csv"
    )
    res: pd.DataFrame = pd.read_csv(filepath)
    res.columns = res.columns.str.lower()
    res = res.set_index(["model", "scenario", "variable", "unit", "region"])
    res.columns = res.columns.astype(int)

    res = pix.assignlevel(
        res,
        variable=res.pix.unique("variable").map(  # type: ignore
            lambda x: x.replace("AR6 climate diagnostics|", "").replace(
                "|Unharmonized", ""
            )
        ),
    )

    # Strip out any units that won't play nice with pint
    res = strip_pint_incompatible_characters_from_units(res, units_index_level="unit")

    # We only care about data from 1990 onwards (really, only 2015, but ok).
    res = res.loc[:, 1990:]

    return res


def harmonise_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.DataFrame | None,
    calc_scaling_year: int,
) -> pd.DataFrame:
    """
    Harmonise a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    history
        History to harmonise to

    year
        Year to use for harmonisation

    overrides
        Overrides to pass to aneris

    calc_scaling_year
        Year to use for calculating scaling if `year` is not in `indf`

    Returns
    -------
    :
        Harmonised scenario
    """
    assert_only_working_on_variable_unit_variations(indf)

    # TODO: split this out
    # A bunch of other fix ups that were applied in AR6
    if year not in indf:
        emissions_to_harmonise = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf,
            emissions_history=history,
        )

    elif indf[year].isnull().any():
        null_emms_in_harm_year = indf[year].isnull()

        dont_change = indf[~null_emms_in_harm_year]

        updated = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf[null_emms_in_harm_year].drop(year, axis="columns"),
            emissions_history=history,
        )

        emissions_to_harmonise = pd.concat([dont_change, updated])

    else:
        emissions_to_harmonise = indf

    # In AR6, any emissions with zero in the harmonisation year were dropped
    emissions_to_harmonise = emissions_to_harmonise[
        ~(emissions_to_harmonise[year] == 0.0)
    ]

    ### In AR6, we interpolated before harmonising

    # First, check that there are no nans in the max year.
    # I don't know what happens in that case.
    if emissions_to_harmonise[emissions_to_harmonise.columns.max()].isnull().any():
        raise NotImplementedError

    # Then, interpolate
    out_interp_years = list(range(year, emissions_to_harmonise.columns.max() + 1))
    emissions_to_harmonise = emissions_to_harmonise.reindex(
        columns=out_interp_years
    ).interpolate(method="slinear", axis="columns")

    harmonised = harmonise_all(
        emissions_to_harmonise,
        history=history,
        year=year,
        overrides=overrides,
    )

    return harmonised


@define
class AR6Harmoniser:
    """
    Harmoniser that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
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
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in raw_emissions
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        harmonised_df = pix.concat(
            run_parallel(
                func_to_call=harmonise_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                ),
                input_desc="model-scenario combinations to harmonise",
                n_processes=self.n_processes,
                history=self.historical_emissions,
                year=self.harmonisation_year,
                overrides=self.aneris_overrides,
                calc_scaling_year=self.calc_scaling_year,
            )
        )

        # Not sure why this is happening, anyway
        harmonised_df.columns = harmonised_df.columns.astype(int)

        # Apply AR6 naming scheme
        out: pd.DataFrame = harmonised_df.pix.format(
            variable="AR6 climate diagnostics|Harmonized|{variable}"
        )

        # TODO:
        #   - enable optional checks for:
        #       - input and output metadata is identical
        #           - no mangled variable names
        #           - no mangled units
        #           - output timesteps are from harmonisation year onwards only
        #       - output scenarios all have common starting point

        return out

    @classmethod
    def from_ar6_like_config(
        cls, run_checks: bool = True, n_processes: int = multiprocessing.cpu_count()
    ) -> AR6Harmoniser:
        """
        Initialise from config (exactly) like what was used in AR6

        Parameters
        ----------
        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        n_processes
            Number of processes to use for parallel processing.

            Set to 1 to process in serial.

        Returns
        -------
        :
            Initialised harmoniser
        """
        historical_emissions = load_ar6_historical_emissions()

        # All variables not mentioned here use aneris' default decision tree
        aneris_overrides_ar6 = pd.DataFrame(
            [
                {
                    # high historical variance (cov=16.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC",
                },
                {
                    # high historical variance (cov=16.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C2F6",
                },
                {
                    # high historical variance (cov=15.4)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C6F14",
                },
                {
                    # high historical variance (cov=11.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|CF4",
                },
                {
                    # high historical variance (cov=15.4)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|CO",
                },
                {
                    # always ratio method by choice
                    "method": "reduce_ratio_2080",
                    "variable": "Emissions|CO2",
                },
                {
                    # high historical variance,
                    # but using offset method to prevent diff
                    # from increasing when going negative rapidly (cov=23.2)
                    "method": "reduce_offset_2150_cov",
                    "variable": "Emissions|CO2|AFOLU",
                },
                {
                    # always ratio method by choice
                    "method": "reduce_ratio_2080",
                    "variable": "Emissions|CO2|Energy and Industrial Processes",
                },
                {
                    # basket not used in infilling
                    # (sum of f-gases with low model reporting confidence)
                    "method": "constant_ratio",
                    "variable": "Emissions|F-Gases",
                },
                {
                    # basket not used in infilling
                    # (sum of subset of f-gases with low model reporting confidence)
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC125",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC134a",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC143a",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC227ea",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC23",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC32",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC43-10",
                },
                {
                    # high historical variance (cov=18.5)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|OC",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|SF6",
                },
                {
                    # high historical variance (cov=12.0)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|VOC",
                },
            ]
        )

        # TODO: turn checks back on
        return cls(
            historical_emissions=historical_emissions,
            harmonisation_year=2015,
            calc_scaling_year=2010,
            aneris_overrides=aneris_overrides_ar6,
            run_checks=run_checks,
            n_processes=n_processes,
        )
