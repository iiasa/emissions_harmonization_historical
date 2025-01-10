"""
Harmonisation part of the AR6 workflow
"""

from __future__ import annotations

import importlib
import multiprocessing

import aneris.convenience  # type: ignore
import pandas as pd
import pandas_indexing as pix  # type: ignore
from attrs import define

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
    res: pd.DataFrame = pd.read_csv(
        importlib.resources.open_binary("gcages", "history_ar6.csv")
    )
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


def add_historical_year_based_on_scaling(
    year_to_add: int,
    year_calc_scaling: int,
    emissions: pd.DataFrame,
    emissions_historical: pd.DataFrame,
    ms: tuple[str, ...] = ("model", "scenario"),
) -> pd.DataFrame:
    """
    Add a historical emissions year based on scaling

    Parameters
    ----------
    year_to_add
        Year to add

    year_calc_scaling
        Year to use to calculate the scaling

    emissions
        Emissions to which to add data for `year_to_add`

    emissions_historical
        Historical emissions to use to calculate
        the fill values based on scaling

    ms
        Name of the model and scenario columns.

        These have to be dropped from `emissions_historical`
        before everything will line up.

    Returns
    -------
    :
        `emissions` with data for `year_to_add`
        based on the scaling between `emissions`
        and `emissions_historical` in `year_calc_scaling`.
    """
    if emissions.pix.unique(["model", "scenario"]).shape[0] > 1:  # type: ignore
        # Processing is much trickier with multiple scenarios
        raise NotImplementedError

    ms = ("model", "scenario")
    # emissions_no_ms = emissions.reset_index(ms, drop=True)
    emissions_historical_common_vars = emissions_historical.loc[
        pix.isin(variable=emissions.pix.unique("variable"))  # type: ignore
    ]

    emissions_historical_no_ms = emissions_historical_common_vars.reset_index(
        ms, drop=True
    )

    scale_factor = emissions[year_calc_scaling].divide(
        emissions_historical_no_ms[year_calc_scaling]
    )
    fill_value = scale_factor.multiply(emissions_historical_no_ms[year_to_add])
    fill_value.name = year_to_add

    out = pd.concat([emissions, fill_value], axis="columns").sort_index(axis="columns")

    return out


def harmonise_scenario(
    indf: pd.DataFrame, history: pd.DataFrame, year: int, overrides: pd.DataFrame | None
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


    Returns
    -------
    :
        Harmonised scenario
    """
    assert_only_working_on_variable_unit_variations(indf)

    harmonised = aneris.convenience.harmonise_all(
        indf,
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

        # May need to drop all nan times here
        # May need to drop out variables which are all zero
        # May need to drop out variables which are zero in the harmonisation year

        # TODO: move into apply_various_ar6_fix_ups
        # or similar
        if self.harmonisation_year not in in_emissions:
            emissions_to_harmonise = add_historical_year_based_on_scaling(
                year_to_add=self.harmonisation_year,
                year_calc_scaling=self.calc_scaling_year,
                emissions=in_emissions,
                emissions_historical=self.historical_emissions,
            )

        elif in_emissions[self.harmonisation_year].isnull().any():
            null_emms_in_harm_year = in_emissions[self.harmonisation_year].isnull()

            dont_change = in_emissions[~null_emms_in_harm_year]

            updated = add_historical_year_based_on_scaling(
                year_to_add=self.harmonisation_year,
                year_calc_scaling=self.calc_scaling_year,
                emissions=in_emissions[null_emms_in_harm_year].drop(
                    self.harmonisation_year, axis="columns"
                ),
                emissions_historical=self.historical_emissions,
            )

            emissions_to_harmonise = pd.concat([dont_change, updated])

        else:
            emissions_to_harmonise = in_emissions

        # In AR6, any emissions with zero in the harmonisation year were dropped
        emissions_to_harmonise = emissions_to_harmonise[
            ~(emissions_to_harmonise[self.harmonisation_year] == 0.0)
        ]

        # # In AR6, any emissions still with nan were dropped
        # # (I think, might be wrong)
        # emissions_to_harmonise = emissions_to_harmonise[
        #     ~emissions_to_harmonise.isnull().any(axis="columns")
        # ]

        # In AR6, we interpolated before harmonising
        # Check that there are no nans in the max year.
        # I don't know what happens in that case.
        if emissions_to_harmonise[emissions_to_harmonise.columns.max()].isnull().any():
            raise NotImplementedError

        out_interp_years = list(
            range(self.harmonisation_year, in_emissions.columns.max() + 1)
        )
        emissions_to_harmonise = emissions_to_harmonise.reindex(
            columns=out_interp_years
        ).interpolate(method="slinear", axis="columns")

        harmonised_df = pix.concat(
            run_parallel(  # type: ignore
                func_to_call=harmonise_scenario,
                iterable_input=(
                    gdf
                    for _, gdf in emissions_to_harmonise.groupby(["model", "scenario"])
                ),
                input_desc="model-scenario combinations to harmonise",
                n_processes=self.n_processes,
                history=self.historical_emissions,
                year=self.harmonisation_year,
                overrides=self.aneris_overrides,
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
        #         (except maybe a stage indicator)
        #           - no mangled variable names
        #           - no mangled units
        #           - output timesteps are from harmonisation year onwards,
        #             otherwise identical to input
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
