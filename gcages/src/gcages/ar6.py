"""
Reproduction of the AR6 workflow
"""

from __future__ import annotations

import importlib

import aneris.convenience
import pandas as pd
import pandas_indexing as pix
import tqdm.autonotebook as tqdman
from attrs import define

AR6_RAW_VARIABLES: tuple[str, ...] = (
    "Emissions|BC",
    "Emissions|PFC|C2F6",
    "Emissions|PFC|C6F14",
    "Emissions|PFC|CF4",
    "Emissions|CO",
    # "Emissions|CO2",  # Don't harmonise the totals because we don't use them anyway
    "Emissions|CO2|AFOLU",
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CH4",
    # "Emissions|F-Gases",  # Don't harmonise the totals because we don't use them anyway
    # "Emissions|HFC",  # Don't harmonise the totals because we don't use them anyway
    "Emissions|HFC|HFC125",
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC143a",
    "Emissions|HFC|HFC227ea",
    "Emissions|HFC|HFC23",
    # 'Emissions|HFC|HFC245ca',  # all nan in historical dataset (RCMIP)
    # "Emissions|HFC|HFC245fa",  # not in historical dataset (RCMIP)
    "Emissions|HFC|HFC32",
    "Emissions|HFC|HFC43-10",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NOx",
    "Emissions|OC",
    # "Emissions|PFC",  # Don't harmonise the totals because we don't use them anyway
    "Emissions|SF6",
    "Emissions|Sulfur",
    "Emissions|VOC",
)
"""
Raw variables that were used in the AR6 workflow

Many variables were dropped before the workflow was entered.
For example, most sectoral detail.
"""


def load_ar6_historical_emissions() -> pd.DataFrame:
    res = pd.read_csv(importlib.resources.open_binary("gcages", "history_ar6.csv"))
    res = res.set_index(["Model", "Scenario", "Variable", "Unit", "Region"])
    res.columns = res.columns.astype(int)

    res = pix.assignlevel(
        res,
        Variable=res.pix.unique("Variable").map(
            lambda x: x.replace("AR6 climate diagnostics|", "").replace(
                "|Unharmonized", ""
            )
        ),
    )

    # May need to clean up unit too

    # We only care about data from 1990 onwards
    # (really, only 2015, but ok)
    res = res.loc[:, 1990:]

    return res


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

        if self.harmonisation_year not in raw_emissions:
            # Need to add pre-processing in its own class
            raise NotImplementedError

        harmonised_dfs = []
        for _, msdf in tqdman.tqdm(raw_emissions.groupby(["model", "scenario"])):
            harmonised_dfs.append(
                aneris.convenience.harmonise_all(
                    msdf,
                    history=self.historical_emissions,
                    year=self.harmonisation_year,
                    overrides=self.aneris_overrides,
                )
            )
        breakpoint()
        harmonised_df = pd.concat(harmonised_dfs).reset_index()

        # # Parallel version something like this
        # with parallel_progress_bar(tqdm.tqdm(desc="Harmonisation")):
        #     LOGGER.info("Harmonising in parallel")
        #     # TODO: remove hard-coding of n_jobs
        #     scenarios_harmonized = Parallel(n_jobs=-1)(
        #         delayed(aneris.convenience.harmonise_all)(
        #             msdf,
        #             history=history,
        #             harmonisation_year=harmonization_year,
        #             overrides=overrides,
        #         )
        #         for _, msdf in scenarios.groupby(["model", "scenario"])
        #     )

        # TODO:
        #   - enable optional checks for:
        #       - input and output metadata is identical (except maybe a stage indicator)
        #           - no mangled variable names
        #           - no mangled units
        #           - output timesteps are from harmonisation year onwards, otherwise identical to input
        #       - output scenarios all have common starting point

        raise NotImplementedError

    @classmethod
    def from_ar6_like_config(cls, run_checks: bool = True) -> AR6Harmoniser:
        """
        Initialise from config (exactly) like what was used in AR6

        Parameters
        ----------
        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

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
                    # basket not used in infilling (sum of f-gases with low model reporting confidence)
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
            aneris_overrides=aneris_overrides_ar6,
            run_checks=run_checks,
        )
