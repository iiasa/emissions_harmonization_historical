"""
Harmonisation part of the AR6 workflow
"""

from __future__ import annotations

import importlib

import aneris.convenience
import pandas as pd
import pandas_indexing as pix
import tqdm.autonotebook as tqdman
from attrs import define

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
    res = pd.read_csv(importlib.resources.open_binary("gcages", "history_ar6.csv"))
    res.columns = res.columns.str.lower()
    res = res.set_index(["model", "scenario", "variable", "unit", "region"])
    res.columns = res.columns.astype(int)

    res = pix.assignlevel(
        res,
        variable=res.pix.unique("variable").map(
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

        if self.harmonisation_year not in in_emissions:
            # Need to add pre-processing in its own class
            raise NotImplementedError

        # In AR6, we interpolated before harmonising
        # TODO: add check that there are no nans in the max year.
        # I don't know what happens in that case.
        out_interp_years = list(
            range(self.harmonisation_year, in_emissions.columns.max() + 1)
        )
        emissions_to_harmonise = in_emissions.reindex(
            columns=out_interp_years
        ).interpolate(method="slinear", axis="columns")

        # TODO: Split out into separate function
        # (also makes it possible to fix up the aneris dependence on pyam,
        # i.e. write new convenience module)
        harmonised_df_l = []
        for _, msdf in tqdman.tqdm(
            emissions_to_harmonise.groupby(["model", "scenario"])
        ):
            harmonised_df_l.append(
                aneris.convenience.harmonise_all(
                    msdf,
                    history=self.historical_emissions,
                    year=self.harmonisation_year,
                    overrides=self.aneris_overrides,
                )
            )

        # # Parallel version of the above something like this
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

        harmonised_df = pd.concat(harmonised_df_l)
        # Not sure why this is happening, anyway
        harmonised_df.columns = harmonised_df.columns.astype(int)

        # Apply AR6 naming scheme
        out = harmonised_df.pix.format(
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
            aneris_overrides=aneris_overrides_ar6,
            run_checks=run_checks,
        )
