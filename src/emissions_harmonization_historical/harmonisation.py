"""
Harmonisation configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
from attrs import define
from gcages.harmonisation import harmonise_scenario
from gcages.io import load_timeseries_csv
from gcages.parallelisation import run_parallel
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

HARMONISATION_YEAR = 2021

HARMONISATION_YEAR_MISSING_SCALING_YEAR = 2015
"""
Year to scale if the harmonisation year is missing from a submission
"""


@define
class AR7FTHarmoniser:
    """
    AR7 fast-track harmoniser
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
                iterable_input=(gdf for _, gdf in in_emissions.groupby(["model", "scenario"])),
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

    @classmethod
    def from_default_config(cls, data_root: Path, n_processes: int = multiprocessing.cpu_count()) -> AR7FTHarmoniser:
        """
        Initialise from default, hard-coded configuration
        """
        from emissions_harmonization_historical.constants import COMBINED_HISTORY_ID, HARMONISATION_VALUES_ID

        history_path = (
            data_root
            / "global-composite"
            / f"cmip7-harmonisation-history_world_{COMBINED_HISTORY_ID}_{HARMONISATION_VALUES_ID}.csv"
        )

        history = strip_pint_incompatible_characters_from_units(
            load_timeseries_csv(
                history_path,
                index_columns=["model", "scenario", "region", "variable", "unit"],
                out_column_type=int,
            )
        )

        # As at 2024-01-30, just the list from AR6.
        # We can tweak from here.
        aneris_overrides = pd.DataFrame(
            [
                # depending on the decision tree in aneris/method.py
                #     {'method': 'default_aneris_tree', 'variable': 'Emissions|BC'},
                {
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC",
                },  # high historical variance (cov=16.2)
                {
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C2F6",
                },  # high historical variance (cov=16.2)
                {
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C6F14",
                },  # high historical variance (cov=15.4)
                {
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|CF4",
                },  # high historical variance (cov=11.2)
                {
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|CO",
                },  # high historical variance (cov=15.4)
                {
                    "method": "reduce_ratio_2080",
                    "variable": "Emissions|CO2",
                },  # always ratio method by choice
                {
                    # high historical variance,
                    # but using offset method to prevent diff from increasing
                    # when going negative rapidly (cov=23.2)
                    "method": "reduce_offset_2150_cov",
                    "variable": "Emissions|CO2|AFOLU",
                },
                {
                    "method": "reduce_ratio_2080",  # always ratio method by choice
                    "variable": "Emissions|CO2|Energy and Industrial Processes",
                },
                # depending on the decision tree in aneris/method.py
                #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'},
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|F-Gases",
                },  # basket not used in infilling (sum of f-gases with low model reporting confidence)
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC",
                },  # basket not used in infilling (sum of subset of f-gases with low model reporting confidence)
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC125",
                },  # minor f-gas with low model reporting confidence
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC134a",
                },  # minor f-gas with low model reporting confidence
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC143a",
                },  # minor f-gas with low model reporting confidence
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC227ea",
                },  # minor f-gas with low model reporting confidence
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC23",
                },  # minor f-gas with low model reporting confidence
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC32",
                },  # minor f-gas with low model reporting confidence
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC43-10",
                },  # minor f-gas with low model reporting confidence
                # depending on the decision tree in aneris/method.py
                #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'},
                # depending on the decision tree in aneris/method.py
                #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'},
                # depending on the decision tree in aneris/method.py
                #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'},
                {
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|OC",
                },  # high historical variance (cov=18.5)
                {
                    "method": "constant_ratio",
                    "variable": "Emissions|SF6",
                },  # minor f-gas with low model reporting confidence
                # depending on the decision tree in aneris/method.py
                #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'},
                {
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|VOC",
                },  # high historical variance (cov=12.0)
            ]
        )

        return AR7FTHarmoniser(
            historical_emissions=history,
            harmonisation_year=HARMONISATION_YEAR,
            calc_scaling_year=HARMONISATION_YEAR_MISSING_SCALING_YEAR,
            aneris_overrides=aneris_overrides,
            n_processes=n_processes,
            # TODO: implement and turn on
            run_checks=False,
        )
