"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import pandas as pd
import pandas_indexing as pix
from attrs import define

from gcages.units_helpers import strip_pint_incompatible_characters_from_units


@define
class AR6PreProcessor:
    """
    Pre-processor that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
    """

    emissions_out: tuple[str, ...]
    """
    Names of emissions that can be included in the result of pre-processing

    Not all these emissions need to be there,
    but any names which are not in this list will be flagged
    (if `self.run_checks` is `True`).
    """

    negative_value_not_small_threshold: float
    """
    Threshold which defines when a negative value is not small

    Non-CO2 emissions less than this that are negative
    are not automatically set to zero.
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
        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in the output
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        if in_emissions.pix.unique(["model", "scenario"]).shape[0] > 1:
            # Mapping is much trickier with multiple scenarios
            raise NotImplementedError

        # Rmove any rows with only zero
        in_emissions = in_emissions[~((in_emissions == 0.0).all(axis="columns"))]

        # TODO: add some configuration for this mapping
        conditional_sums = (
            # Variable to create: variables it depends on
            (
                "Emissions|CO2|Energy and Industrial Processes",
                (
                    "Emissions|CO2|Industrial Processes",
                    "Emissions|CO2|Energy",
                ),
            ),
        )
        for v_target, v_sources in conditional_sums:
            existing_vars = in_emissions.pix.unique("variable")
            if v_target not in existing_vars:
                if all(v in existing_vars for v in v_sources):
                    locator_sources = pix.isin(variable=v_sources)
                    to_add = in_emissions.loc[locator_sources]

                    tmp = in_emissions.loc[pix.isin(variable=v_sources[0])]
                    tmp[:] = to_add.sum()
                    tmp = tmp.pix.assign(variable=v_target)
                    in_emissions = pd.concat(
                        [in_emissions.loc[~locator_sources], tmp], axis="rows"
                    )

        reclassifications = {
            "Emissions|CO2|Energy and Industrial Processes": (
                "Emissions|CO2|Other",
                "Emissions|CO2|Waste",
            )
        }
        for v_target, v_sources in reclassifications.items():
            if any(
                reclassify_v in in_emissions.pix.unique("variable")
                for reclassify_v in v_sources
            ):
                locator_sources = pix.isin(variable=v_sources)
                to_add = in_emissions.loc[locator_sources]
                in_emissions.loc[pix.isin(variable=v_target)] += to_add.sum()
                in_emissions = in_emissions.loc[~locator_sources]

        conditional_keepers = (
            # (
            #     Variable to potentially remove,
            #     remove if all of these variables are present
            # )
            (
                "Emissions|CO2",
                (
                    "Emissions|CO2|Energy and Industrial Processes",
                    "Emissions|CO2|AFOLU",
                ),
            ),
        )
        for v_drop, v_sub_components in conditional_keepers:
            existing_vars = in_emissions.pix.unique("variable")
            if v_drop in existing_vars:
                if all(v in existing_vars for v in v_sub_components):
                    in_emissions = in_emissions.loc[~pix.isin(variable=v_drop)]

        drop_if_identical = (
            # (
            #     Variable to potentially remove,
            #     remove if identical to this variable
            # )
            ("Emissions|CO2", "Emissions|CO2|Energy and Industrial Processes"),
            ("Emissions|CO2", "Emissions|CO2|AFOLU"),
        )
        for v_drop, v_check in drop_if_identical:
            existing_vars = in_emissions.pix.unique("variable")
            if all(v in existing_vars for v in (v_drop, v_check)):
                # Should really use isclose here, but we didn't in AR6
                # so we get some funny reporting for weird scenarios
                # e.g. C3IAM 2.0 2C-hybrid
                if (
                    (
                        in_emissions.loc[pix.isin(variable=v_drop)].reset_index(
                            "variable", drop=True
                        )
                        == in_emissions.loc[pix.isin(variable=v_check)].reset_index(
                            "variable", drop=True
                        )
                    )
                    .all()
                    .all()
                ):
                    in_emissions = in_emissions.loc[~pix.isin(variable=v_drop)]

        # Negative value handling
        co2_locator = pix.ismatch(variable="**CO2**")
        in_emissions.loc[~co2_locator] = in_emissions.loc[~co2_locator].where(
            # Where these conditions are true, keep the original data.
            (in_emissions.loc[~co2_locator] > 0)
            | (in_emissions.loc[~co2_locator] < self.negative_value_not_small_threshold)
            | in_emissions.loc[~co2_locator].isnull(),
            # Otherwise, set to zero
            other=0.0,
        )

        res = in_emissions.loc[pix.isin(variable=self.emissions_out)]

        # Strip out any units that won't play nice with pint
        res = strip_pint_incompatible_characters_from_units(
            res, units_index_level="unit"
        )

        return res

    @classmethod
    def from_ar6_like_config(cls, run_checks: bool) -> AR6PreProcessor:
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
            Initialised Pre-processor
        """
        ar6_emissions_for_harmonisation = (
            "Emissions|BC",
            "Emissions|PFC|C2F6",
            "Emissions|PFC|C6F14",
            "Emissions|PFC|CF4",
            "Emissions|CO",
            "Emissions|CO2",
            "Emissions|CO2|AFOLU",
            "Emissions|CO2|Energy and Industrial Processes",
            "Emissions|CH4",
            # "Emissions|F-Gases",  # Not used
            # "Emissions|HFC",  # Not used
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
            # "Emissions|PFC",  # Not used
            "Emissions|SF6",
            "Emissions|Sulfur",
            "Emissions|VOC",
        )

        return cls(
            emissions_out=ar6_emissions_for_harmonisation,
            negative_value_not_small_threshold=-0.1,
            run_checks=run_checks,
        )
