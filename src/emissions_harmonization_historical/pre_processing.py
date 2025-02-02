"""
Pre-processing before our more standard pre-processing.

We might blend this in in future, let's see.
"""

from __future__ import annotations

import multiprocessing
import warnings
from typing import Any

import pandas as pd
import pandas_indexing as pix
from attrs import define
from gcages.pre_processing import reclassify_variables
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


def reclassify_co2_if_needed(
    indf: pd.DataFrame, co2_ei_check_rtol: float = 1e-3, raise_on_co2_ei_difference: bool = False, silent: bool = False
) -> pd.DataFrame:
    """
    Re-classify CO2 variables if needed

    This really shouldn't be needed here and should be checked as part of scenario submission.
    The tools provided by nomenclature are also ideal for such checking.
    Nonetheless, for now, we deal with this.

    Parameters
    ----------
    indf
        Input data to process

    co2_ei_check_rtol
        Relative tolerance to apply when checking
        if CO2 energy and industrial is the sum of energy and industrial.

    raise_on_co2_ei_difference
        Should an error be raised if CO2 energy and industrial
        is not the sum of reported energy and industrial?

        If `False`, a warning is raised instead.

    silent
        If `True`, no output is shown.

    Returns
    -------
    :
        Re-classified data
    """
    pre_pre_processed_l = []
    for (model, scenario), msdf in indf.groupby(["model", "scenario"]):
        if "Emissions|CO2|Energy and Industrial Processes" not in msdf.pix.unique("variable"):
            if not silent:
                warnings.warn(
                    f"Excluding {model=} {scenario=} because there are no CO2 fossil emissions."
                    f"\nAvailable variables: {sorted(msdf.pix.unique('variable').tolist())}.\n"
                )
            continue

        co2_energy_and_industrial = msdf.loc[pix.isin(variable=["Emissions|CO2|Energy and Industrial Processes"])]

        all_nan_or_zero = (msdf.isnull() | (msdf == 0.0)).all(axis=1)

        msdf_use = msdf[~all_nan_or_zero]

        # If this doesn't hold, the check doesn't do anything so don't bother running
        run_check = raise_on_co2_ei_difference or not silent
        if run_check and all(
            v in msdf_use.pix.unique("variable") for v in ["Emissions|CO2|Energy", "Emissions|CO2|Industrial Processes"]
        ):
            co2_fossil_sum_of_energy_and_industrial_checker = (
                msdf_use.loc[pix.isin(variable=["Emissions|CO2|Energy", "Emissions|CO2|Industrial Processes"])]
                .groupby(msdf_use.index.names.difference(["variable"]))
                .sum(min_count=2)
                .pix.assign(variable="Emissions|CO2|Energy and Industrial Processes")
                .reorder_levels(msdf_use.index.names)
            )

            try:
                pd.testing.assert_frame_equal(
                    co2_fossil_sum_of_energy_and_industrial_checker,
                    co2_energy_and_industrial,
                    rtol=co2_ei_check_rtol,
                )
            except AssertionError as exc:
                msg = f"CO2 energy and industrial is not the sum of energy and industrial.\n{model=} {scenario=}"
                if raise_on_co2_ei_difference:
                    raise AssertionError(msg) from exc
                else:
                    if silent:
                        msg = "Should never get here"
                        raise AssertionError(msg)
                    warnings.warn(f"{msg}. {exc}")

        co2_fossil_component_vars = msdf_use.loc[
            pix.ismatch(variable="*|CO2|*")
            & ~pix.ismatch(variable=["*|CO2|AFOLU", "*|CO2|Energy and Industrial Processes"])
        ].pix.unique("variable")

        variables_to_reclassify = set(co2_fossil_component_vars).difference(
            {"Emissions|CO2|Energy", "Emissions|CO2|Industrial Processes"}
        )
        if variables_to_reclassify:
            co2_fossil_checker = (
                msdf_use.loc[pix.isin(variable=co2_fossil_component_vars)]
                .groupby(msdf_use.index.names.difference(["variable"]))
                .sum(min_count=len(co2_fossil_component_vars))
                .pix.assign(variable="Emissions|CO2|Energy and Industrial Processes")
                .reorder_levels(msdf_use.index.names)
            )

            try:
                pd.testing.assert_frame_equal(
                    co2_fossil_checker,
                    co2_energy_and_industrial,
                    rtol=co2_ei_check_rtol,
                )
                # Sum is already reported as intended,
                # don't reclassify.
                if not silent:
                    msg = (
                        "CO2 energy and industrial is misreported. "
                        f"It is already the sum of everything ({sorted(co2_fossil_component_vars)}), "
                        "not just energy and industrial.\n"
                        f"{model=} {scenario=}"
                    )
                    warnings.warn(f"{msg}")

            except AssertionError:
                # The sum doesn't include the variables to reclassify,
                # hence do the reclassification.
                msdf_use = reclassify_variables(
                    msdf_use,
                    reclassifications={"Emissions|CO2|Energy and Industrial Processes": tuple(variables_to_reclassify)},
                )

        pre_pre_processed_l.append(msdf_use)

    pre_pre_processed = pix.concat(pre_pre_processed_l)

    return pre_pre_processed


@define
class AR7FTPreProcessor:
    """
    AR7 fast-track pre-processor
    """

    emissions_out: tuple[str, ...]
    """
    Names of emissions that can be included in the result of pre-processing

    Not all these emissions need to be there,
    but any names which are not in this list will be flagged
    (if `self.run_checks` is `True`).
    """

    run_co2_reclassification: bool | dict[str, Any] = False
    """
    Should we run `reclassify_co2_if_needed`?

    If this value is a dictionary, we pass it as keyword-arguments
    to `reclassify_co2_if_needed`.
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
        if self.run_co2_reclassification:
            if isinstance(self.run_co2_reclassification, dict):
                reclassified = reclassify_co2_if_needed(
                    in_emissions,
                    **self.run_co2_reclassification,
                )
            else:
                reclassified = reclassify_co2_if_needed(in_emissions)
        else:
            reclassified = in_emissions

        res: pd.DataFrame = reclassified.loc[pix.isin(variable=self.emissions_out)]

        res = strip_pint_incompatible_characters_from_units(res, units_index_level="unit")

        return res

    @classmethod
    def from_default_config(cls) -> AR7FTPreProcessor:
        """
        Initialise from default, hard-coded configuration
        """
        all_known_variables = (
            "Emissions|BC",
            "Emissions|C2F6",
            "Emissions|C3F8",
            "Emissions|C4F10",
            "Emissions|C5F12",
            "Emissions|C6F14",
            "Emissions|C7F16",
            "Emissions|C8F18",
            "Emissions|CF4",
            "Emissions|CH4",
            "Emissions|CO",
            "Emissions|CO2|AFOLU",
            "Emissions|CO2|Energy and Industrial Processes",
            "Emissions|HFC|HFC125",
            "Emissions|HFC|HFC134a",
            "Emissions|HFC|HFC143a",
            "Emissions|HFC|HFC152a",
            "Emissions|HFC|HFC227ea",
            "Emissions|HFC|HFC23",
            "Emissions|HFC|HFC236fa",
            "Emissions|HFC|HFC245fa",
            "Emissions|HFC|HFC32",
            "Emissions|HFC|HFC365mfc",
            "Emissions|HFC|HFC43-10",
            "Emissions|Montreal Gases|CCl4",
            "Emissions|Montreal Gases|CFC|CFC11",
            "Emissions|Montreal Gases|CFC|CFC113",
            "Emissions|Montreal Gases|CFC|CFC114",
            "Emissions|Montreal Gases|CFC|CFC115",
            "Emissions|Montreal Gases|CFC|CFC12",
            "Emissions|Montreal Gases|CH2Cl2",
            "Emissions|Montreal Gases|CH3Br",
            "Emissions|Montreal Gases|CH3CCl3",
            "Emissions|Montreal Gases|CH3Cl",
            "Emissions|Montreal Gases|CHCl3",
            "Emissions|Montreal Gases|HCFC141b",
            "Emissions|Montreal Gases|HCFC142b",
            "Emissions|Montreal Gases|HCFC22",
            "Emissions|Montreal Gases|Halon1202",
            "Emissions|Montreal Gases|Halon1211",
            "Emissions|Montreal Gases|Halon1301",
            "Emissions|Montreal Gases|Halon2402",
            "Emissions|N2O",
            "Emissions|NF3",
            "Emissions|NH3",
            "Emissions|NOx",
            "Emissions|OC",
            "Emissions|SF6",
            "Emissions|SO2F2",
            "Emissions|Sulfur",
            "Emissions|VOC",
            "Emissions|cC4F8",
        )

        return AR7FTPreProcessor(
            emissions_out=all_known_variables,
            run_co2_reclassification=dict(
                co2_ei_check_rtol=1e-3,
                raise_on_co2_ei_difference=False,
                silent=True,
            ),
            # TODO: implement and turn on
            run_checks=False,
            n_processes=1,
        )
