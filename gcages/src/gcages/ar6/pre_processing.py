"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import multiprocessing
from typing import Callable, Concatenate, ParamSpec

import pandas as pd
import pandas_indexing as pix
from attrs import define

from gcages.parallelisation import run_parallel
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

P = ParamSpec("P")


def assert_only_working_on_variable_unit_variations(indf: pd.DataFrame) -> None:
    """
    Assert that we're only working on variations in variable and unit

    In other words, we don't have variations in scenarios, models etc.

    Parameters
    ----------
    indf
        Data to verify

    Raises
    ------
    AssertionError
        There are variations in columns other than variable and unit
    """
    non_v_u_cols = list(set(indf.index.names).difference(["variable", "unit"]))
    variations_in_other_cols = indf.pix.unique(non_v_u_cols)

    if variations_in_other_cols.shape[0] > 1:
        raise AssertionError(f"{variations_in_other_cols=}")


def add_conditional_sums(
    indf: pd.DataFrame,
    conditional_sums: tuple[tuple[str, tuple[str, ...]]],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Add sums to a `pd.DataFrame` if all components are present

    Parameters
    ----------
    indf
        Data to add sums to

    conditional_sums
        Definition of the conditional sums.

        The first element of each sub-tuple is the name of the variable to add.
        The second element are its components.
        All components must be present for the variable to be added.
        If the variable is already there, the sum is not re-calculated or checked.

    copy_on_entry
        Should the data be copied on entry?


    Returns
    -------
    :
        `indf` with conditional sums added if all enabling conditions were fulfilled.
    """
    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    existing_vars = out.pix.unique("variable")
    for v_target, v_sources in conditional_sums:
        if v_target not in existing_vars:
            if all(v in existing_vars for v in v_sources):
                locator_sources = pix.isin(variable=v_sources)
                to_add = out.loc[locator_sources]

                # Need to line up with model-scenario here
                tmp = out.loc[pix.isin(variable=v_sources[0])]
                tmp[:] = to_add.sum()
                tmp = tmp.pix.assign(variable=v_target)
                out = pd.concat([out.loc[~locator_sources], tmp], axis="rows")

    return out


def run_parallel_pre_processing(
    indf: pd.DataFrame,
    func_to_call: Callable[Concatenate[pd.DataFrame, P], pd.DataFrame],
    groups: tuple[str, ...] = ("model", "scenario"),
    n_processes: int = multiprocessing.cpu_count(),
    **kwargs: P.kwargs,
) -> pd.DataFrame:
    """
    Run a pre-processing step in parallel

    Parameters
    ----------
    indf
        Input data to process

    func_to_call
        Function to apply to each group in `indf`

    groups
        Columns to use to group the data in `indf`

    n_processes
        Number of parallel processes to use

    **kwargs
        Passed to `func_to_call`

    Returns
    -------
    :
        Result of calling `func_to_call` on each group in `indf`.
    """
    res = pd.concat(
        run_parallel(
            func_to_call=func_to_call,
            **kwargs,
            iterable_input=(gdf for _, gdf in indf.groupby(list(groups))),
            input_desc=f"{', '.join(groups)} combinations",
            n_processes=n_processes,
        )
    )

    return res


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

    conditional_sums: tuple[tuple[str, tuple[str, ...]]] | None = None
    """
    Specification for variables that can be created from other variables

    Form:

    ```python
    (
        (variable_that_can_be_created, (component_1, component_2)),
        ...
    )
    ```

    The variable that can be created is only created
    if all the variables it depends on are present.
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

        # Remove any rows with only zero
        in_emissions = in_emissions[~((in_emissions == 0.0).all(axis="columns"))]

        # Remove any rows that have NaN in required years
        required_years = list(range(2020, 2100 + 1, 10))
        in_emissions = in_emissions[
            ~in_emissions[required_years].isnull().any(axis="columns")
        ]

        if self.conditional_sums is not None:
            in_emissions = run_parallel_pre_processing(
                in_emissions,
                func_to_call=add_conditional_sums,
                conditional_sums=self.conditional_sums,
            )

        # in_emissions = reclassify_variables(in_emissions, self.reclassifications)
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

        # in_emissions = condtionally_remove_variables(
        #     in_emissions, self.conditional_removals
        # )
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

        # in_emissions = drop_if_identical(in_emissions, self.drop_if_identical)
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
        conditional_sums = (
            (
                "Emissions|CO2|Energy and Industrial Processes",
                (
                    "Emissions|CO2|Industrial Processes",
                    "Emissions|CO2|Energy",
                ),
            ),
        )

        return cls(
            emissions_out=ar6_emissions_for_harmonisation,
            negative_value_not_small_threshold=-0.1,
            conditional_sums=conditional_sums,
            run_checks=run_checks,
        )
