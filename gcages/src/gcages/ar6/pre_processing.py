"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping
from functools import partial
from typing import Callable, Concatenate, ParamSpec

import pandas as pd
import pandas_indexing as pix  # type: ignore
from attrs import define

from gcages.parallelisation import (
    assert_only_working_on_variable_unit_variations,
    run_parallel,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

P = ParamSpec("P")


def add_conditional_sums(
    indf: pd.DataFrame,
    conditional_sums: tuple[tuple[str, tuple[str, ...]], ...],
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
        If the variable is added, all the sub-components are dropped.
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

    for v_target, v_sources in conditional_sums:
        existing_vars: pd.MultiIndex = out.pix.unique("variable")  # type: ignore
        if v_target not in existing_vars:
            if all(v in existing_vars for v in v_sources):
                locator_sources = pix.isin(variable=v_sources)
                to_add = out.loc[locator_sources]

                tmp = to_add.groupby(list(set(to_add.index.names) - {"variable"})).sum(
                    min_count=len(v_sources)
                )
                tmp = tmp.pix.assign(variable=v_target)
                out = pix.concat([out.loc[~locator_sources], tmp], axis="index")

    return out


def reclassify_variables(
    indf: pd.DataFrame,
    reclassifications: Mapping[str, tuple[str, ...]],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Reclassify variables

    Parameters
    ----------
    indf
        Data to add sums to

    reclassifications
        Definition of the reclassifications.

        For each variable (key) in `reclassifications`, the variables in its value
        will be reclassified as part of its total.

        For example, if `reclassifications` is

        ```python
        {"var_a": ("var_b", "var_c")}
        ```

        then if "var_b" or "var_c" (or both) is in `indf`,
        they will be removed and their contents will be added to the total of `var_a`.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf`, reclassified as needed.
    """
    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_target, v_sources in reclassifications.items():
        locator_sources = pix.isin(variable=v_sources)
        to_add = out.loc[locator_sources]
        if not to_add.empty:
            out.loc[pix.isin(variable=v_target)] += to_add.sum()
            out = out.loc[~locator_sources]

    return out


def condtionally_remove_variables(
    indf: pd.DataFrame,
    conditional_removals: tuple[tuple[str, tuple[str, ...]], ...],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Conditionally remove variables

    Parameters
    ----------
    indf
        Data to add sums to

    conditional_removals
        Definition of the conditional removals.

        For each tuple, the first element defines the variable that can be removed.
        This variable will be removed if all variables in the tuple's second element
        are present in `indf`.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf` with variables removed according to this function's logic.
    """
    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_drop, v_sub_components in conditional_removals:
        existing_vars: pd.MultiIndex = out.pix.unique("variable")  # type: ignore
        if v_drop in existing_vars and all(
            v in existing_vars for v in v_sub_components
        ):
            out = out.loc[~pix.isin(variable=v_drop)]

    return out


def drop_variables_if_identical(
    indf: pd.DataFrame,
    drop_if_identical: tuple[tuple[str, str], ...],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Drop variables if they are identical to another variable

    Parameters
    ----------
    indf
        Data to add sums to

    drop_if_identical
        Definition of the variables that can be dropped.

        For each tuple, the first element defines the variable that can be removed
        and the second element defines the variable to compare it to.
        If the variable to drop has the same values as the variable to compare to,
        it is dropped.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf` with variables removed according to this function's logic.
    """
    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_drop, v_check in drop_if_identical:
        existing_vars: pd.MultiIndex = out.pix.unique("variable")  # type: ignore
        if all(v in existing_vars for v in (v_drop, v_check)):
            # Should really use isclose here, but we didn't in AR6
            # so we get some funny reporting for weird scenarios
            # e.g. C3IAM 2.0 2C-hybrid
            if (
                (
                    out.loc[pix.isin(variable=v_drop)]
                    .reset_index("variable", drop=True)
                    .dropna(axis="columns")
                    == out.loc[pix.isin(variable=v_check)]
                    .reset_index("variable", drop=True)
                    .dropna(axis="columns")
                )
                .all()
                .all()
            ):
                out = out.loc[~pix.isin(variable=v_drop)]

    return out


def run_parallel_pre_processing(
    indf: pd.DataFrame,
    func_to_call: Callable[Concatenate[pd.DataFrame, P], pd.DataFrame],
    groups: tuple[str, ...] = ("model", "scenario"),
    n_processes: int = multiprocessing.cpu_count(),
    *args: P.args,
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
        Passed to `run_parallel`

    Returns
    -------
    :
        Result of calling `func_to_call` on each group in `indf`.
    """
    res = pd.concat(
        run_parallel(  # type: ignore
            func_to_call=func_to_call,
            iterable_input=(gdf for _, gdf in indf.groupby(list(groups))),
            input_desc=f"{', '.join(groups)} combinations",
            n_processes=n_processes,
            **kwargs,  # type: ignore
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

    conditional_sums: tuple[tuple[str, tuple[str, ...]], ...] | None = None
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

    reclassifications: Mapping[str, tuple[str, ...]] | None = None
    """
    Specification for variables that should be reclassified as being another variable

    Form:

    ```python
    {
        variable_to_add_to: (variable_to_rename_1, variable_to_rename_2),
        ...
    }
    ```
    """

    conditional_removals: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    """
    Specification for variables that can be removed if other variables are present

    Form:

    ```python
    (
        (variable_that_can_be_removed, (component_1, component_2)),
        ...
    )
    ```

    The variable that can be removed is only removed
    if all the variables it depends on are present.
    """

    drop_if_identical: tuple[tuple[str, str], ...] | None = None
    """
    Variables that can be dropped if they are idential to another variable

    Form:

    ```python
    (
        (variable_that_can_be_removed, variable_to_compare_to),
        ...
    )
    ```

    The variable that can be removed is only removed
    if its values are identical to the variable it is compared to.
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

        # Remove any rows with only zero
        in_emissions = in_emissions[
            ~(((in_emissions == 0.0) | in_emissions.isnull()).all(axis="columns"))
        ]

        # Remove any rows that have NaN in required years
        required_years = list(range(2020, 2100 + 1, 10))
        in_emissions = in_emissions[
            ~in_emissions[required_years].isnull().any(axis="columns")
        ]

        rp = partial(run_parallel_pre_processing, n_processes=self.n_processes)
        if self.conditional_sums is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=add_conditional_sums,
                conditional_sums=self.conditional_sums,
            )

        if self.reclassifications is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=reclassify_variables,
                reclassifications=self.reclassifications,
            )

        if self.conditional_removals is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=condtionally_remove_variables,
                conditional_removals=self.conditional_removals,
            )

        if self.drop_if_identical is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=drop_variables_if_identical,
                drop_if_identical=self.drop_if_identical,
            )

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

        res: pd.DataFrame = in_emissions.loc[pix.isin(variable=self.emissions_out)]

        # Strip out any units that won't play nice with pint
        res = strip_pint_incompatible_characters_from_units(
            res, units_index_level="unit"
        )

        return res

    @classmethod
    def from_ar6_like_config(
        cls, run_checks: bool, n_processes: int = multiprocessing.cpu_count()
    ) -> AR6PreProcessor:
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
        reclassifications = {
            "Emissions|CO2|Energy and Industrial Processes": (
                "Emissions|CO2|Other",
                "Emissions|CO2|Waste",
            )
        }
        conditional_removals = (
            (
                "Emissions|CO2",
                (
                    "Emissions|CO2|Energy and Industrial Processes",
                    "Emissions|CO2|AFOLU",
                ),
            ),
        )
        drop_if_identical = (
            ("Emissions|CO2", "Emissions|CO2|Energy and Industrial Processes"),
            ("Emissions|CO2", "Emissions|CO2|AFOLU"),
        )

        return cls(
            emissions_out=ar6_emissions_for_harmonisation,
            negative_value_not_small_threshold=-0.1,
            conditional_sums=conditional_sums,
            reclassifications=reclassifications,
            conditional_removals=conditional_removals,
            drop_if_identical=drop_if_identical,
            run_checks=run_checks,
            n_processes=n_processes,
        )
