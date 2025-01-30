"""
Pre-processing before our more standard pre-processing.

We might blend this in in future, let's see.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pandas_indexing as pix
from gcages.pre_processing import reclassify_variables


def pre_pre_process(
    indf: pd.DataFrame, co2_ei_check_rtol: float = 1e-3, raise_on_co2_ei_difference: bool = False, silent: bool = False
) -> pd.DataFrame:
    """
    Pre-pre-process

    We might put this in a pre-processor in future,
    but it feels quite custom.

    Parameters
    ----------
    indf
        Input data to pre-pre-process

    co2_ei_check_rtol
        Relative tolerance to apply when checking
        if CO2 energy and industrial is the sum of energy
        and industrial.

    raise_on_co2_ei_difference
        Should an error be raised if CO2 energy and industrial
        is not the sum of reported energy and industrial?

        If `False`, a warning is raised instead.

    silent
        If `True`, not output is shown.

    Returns
    -------
    :
        Pre-pre-processed data
    """
    pre_pre_processed_l = []
    # all_nan_or_zero_l = []
    for (model, scenario), msdf in indf.groupby(["model", "scenario"]):
        if "Emissions|CO2|Energy and Industrial Processes" not in msdf.pix.unique("variable"):
            if not silent:
                warnings.warn(
                    f"Excluding {model=} {scenario=} because there are no CO2 fossil emissions."
                    # f"\nAvailable variables: {sorted(msdf.pix.unique('variable').tolist())}.\n"
                )
            continue

        co2_energy_and_industrial = msdf.loc[pix.isin(variable=["Emissions|CO2|Energy and Industrial Processes"])]

        all_nan_or_zero = (msdf.isnull() | (msdf == 0.0)).all(axis=1)
        # if all_nan_or_zero.any():
        #     all_nan_or_zero_l.append(msdf[all_nan_or_zero])

        msdf_use = msdf[~all_nan_or_zero]

        if all(
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
                if not silent:
                    msg = f"CO2 energy and industrial is not the sum of energy and industrial.\n{model=} {scenario=}"
                    if raise_on_co2_ei_difference:
                        raise AssertionError(msg) from exc
                    else:
                        warnings.warn(f"{msg}. {exc}")

        co2_fossil_component_vars = msdf_use.loc[
            pix.ismatch(variable="*|CO2|*")
            & ~pix.ismatch(variable=["*|CO2|AFOLU", "*|CO2|Energy and Industrial Processes"])
        ].pix.unique("variable")

        variables_to_reclassify = set(co2_fossil_component_vars).difference(
            {"Emissions|CO2|Energy", "Emissions|CO2|Industrial Processes"}
        )
        if variables_to_reclassify:
            # print(f"{model=} {scenario=} {variables_to_reclassify=}")
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
                if not silent:
                    msg = (
                        "CO2 energy and industrial is misreported. "
                        f"It is already the sum of everything ({sorted(co2_fossil_component_vars)}), "
                        "not just energy and industrial.\n"
                        f"{model=} {scenario=}"
                    )
                    warnings.warn(f"{msg}")

            except AssertionError:
                msdf_use = reclassify_variables(
                    msdf_use,
                    reclassifications={
                        "Emissions|CO2|Energy and Industrial Processes": tuple(variables_to_reclassify)
                        # (
                        #     "Emissions|CO2|Other",
                        #     "Emissions|CO2|Other Capture and Removal",
                        #     "Emissions|CO2|Product Use",
                        #     "Emissions|CO2|Waste",
                        # )
                    },
                )

        pre_pre_processed_l.append(msdf_use)

    # all_nan_or_zero = pix.concat(all_nan_or_zero_l)
    pre_pre_processed = pix.concat(pre_pre_processed_l)

    return pre_pre_processed
