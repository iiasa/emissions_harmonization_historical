"""
Harmonisation configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import aneris.utils
import openscm_units
import pandas as pd
import pandas_indexing as pix
import tqdm.auto
from aneris.methods import default_methods
from attrs import define
from gcages.aneris_helpers import _convert_units_to_match, harmonise_all
from gcages.harmonisation.common import align_history_to_data_at_time
from gcages.testing import compare_close
from pandas_openscm.indexing import multi_index_lookup

if TYPE_CHECKING:
    from gcages.typing import PINT_SCALAR

HARMONISATION_YEAR = 2023


def get_aneris_defaults(  # noqa: PLR0913
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
    harmonisation_year: int,
    region_level: str = "region",
    unit_level: str = "unit",
    scenario_grouper: tuple[str, ...] = ("model", "scenario"),
) -> pd.DataFrame:
    """
    Get aneris' defaults

    Parameters
    ----------
    scenarios
        Scenarios for which to get the defaults

    history
        History to use

    harmonisation_year
        Harmonisation year

    region_level
        Level which holds region metadata in the data indices

    unit_level
        Level which holds unit metadata in the data indices

    scenario_grouper
        Levels to use to group `scenarios` into scenarios
        which would be run through a simple climate model

    Returns
    -------
    :
        Default harmonisation methods used by aneris for each timeseries in `scenarios`
    """
    scenario_grouper_l = list(scenario_grouper)

    default_l = []
    for (model, scenario), msdf in tqdm.auto.tqdm(scenarios.groupby(scenario_grouper_l)):
        msdf_relevant_aneris = msdf.reset_index(scenario_grouper_l, drop=True)

        history_model_relevant_aneris = _convert_units_to_match(
            start=(
                history.loc[pix.isin(region=msdf_relevant_aneris.index.get_level_values(region_level).unique())]
                .reset_index(scenario_grouper_l, drop=True)
                .reorder_levels(msdf_relevant_aneris.index.names)
            ),
            match=msdf_relevant_aneris,
        )
        # Supress warnings about divide by zero
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            msdf_default_overrides = default_methods(
                hist=history_model_relevant_aneris, model=msdf_relevant_aneris, base_year=harmonisation_year
            )

        default_l.append(msdf_default_overrides[0].pix.assign(model=model, scenario=scenario))

    default = pix.concat(default_l).reset_index(unit_level, drop=True)

    return default


def avoid_offset_with_negative_results(  # noqa: PLR0913
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
    harmonisation_year: int,
    overrides_in: pd.Series,
    unit_level: str = "unit",
    scenario_grouper: tuple[str, ...] = ("model", "scenario"),
    do_not_update: tuple[str, ...] = ("**CO2**",),
    offset_methods_to_update: tuple[str, ...] = ("reduce_offset_2150_cov",),
    offset_methods_replacement: str = "constant_ratio",
) -> pd.Series:
    """
    Update overrides to avoid using offset methods where negative values will be the result

    The default harmonisation gives unwanted negative values
    where the offset between the model and history
    is larger than the historical emissions themselves
    and the default choice is some sort of offset.
    Hence we use this function to override these cases.

    Parameters
    ----------
    scenarios
        Scenarios to be harmonised

    history
        History to use for harmonisation

    harmonisation_year
        Harmonisation year

    overrides_in
        Starting harmonisation method overrides

    unit_level
        Level which holds unit metadata in the data indices

    scenario_grouper
        Levels to use to group `scenarios` into scenarios
        which would be run through a simple climate model

    do_not_update
        Variable selectors which will not be updated,
        even if the result of using `overrides_in` is negative values.

    offset_methods_to_update
        Offset methods which should be updated if the result will be negative values

    offset_methods_replacement
        Method to use if the result of using aneris' default method will be negative values

    Returns
    -------
    :
        `overrides_in`, updated to avoid giving negative values
    """
    scenario_grouper_l = list(scenario_grouper)

    offsets = scenarios[harmonisation_year].subtract(
        history[harmonisation_year].reset_index(scenario_grouper_l, drop=True)
    )

    scenarios_minus_offset = scenarios.subtract(offsets, axis="rows")
    negative_with_offset = scenarios_minus_offset[scenarios_minus_offset.min(axis="columns") < 0.0]
    # Drop out anything we don't want to update
    negative_with_offset = negative_with_offset.loc[~pix.ismatch(variable=list(do_not_update))].index.droplevel(
        unit_level
    )

    overrides = overrides_in.reorder_levels(negative_with_offset.names).copy()

    overrides_negative_with_offset = overrides.loc[negative_with_offset]
    reduce_offset_despite_negative_with_offset = overrides_negative_with_offset[
        overrides_negative_with_offset.isin(offset_methods_to_update)
    ]

    # Override with ratio instead to avoid unintended negatives
    overrides.loc[reduce_offset_despite_negative_with_offset.index] = offset_methods_replacement

    return overrides


@define
class HarmonisationResult:
    """
    Results of harmonisation

    Includes the overrides used too, so we can iterate with IAM teams
    """

    timeseries: pd.DataFrame
    """The harmonised timeseries"""

    overrides: pd.Series[str]
    """The overrides used during the harmonisation"""


def harmonise(  # noqa: PLR0913
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
    harmonisation_year: int,
    user_overrides: pd.Series[str] | None,
    do_not_update_negative_after_offset: tuple[str, ...] = ("**CO2**",),
    offset_methods_to_update: tuple[str, ...] = ("reduce_offset_2150_cov",),
    offset_methods_replacement: str = "constant_ratio",
    region_level: str = "region",
    unit_level: str = "unit",
    scenario_grouper: tuple[str, ...] = ("model", "scenario"),
    silence_aneris: bool = True,
) -> HarmonisationResult:
    """
    Harmonise scenarios

    Parameters
    ----------
    scenarios
        Scenarios to be harmonised

    history
        History to use for harmonisation

    harmonisation_year
        Harmonisation year

    user_overrides
        User-supplied overrides of harmonisation methods

    do_not_update_negative_after_offset
        Variable selectors which will not be updated,
        even if the result of using aneris' default harmonisation
        methods is negative values.

    offset_methods_to_update
        Default aneris offset methods
        which should be updated if the result will be negative values

    offset_methods_replacement
        Method to use if the result of using aneris' default method will be negative values

    region_level
        Level which holds region metadata in the data indices

    unit_level
        Level which holds unit metadata in the data indices

    scenario_grouper
        Levels to use to group `scenarios` into scenarios
        which would be run through a simple climate model

    silence_aneris
        Silence aneris' logging messages

    Returns
    -------
    :
        Harmonisation results
    """
    scenario_grouper_l = list(scenario_grouper)

    # Only keep history relevant to the scenarios we're harmonising
    history_for_harmonisation = multi_index_lookup(history, scenarios.index.droplevel(scenario_grouper_l))

    aneris_defaults = get_aneris_defaults(
        scenarios=scenarios,
        history=history_for_harmonisation,
        harmonisation_year=harmonisation_year,
        region_level=region_level,
        unit_level=unit_level,
        scenario_grouper=scenario_grouper_l,
    )
    # The default harmonisation gives unwanted negative values
    # where the difference offset between the model and history
    # is larger than the historical emissions themselves
    # and the default choice is reduce offset.
    # Hence we override these here
    # for all species except CO<sub>2</sub>.
    overrides_auto_inferred = avoid_offset_with_negative_results(
        scenarios=scenarios,
        history=history_for_harmonisation,
        harmonisation_year=harmonisation_year,
        overrides_in=aneris_defaults,
        unit_level=unit_level,
        scenario_grouper=scenario_grouper_l,
        do_not_update=("**CO2**",),
        offset_methods_to_update=("reduce_offset_2150_cov",),
        offset_methods_replacement="constant_ratio",
    )

    if user_overrides is None:
        overrides_to_use = overrides_auto_inferred

    else:
        # Could be more flexible and provide helpers for getting into the right format.
        # If we do this, put them outside this function.
        overrides_auto_inferred = overrides_auto_inferred.reorder_levels(user_overrides.index.names)
        wont_be_used_locator = ~user_overrides.index.isin(overrides_auto_inferred.index)
        if wont_be_used_locator.any():
            msg = f"The following overrides will not be used: {user_overrides.loc[wont_be_used_locator]}"
            raise AssertionError(msg)

        overrides_to_use = pix.concat(
            [overrides_auto_inferred.loc[~overrides_auto_inferred.index.isin(user_overrides.index)], user_overrides]
        )

    if silence_aneris:
        aneris_logger = aneris.utils.logger()
        aneris_logger_level_in = aneris_logger.level
        aneris_logger.setLevel(logging.WARNING)

    # Supress warnings about divide by zero
    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        harmonised = harmonise_all(
            scenarios=scenarios,
            history=history_for_harmonisation,
            year=harmonisation_year,
            overrides=overrides_to_use,
        )

    if silence_aneris:
        # Reset logger
        aneris_logger.setLevel(aneris_logger_level_in)

    return HarmonisationResult(timeseries=harmonised, overrides=overrides_to_use)


def assert_harmonised(
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
    species_tolerances: dict[str, dict[str, float | PINT_SCALAR]] | None = None,
) -> None:
    """
    Assert that scenarios are harmonised to a given history

    TODO: move to gcages/update the existing gcages function

    Parameters
    ----------
    scenarios
        Scenarios

    history
        History

    species_tolerances
        Tolerance to apply while checking harmonisation of different species
    """
    # Protect this with try except in gcagees
    Q = openscm_units.unit_registry.Quantity

    if species_tolerances is None:
        species_tolerances = {
            "BC": dict(rtol=1e-3, atol=Q(1e-3, "Mt BC/yr")),
            "CH4": dict(rtol=1e-3, atol=Q(1e-2, "Mt CH4/yr")),
            "CO": dict(rtol=1e-3, atol=Q(1e-1, "Mt CO/yr")),
            "CO2": dict(rtol=1e-3, atol=Q(1e-3, "Gt CO2/yr")),
            "NH3": dict(rtol=1e-3, atol=Q(1e-2, "Mt NH3/yr")),
            "NOx": dict(rtol=1e-3, atol=Q(1e-2, "Mt NO2/yr")),
            "OC": dict(rtol=1e-3, atol=Q(1e-3, "Mt OC/yr")),
            "Sulfur": dict(rtol=1e-3, atol=Q(1e-2, "Mt SO2/yr")),
            "VOC": dict(rtol=1e-3, atol=Q(1e-2, "Mt VOC/yr")),
            "N2O": dict(rtol=1e-3, atol=Q(1e-1, "kt N2O/yr")),
        }

    scenarios_a, history_a = align_history_to_data_at_time(
        scenarios,
        history=history.loc[pix.isin(variable=scenarios.pix.unique("variable"))].reset_index(
            ["model", "scenario"], drop=True
        ),
        time=HARMONISATION_YEAR,
    )
    for variable, scen_a_vdf in scenarios_a.groupby("variable"):
        history_a_vdf = history_a.loc[pix.isin(variable=variable)]
        species = variable.split("|")[1]
        if species in species_tolerances:
            unit_l = scen_a_vdf.pix.unique("unit").tolist()
            if len(unit_l) != 1:
                raise AssertionError(unit_l)
            unit = unit_l[0]

            rtol = species_tolerances[species]["rtol"]
            atol = species_tolerances[species]["atol"].to(unit).m

        else:
            rtol = 1e-4
            atol = 1e-6

        compare_close(
            scen_a_vdf.unstack("region"),
            history_a_vdf.unstack("region"),
            left_name="scenario",
            right_name="history",
            rtol=rtol,
            atol=atol,
        )
