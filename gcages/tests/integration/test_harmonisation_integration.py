"""
Integration tests of harmonisation
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.ar6 import AR6Harmoniser, AR6PreProcessor

TEST_DATA_DIR = Path(__file__).parents[1] / "test-data"


def create_harmonisation_test_cases():
    model_scenarios = pd.read_csv(
        TEST_DATA_DIR / "ar6_scenarios_raw_model_scenario_combinations.csv"
    )

    return tuple(
        pytest.param(model, scenario, id=f"{model}__{scenario}")
        for (model, scenario), _ in model_scenarios.groupby(["Model", "Scenario"])
    )


harmonisation_cases = pytest.mark.parametrize(
    "model, scenario", create_harmonisation_test_cases()
)


@functools.cache
def get_ar6_all_emissions(model: str, scenario: str) -> pd.DataFrame:
    filename_emissions = f"ar6_scenarios__{model}__{scenario}__emissions.csv"
    filename_emissions = filename_emissions.replace("/", "_").replace(" ", "_")
    emissions_file = TEST_DATA_DIR / filename_emissions

    res = pd.read_csv(emissions_file)
    res.columns = res.columns.str.lower()
    res = res.set_index(["model", "scenario", "variable", "region", "unit"])
    res.columns = res.columns.astype(int)

    return res


@functools.cache
def get_ar6_raw_emissions(model: str, scenario: str) -> pd.DataFrame:
    all_emissions = get_ar6_all_emissions(model, scenario)
    res = all_emissions.loc[pix.ismatch(variable="Emissions**")].dropna(
        how="all", axis="columns"
    )

    return res


@functools.cache
def get_ar6_harmonised_emissions(model: str, scenario: str) -> pd.DataFrame:
    all_emissions = get_ar6_all_emissions(model, scenario)
    res = all_emissions.loc[pix.ismatch(variable="**Harmonized**")].dropna(
        how="all", axis="columns"
    )

    return res


@harmonisation_cases
def test_harmonisation_single_model_scenario(model, scenario):
    raw = get_ar6_raw_emissions(model, scenario)
    if raw.empty:
        msg = f"No test data for {model=} {scenario=}?"
        raise AssertionError(msg)

    pre_processor = AR6PreProcessor.from_ar6_like_config(run_checks=False)
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False)

    pre_processed = pre_processor(raw)
    res = harmoniser(pre_processed)

    exp = (
        get_ar6_harmonised_emissions(model, scenario)
        .loc[~pix.ismatch(variable="**CO2")]  # Not used downstream
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    # TODO: split this out to make it reusable
    for idx_name in res.index.names:
        idx_diffs = res.pix.unique(idx_name).symmetric_difference(
            exp.pix.unique(idx_name)
        )
        if not idx_diffs.empty:
            msg = f"Differences in the {idx_name} (res on the left): {idx_diffs=}"
            raise AssertionError(msg)

    pd.testing.assert_frame_equal(res.T, exp.T, check_exact=False, rtol=1e-8)
    # try:
    #     pd.testing.assert_frame_equal(
    #         res.T, exp.T, check_exact=False, rtol=rtol, atol=atol
    #     )
    # except AssertionError:
    #     import matplotlib
    #     import matplotlib.pyplot as plt
    #
    #     matplotlib.use("MacOSX")
    #
    #     variable_to_plot = ""
    #
    #     fig, ax = plt.subplots()
    #
    #     exp.loc[(model, scenario, variable_to_plot)].T.plot(ax=ax)
    #     res.loc[(model, scenario, variable_to_plot)].T.plot(ax=ax)
    #
    #     plt.show()
    #
    #     raise


def test_harmonisation_ips_simultaneously():
    # Test that we can harmonise all the IPs at once.
    # Tests parallel harmonisation.
    # Without harmonising all scenarios at once (overkill).
    raise NotImplementedError
