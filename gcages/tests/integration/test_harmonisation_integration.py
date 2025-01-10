"""
Integration tests of harmonisation
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.ar6 import AR6Harmoniser

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
    res = all_emissions.loc[pix.ismatch(variable="Emissions**")]

    return res


@functools.cache
def get_ar6_harmonised_emissions(model: str, scenario: str) -> pd.DataFrame:
    all_emissions = get_ar6_all_emissions(model, scenario)
    res = all_emissions.loc[pix.ismatch(variable="**Harmonized**")]

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

    exp = get_ar6_harmonised_emissions(model, scenario)

    pd.testing.assert_frame_equal(res, exp)


def test_harmonisation_ips_simultaneously():
    # Test that we can harmonise all the IPs at once.
    # Tests parallel harmonisation.
    # Without harmonising all scenarios at once (overkill).
    raise NotImplementedError
