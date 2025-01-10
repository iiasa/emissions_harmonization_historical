"""
Integration tests of harmonisation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.ar6 import AR6_RAW_VARIABLES, get_ar6_harmoniser

TEST_DATA_DIR = Path(__file__).parents[1] / "test-data"


@pytest.fixture(scope="session")
def ar6_all_emissions():
    res_l = [
        pd.read_csv(f) for f in TEST_DATA_DIR.glob("ar6_scenarios__*__emissions.csv")
    ]
    res = pd.concat(res_l)

    res.columns = res.columns.str.lower()
    res = res.set_index(["model", "scenario", "variable", "region", "unit"])
    res.columns = res.columns.astype(int)

    return res


@pytest.fixture(scope="session")
def ar6_raw(ar6_all_emissions):
    res = ar6_all_emissions.loc[pix.ismatch(variable="Emissions**")]

    return res


@pytest.fixture(scope="session")
def ar6_harmonised(ar6_all_emissions):
    res = ar6_all_emissions.loc[pix.ismatch(variable="**Harmonized**")]

    return res


@pytest.fixture(scope="session")
def ar6_infilled(ar6_all_emissions):
    res = ar6_all_emissions.loc[pix.ismatch(variable="**Infilled**")]

    return res


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


@harmonisation_cases
def test_harmonisation(
    model,
    scenario,
    ar6_raw,
    ar6_harmonised,
):
    raw = ar6_raw.loc[
        pix.isin(model=model)
        & pix.isin(scenario=scenario)
        & pix.isin(variable=AR6_RAW_VARIABLES)
    ]
    if raw.empty:
        pytest.skip(f"No raw data for {model=} and {scenario=}")
        return

    harmoniser = get_ar6_harmoniser()

    res = harmoniser.harmonise(raw)

    exp = ar6_harmonised.loc[pix.isin(model=model) & pix.isin(scenario=scenario)]

    pd.testing.assert_frame_equal(res, exp)
