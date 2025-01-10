"""
Integration tests of harmonisation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.ar6 import AR6Harmoniser, AR6PreProcessor
from gcages.testing import (
    AR6_IPS,
    assert_frame_equal,
    create_model_scenario_test_cases,
    get_all_model_scenarios,
    get_ar6_harmonised_emissions,
    get_ar6_raw_emissions,
)

TEST_DATA_DIR = Path(__file__).parents[1] / "test-data"
MODEL_SCENARIO_COMBOS_FILE = (
    TEST_DATA_DIR / "ar6_scenarios_raw_model_scenario_combinations.csv"
)


harmonisation_cases = pytest.mark.parametrize(
    "model, scenario",
    create_model_scenario_test_cases(
        get_all_model_scenarios(MODEL_SCENARIO_COMBOS_FILE)
    ),
)


@pytest.mark.slow
@harmonisation_cases
def test_harmonisation_single_model_scenario(model, scenario):
    raw = get_ar6_raw_emissions(
        model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
    )
    if raw.empty:
        msg = f"No test data for {model=} {scenario=}?"
        raise AssertionError(msg)

    pre_processor = AR6PreProcessor.from_ar6_like_config(
        run_checks=False, n_processes=1
    )
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False, n_processes=1)

    pre_processed = pre_processor(raw)
    res = harmoniser(pre_processed)

    exp = (
        get_ar6_harmonised_emissions(
            model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)


def test_harmonisation_ips_simultaneously():
    raw = pd.concat(
        [
            get_ar6_raw_emissions(
                model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
            )
            for model, scenario in AR6_IPS
        ]
    )

    pre_processor = AR6PreProcessor.from_ar6_like_config(run_checks=False)
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False)

    pre_processed = pre_processor(raw)
    res = harmoniser(pre_processed)

    exp = (
        pd.concat(
            [
                get_ar6_harmonised_emissions(
                    model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
                )
                for model, scenario in AR6_IPS
            ]
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)


@pytest.mark.slow
def test_harmonisation_all_simultaneously():
    model_scenarios = get_all_model_scenarios(MODEL_SCENARIO_COMBOS_FILE).values

    raw = pd.concat(
        [
            get_ar6_raw_emissions(
                model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
            )
            for model, scenario in model_scenarios
        ]
    )

    pre_processor = AR6PreProcessor.from_ar6_like_config(run_checks=False)
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False)

    pre_processed = pre_processor(raw)
    res = harmoniser(pre_processed)

    exp = (
        pd.concat(
            [
                get_ar6_harmonised_emissions(
                    model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
                )
                for model, scenario in model_scenarios
            ]
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)
