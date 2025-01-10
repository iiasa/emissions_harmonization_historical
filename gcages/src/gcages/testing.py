"""
Support for our testing

You probably will never use this as a user of the package.

It's not in the tests folder,
because we don't put `__init__.py` files in our `tests`
because we've been bitten
by the subtleties of that too many times
(further reading here:
https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#choosing-an-import-mode).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pandas_indexing as pix  # type: ignore

if TYPE_CHECKING:
    import _pytest

AR6_IPS: tuple[tuple[str, str], ...] = (
    ("AIM/CGE 2.2", "EN_NPi2020_900f"),
    ("COFFEE 1.1", "EN_NPi2020_400f_lowBECCS"),
    ("GCAM 5.3", "NGFS2_Current Policies"),
    ("IMAGE 3.0", "EN_INDCi2030_3000f"),
    ("MESSAGEix-GLOBIOM 1.0", "LowEnergyDemand_1.3_IPCC"),
    ("MESSAGEix-GLOBIOM_GEI 1.0", "SSP2_openres_lc_50"),
    ("REMIND-MAgPIE 2.1-4.2", "SusDev_SDP-PkBudg1000"),
    ("REMIND-MAgPIE 2.1-4.3", "DeepElec_SSP2_ HighRE_Budg900"),
    ("WITCH 5.0", "CO_Bridge"),
)
"""
AR6 illustrative pathway scenarios

In each sub-tuple, the first element is the model, second is the scenario name.
"""


@functools.cache
def get_all_model_scenarios(filepath: Path) -> pd.DataFrame:
    """
    Load the file with all the model-scenario combinations

    Parameters
    ----------
    filepath
        File to load from

    Returns
    -------
    :
        Loaded model-scenario combinations
    """
    model_scenarios = pd.read_csv(filepath)

    return model_scenarios


def create_model_scenario_test_cases(
    model_scenarios: pd.DataFrame,
) -> tuple[_pytest.mark.structures.ParameterSet, ...]:
    """
    Create a test case for each model-scenario combination

    Parameters
    ----------
    model_scenarios
        Model-scenarios combinations from which to generate the tests cases

    Returns
    -------
    :
        Created test cases
    """
    # Late import to avoid explosions if pytest isn't installed
    import pytest

    return tuple(
        pytest.param(model, scenario, id=f"{model}__{scenario}")
        for (model, scenario), _ in model_scenarios.groupby(["Model", "Scenario"])
    )


@functools.cache
def get_ar6_all_emissions(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    filename_emissions = f"ar6_scenarios__{model}__{scenario}__emissions.csv"
    filename_emissions = filename_emissions.replace("/", "_").replace(" ", "_")
    emissions_file = test_data_dir / filename_emissions

    res = pd.read_csv(emissions_file)
    res.columns = res.columns.str.lower()
    res = res.set_index(["model", "scenario", "variable", "region", "unit"])
    res.columns = res.columns.astype(int)

    return res


@functools.cache
def get_ar6_raw_emissions(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    all_emissions = get_ar6_all_emissions(
        model=model, scenario=scenario, test_data_dir=test_data_dir
    )
    res: pd.DataFrame = all_emissions.loc[pix.ismatch(variable="Emissions**")].dropna(
        how="all", axis="columns"
    )

    return res


@functools.cache
def get_ar6_harmonised_emissions(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    all_emissions = get_ar6_all_emissions(
        model=model, scenario=scenario, test_data_dir=test_data_dir
    )
    res: pd.DataFrame = all_emissions.loc[
        pix.ismatch(variable="**Harmonized**")
    ].dropna(how="all", axis="columns")

    return res


def assert_frame_equal(
    res: pd.DataFrame, exp: pd.DataFrame, rtol: float = 1e-8, **kwargs: Any
) -> None:
    for idx_name in res.index.names:
        idx_diffs = res.pix.unique(idx_name).symmetric_difference(  # type: ignore
            exp.pix.unique(idx_name)  # type: ignore
        )
        if not idx_diffs.empty:
            msg = f"Differences in the {idx_name} (res on the left): {idx_diffs=}"
            raise AssertionError(msg)

    pd.testing.assert_frame_equal(
        res.T, exp.T, check_like=True, check_exact=False, rtol=rtol, **kwargs
    )
