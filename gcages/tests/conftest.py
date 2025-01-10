"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

TEST_DATA_DIR = Path(__file__).parents[0] / "test-data"


@pytest.fixture(scope="session", autouse=True)
def pandas_terminal_width():
    # Set pandas terminal width so that doctests don't depend on terminal width.

    # We set the display width to 120 because examples should be short,
    # anything more than this is too wide to read in the source.
    pd.set_option("display.width", 120)

    # Display as many columns as you want (i.e. let the display width do the
    # truncation)
    pd.set_option("display.max_columns", 1000)


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
