"""
Convert the test file from xls to csv because it's way faster to read
"""

from pathlib import Path

import pandas as pd

TEST_DATA_DIR = Path(__file__).parents[1] / "tests" / "test-data"
raw = pd.read_excel(TEST_DATA_DIR / "20220314_ar6emissions_harmonized_infilled.xlsx")

raw.to_csv(TEST_DATA_DIR / "20220314_ar6emissions_harmonized_infilled.csv", index=False)
raw[["Model", "Scenario"]].drop_duplicates().to_csv(
    TEST_DATA_DIR / "20220314_ar6emissions_harmonized_infilled_model_scenarios.csv",
    index=False,
)
