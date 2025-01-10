"""
Convert the test file from xls to csv because it's way faster to read
"""

from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import tqdm

TEST_DATA_DIR = Path(__file__).parents[1] / "tests" / "test-data"
# This file is available from the scenario explorer.
# It's nearly 1GB, which is why we don't include it here.
# It's md5sum is:
# 4b8aea8270566584db9d893ff7b4562e  tests/test-data/AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv  # noqa: E501
# (Generated with `md5sum tests/test-data/AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv`)  # noqa: E501

raw = pd.read_csv(
    TEST_DATA_DIR / "AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv"
).set_index(["Model", "Scenario", "Variable", "Unit", "Region"])

emissions = raw[pix.ismatch(Variable="**Emissions**")]
temperature_magicc = raw[pix.ismatch(Variable="**|Surface Temperature**MAGICC**")]

temperature_magicc.index.to_frame()[["Model", "Scenario"]].drop_duplicates().to_csv(
    TEST_DATA_DIR / "ar6_scenarios_raw_model_scenario_combinations.csv", index=False
)

for (model, scenario), msdf in tqdm.tqdm(
    temperature_magicc.groupby(["Model", "Scenario"])
):
    filename = f"ar6_scenarios__{model}__{scenario}__temperatures.csv"
    filename = filename.replace("/", "_").replace(" ", "_")
    out_file = TEST_DATA_DIR / filename

    msdf.to_csv(out_file)

    filename_emissions = f"ar6_scenarios__{model}__{scenario}__emissions.csv"
    filename_emissions = filename_emissions.replace("/", "_").replace(" ", "_")
    out_file_emissions = TEST_DATA_DIR / filename_emissions
    emissions[pix.ismatch(Model=model) & pix.ismatch(Scenario=scenario)].to_csv(
        out_file_emissions
    )
