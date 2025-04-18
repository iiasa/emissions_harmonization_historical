"""
Constants for use in our notebooks

These are just a way of avoiding overwriting and identifying data
as we update our different processing steps.
If you update the way something is done
and want to avoid overwriting your old data,
just update the ID in here
(because then the data will be written to a different file).
There are much more sophisticated ways this could be done,
but they also require more effort :)
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
DATA_ROOT = REPO_ROOT / "data"

CEDS_PROCESSING_ID = "0010"
CEDS_EXPECTED_NUMBER_OF_REGION_VARIABLE_PAIRS_IN_GLOBAL_HARMONIZATION = 2

CAMS_PROCESSING_ID = "0010"

GFED_PROCESSING_ID = "0011"

GCB_PROCESSING_ID = "0010"

VELDERS_ET_AL_2022_PROCESSING_ID = "0010"

ADAM_ET_AL_2024_PROCESSING_ID = "0010"

WMO_2022_PROCESSING_ID = "0010"

EDGAR_PROCESSING_ID = "0010"

CMIP_CONCENTRATION_INVERSION_ID = "0012"

COMBINED_HISTORY_ID = "0021"

PRIMAP_HIST_PROCESSING_ID = "0001"

IAMC_REGION_PROCESSING_ID = "0020"

FOLLOWER_SCALING_FACTORS_ID = "0010"

SCENARIO_TIME_ID = "20250416-112220"

HARMONISATION_ID = "0020"

HARMONISATION_VALUES_ID = "0020"

INFILLING_SILICONE_ID = "0011"

INFILLING_WMO_ID = "0011"

INFILLING_LEFTOVERS_ID = "0011"

MAGICC_RUN_ID = "0020"

WORKFLOW_ID = "0020"

# Chosen to match the CMIP experiment ID
HISTORY_SCENARIO_NAME = "historical"
