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

DATA_ROOT = Path(__file__).parents[2] / "data"

# CEDS_PROCESSING_ID = "0011" # v_2025_03_11
CEDS_PROCESSING_ID = "0012"  # v_2025_03_18
CEDS_EXPECTED_NUMBER_OF_REGION_VARIABLE_PAIRS_IN_GLOBAL_HARMONIZATION = 0
# all domestic aviation is on the country-level since v_2025_03_18;
# 2 for v_2025_03_11; for v2024_07_08 this is also 2; see https://github.com/JGCRI/CEDS/issues/54

CAMS_PROCESSING_ID = "0010"

GFED_PROCESSING_ID = "0012" # move BB4CMIP from CMIP6Plus:DRES-CMIP-BB4CMIP7-1-0 to CMIP7:DRES-CMIP-BB4CMIP7-2-0

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

SCENARIO_TIME_ID = "20250313-140552"

HARMONISATION_ID = "0020"

HARMONISATION_VALUES_ID = "0020"

INFILLING_SILICONE_ID = "0011"

INFILLING_WMO_ID = "0011"

INFILLING_LEFTOVERS_ID = "0011"

MAGICC_RUN_ID = "0020"

WORKFLOW_ID = "0020"

# Chosen to match the CMIP experiment ID
HISTORY_SCENARIO_NAME = "historical"

EDGAR_PROCESSING_ID = "0010" # processed data received from Steve Smith 

FAO_PROCESSING_ID = "0010" # processed data received from Steve Smith

GAINS_PROCESSING_ID = "0010" # processed data received from Shaohui Zhang