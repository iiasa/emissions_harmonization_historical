"""
Constants for use in the 5000 notebooks

These are a way of ensuring that shared ideas are consistent across notebooks
and identifying the processing
that was done to create different data
as we update our processing steps.
If you update the way something is done
and want to avoid overwriting your old data,
just update the ID in here
(because then the data will be written to a different file).
There are much more sophisticated ways this could be done,
but they also require more effort :)
"""

from pathlib import Path

from pandas_openscm.db import (
    CSVDataBackend,
    CSVIndexBackend,
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)

# Chosen to match the CMIP experiment ID
HISTORY_SCENARIO_NAME = "historical"

REPO_ROOT = Path(__file__).parents[2]
DATA_ROOT = REPO_ROOT / "data"

CEDS_VERSION_ID = "v_2025_03_18"

# ID for the CEDS processing step
# CEDS_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
CEDS_PROCESSING_ID = "0002"
# Updated Aircraft handling
CEDS_PROCESSING_ID = "0003"

CEDS_TOP_LEVEL_RAW_PATH = DATA_ROOT / "raw" / "ceds"
CEDS_RAW_PATH = CEDS_TOP_LEVEL_RAW_PATH / CEDS_VERSION_ID

# Database into which the processed CEDS data is saved
CEDS_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "ceds" / CEDS_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

GFED4_INTERIM_OUTPUT_DIR = DATA_ROOT / "interim" / "gfed4"

# ID for the GFED4 processing step
# GFED4_PROCESSING_ID = "0001"
# # Fix a bug in regridding handling
# GFED4_PROCESSING_ID = "0002"
# Split into multiple steps now

# ID for the GFED4 annual sum step
GFED4_ANNUAL_SUM_ID = "0001"
GFED4_ANNUAL_SUM_OUTPUT_DIR = GFED4_INTERIM_OUTPUT_DIR / GFED4_ANNUAL_SUM_ID

# ID for the GFED4 regridding, smoothing and extension step
GFED4_REGRIDDING_SMOOTHING_EXTENSION_ID = "_".join([GFED4_ANNUAL_SUM_ID, "0001"])
GFED4_REGRIDDING_SMOOTHING_EXTENSION_OUTPUT_DIR = GFED4_INTERIM_OUTPUT_DIR / GFED4_REGRIDDING_SMOOTHING_EXTENSION_ID

GFED4_SPLIT_INTO_SPECIES_AND_COUNTRIES_ID = "_".join(
    [
        GFED4_ANNUAL_SUM_ID,
        GFED4_REGRIDDING_SMOOTHING_EXTENSION_ID,
        # "0001",
        # Moved to portable OpenSCMDB
        "0002",
    ]
)

GFED4_TOP_LEVEL_RAW_PATH = DATA_ROOT / "raw" / "gfed4"
GFED4_RAW_PATH = GFED4_TOP_LEVEL_RAW_PATH

# Database into which the processed GFED4 data is saved
GFED4_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "gfed4" / GFED4_SPLIT_INTO_SPECIES_AND_COUNTRIES_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

BB4CMIP7_INTERIM_OUTPUT_DIR = DATA_ROOT / "interim" / "bb4cmip7"

# ID for the processing to annual, sectoral emissions
# BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_ID = "0001"
# Process all the way back to 1750
BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_ID = "0002"
BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR = BB4CMIP7_INTERIM_OUTPUT_DIR / BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_ID

# ID for formatting BB4CMIP7 data into the required format
# BB4CMIP7_FORMATTING_ID = "0001"
# Moved to portable OpenSCMDB
BB4CMIP7_FORMATTING_ID = "0002"
# Update after investigation of Annika in https://github.com/iiasa/emissions_harmonization_historical/pull/110/files
# ... not because of content changes, but because the current version was produced on that branch
BB4CMIP7_FORMATTING_ID = "0003"

# Database into which the processed BB4CMIP7 data is saved
BB4CMIP7_PROCESSED_DIR = DATA_ROOT / "processed" / "bb4cmip7" / BB4CMIP7_FORMATTING_ID
BB4CMIP7_PROCESSED_DB = OpenSCMDB(
    db_dir=BB4CMIP7_PROCESSED_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for including other modifications that affect
# the creation of historical emissions dataset for gridding
# Gcages CDR split
MOD_HISTORY_FOR_GRIDDING_ID = "0001"
# ID for the creation of a historical emissions dataset for gridding
# CREATE_HISTORY_FOR_GRIDDING_ID = "0001"
# Update to make the smoothing consistent with CMIP7
# CREATE_HISTORY_FOR_GRIDDING_ID = "0002"
# Update to use BB4CMIP7 data
CREATE_HISTORY_FOR_GRIDDING_ID = "_".join(
    [
        CEDS_PROCESSING_ID,
        BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_ID,
        BB4CMIP7_FORMATTING_ID,
        # MOD_HISTORY_FOR_GRIDDING_ID,
    ]
)

COUNTRY_LEVEL_HISTORY = DATA_ROOT / "processed" / "cmip7_history_countrylevel_250721.csv"

GCB_VERSION = "2024v1.0"

# ID for the GCB processing step
# GCB_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
GCB_PROCESSING_ID = "0002"

GCB_RAW_PATH = DATA_ROOT / "raw" / "gcb" / GCB_VERSION

# Database into which the processed GCB data is saved
GCB_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "gcb" / GCB_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

WMO_2022_RAW_PATH = DATA_ROOT / "raw" / "wmo-2022"

# ID for the WMO 2022 processing step
# WMO_2022_PROCESSING_ID = "0001"
# # Fix negative values in smoothing
# WMO_2022_PROCESSING_ID = "0002"
# Moved to portable OpenSCMDB
WMO_2022_PROCESSING_ID = "0003"

WMO_2022_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "wmo-2022" / WMO_2022_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

VELDERS_ET_AL_2022_RAW_PATH = DATA_ROOT / "raw" / "velders-et-al-2022"

# ID for the Velders et al. 2022 processing step
# VELDERS_ET_AL_2022_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
VELDERS_ET_AL_2022_PROCESSING_ID = "0002"

VELDERS_ET_AL_2022_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "velders-et-al-2022" / VELDERS_ET_AL_2022_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

ADAM_ET_AL_2024_RAW_PATH = DATA_ROOT / "raw" / "adam-et-al-2024"

# ID for the adam et al. 2024 processing step
# ADAM_ET_AL_2024_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
ADAM_ET_AL_2024_PROCESSING_ID = "0002"

ADAM_ET_AL_2024_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "adam-et-al-2024" / ADAM_ET_AL_2024_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

CMIP7_GHG_VERSION_ID = "CR-CMIP-1-0-0"
CMIP7_GHG_PUB_DATE = "v20250228"

CMIP7_GHG_RAW_PATH = DATA_ROOT / "raw" / "cmip7-ghgs" / CMIP7_GHG_VERSION_ID

# ID for the CMIP7 GHG processing step
# CMIP7_GHG_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
CMIP7_GHG_PROCESSING_ID = "0002"

CMIP7_GHG_PROCESSED_DIR = DATA_ROOT / "processed" / "cmip7-ghgs" / CMIP7_GHG_PROCESSING_ID
CMIP7_GHG_PROCESSED_DB = OpenSCMDB(
    db_dir=CMIP7_GHG_PROCESSED_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for creating history from the global workflow
CREATE_HISTORY_FOR_GLOBAL_WORKFLOW_ID = "_".join(
    [
        GCB_PROCESSING_ID,
        WMO_2022_PROCESSING_ID,
        VELDERS_ET_AL_2022_PROCESSING_ID,
        ADAM_ET_AL_2024_PROCESSING_ID,
        CMIP7_GHG_PROCESSING_ID,
        # "0001",
        # Moved to portable OpenSCMDB
        "0002",
    ]
)

RCMIP_VERSION_ID = "v5.1.0"

RCMIP_RAW_PATH = DATA_ROOT / "raw" / "rcmip" / RCMIP_VERSION_ID

# ID for the CMIP7 GHG processing step
# RCMIP_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
RCMIP_PROCESSING_ID = "0002"

RCMIP_PROCESSED_DIR = DATA_ROOT / "processed" / "rcmip" / RCMIP_PROCESSING_ID
RCMIP_PROCESSED_DB = OpenSCMDB(
    db_dir=RCMIP_PROCESSED_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)


# Commit from https://github.com/IAMconsortium/common-definitions
# COMMON_DEFINITIONS_COMMIT = get_latest_commit_hash(
#     "IAMconsortium", "common-definitions", fallback_commit="cc69ed0a415a63c7ce7372d5a36c088d9cbee055"
# )
# Hard-code to ensure stability
COMMON_DEFINITIONS_COMMIT = "7e32405ade790677a6022ff498395bff00d9792d"
COMMON_DEFINITIONS_PATH = REPO_ROOT / "common-definitions"


REGION_MAPPING_FILE = (
    DATA_ROOT
    / "processed"
    / "region-mapping"
    / COMMON_DEFINITIONS_COMMIT
    / f"region-mapping_{COMMON_DEFINITIONS_COMMIT}.csv"
)

REGION_MAPPING_PATH = DATA_ROOT / "processed" / "region-mapping" / COMMON_DEFINITIONS_COMMIT


# ID for the created history for harmonisation
HISTORY_FOR_HARMONISATION_ID = "_".join(
    [
        CREATE_HISTORY_FOR_GRIDDING_ID,
        CREATE_HISTORY_FOR_GLOBAL_WORKFLOW_ID,
        COMMON_DEFINITIONS_COMMIT,
    ]
)

# Directory in which the history for harmonisation information lives
HISTORY_HARMONISATION_DIR = DATA_ROOT / "processed" / "history-for-harmonisation" / HISTORY_FOR_HARMONISATION_ID

# Database to hold historical emissions for harmonisation
HISTORY_HARMONISATION_DB = OpenSCMDB(
    db_dir=HISTORY_HARMONISATION_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)


# # ID for the scenario download step
# DOWNLOAD_SCENARIOS_ID = "0001"
# # Updated May 6 for revised AIM scenarios
# DOWNLOAD_SCENARIOS_ID = "0002"
# # Chris workaround
# DOWNLOAD_SCENARIOS_ID = "0003"
# # Update hashing
# DOWNLOAD_SCENARIOS_ID = "0004"
# # Moved to portable OpenSCMDB
# DOWNLOAD_SCENARIOS_ID = "0005"
# # Scratch while we wait for new submissions
# DOWNLOAD_SCENARIOS_ID = "0006-zn-rc0"
# 2025-05-23 submissions
# (still problems with how we harmonise novel CDR,
# but good interim step)
DOWNLOAD_SCENARIOS_ID = "0006"
# DOWNLOAD_SCENARIOS_ID = "0006-MZ"
# 2025-05-26 run
# (still problems with how we harmonise novel CDR,
# but good interim step)

DOWNLOAD_SCENARIOS_ID = "0008"
# Development helper
DOWNLOAD_SCENARIOS_ID = "0009-zn"
# Temporary to investigate code
DOWNLOAD_SCENARIOS_ID = "0010-jk"
# Run intermediary submission REMIND 20250617
DOWNLOAD_SCENARIOS_ID = "0011-REMIND-jk"
# Current for this branch, a bit outdated...
DOWNLOAD_SCENARIOS_ID = "20250602-123212"
# Run 20250710 by Marco (prepared with/by Jarmo)
DOWNLOAD_SCENARIOS_ID = "08Snapshot"
DOWNLOAD_SCENARIOS_ID = "20250818"
# Run by Marco
DOWNLOAD_SCENARIOS_ID = "All_23Oct"

# Database into which raw scenarios are saved
RAW_SCENARIO_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the pre-processing step
# PRE_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
PRE_PROCESSING_ID = "0002"
# Various hacks to deal with issues in the 20250710 run
PRE_PROCESSING_ID = "0003"
# Upgrade to gcages which puts CDR in the Emissions tree
PRE_PROCESSING_ID = "0004"
# PRE_PROCESSING_ID = "0005"

# Database into which pre-processed scenarios are saved
PRE_PROCESSED_SCENARIO_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "pre-processed" / f"{DOWNLOAD_SCENARIOS_ID}_{PRE_PROCESSING_ID}" / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the harmonisation step
# HARMONISATION_ID = "0001"
# Fixed up override passing
# HARMONISATION_ID = "0002"
# Moved to portable OpenSCMDB
HARMONISATION_ID = "0003"

HARMONISED_OUT_DIR = (
    DATA_ROOT
    / "processed"
    / "harmonised"
    / f"{DOWNLOAD_SCENARIOS_ID}_{PRE_PROCESSING_ID}_{HISTORY_FOR_HARMONISATION_ID}_{HARMONISATION_ID}"
)

# Database into which harmonised scenarios are saved
HARMONISED_SCENARIO_DB = OpenSCMDB(
    db_dir=HARMONISED_OUT_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the infilling database creation step
# INFILLING_DB_CREATION_ID = "0001"
# Moved to portable OpenSCMDB
INFILLING_DB_CREATION_ID = "0002"

INFILLING_DB_DIR = (
    DATA_ROOT
    / "processed"
    / "infilling-db"
    / f"{DOWNLOAD_SCENARIOS_ID}_{WMO_2022_PROCESSING_ID}_{HARMONISATION_ID}_{INFILLING_DB_CREATION_ID}"
)

# Database into which infilled emissions are saved
INFILLING_DB = OpenSCMDB(
    db_dir=INFILLING_DB_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the infilling step
# INFILLING_ID = "0001"
# Fixed a bug in how some Montreal gases were infilled
# (some had emissions that were becoming way too low in future).
# INFILLING_ID = "0002"
# Moved to portable OpenSCMDB
INFILLING_ID = "0003"

INFILLED_OUT_DIR_ID = "_".join(
    [
        DOWNLOAD_SCENARIOS_ID,
        PRE_PROCESSING_ID,
        HISTORY_FOR_HARMONISATION_ID,
        HARMONISATION_ID,
        INFILLING_ID,
    ]
)

INFILLED_OUT_DIR = DATA_ROOT / "processed" / "infilled" / INFILLED_OUT_DIR_ID
# print(f"Using infilled output dir: {INFILLED_OUT_DIR}")
# print(f"DOWNLOAD_SCENARIOS_ID: {DOWNLOAD_SCENARIOS_ID}")
# print(f"PRE_PROCESSING_ID: {PRE_PROCESSING_ID}")
# print(f"HISTORY_FOR_HARMONISATION_ID: {HISTORY_FOR_HARMONISATION_ID}")
# print(f"HARMONISATION_ID: {HARMONISATION_ID}")
# print(f"INFILLING_ID: {INFILLING_ID}")
# sys.exit(4)

# Database into which infilled data is saved
INFILLED_SCENARIOS_DB = OpenSCMDB(
    db_dir=INFILLED_OUT_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the simple climate model running step
# SCM_RUNNING_ID = "0001"
# Moved to portable OpenSCMDB
SCM_RUNNING_ID = "0002"

SCM_OUT_DIR = (
    DATA_ROOT
    / "processed"
    / "scm-output"
    / "_".join(
        [
            DOWNLOAD_SCENARIOS_ID,
            PRE_PROCESSING_ID,
            HISTORY_FOR_HARMONISATION_ID,
            HARMONISATION_ID,
            INFILLING_ID,
            SCM_RUNNING_ID,
        ]
    )
)

# Database into which SCM output is saved
SCM_OUTPUT_DB = OpenSCMDB(
    db_dir=SCM_OUT_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the scenario extension step
# Moved to portable OpenSCMDB
EXTENSIONS_ID = "0001"

EXTENSIONS_OUT_DIR = (
    DATA_ROOT
    / "processed"
    / "extension-output"
    / "_".join(
        [
            DOWNLOAD_SCENARIOS_ID,
            PRE_PROCESSING_ID,
            HISTORY_FOR_HARMONISATION_ID,
            HARMONISATION_ID,
            INFILLING_ID,
            EXTENSIONS_ID,
        ]
    )
)

# Database into which extensions output is saved
EXTENSIONS_OUTPUT_DB = OpenSCMDB(
    db_dir=EXTENSIONS_OUT_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)


# ID for the post-processing step
# POST_PROCESSING_ID = "0001"
# Moved to portable OpenSCMDB
POST_PROCESSING_ID = "0002"

POST_PROCESSING_DIR = (
    DATA_ROOT
    / "processed"
    / "post-processed"
    / "_".join(
        [
            DOWNLOAD_SCENARIOS_ID,
            PRE_PROCESSING_ID,
            HISTORY_FOR_HARMONISATION_ID,
            HARMONISATION_ID,
            INFILLING_ID,
            SCM_RUNNING_ID,
            POST_PROCESSING_ID,
        ]
    )
)

# Databases intwo which post-processed output is saved.
# There are lots of these because each kind of data
# has slightly different axes,
# so we don't want to use the same ones.

POST_PROCESSED_METADATA_CATEGORIES_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-metadata-categories",
    backend_data=CSVDataBackend(),
    backend_index=CSVIndexBackend(),
)

POST_PROCESSED_METADATA_EXCEEDANCE_PROBABILITIES_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-metadata-exceedance-probabilities-db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

POST_PROCESSED_METADATA_QUANTILE_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-metadata-quantile",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

POST_PROCESSED_METADATA_RUN_ID_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-metadata-run-id",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

POST_PROCESSED_TIMESERIES_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-timeseries",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

POST_PROCESSED_TIMESERIES_EXCEEDANCE_PROBABILITIES_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-timeseries-exceedance-probabilities",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

POST_PROCESSED_TIMESERIES_QUANTILE_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-timeseries-quantile",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

POST_PROCESSED_TIMESERIES_RUN_ID_DB = OpenSCMDB(
    db_dir=POST_PROCESSING_DIR / "db-timeseries-run-id",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)
