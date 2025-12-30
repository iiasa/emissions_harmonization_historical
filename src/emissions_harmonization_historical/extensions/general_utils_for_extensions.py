import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_indexing as pix
from pandas_openscm.db import OpenSCMDB


def interpolate_to_annual(idf: pd.DataFrame, max_supplement: float = 1e-5) -> pd.DataFrame:
    """
    Interpolate pd.DataFrame of emissions input that might not have data for all years

    Parameters
    ----------
    idf : pd.DataFrame
        Input DataFrame with potentially missing years.
    max_supplement : float, optional
        Maximum supplement for year range, default 1e-5.

    Returns
    -------
    pd.DataFrame
        Interpolated DataFrame with all years filled.
    """
    missing_cols = np.setdiff1d(np.arange(idf.columns.min(), idf.columns.max() + max_supplement), idf.columns)

    out = idf.copy()
    out.loc[:, missing_cols] = np.nan
    out = out.sort_index(axis="columns").T.interpolate("index").T

    return out


def glue_with_historical(scen_df: pd.DataFrame, hist_df: pd.DataFrame, history_end=2023) -> pd.DataFrame:
    """
    Glue historical data to the beginning of scenario data to get complete timeseries

    Parameters
    ----------
    scen_df : pd.DataFrame
        Scenario data.
    hist_df : pd.DataFrame
        Historical data.
    history_end : int, optional
        End year of history, default 2023.

    Returns
    -------
    pd.DataFrame
        Combined timeseries.
    """
    out = interpolate_to_annual(scen_df.copy())
    orig_len = len(out.columns)
    missing_years = np.arange(int(hist_df.columns[0]), int(scen_df.columns[0]))
    out.loc[:, missing_years] = np.nan  # hist_df.loc[:, missing_years].values
    for index, df_row in out.iterrows():
        df_row.iloc[orig_len:] = hist_df.loc[pix.ismatch(variable=index[3])].values[
            0, : int(scen_df.columns[0] - hist_df.columns[0])
        ]
        for year in range(scen_df.columns[0], history_end + 1):
            if np.isnan(df_row.iloc[year - scen_df.columns[0]]):
                df_row.iloc[year - scen_df.columns[0]] = hist_df.loc[pix.ismatch(variable=index[3])].values[
                    0, int(year - hist_df.columns[0])
                ]
        out.loc[index, :] = df_row.values
    out = out.sort_index(axis="columns")
    return out


def standardize_year_columns(df: pd.DataFrame, startyr=2023) -> pd.DataFrame:
    """Convert all year columns to the same data type

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    startyr : int, optional
        Starting year, default 2023.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized year columns.
    """
    df_copy = df.copy()
    new_columns = []
    drop_columns = []
    for col in df_copy.columns:
        try:
            # Try to convert to target type
            if isinstance(col, str):
                # Remove '.0' suffix if present and convert
                year_val = float(col.replace(".0", ""))
            else:
                year_val = float(col)
            if year_val < startyr:
                drop_columns.append(col)
                continue
            new_columns.append(year_val)
        except (ValueError, TypeError):
            # Keep non-year columns as-is
            new_columns.append(col)
    df_copy = df_copy.drop(columns=drop_columns)
    df_copy.columns = new_columns
    return df_copy


def add_region_level_to_index(df: pd.DataFrame, region="World") -> pd.DataFrame:
    """Add a 'region' level to a DataFrame index after 'scenario'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    region : str, optional
        Region name, default "World".

    Returns
    -------
    pd.DataFrame
        DataFrame with region level added.
    """
    if "region" in df.index.names:
        return df

    # Reset index to work with it
    df_reset = df.reset_index()

    # Add region column in the correct position (after scenario, before variable)
    cols = list(df_reset.columns)
    scenario_idx = cols.index("scenario")
    cols.insert(scenario_idx + 1, "region")
    df_reset["region"] = region
    df_reset = df_reset[cols]

    # Set the index back with the correct order
    index_cols = ["model", "scenario", "region", "variable", "unit"]
    return df_reset.set_index(index_cols)


def add_workflow_level_to_index(df: pd.DataFrame, workflow="for_scms") -> pd.DataFrame:
    """Add a 'workflow' level to a DataFrame index after 'scenario'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    workflow : str, optional
        Workflow name, default "for_scms".

    Returns
    -------
    pd.DataFrame
        DataFrame with workflow level added.
    """
    if "workflow" in df.index.names:
        return df

    # Reset index to work with it
    df_reset = df.reset_index()
    # Add region column in the correct position (after scenario, before variable)
    cols = list(df_reset.columns)
    if "region" in df.index.names:
        if len(df.pix.unique("region")) > 1:
            workflow = "gridding"
        placement_idx = cols.index("region") + 1
        index_cols = ["model", "scenario", "region", "workflow", "variable", "unit"]
    else:
        placement_idx = cols.index("scenario") + 1
        index_cols = ["model", "scenario", "workflow", "variable", "unit"]

    cols.insert(placement_idx, "workflow")
    df_reset["workflow"] = workflow
    df_reset = df_reset[cols]

    # Set the index back with the correct order
    return df_reset.set_index(index_cols)


def fix_up_and_concatenate_extensions(extended_dfs_dict: dict) -> pd.DataFrame:
    """
    Fix up and concatenate extended DataFrames from different components.

    Ensures consistent indexing and concatenation.

    Parameters
    ----------
    extended_dfs_dict : dict
        Dictionary of extended DataFrames.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame.
    """
    fixed_dfs = []
    for component_name, df in extended_dfs_dict.items():
        if component_name in ["gross_positive_extensions", "cdr_extensions"]:
            df_fixed = add_region_level_to_index(df.copy(), region="World")
        elif component_name == "afolu_extensions":
            if "model" in df.columns:
                index_cols = ["model", "scenario", "region", "variable", "unit"]
                df_fixed = df.copy().set_index(index_cols)
                print(f"df_afolu_fixed index: {df_fixed.index.names}")
            else:
                df_fixed = df.copy()
                print(f"df_afolu already has proper index: {df.index.names}")
        else:
            df_fixed = df.copy()
        if "workflow" not in df_fixed.index.names:
            df_fixed = add_workflow_level_to_index(df_fixed)
        df_fixed = standardize_year_columns(df_fixed)
        fixed_dfs.append(df_fixed)

    concatenated_df = pix.concat(fixed_dfs)
    return concatenated_df


def save_continuous_timeseries_to_csv(data: pd.DataFrame, filename="continuous_timeseries_historical_future") -> dict:
    """
    Save the continuous timeseries data to CSV with metadata.

    Clean, simple approach focused on CSV output as final deliverable.

    Parameters
    ----------
    data : pd.DataFrame
        Data to save.
    filename : str, optional
        Base filename, default "continuous_timeseries_historical_future".

    Returns
    -------
    dict
        Dictionary with file paths and metadata.
    """
    print("=== SAVING CONTINUOUS TIMESERIES TO CSV ===")

    # Create filename without timestamp
    csv_filename = f"{filename}.csv"
    csv_path = os.path.join(os.getcwd(), csv_filename)

    # Convert to long format for CSV
    data_long = data.reset_index()

    # Save to CSV
    data_long.to_csv(csv_path, index=False)

    # Create summary metadata
    metadata = {
        "filename": csv_filename,
        "created": datetime.now().isoformat(),
        "shape": f"{data.shape[0]} rows x {data.shape[1]} columns",
        "time_coverage": f"{data.columns[0]} to {data.columns[-1]}",
        "total_years": len([col for col in data.columns if isinstance(col, int | float)]),
        "variables": len(data.index.get_level_values("variable").unique()),
        "scenarios": len(data.index.droplevel(["region", "variable", "unit"]).drop_duplicates()),
        "description": (
            "Continuous emissions timeseries merging historical data (1750-2023) with future projections (2024-2500)"
        ),
    }

    print(f"📂 Location: {csv_path}")
    print(f"📊 Size: {metadata['shape']}")
    print(f"🕐 Coverage: {metadata['time_coverage']} ({metadata['total_years']} years)")
    print(f"📈 Variables: {metadata['variables']}")
    print(f"🎯 Scenarios: {metadata['scenarios']}")

    # Save metadata as JSON
    metadata_filename = f"{filename}_metadata.json"
    metadata_path = os.path.join(os.getcwd(), metadata_filename)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Metadata: {metadata_filename}")

    return {
        "csv_file": csv_filename,
        "csv_path": csv_path,
        "metadata_file": metadata_filename,
        "metadata": metadata,
    }


def dump_data_per_model(
    extended_data: pd.DataFrame, model: str, output_dir: str, output_db: OpenSCMDB, scenario_model_match=None
):
    """
    Dump extended data for a specific model to CSV.

    Parameters
    ----------
    extended_data : pd.DataFrame
        The complete extended data with MultiIndex.
    model : str
        The model name to filter and dump data for.
    output_dir : str
        Output directory path.
    output_db : OpenSCMDB
        Database object for saving.
    scenario_model_match : dict, optional
        Mapping for scenario names, default None.

    Returns
    -------
    None
    """
    model_short = model.split(" ")[0]
    model_short = model.split(" ")[0]
    if model.startswith("MESSAGE"):
        model_short = "MESSAGE"
    print(f"=== DUMPING DATA FOR MODEL: {model} ({model_short}) ===")
    output_dir_model = output_dir / model_short
    output_dir_model.mkdir(exist_ok=True, parents=True)
    model_data = extended_data.loc[extended_data.index.get_level_values("model") == model].copy()

    if scenario_model_match is not None:
        # Build mapping from long scenario names to short codes
        long_to_short = {v[0]: k for k, v in scenario_model_match.items()}
        if "scenario" in model_data.index.names:
            # Get index level number for 'scenario'
            scenario_level = model_data.index.names.index("scenario")
            # Create new MultiIndex with scenario values mapped
            new_index = [
                tuple(
                    (long_to_short.get(idx[scenario_level], idx[scenario_level]) if i == scenario_level else v)
                    for i, v in enumerate(idx)
                )
                for idx in model_data.index
            ]
            model_data.index = pd.MultiIndex.from_tuples(new_index, names=model_data.index.names)
            print("Renamed scenario values in index using scenario_model_match mapping.")
        else:
            print("No 'scenario' level in index; skipping scenario renaming.")

    # Fix mixed column types warning by converting ALL columns to strings
    # This ensures consistent typing for PyArrow/database storage
    model_data.columns = [str(col) for col in model_data.columns]

    output_db.save(model_data, allow_overwrite=True)
