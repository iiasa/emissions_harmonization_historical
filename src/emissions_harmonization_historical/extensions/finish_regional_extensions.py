import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_indexing as pix
import tqdm.auto


def standardize_year_columns(df, target_type=float, startyr=2023):
    """Convert all year columns to the same data type"""
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


def add_region_level_to_index(df, region="World"):
    """Add a 'region' level to a DataFrame index after 'scenario'"""
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


def add_workflow_level_to_index(df, workflow="for_scms"):
    """Add a 'workflow' level to a DataFrame index after 'scenario'"""
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


def fix_up_and_concatenate_extensions(extended_dfs_dict):
    """
    Fix up and concatenate extended DataFrames from different components.

    Ensures consistent indexing and concatenation.
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


def extend_regional_for_missing(df_everything, scenarios_regional, fractions_list, end_year=2500):
    """Extend regional data for scenarios missing region level."""
    df_extended_list = []

    for variable in tqdm.auto.tqdm(scenarios_regional.pix.unique("variable").values):
        if not variable.startswith("Emissions|CO2"):
            continue
        unique_meta = (
            scenarios_regional.loc[pix.ismatch(variable=variable)]
            .index.droplevel(["workflow", "variable", "region", "unit"])
            .drop_duplicates()
        )
        for model, scen in unique_meta.to_list():
            data_regional = scenarios_regional.loc[
                pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}**")
            ]
            data_existing = df_everything.loc[
                pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}**")
            ]
            gross_pos_traj = df_everything.loc[
                pix.ismatch(
                    scenario=f"{scen}",
                    model=f"{model}",
                    variable="Emissions|CO2|Gross Positive Emissions",
                    region="World",
                )
            ]
            regions_total = data_regional.pix.unique("region").values
            regions_existing = data_existing.pix.unique("region").values
            if set(regions_total) == set(regions_existing):
                continue
            if variable in fractions_list[(model, scen)]["fractions_fossil_nocdr"].index.get_level_values("variable"):
                print(f"Now extending {variable} for model {model}, scenario {scen}")
                fractions_f_em_df = fractions_list[(model, scen)]["fractions_fossil_nocdr"]
                for region in regions_total:
                    if region in regions_existing:
                        continue
                    data_frac_extend = (
                        gross_pos_traj
                        * fractions_f_em_df.loc[pix.ismatch(variable=variable, region=f"{region}")].values[0, 0]
                    )
                    full_years = np.arange(df_everything.columns[0], end_year + 1)
                    data_extend = np.zeros((1, len(full_years)))
                    values_exist = data_regional.loc[pix.ismatch(region=region)].values[0, :]
                    extend_year_idx = len(values_exist)
                    data_extend[:, :extend_year_idx] = values_exist
                    data_extend[:, extend_year_idx:] = data_frac_extend.values[:, extend_year_idx:]
                    df_regional = pd.DataFrame(
                        data=data_extend, columns=full_years, index=data_regional.loc[pix.ismatch(region=region)].index
                    )
                    df_extended_list.append(df_regional)
    df_extended = pix.concat(df_extended_list)
    return pix.concat([df_everything, df_extended])


def dump_data_per_model(extended_data, model, output_dir, output_db, scenario_model_match=None):
    """
    Dump extended data for a specific model to CSV.

    Parameters
    ----------
    extended_data : pd.DataFrame
        The complete extended data with MultiIndex.
    model : str
        The model name to filter and dump data for.
    """
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


def save_continuous_timeseries_to_csv(data, filename="continuous_timeseries_historical_future"):
    """
    Save the continuous timeseries data to CSV with metadata.

    Clean, simple approach focused on CSV output as final deliverable.
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


def merge_historical_future_timeseries(history_data, extensions_data, overlap_year=2023):
    """
    Merge historical and future emissions data.

    Key learnings applied:
    - Remove duplicates from extensions data first
    - Replicate historical data for each scenario
    - Simple concatenation along time axis
    """
    print("=== CONCISE HISTORICAL-FUTURE MERGE ===")

    # Step 1: Clean extensions data (remove duplicates)
    extensions_clean = extensions_data[~extensions_data.index.duplicated(keep="first")]
    print(f"Removed {extensions_data.shape[0] - extensions_clean.shape[0]} duplicate extension rows")
    print("Extensions data shape after cleaning:", extensions_clean.shape)

    # Step 2: Define time splits
    hist_years = [col for col in history_data.columns if isinstance(col, int | float) and col <= overlap_year]
    future_years = [col for col in extensions_clean.columns if isinstance(col, int | float) and col > overlap_year]

    # Step 3: Get scenario list from extensions
    scenarios = extensions_clean.index.droplevel(["region", "variable", "unit", "workflow"]).drop_duplicates()

    # Step 4: Replicate historical data for each scenario
    historical_expanded = []
    for model, scenario in scenarios:
        hist_copy = history_data[hist_years].copy()

        # Update index to match scenario structure
        new_index = []
        for idx in hist_copy.index:
            new_idx = (
                model,
                scenario,
                idx[2],
                "for_scms",  # workflow
                idx[3],
                idx[4],
            )  # model, scenario, region, variable, unit
            new_index.append(new_idx)
        hist_copy.index = pd.MultiIndex.from_tuples(
            new_index, names=["model", "scenario", "region", "workflow", "variable", "unit"]
        )
        historical_expanded.append(hist_copy)

    historical_replicated = pd.concat(historical_expanded)
    historical_replicated.index = historical_replicated.index.reorder_levels(extensions_clean.index.names)
    print(f"Replicated historical data shape: {historical_replicated.shape}")
    print(f"Time range: {historical_replicated.columns[0]}-{historical_replicated.columns[-1]}")

    # Step 5: Get future data and merge
    future_data = extensions_clean[future_years]

    # Step 6: Find common variables and merge
    hist_vars = set(historical_replicated.index.get_level_values("variable"))
    future_vars = set(future_data.index.get_level_values("variable"))
    common_vars = hist_vars & future_vars

    # Filter to common variables
    hist_common = historical_replicated.loc[historical_replicated.index.get_level_values("variable").isin(common_vars)]
    future_common = future_data.loc[future_data.index.get_level_values("variable").isin(common_vars)]
    # Step 7: Concatenate along time axis
    continuous_data = pd.concat([hist_common, future_common], axis=1).sort_index(axis=1)

    # Step 8: Cut superfluous global data:
    continuous_data = continuous_data.loc[~pix.ismatch(workflow="global")]

    print(f"Merged data: {continuous_data.shape} ({len(common_vars)} variables, {len(scenarios)} scenarios)")
    print(f"Time range: {continuous_data.columns[0]}-{continuous_data.columns[-1]}")
    return continuous_data
