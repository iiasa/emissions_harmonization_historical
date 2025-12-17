import sys

import pandas_indexing as pix
import tqdm.auto


def standardize_year_columns(df, target_type=float):
    """Convert all year columns to the same data type"""
    df_copy = df.copy()
    new_columns = []

    for col in df_copy.columns:
        try:
            # Try to convert to target type
            if isinstance(col, str):
                # Remove '.0' suffix if present and convert
                year_val = float(col.replace(".0", ""))
            else:
                year_val = float(col)
            new_columns.append(year_val)
        except (ValueError, TypeError):
            # Keep non-year columns as-is
            new_columns.append(col)

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
                print(f"df_afolu_fixed index: {df.index.names}")
            else:
                df_fixed = df.copy()
                print(f"df_afolu already has proper index: {df.index.names}")
        elif component_name in ["beccs_extensions", "dacc_extensions", "ocean_extensions", "ew_extensions"]:
            if "workflow" in df.index.names:
                df_fixed = df.copy().droplevel(["workflow"])
        df_fixed = standardize_year_columns(df_fixed)
        fixed_dfs.append(df_fixed)

    concatenated_df = pix.concat(fixed_dfs)
    return concatenated_df


def extend_regional_for_missing(df_everything, scenarios_regional):
    """Extend regional data for scenarios missing region level."""
    variables_exist = df_everything.pix.unique("variable").values
    print(variables_exist)
    df_extended_list = []
    for variable in tqdm.auto.tqdm(scenarios_regional.pix.unique("variable").values):
        print_done_empty = False
        print_done_regional = False
        print_done_in = False
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
            regions_total = data_regional.pix.unique("region").values
            regions_existing = data_existing.pix.unique("region").values
            if regions_existing.size == 0:
                if not print_done_empty:
                    print(f"{variable} has no globally extended data")
                    print_done_empty = True
                # sys.exit(4)
            if set(regions_total) == set(regions_existing):
                if not print_done_in:
                    print(f"{variable} already has all regional data")
                    print_done_in = True
                continue
            if len(regions_total) == 1:
                if not print_done_regional:
                    print(f"{variable} only has World region, skipping")
                    print_done_regional = True

            # fractions = get_2100_compound_composition(data_regional[2100].copy(), variable)

    sys.exit(4)

    df_extended = pix.concat(df_extended_list)
    return df_extended
