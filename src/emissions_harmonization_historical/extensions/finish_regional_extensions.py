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
    # print(df.index.names)
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
    # print(cols)
    cols.insert(placement_idx, "workflow")
    df_reset["workflow"] = workflow
    df_reset = df_reset[cols]
    # print(cols)
    # print(index_cols)
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


def extend_regional_for_missing(df_everything, scenarios_regional, fractions_list):
    """Extend regional data for scenarios missing region level."""
    variables_exist = df_everything.pix.unique("variable").values
    print(variables_exist)
    df_extended_list = []

    for variable in tqdm.auto.tqdm(scenarios_regional.pix.unique("variable").values):
        print_done_empty = False
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
            gross_pos_traj = df_everything.loc[
                pix.ismatch(
                    scenario=f"{scen}",
                    model=f"{model}",
                    variable="Emissions|CO2|Gross Positive Emissions",
                    region="World",
                )
            ]
            # sys.exit(4)
            regions_total = data_regional.pix.unique("region").values
            regions_existing = data_existing.pix.unique("region").values
            if regions_existing.size == 0:
                if not print_done_empty:
                    print(f"{variable} has no globally extended data, needs {len(regions_total)} regions")
                    print_done_empty = True
                # sys.exit(4)
            if set(regions_total) == set(regions_existing):
                if not print_done_in:
                    print(f"{variable} already has all regional data ({len(regions_total)})")
                    print_done_in = True
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
                    full_years = np.arange(df_everything.columns[0], 2501)
                    data_extend = np.zeros((1, len(full_years)))
                    values_exist = data_regional.loc[pix.ismatch(region=region)].values[0, :]
                    extend_year_idx = len(values_exist)
                    data_extend[:, :extend_year_idx] = values_exist
                    data_extend[:, extend_year_idx:] = data_frac_extend.values[:, extend_year_idx:]
                    df_regional = pd.DataFrame(
                        data=data_extend, columns=full_years, index=data_regional.loc[pix.ismatch(region=region)].index
                    )
                    # sys.exit(4)
                    df_extended_list.append(df_regional)
            # if len(regions_total) == 1:
            #     if not print_done_regional:
            #         print(f"{variable} only has World region, skipping")
            #         print_done_regional = True

            # fractions = get_2100_compound_composition(data_regional[2100].copy(), variable)

    # sys.exit(4)

    df_extended = pix.concat(df_extended_list)
    print(df_extended.index.names)
    return pix.concat([df_everything, df_extended])
