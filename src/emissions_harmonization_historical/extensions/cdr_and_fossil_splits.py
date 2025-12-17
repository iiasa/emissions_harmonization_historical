import pandas as pd


def extend_cdr_components_vectorized(cdr_components_dict, global_cdr_ext, baseline_year=2100):
    """
    Ultra-efficient CDR extension using pure vectorized operations.

    Eliminates all DataFrame fragmentation warnings by using bulk operations.

    FIXED: Handles MultiIndex alignment properly for regional fraction calculations.
    """
    extension_years = [col for col in global_cdr_ext.columns if isinstance(col, int | float) and col > baseline_year]

    print(f"Extension: {len(extension_years)} years ({min(extension_years)}-{max(extension_years)})")

    extended_components = {}

    for component_name, cdr_df in cdr_components_dict.items():
        print(component_name)

        # Validate inputs and find common scenarios
        validation_result = _validate_extension_inputs(cdr_df, global_cdr_ext, baseline_year)
        if not validation_result["is_valid"]:
            extended_components[component_name] = cdr_df.copy()
            continue

        # Calculate baseline ratios
        ratios = _calculate_baseline_ratios(cdr_df, global_cdr_ext, baseline_year)

        # Build extension data
        extension_data = _build_extension_data(
            extension_years,
            global_cdr_ext,
            ratios["baseline_data"],
            ratios["component_fractions"],
            ratios["regional_fractions"],
        )

        # Construct final DataFrame
        final_df = _construct_final_dataframe(cdr_df, extension_data, baseline_year)
        extended_components[component_name] = final_df

    return extended_components


def _validate_extension_inputs(cdr_df, global_cdr_ext, baseline_year):
    """Validate inputs and find common scenarios."""
    if baseline_year not in cdr_df.columns:
        print(f"  ⚠️  {baseline_year} not found, copying original")
        return {"is_valid": False}

    cdr_scenarios = set(cdr_df.index.get_level_values("scenario"))
    global_scenarios = set(global_cdr_ext.index.get_level_values("scenario"))
    common_scenarios = cdr_scenarios & global_scenarios

    if not common_scenarios:
        print("  ⚠️  No common scenarios, copying original")
        return {"is_valid": False}

    return {"is_valid": True, "common_scenarios": common_scenarios}


def _calculate_baseline_ratios(cdr_df, global_cdr_ext, baseline_year):
    """Calculate baseline ratios for component and regional fractions."""
    baseline_data = cdr_df[baseline_year]
    component_totals_2100 = baseline_data.groupby("scenario").sum()
    global_data_2100 = global_cdr_ext[baseline_year].groupby("scenario").first()
    component_fractions = (component_totals_2100 / global_data_2100).fillna(0)

    # Calculate regional fractions
    regional_fractions = baseline_data.copy()
    for scenario in baseline_data.index.get_level_values("scenario").unique():
        scenario_mask = baseline_data.index.get_level_values("scenario") == scenario
        scenario_data = baseline_data[scenario_mask]
        component_total = component_totals_2100[scenario]

        if component_total != 0:
            regional_fractions[scenario_mask] = scenario_data / component_total
        else:
            regional_fractions[scenario_mask] = 0

    regional_fractions = regional_fractions.fillna(0)

    return {
        "baseline_data": baseline_data,
        "component_fractions": component_fractions,
        "regional_fractions": regional_fractions,
    }


def _build_extension_data(
    extension_years,
    global_cdr_ext,
    baseline_data,
    component_fractions,
    regional_fractions,
):
    """Build extension data for all years."""
    extension_data_dict = {}

    for year in extension_years:
        if year in global_cdr_ext.columns:
            global_year_totals = global_cdr_ext[year].groupby("scenario").first()
            component_year_totals = global_year_totals * component_fractions

            year_values = baseline_data.copy()
            for scenario in baseline_data.index.get_level_values("scenario").unique():
                scenario_mask = baseline_data.index.get_level_values("scenario") == scenario
                if scenario in component_year_totals.index:
                    component_total_year = component_year_totals[scenario]
                    year_values[scenario_mask] = regional_fractions[scenario_mask] * component_total_year
                else:
                    year_values[scenario_mask] = 0

            extension_data_dict[year] = year_values

    return extension_data_dict


def _construct_final_dataframe(cdr_df, extension_data_dict, baseline_year):
    """Construct final DataFrame with original and extension data."""
    if extension_data_dict:
        original_years = [col for col in cdr_df.columns if isinstance(col, int | float) and col <= baseline_year]
        original_data = cdr_df[original_years]

        extension_df = pd.DataFrame(extension_data_dict, index=cdr_df.index)
        all_year_data = pd.concat([original_data, extension_df], axis=1)

        non_year_cols = [col for col in cdr_df.columns if not isinstance(col, int | float)]

        final_df = pd.concat([all_year_data, cdr_df[non_year_cols]], axis=1)
    else:
        final_df = cdr_df.copy()
        print("  ⚠️  No extension data created")

    return final_df
