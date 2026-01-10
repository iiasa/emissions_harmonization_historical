"""Regional and sectoral extension routines for non-CO2 species and helpers to combine results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix

from .extension_functionality import (
    do_simple_sigmoid_or_exponential_extension_to_target,
    sigmoid_function,
)
from .general_utils_for_extensions import add_workflow_level_to_index, glue_with_historical, interpolate_to_annual


def do_simple_sigmoid_extension_to_target(
    scen_full: pd.DataFrame,
    target: float,
    sigmoid_shift=40,
    end_year=2500,
    sigmoid_end_min_year=2150,
) -> np.ndarray:
    """
    Calculate extension function by calling sigmoid functionality to extend

    This function extends a given scenario data using a sigmoid function to reach a specified target value.
    It creates an array of extended values from the start year to the end year, using the sigmoid to smoothly
    transition from the last value in the scenario to the target.

    Parameters
    ----------
    scen_full : pd.DataFrame
        The full scenario data with years as columns and a single row of values.
    target : float
        The target value to extend towards.
    sigmoid_shift : int, optional
        The shift parameter for the sigmoid function, default is 40.
    end_year : int, optional
        The final year to extend to, default is 2500.
    sigmoid_end_min_year : int, optional
        The minimum year for the sigmoid end, default is 2150.

    Returns
    -------
    np.ndarray
        An array of extended values from the start year to end_year.
    """
    full_years = np.arange(scen_full.columns[0], end_year + 1)
    data_extend = np.zeros(len(full_years))
    data_extend[: len(scen_full.columns)] = scen_full.values[0, :]
    data_extend[len(scen_full.columns) :] = sigmoid_function(
        target,
        scen_full.values[0, -1],
        scen_full.columns[-1] + sigmoid_shift,
        sigmoid_end_min_year + sigmoid_shift,
        full_years[len(scen_full.columns) :],
    )
    return data_extend


def get_2100_compound_composition(data_regional: pd.DataFrame, variable: str):
    """
    Find fractional composition of values in 2100 to allocate the residual end point emissions accordingly

    This function calculates the fractional composition of regional data for a given variable in the year 2100.
    It excludes the total variable (e.g., Emissions|CO) and computes fractions based on the sum of other sectors.

    Parameters
    ----------
    data_regional : pd.DataFrame
        Regional data with variables and regions as index, years as columns.
    variable : str
        The variable name to exclude from the total (e.g., 'Emissions|CO').

    Returns
    -------
    pd.DataFrame
        A DataFrame with fractional compositions for each sector and region in 2100.
    """
    data_rest = data_regional.loc[~pix.ismatch(variable=f"{variable}")]
    total_from_sectors = data_rest.sum(axis=0)
    fractions = data_rest.values / total_from_sectors
    fractions_df = pd.DataFrame(data=fractions, columns=["fractions"], index=data_rest.index)
    return fractions_df


def do_single_component_for_scenario_model_regionally(  # noqa: PLR0913, PLR0912
    scen: str,
    model: str,
    variable: str,
    scenarios_regional: pd.DataFrame,
    scenarios_complete_global: pd.DataFrame,
    history: pd.DataFrame,
    global_target=None,
    end_year=2500,
    end_scenario_year=2100,
):
    """
    For given scenario, model and variable, do extensions per sector and region and combine

    This function performs regional extensions for a specific scenario, model, and variable.
    It handles both global and regional data, extending emissions using sigmoid or exponential functions
    to reach targets, and combines the results into a single DataFrame.

    Parameters
    ----------
    scen : str
        The scenario name.
    model : str
        The model name.
    variable : str
        The variable name (e.g., 'Emissions|CO').
    scenarios_regional : pd.DataFrame
        Regional scenario data.
    scenarios_complete_global : pd.DataFrame
        Complete global scenario data.
    history : pd.DataFrame
        Historical data.
    global_target : float, optional
        The global target value. If None, it is calculated from data.
    end_year : int, optional
        The year to extend to, default is 2500.
    end_scenario_year : int, optional
        The end year of the scenario, default is 2100.

    Returns
    -------
    pd.DataFrame
        Extended regional data combined with global extensions.
    """
    if scenarios_regional.shape[0] == 0:
        return pd.DataFrame()
    if scenarios_complete_global.shape[0] == 0:
        return pd.DataFrame()
    data_scenario_global = scenarios_complete_global.loc[
        pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}")
    ]
    data_historical = history.loc[pix.ismatch(variable=f"{variable}")]
    data_min = np.nanmin(data_scenario_global.values[0, :])
    if global_target is None:
        global_target = np.min((data_min, data_historical.values[0, 0]))
    else:
        global_target = np.min((global_target, data_scenario_global.values[0, -1]))

    data_regional = scenarios_regional.loc[pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}**")]

    # For CO the regex above also picks up CO2, so remove that here
    if variable == "Emissions|CO":
        data_regional = data_regional.loc[~pix.ismatch(variable="Emissions|CO2**")]
    data_regional = interpolate_to_annual(data_regional)

    full_years = np.arange(data_regional.columns[0], end_year + 1)

    fractions = get_2100_compound_composition(data_regional[end_scenario_year].copy(), variable)

    sectors = data_regional.pix.unique("variable")
    regions = data_regional.pix.unique("region")

    # Hook for species that are not regional, and just infilled
    if len(regions) <= 1 and len(sectors) <= 1:
        scen_full = interpolate_to_annual(data_scenario_global)
        data_extend = do_simple_sigmoid_or_exponential_extension_to_target(
            scen_full.values[0, :],
            np.arange(scen_full.columns[0], end_year + 1),
            end_scenario_year - int(scen_full.columns[0]),
            global_target,
        )
        df_regional = pd.DataFrame(data=[data_extend], columns=full_years, index=data_scenario_global.index)
        if "workflow" not in df_regional.index.names:
            df_regional = add_workflow_level_to_index(df_regional)
        return df_regional
    temp_list_for_regional = []
    total_sector = None
    world_sector = None
    target_sum = 0
    for sector in sectors:
        if sector == variable:
            continue
        data_sector = data_regional.loc[pix.ismatch(variable=f"{sector}")]
        regions = data_sector.pix.unique("region")
        for region in regions:
            if region == "World" and len(regions) > 1:
                continue
            data = data_sector.loc[pix.ismatch(region=f"{region}")]
            target = global_target * fractions.loc[pix.ismatch(variable=f"{sector}", region=f"{region}")].values[0, 0]
            target_sum = target_sum + target
            data_extend = do_simple_sigmoid_or_exponential_extension_to_target(
                data.values[0, :],
                full_years,
                end_scenario_year - int(data.columns[0]),
                target,
            )
            df_regional = pd.DataFrame(data=[data_extend], columns=full_years, index=data.index)
            if total_sector is None:
                total_sector = data_extend
                world_sector = data_extend
            else:
                total_sector = total_sector + data_extend
                world_sector = total_sector + data_extend

            temp_list_for_regional.append(df_regional)
    df_total = pd.DataFrame(
        data=[world_sector, total_sector],
        columns=full_years,
        index=data_regional.loc[pix.ismatch(variable=f"{variable}")].index,
    )

    temp_list_for_regional.append(df_total)
    df_all = pd.concat(temp_list_for_regional)
    return df_all


def plot_just_global(  # noqa: PLR0913
    scen: str,
    model: str,
    variable: str,
    df_extended: pd.DataFrame,
    scenarios_complete_global: pd.DataFrame,
    history: pd.DataFrame,
    scenarios_regional: pd.DataFrame,
):
    """
    Make global value plots

    This function creates a plot of global values for a given scenario, model, and variable.
    It plots the harmonized historical and scenario data, the extended data,
    and optionally the unextended regional data.

    Parameters
    ----------
    scen : str
        The scenario name.
    model : str
        The model name.
    variable : str
        The variable name.
    df_extended : pd.DataFrame
        The extended data.
    scenarios_complete_global : pd.DataFrame
        Complete global scenario data.
    history : pd.DataFrame
        Historical data.
    scenarios_regional : pd.DataFrame
        Regional scenario data.

    Returns
    -------
    None
        Saves a plot to a PNG file.
    """
    ax = plt.subplot()
    total_harmon = glue_with_historical(
        scenarios_complete_global.loc[pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}")],
        history.loc[pix.ismatch(variable=f"{variable}")],
    )
    ax.plot(total_harmon.columns, total_harmon.values[0, :])
    extended_to_plot = df_extended.loc[pix.ismatch(region="World", variable=f"{variable}")]
    unextended = scenarios_regional.loc[
        pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}", region="World")
    ]
    ax.plot(extended_to_plot.columns, extended_to_plot.values[-1, :])
    if unextended.shape[0] > 0:
        ax.plot(unextended.columns, unextended.values[-1, :], linestyle="--", alpha=0.7)
    plt.savefig(f"extended_match_totals_{scen.replace(' ', '')}_{model.replace(' ', '')}_{variable.split('|')[-1]}.png")
    plt.clf()
