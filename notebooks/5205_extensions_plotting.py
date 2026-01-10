# %%
# +
# ruff: noqa: E402
# -

# %% [markdown]
# # Extensions plotting
#
# ## Imports

# %%

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas_indexing as pix

# Add notebooks directory to path for helper function imports
# When run by papermill, we need to find the notebooks directory relative to the repo root
repo_root = Path.cwd()
notebooks_dir = repo_root / "notebooks"
if notebooks_dir.exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))
elif str(Path.cwd()) not in sys.path:
    # Fallback: add current directory
    sys.path.insert(0, str(Path.cwd()))

from emissions_harmonization_historical.constants_5000 import (
    EXTENSIONS_OUTPUT_DB,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_SCENARIOS_DB,
)
from emissions_harmonization_historical.extensions.cdr_and_fossil_splits import (
    add_removals_and_positive_fossil_emissions_to_historical,
)

# -
from emissions_harmonization_historical.extensions.finish_regional_extensions import merge_historical_future_timeseries

# %% [markdown]
# ## Loading in datasets and preparing for plots

# %%

scenarios_pre_extensions = INFILLED_SCENARIOS_DB.load()  # (pix.isin(stage="complete")).reset_index("stage", drop=True)
scenarios_post_extensions = INFILLED_SCENARIOS_DB.load()
scenarios_harmonised = HARMONISED_SCENARIO_DB.load()
scenarios_ext_out = EXTENSIONS_OUTPUT_DB.load()
harmonised_history = history = HISTORY_HARMONISATION_DB.load(
    pix.ismatch(purpose="global_workflow_emissions")
).reset_index("purpose", drop=True)
history = add_removals_and_positive_fossil_emissions_to_historical(harmonised_history)
scenarios_ext_out.columns = [
    float(col)
    if isinstance(col, int | str) and str(col).replace(".", "").replace("-", "").replace("+", "").isdigit()
    else col
    for col in scenarios_ext_out.columns
]
raw_output = merge_historical_future_timeseries(history, scenarios_ext_out)

# %% [markdown]
# ## Constants

# %%
# Constants
BASELINE_YEAR = 2100
BASELINE_YEAR_FLOAT = 2100.0
YEAR_GRID_COLS = 3
CDR_LIMIT = -1460  # Gt CO2
PROVED_FOSSIL_RESERVES = 2032 + 2400  # Gt CO2
PROBABLE_FOSSIL_RESERVES = 8036 + 2400  # Gt CO2
MAX_YEARS_FOR_MARKERS = 50
make_plots = True

# %% [markdown]
# Marker definitions

# %%
scenario_model_match = {
    "VL": [
        "SSP1 - Very Low Emissions",
        "REMIND-MAgPIE 3.5-4.11",
        "tab:blue",
    ],  # old VLLO
    "LN": ["SSP2 - Low Overshoot_a", "AIM 3.0", "tab:cyan"],  # old VLHO
    "L": ["SSP2 - Low Emissions", "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "tab:green"],
    "ML": ["SSP2 - Medium-Low Emissions", "COFFEE 1.6", "tab:pink"],
    "M": ["SSP2 - Medium Emissions", "IMAGE 3.4", "tab:purple"],
    "H": ["SSP3 - High Emissions", "GCAM 8s", "tab:red"],
    "HL": ["SSP5 - Medium-Low Emissions_a", "WITCH 6.0", "tab:brown"],
}

# %% [markdown]
# Create inverse dictionaries for scenario_model_match

# %%

scenario_to_code = {}  # Maps scenario name to short code
model_to_code = {}  # Maps model name to short code
code_to_color = {}  # Maps short code to color

for code, info in scenario_model_match.items():
    scenario_name, model_name, color = info
    scenario_to_code[scenario_name] = code
    model_to_code[model_name] = code
    code_to_color[code] = color


# %% [markdown]
# Configuration classes for reducing function arguments


# %%
class HistoryPlotConfig:
    """Configuration for plotting scenarios with historical data."""

    def __init__(self, all_years, future_years, axes, colors):
        self.all_years = all_years
        self.future_years = future_years
        self.axes = axes
        self.colors = colors


# %% [markdown]
# ##

# %% [markdown]
# ## Various helper functions
#
# Defineing cdr extension dataframes

# %%
co2_gross_positive_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Gross Positive Emissions")]
co2_beccs_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|BECCS")]
co2_dacc_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Direct Air Capture")]
co2_ocean_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Ocean")]
co2_ew_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Enhanced Weathering")]
co2_soil_carbon_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Soil Carbon Management")]
co2_biochar_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Biochar")]
co2_other_cdr_ext = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Other CDR")]
fossil_extension_df = scenarios_ext_out.loc[pix.ismatch(variable="Emissions|CO2|Energy and Industrial Processes")]


# %%
# CDR components: zero for historical, then future values
def pad_future_with_zeros(df_ext, all_years, future_years, scenario):
    """Pad future extension data with zeros for historical period."""
    zeros = np.zeros(len(all_years) - len(future_years))
    vals = df_ext.loc[df_ext.index.get_level_values("scenario") == scenario][future_years].sum().values
    return np.concatenate([zeros, vals])


def _get_annual_data_for_scenario(scenario, year_cols):
    """Get all annual data for a scenario."""
    gross_pos_annual = co2_gross_positive_ext.loc[
        co2_gross_positive_ext.index.get_level_values("scenario") == scenario
    ][year_cols].sum()

    beccs_annual = co2_beccs_ext.loc[co2_beccs_ext.index.get_level_values("scenario") == scenario][year_cols].sum()

    daccs_annual = co2_dacc_ext.loc[co2_dacc_ext.index.get_level_values("scenario") == scenario][year_cols].sum()

    ocean_annual = co2_ocean_ext.loc[co2_ocean_ext.index.get_level_values("scenario") == scenario][year_cols].sum()

    ew_annual = co2_ew_ext.loc[co2_ew_ext.index.get_level_values("scenario") == scenario][year_cols].sum()

    # Get fossil extension data for comparison
    fossil_annual = fossil_extension_df.loc[fossil_extension_df.index.get_level_values("scenario") == scenario][
        year_cols
    ]
    if len(fossil_annual) > 0:
        fossil_annual = fossil_annual.iloc[0]
    else:
        fossil_annual = None

    return {
        "gross_pos": gross_pos_annual,
        "beccs": beccs_annual,
        "daccs": daccs_annual,
        "ocean": ocean_annual,
        "ew": ew_annual,
        "fossil": fossil_annual,
    }


# %% [markdown]
# ## CO2-related plotting


# %%
def _plot_scenarios_comprehensive(scenarios, year_cols, axes, colors):
    """Plot all scenarios with comprehensive analysis."""
    for i, scenario in enumerate(scenarios):
        _plot_single_scenario_comprehensive(i, scenario, year_cols, axes, colors)

    plt.tight_layout()
    fig = axes[0, 0].figure
    return fig


def _plot_single_scenario_comprehensive(i, scenario, year_cols, axes, colors):
    """Plot a single scenario with annual and cumulative views."""
    # === LEFT COLUMN: ANNUAL FLUXES ===
    ax_annual = axes[i, 0]

    # Get annual data for this scenario - sum over regions
    annual_data = _get_annual_data_for_scenario(scenario, year_cols)

    # Stack the data for annual plot
    years = np.array(year_cols)

    # Positive fluxes (above zero)
    y1_pos = annual_data["gross_pos"].values

    # Negative fluxes (below zero) - stack downwards
    y1_neg = annual_data["beccs"].values
    y2_neg = y1_neg + annual_data["daccs"].values
    y3_neg = y2_neg + annual_data["ocean"].values
    y4_neg = y3_neg + annual_data["ew"].values

    # Plot annual stacked areas
    ax_annual.fill_between(
        years,
        0,
        y1_pos,
        alpha=0.7,
        color=colors["Gross_Positive"],
        label="Gross Positive",
    )
    ax_annual.fill_between(years, 0, y1_neg, alpha=0.7, color=colors["BECCS"], label="BECCS")
    ax_annual.fill_between(years, y1_neg, y2_neg, alpha=0.7, color=colors["DACCS"], label="DACCS")
    ax_annual.fill_between(years, y2_neg, y3_neg, alpha=0.7, color=colors["Ocean"], label="Ocean CDR")
    ax_annual.fill_between(
        years,
        y3_neg,
        y4_neg,
        alpha=0.7,
        color=colors["Enhanced_Weathering"],
        label="Enhanced Weathering",
    )

    # Overlay fossil extension line
    if annual_data["fossil"] is not None:
        ax_annual.plot(
            years,
            annual_data["fossil"].values,
            "k-",
            linewidth=2,
            alpha=0.8,
            label="Net Fossil (Total)",
        )

    # === RIGHT COLUMN: CUMULATIVE FLUXES ===
    ax_cumul = axes[i, 1]

    _plot_cumulative_fluxes(ax_cumul, annual_data, years, colors)

    # === FORMATTING FOR BOTH COLUMNS ===
    _format_both_axes(ax_annual, ax_cumul, scenario, i)


def _plot_cumulative_fluxes(ax_cumul, annual_data, years, colors):
    """Plot cumulative fluxes for a scenario."""
    # Calculate cumulative sums
    gross_pos_cumul = annual_data["gross_pos"].cumsum()
    beccs_cumul = annual_data["beccs"].cumsum()
    daccs_cumul = annual_data["daccs"].cumsum()
    ocean_cumul = annual_data["ocean"].cumsum()
    ew_cumul = annual_data["ew"].cumsum()

    if annual_data["fossil"] is not None:
        fossil_cumul = annual_data["fossil"].cumsum()
    else:
        fossil_cumul = None

    # Stack cumulative data
    y1_pos_cumul = gross_pos_cumul.values
    y1_neg_cumul = beccs_cumul.values
    y2_neg_cumul = y1_neg_cumul + daccs_cumul.values
    y3_neg_cumul = y2_neg_cumul + ocean_cumul.values
    y4_neg_cumul = y3_neg_cumul + ew_cumul.values

    # Plot cumulative stacked areas
    ax_cumul.fill_between(
        years,
        0,
        y1_pos_cumul,
        alpha=0.7,
        color=colors["Gross_Positive"],
        label="Gross Positive",
    )
    ax_cumul.fill_between(years, 0, y1_neg_cumul, alpha=0.7, color=colors["BECCS"], label="BECCS")
    ax_cumul.fill_between(
        years,
        y1_neg_cumul,
        y2_neg_cumul,
        alpha=0.7,
        color=colors["DACCS"],
        label="DACCS",
    )
    ax_cumul.fill_between(
        years,
        y2_neg_cumul,
        y3_neg_cumul,
        alpha=0.7,
        color=colors["Ocean"],
        label="Ocean CDR",
    )
    ax_cumul.fill_between(
        years,
        y3_neg_cumul,
        y4_neg_cumul,
        alpha=0.7,
        color=colors["Enhanced_Weathering"],
        label="Enhanced Weathering",
    )

    # Overlay cumulative fossil line
    if fossil_cumul is not None:
        ax_cumul.plot(
            years,
            fossil_cumul.values,
            "k-",
            linewidth=2,
            alpha=0.8,
            label="Net Fossil (Cumulative)",
        )


def _format_both_axes(ax_annual, ax_cumul, scenario, i):
    """Format both annual and cumulative axes."""
    for ax, title_suffix in [
        (ax_annual, "Annual Fluxes"),
        (ax_cumul, "Cumulative Fluxes"),
    ]:
        ax.set_title(
            scenario_to_code[scenario] + " " + title_suffix,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(
            "CO₂ Flux (Gt CO₂/yr)" if ax == ax_annual else "Cumulative CO₂ (Gt CO₂)",
            fontsize=11,
        )
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2020, 2500)

        # Add vertical line at baseline year
        ax.axvline(x=BASELINE_YEAR, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        # Add legend for first row only
        if i == 0:
            if ax == ax_annual:
                ax.legend(loc="upper right", fontsize=10)


def plot_comprehensive_co2_analysis_with_history():
    """
    Plot annual and cumulative CO₂ fluxes including historical period.

    Gross positive emissions use the full historical+future timeseries.
    CDR components are zero in the historical period.
    """
    # Get all year columns from raw_output
    all_years = [col for col in raw_output.columns if isinstance(col, int | float)]
    all_years.sort()

    # Get future-only year columns (from CDR extension)
    future_years = [col for col in co2_beccs_ext.columns if isinstance(col, int | float)]
    future_years.sort()

    # Get scenarios that exist in all datasets
    scenarios = _get_common_scenarios_with_history()
    n_scenarios = len(scenarios)

    print(
        f"Creating comprehensive flux analysis (with history) for {n_scenarios} scenarios across {len(all_years)} years"
    )

    fig, axes = plt.subplots(n_scenarios, 2, figsize=(10, 3 * n_scenarios))
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    colors = _get_color_scheme_with_afolu()

    config = HistoryPlotConfig(all_years, future_years, axes, colors)
    for i, scenario in enumerate(scenarios):
        _plot_single_scenario_with_history(i, scenario, config)

    plt.tight_layout()
    return fig


def _get_common_scenarios_with_history():
    """Get scenarios common to all datasets including historical."""
    scenarios = (
        set(co2_gross_positive_ext.index.get_level_values("scenario"))
        & set(co2_beccs_ext.index.get_level_values("scenario"))
        & set(co2_dacc_ext.index.get_level_values("scenario"))
        & set(co2_ocean_ext.index.get_level_values("scenario"))
        & set(co2_ew_ext.index.get_level_values("scenario"))
        & set(fossil_extension_df.index.get_level_values("scenario"))
    )
    return sorted(list(scenarios))


def _get_color_scheme_with_afolu():
    """Get color scheme including AFOLU."""
    return {
        "Gross_Positive": "#8B4513",
        "BECCS": "#BEDB3C",
        "DACCS": "#DF23D9",
        "Ocean": "#4D3EBD",
        "Enhanced_Weathering": "#A6A6A6",
        "AFOLU": "#51E390",
    }


def plot_comprehensive_co2_analysis():
    """
    Create dual-column plot showing annual fluxes (left) and cumulative fluxes (right).

    For each scenario, with gross positive and CDR components.
    """
    # Get year columns (numeric only)
    year_cols = [col for col in co2_gross_positive_ext.columns if isinstance(col, int | float)]
    year_cols.sort()

    # Get scenarios that exist in all datasets
    scenarios = (
        set(co2_gross_positive_ext.index.get_level_values("scenario"))
        & set(co2_beccs_ext.index.get_level_values("scenario"))
        & set(co2_dacc_ext.index.get_level_values("scenario"))
        & set(co2_ocean_ext.index.get_level_values("scenario"))
        & set(co2_ew_ext.index.get_level_values("scenario"))
        & set(fossil_extension_df.index.get_level_values("scenario"))
    )

    scenarios = sorted(list(scenarios))
    n_scenarios = len(scenarios)

    print(f"Creating comprehensive flux analysis for {n_scenarios} scenarios across {len(year_cols)} years")

    # Create subplot grid: 2 columns (annual, cumulative), n_scenarios rows
    _fig, axes = plt.subplots(n_scenarios, 2, figsize=(10, 3 * n_scenarios))
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    # Colors for components
    colors = {
        "Gross_Positive": "#8B4513",  # Brown
        "BECCS": "#2E8B57",  # Sea Green
        "DACCS": "#4682B4",  # Steel Blue
        "Ocean": "#20B2AA",  # Light Sea Green
        "Enhanced_Weathering": "#9370DB",  # Medium Purple
    }

    return _plot_scenarios_comprehensive(scenarios, year_cols, axes, colors)


def _plot_single_scenario_with_history(i, scenario, config):
    """Plot a single scenario with historical data."""
    # --- Annual fluxes ---
    ax_annual = config.axes[i, 0]

    # Get historical + future data
    historical_data = _get_historical_data_for_scenario(scenario, config.all_years, config.future_years)
    years = np.array(config.all_years)

    # Plot annual data
    _plot_annual_fluxes_with_history(ax_annual, historical_data, years, config.colors)

    # --- Cumulative fluxes ---
    ax_cumul = config.axes[i, 1]
    _plot_cumulative_fluxes_with_history(ax_cumul, historical_data, years, config.colors)

    # --- Formatting ---
    _format_axes_with_history(ax_annual, ax_cumul, scenario, i, config.all_years)


def _plot_annual_fluxes_with_history(ax_annual, data, years, colors):
    """Plot annual fluxes including historical data."""
    afolu_pos = np.clip(data["afolu"].values[:, 0], 0, None)
    afolu_neg = np.clip(data["afolu"].values[:, 0], None, 0)

    # Stack for annual plot
    y1_pos = data["gross_pos"].values
    y2_pos = afolu_pos + y1_pos
    y1_neg = data["beccs"]
    y2_neg = y1_neg + data["daccs"]
    y3_neg = y2_neg + data["ocean"]
    y4_neg = y3_neg + data["ew"]
    y5_neg = y4_neg + afolu_neg

    # Plot stacked areas
    ax_annual.fill_between(years, 0, y1_pos, alpha=0.7, color=colors["Gross_Positive"], label="Gross FF&I")
    ax_annual.fill_between(years, y1_pos, y2_pos, alpha=0.7, color=colors["AFOLU"], label="AFOLU")
    ax_annual.fill_between(years, 0, y1_neg, alpha=0.7, color=colors["BECCS"], label="BECCS")
    ax_annual.fill_between(years, y1_neg, y2_neg, alpha=0.7, color=colors["DACCS"], label="DACCS")
    ax_annual.fill_between(years, y2_neg, y3_neg, alpha=0.7, color=colors["Ocean"], label="Ocean CDR")
    ax_annual.fill_between(
        years,
        y3_neg,
        y4_neg,
        alpha=0.7,
        color=colors["Enhanced_Weathering"],
        label="Enhanced Weathering",
    )
    ax_annual.fill_between(years, y4_neg, y5_neg, alpha=0.7, color=colors["AFOLU"])

    if data["fossil"] is not None:
        ax_annual.plot(
            years,
            data["fossil"] + data["afolu"].values,
            "k-",
            linewidth=2,
            alpha=0.8,
            label="Net Emissions (Total)",
        )


def _get_historical_data_for_scenario(scenario, all_years, future_years):
    """Get all data for scenario including historical padding."""
    # Gross positive: full historical+future
    gross_pos_annual = (
        raw_output.loc[
            (
                slice(None),
                scenario,
                slice(None),
                "Emissions|CO2|Gross Positive Emissions",
                slice(None),
            ),
            all_years,
        ].sum()
        / 1000
    )

    beccs_annual = pad_future_with_zeros(co2_beccs_ext, all_years, future_years, scenario) / 1000
    daccs_annual = pad_future_with_zeros(co2_dacc_ext, all_years, future_years, scenario) / 1000
    ocean_annual = pad_future_with_zeros(co2_ocean_ext, all_years, future_years, scenario) / 1000
    ew_annual = pad_future_with_zeros(co2_ew_ext, all_years, future_years, scenario) / 1000

    # Fossil and AFOLU data
    fossil_annual = (
        raw_output.loc[
            (
                slice(None),
                scenario,
                slice(None),
                "Emissions|CO2|Energy and Industrial Processes",
                slice(None),
            ),
            all_years,
        ].T
        / 1000
    )
    afolu_annual = (
        raw_output.loc[
            (slice(None), scenario, slice(None), "Emissions|CO2|AFOLU", slice(None)),
            all_years,
        ].T
        / 1000
    )

    return {
        "gross_pos": gross_pos_annual,
        "beccs": beccs_annual,
        "daccs": daccs_annual,
        "ocean": ocean_annual,
        "ew": ew_annual,
        "fossil": fossil_annual,
        "afolu": afolu_annual,
    }


def _plot_cumulative_fluxes_with_history(ax_cumul, data, years, colors):
    """Plot cumulative fluxes including historical data."""
    afolu_pos = np.clip(data["afolu"].values[:, 0], 0, None)
    afolu_neg = np.clip(data["afolu"].values[:, 0], None, 0)

    gross_pos_cumul = np.cumsum(data["gross_pos"].values)
    afolu_cumul = np.cumsum(afolu_pos + afolu_neg)
    beccs_cumul = np.cumsum(data["beccs"])
    daccs_cumul = np.cumsum(data["daccs"])
    ocean_cumul = np.cumsum(data["ocean"])
    ew_cumul = np.cumsum(data["ew"])

    if data["fossil"] is not None:
        fossil_cumul = np.cumsum(data["fossil"])
    else:
        fossil_cumul = None

    y1_pos_cumul = gross_pos_cumul
    y2_pos_cumul = afolu_cumul + y1_pos_cumul
    y1_neg_cumul = beccs_cumul
    y2_neg_cumul = y1_neg_cumul + daccs_cumul
    y3_neg_cumul = y2_neg_cumul + ocean_cumul
    y4_neg_cumul = y3_neg_cumul + ew_cumul

    ax_cumul.fill_between(years, 0, y1_pos_cumul, alpha=0.7, color=colors["Gross_Positive"])
    ax_cumul.fill_between(years, y1_pos_cumul, y2_pos_cumul, alpha=0.7, color=colors["AFOLU"])
    ax_cumul.fill_between(years, 0, y1_neg_cumul, alpha=0.7, color=colors["BECCS"])
    ax_cumul.fill_between(years, y1_neg_cumul, y2_neg_cumul, alpha=0.7, color=colors["DACCS"])
    ax_cumul.fill_between(years, y2_neg_cumul, y3_neg_cumul, alpha=0.7, color=colors["Ocean"])
    ax_cumul.fill_between(
        years,
        y3_neg_cumul,
        y4_neg_cumul,
        alpha=0.7,
        color=colors["Enhanced_Weathering"],
    )

    ax_cumul.plot(
        years,
        fossil_cumul.values[:, 0] + afolu_cumul,
        "k-",
        linewidth=2,
        alpha=0.8,
        label="Net Emissions (Cumulative)",
    )


def _format_axes_with_history(ax_annual, ax_cumul, scenario, i, all_years):
    """Format axes for historical plots."""
    for ax, title_suffix in [
        (ax_annual, "Annual Gross Fluxes"),
        (ax_cumul, "Cumulative Gross Fluxes"),
    ]:
        ax.set_title(
            scenario_to_code[scenario] + " " + title_suffix,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(
            "CO₂ Flux (Gt CO₂/yr)" if ax == ax_annual else "Cumulative CO₂ (Gt CO₂)",
            fontsize=11,
        )
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(all_years[0], all_years[-1])
        ax.axvline(x=BASELINE_YEAR, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=2)

        if ax == ax_cumul:
            ax.axhline(
                y=CDR_LIMIT,
                color="green",
                linestyle="-",
                alpha=0.3,
                linewidth=3,
                label="Cumulative CDR limit",
            )
            ax.axhline(
                y=PROVED_FOSSIL_RESERVES,
                color="red",
                linestyle="-",
                alpha=0.3,
                linewidth=3,
                label="Proved Fossil Reserves",
            )
            ax.axhline(
                y=PROBABLE_FOSSIL_RESERVES,
                color="red",
                linestyle="--",
                alpha=0.3,
                linewidth=3,
                label="Proved + Probable Fossil Reserves",
            )
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)


# %% [markdown]
# ## Create the comprehensive plot with history

# %%
fig_comprehensive_history = plot_comprehensive_co2_analysis_with_history()
plt.show()


# %%
# Create a zoomed-in plot focusing on the historical-future transition period
def plot_co2_transition_period(data, scenario_colors=None, start_year=1990, end_year=2150):
    """
    Plot total CO2 emissions focusing on the historical-future transition period.
    """
    print(f"=== PLOTTING CO2 TRANSITION PERIOD ({start_year}-{end_year}) ===")

    # Filter for CO2 variables and World region
    co2_vars = ["Emissions|CO2|AFOLU", "Emissions|CO2|Energy and Industrial Processes"]
    available_vars = data.index.get_level_values("variable").unique()
    co2_vars_in_data = [var for var in co2_vars if var in available_vars]

    # Filter data
    co2_data = data.loc[
        (data.index.get_level_values("variable").isin(co2_vars_in_data))
        & (data.index.get_level_values("region") == "World")
    ]

    # Get years in the specified range
    all_years = [col for col in co2_data.columns if isinstance(col, int | float)]
    years = [year for year in all_years if start_year <= year <= end_year]
    years = sorted(years)

    # Group by model and scenario
    scenarios = co2_data.index.droplevel(["region", "variable", "unit", "workflow"]).drop_duplicates()

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use scenario colors if provided
    if scenario_colors is None:
        scenario_colors = scenario_model_match

    for i, (model, scenario) in enumerate(scenarios):
        # Get data for this scenario
        scenario_data = co2_data.loc[
            (co2_data.index.get_level_values("model") == model)
            & (co2_data.index.get_level_values("scenario") == scenario)
        ]

        # Sum across CO2 variables
        if len(scenario_data) > 1:
            total_emissions = scenario_data[years].sum(axis=0)
        else:
            total_emissions = scenario_data[years].iloc[0]

        # Find the marker code and color
        marker_code = None
        color = f"C{i}"
        for marker, info in scenario_colors.items():
            if info[1] == model and info[0] == scenario:
                marker_code = marker
                color = info[2]
                break

        label = f"{marker_code} ({model})" if marker_code else f"{model}"

        # Plot with markers to show data points
        ax.plot(
            years,
            total_emissions.values,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=0.8,
            marker="o" if len(years) < MAX_YEARS_FOR_MARKERS else None,
            markersize=3 if len(years) < MAX_YEARS_FOR_MARKERS else 0,
        )

    # Add vertical line at historical/future boundary
    ax.axvline(
        x=2023,
        color="red",
        linestyle="--",
        alpha=0.8,
        label="Historical/Future boundary",
        linewidth=2,
    )

    # Add shaded regions for historical vs future
    ax.axvspan(start_year, 2023, alpha=0.1, color="blue", label="Historical")
    ax.axvspan(2023, end_year, alpha=0.1, color="orange", label="Future projections")

    # Formatting
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Total CO2 Emissions (Mt CO2/yr)", fontsize=12)
    ax.set_title(
        f"Total CO2 Emissions: Historical-Future Transition ({start_year}-{end_year})",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlim(start_year, end_year)
    ax.set_ylim(bottom=0)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    # Add decade markers
    decade_years = [year for year in range(start_year, end_year + 1, 10) if year in years]
    for year in decade_years:
        ax.axvline(x=year, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    plt.show()

    return fig, ax


# %% [markdown]
# ## Plotting various

# %% [markdown]
# Create the transition period plot

# %%
if make_plots:
    fig_transition, ax_transition = plot_co2_transition_period(raw_output, scenario_model_match)

# %% [markdown]
# Create the comprehensive plot

# %%
if make_plots:
    fig_comprehensive = plot_comprehensive_co2_analysis()
    plt.show()
