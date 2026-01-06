# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
# ruff: noqa: E402

# %% [markdown]
# # Extensions of Marker Scenarios

# %% [markdown]
# Regular imports

# %%
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import tqdm.auto

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
    EXTENSIONS_OUT_DIR,
    EXTENSIONS_OUTPUT_DB,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_SCENARIOS_DB,
    INFILLED_SCENARIOS_DB_2100,
)

# Package imports
from emissions_harmonization_historical.extensions.afolu_extension_functions import (
    get_cumulative_afolu,
    get_cumulative_afolu_fill_from_hist,
)
from emissions_harmonization_historical.extensions.cdr_and_fossil_splits import (
    add_removals_and_positive_fossil_emissions_to_historical,
    extend_cdr_components_vectorized,
    get_2100_compound_composition_co2,
)

# from emissions_harmonization_historical.constants import DATA_ROOT
# from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.extensions.extension_functionality import (
    extend_linear_rampdown,
    sigmoid_function,
)
from emissions_harmonization_historical.extensions.extensions_functions_for_non_co2 import (
    do_single_component_for_scenario_model_regionally,
    plot_just_global,
)
from emissions_harmonization_historical.extensions.finish_regional_extensions import (
    extend_regional_for_missing,
    merge_historical_future_timeseries,
)
from emissions_harmonization_historical.extensions.fossil_co2_storyline_functions import (
    extend_co2_for_scen_storyline,
)
from emissions_harmonization_historical.extensions.general_utils_for_extensions import (
    dump_data_per_model,
    fix_up_and_concatenate_extensions,
    glue_with_historical,
    interpolate_to_annual,
    save_continuous_timeseries_to_csv,
)

# Constants
FUTURE_START_YEAR = 2023.0
HISTORICAL_START_YEAR = 1900
SCENARIO_END_YEAR = 2100
TUPLE_LENGTH_WITH_STAGE = 6

# %% tags=["parameters"]
# Papermill parameters
make_plots: bool = False

# %% [markdown]
# More preamble

# %%
save_plots = make_plots
dump_with_full_scenario_names = True

pandas_openscm.register_pandas_accessor()

UR = openscm_units.unit_registry
Q = UR.Quantity

# %% [markdown]
# ## Loading scenarios

# %%
scenarios_complete_global = INFILLED_SCENARIOS_DB_2100.load(pix.isin(stage="complete")).reset_index("stage", drop=True)
scenarios_complete_global  # TODO: drop 2100 end once we have usable scenario data post-2100
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

scenarios_regional = HARMONISED_SCENARIO_DB.load()
history_regional = HISTORY_HARMONISATION_DB.load()


# %%
scenarios_regional.pix.unique("region")

# %%
unique_model_scenario_pairs = scenarios_complete_global.index.droplevel(
    ["region", "variable", "unit"]
).drop_duplicates()

print(f"Number of unique model-scenario pairs: {len(unique_model_scenario_pairs)}")
print("\nUnique model-scenario pairs:")
for i, (model, scenario) in enumerate(unique_model_scenario_pairs, 1):
    print(f"{i:2d}. {model} | {scenario}")

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

# %%
scenarios_regional = scenarios_regional.sort_index(axis="columns").T.interpolate("index").T

fractions_fossil_total = {}
for model, scen in unique_model_scenario_pairs.to_list():
    print(f"Processing {model} | {scen}")
    tot_co2 = scenarios_complete_global.loc[pix.ismatch(scenario=scen, model=model, variable="Emissions|CO2")]
    scen_here = scenarios_regional.loc[pix.ismatch(scenario=scen, model=model, variable="Emissions|CO2**")]
    fractions_list = get_2100_compound_composition_co2(scen_here[2100])
    fractions_fossil_total[(model, scen)] = {
        "fractions_tot_fossil": fractions_list[0],
        "fractions_cdr": fractions_list[1],
        "fractions_fossil_nocdr": fractions_list[2],
    }

# %% [markdown]
# Finally get cumulative CO2 history

# %%
cumulative_history_afolu = get_cumulative_afolu(history, "GCB-extended", "historical")

# %% [markdown]
# ## Main block for AFOLU


# %%
# AFOLU section
def calculate_afolu_extensions(scenarios_complete_global, history, cumulative_history_afolu, plot=True):
    """
    Calculate AFOLU extensions for all scenarios and models
    """
    if plot:
        _fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))
    temp_list_for_new_data_linear_ramp_down = []
    for s, meta in scenario_model_match.items():
        scen = scenarios_complete_global.loc[pix.ismatch(variable="**CO2|AFOLU", model=meta[1], scenario=meta[0])]
        scen_full = glue_with_historical(scen, history.loc[pix.ismatch(variable="Emissions|CO2|AFOLU")])
        cumulative_2100 = get_cumulative_afolu_fill_from_hist(scen, meta[1], meta[0], cumulative_history_afolu)
        em_ext_linear_ramp_down = extend_linear_rampdown(
            scen_full.values[0, :], np.arange(cumulative_2100.columns[0], 2501)
        )
        if plot:
            axs[0].plot(cumulative_2100.columns, cumulative_2100.values[0, :], color=meta[2])
            axs[0].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                np.cumsum(em_ext_linear_ramp_down),
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
            axs[1].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
            axs[1].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                em_ext_linear_ramp_down,
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
            axs[2].plot(cumulative_2100.columns, cumulative_2100.values[0, :], color=meta[2])
            axs[2].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                np.cumsum(em_ext_linear_ramp_down),
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
            axs[3].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
            axs[3].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                em_ext_linear_ramp_down,
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
        df_afolu_linear_ramp_down = pd.DataFrame(
            data=[em_ext_linear_ramp_down],
            columns=np.arange(cumulative_2100.columns[0], 2501),
            index=scen.index,
        )
        temp_list_for_new_data_linear_ramp_down.append(df_afolu_linear_ramp_down)
    extended_data_afolu_linear_ramp_down = pd.concat(temp_list_for_new_data_linear_ramp_down)

    if plot:
        for ax in axs:
            ax.set_xlabel("Year")
            ax.legend()
            ax.axvline(x=2100, ls="--", color="k")
        axs[0].set_ylabel("Cumulative Emissions CO2 AFOLU")
        axs[0].set_title("Cumulative Emissions CO2 AFOLU")
        axs[1].set_ylabel("Emissions CO2 AFOLU")
        axs[1].set_title("Emissions CO2 AFOLU")
        axs[2].set_ylabel("Cumulative Emissions CO2 AFOLU")
        axs[2].set_title("Cumulative Emissions CO2 AFOLU")
        axs[3].set_ylabel("Emissions CO2 AFOLU")
        axs[3].set_title("Emissions CO2 AFOLU")
        axs[2].set_xlim(2000, 2300)
        axs[3].set_xlim(2000, 2300)

        plt.savefig("afolu_first_draft_extensions.png")
    return {
        "linear_afolu_rampdown": extended_data_afolu_linear_ramp_down,
    }


# %% [markdown]
# ## Non-CO2 functionality

# %% [markdown]
# First defining some non-zero end-points for certain gases per marker:

# %%
component_global_targets = {
    "Emissions|CH4": {
        "VL": 95.0,
        "LN": 150.0,
        "L": 95.0,
        "ML": 120.0,
        "M": 450.0,
        "H": 520.0,
        "HL": 110.0,
    },
    "Emissions|Sulfur": {
        "VL": 20.0,
        "LN": 10.0,
        "L": None,
        "ML": 20.0,
        "M": 20.0,
        "H": 50.0,
        "HL": 10.0,
    },
}

# %% [markdown]
# Main functionality for all non-co2 extensions


# %%
def do_all_non_co2_extensions(scenarios_complete_global, history):  # noqa: PLR0912
    """
    Extend all non-CO2 emission variables across scenarios and models using historical data and global targets.

    Iterates over all emission variables (excluding CO2) in the provided global scenarios dataset, matches them with
    scenario-model pairs, and applies regional extension logic. For each variable and scenario-model pair, it computes
    the extended data using historical values and optional global targets, then aggregates the results.
    Optionally, generates diagnostic plots for each variable and scenario-model pair.

    Parameters
    ----------
    scenarios_complete_global : pyam.IamDataFrame
        Complete global scenarios dataset containing emission variables.
    history : pyam.IamDataFrame
        Historical emissions data for matching and extension.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all extended non-CO2 emission variables across scenarios and models.
    """
    total_df_list = []
    look_at_all = False

    for variable in tqdm.auto.tqdm(scenarios_complete_global.pix.unique("variable").values):
        print(variable)
        # print(history.loc[pix.ismatch(variable=f"{variable}")].shape)
        if variable.startswith("Emissions|CO2"):
            continue
        if history.loc[pix.ismatch(variable=f"{variable}")].shape[0] < 1:
            continue
        for s, meta in tqdm.auto.tqdm(scenario_model_match.items()):
            if variable in component_global_targets.keys():
                global_target = component_global_targets[variable][s]
            else:
                global_target = None
            print(f"{s}: {meta}, target: {global_target}")
            df_comp_scen_model = do_single_component_for_scenario_model_regionally(
                meta[0],
                meta[1],
                variable,
                scenarios_regional,
                scenarios_complete_global,
                history,
                global_target=global_target,
            )
            # if "workflow" in df_comp_scen_model.index.names:
            #     print("Dropping workflow level from index")
            #     df_comp_scen_model = df_comp_scen_model.droplevel(["workflow"])
            # # sys.exit(4)
            if "workflow" not in df_comp_scen_model.index.names:
                print(f"Workflow missing for {s}: {meta}, {variable}, adding level")
                print(df_comp_scen_model)
                sys.exit(4)
            total_df_list.append(df_comp_scen_model)
            # print(df_comp_scen_model.columns)
            if look_at_all:
                pdf = df_comp_scen_model.openscm.to_long_data()
                fg = sns.relplot(
                    data=pdf,
                    x="time",
                    y="value",
                    col="variable",
                    col_order=sorted(pdf["variable"].unique()),
                    col_wrap=2,
                    hue="region",
                    hue_order=sorted(pdf["region"].unique()),
                    kind="line",
                    linewidth=2.0,
                    alpha=0.7,
                    facet_kws=dict(sharey=False),
                    errorbar=None,
                )
                for ax in fg.axes.flatten():
                    if "CO2" in ax.get_title():
                        ax.axhline(0.0, linestyle="--", color="gray")
                    else:
                        ax.set_ylim(ymin=0.0)
                        ax.axvline(2100, linestyle="--", color="gray")
                    if ax.get_title().endswith("Emissions|BC"):
                        ax.axhline(2.0814879929813928, linestyle="--", color="gray")
                    # ax.set_xticks(np.arange(2020, 2, 10))
                    ax.grid()
                # fg.savefig(f"regionally_extended_{variable.split('|')[-1]}_{meta[0].replace(' ', '')}
                # _{meta[1].replace(' ', '')}.png")
                plt.show()
                plt.clf()
                plt.close()
            else:
                plot_just_global(
                    meta[0],
                    meta[1],
                    variable,
                    df_comp_scen_model.loc[pix.ismatch(region="World", variable=f"{variable}")],
                    scenarios_complete_global,
                    history,
                    scenarios_regional=scenarios_regional,
                )
    df_all = pix.concat(total_df_list)
    return df_all


# %% [markdown]
# ## Do main block of non-fossil CO2 extensions first

# %%
# Set this to true if running for the first time to generate CSVs
# Otherwise you can set to false to speed-up by not running throuhg
# all the non-CO2 and afolu extensions again
do_and_write_to_csv = False
if do_and_write_to_csv:
    df_all = do_all_non_co2_extensions(scenarios_complete_global, history)
    df_all.to_csv("first_draft_extended_nonCO2_all.csv")
    afolu_dfs = calculate_afolu_extensions(scenarios_complete_global, history, cumulative_history_afolu, plot=True)
    # print(df_all)
    for name, afolu_df in afolu_dfs.items():
        afolu_df.to_csv(f"first_draft_extended_afolu_{name}.csv")


# %%
if not do_and_write_to_csv:
    df_all = pd.read_csv("first_draft_extended_nonCO2_all.csv", index_col=[0, 1, 2, 3, 4, 5])
afolu_dfs = {}
for afolu_file in glob.glob("first_draft_extended_afolu_linear*.csv"):
    print("Reading " + afolu_file)
    name = afolu_file.split("first_draft_extended_")[-1].split(".csv")[0]

    afolu_dfs[name] = pd.read_csv(afolu_file)


# %% [markdown]
# # Total CO2 Storyline dictionaries
# These dictionaries define how total CO2 emissions evolve from 2023 to 2500
# Each storyline type has specific parameters that control the transition phases
# ## Storyline Types (from extensions_fossil_co2_storyline_functions.py):
# - "CS": Constant-then-Sigmoid - holds constant emissions, then smooth transition to zero
# - "ECS": Exponential/linear-then-Constant-then-Sigmoid - initial decay/growth, plateau, then transition to zero
# - "CSCS": Constant-Sigmoid-Constant-Sigmoid - two-phase transition with intermediate plateau
# ## Parameter meanings:
# ### CS storyline: ["CS", stop_const, end_sig, roll_in, roll_out]
# - stop_const: year when constant phase ends
# - end_sig: year when sigmoid transition to zero completes
# - roll_in: years for smooth roll-in to sigmoid (transition smoothing)
# - roll_out: years for smooth roll-out from sigmoid (transition smoothing)
# ### ECS storyline: ["ECS", exp_end, exp_targ, sig_start, sig_end, roll_in, roll_out]
# - exp_end: year when initial exponential/linear phase ends
# - exp_targ: target emission value at exp_end (None = auto-calculated from data trend)
# - sig_start: year when sigmoid transition begins
# - sig_end: year when sigmoid transition to zero completes
# - roll_in, roll_out: transition smoothing parameters (years)
# ### CSCS storyline: ["CSCS", stop_const, sig_targ, end_sig1, start_sig2, end_sig2, roll_in, roll_out]
# - stop_const: year when first constant phase ends
# - sig_targ: target value for intermediate plateau (between two sigmoids)
# - end_sig1: year when first sigmoid completes
# - start_sig2: year when second sigmoid begins
# - end_sig2: year when final sigmoid to zero completes
# - roll_in, roll_out: transition smoothing parameters (years)

# %%
# ["CS",stop_const,end_sig,roll_in,roll_out]
# ["ECS",exp_end,exp_targ,sig_start,sig_end,roll_in,roll_out]
# ["CSCS",stop_const,sig_targ,end_sig1,start_sig2,end_sig2,roll_in,roll_out]

fossil_evolution_dictionary = {
    "VL": ["ECS", 2200, -3.5e3, 2450, 2500, 20, 20],
    "LN": ["ECS", 2120, -24e3, 2200, 2300, 20, 20],
    "L": ["ECS", 2160, None, 2160, 2260, 40, 20],
    "ML": ["ECS", 2150, -13e3, 2230, 2300, 20, 20],
    "M": ["CS", 2100, 2240, 20, 20],
    "H": ["ECS", 2150, None, 2175, 2300, 20, 20],
    "HL": ["ECS", 2150, -22e3, 2200, 2300, 20, 20],
}


# %% [markdown]
# Looping over afolu variants to get CO2

# %%
name = "afolu_linear_afolu_rampdown"
df_afolu = afolu_dfs[name]
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 15))
temp_list_for_new_data = []
for s, meta in scenario_model_match.items():
    print(f"Processing fossil CO2 to match storyline and AFOLU for {s}")
    co2_fossil = interpolate_to_annual(
        scenarios_complete_global.loc[
            pix.ismatch(
                variable="Emissions|CO2|Energy and Industrial Processes",
                model=meta[1],
                scenario=meta[0],
            )
        ]
    )
    year_cols = [
        col
        for col in co2_fossil.columns
        if (isinstance(col, int | float) and col >= FUTURE_START_YEAR)
        or (re.match(r"^\d{4}(?:\.0)?$", str(col)) and float(col) >= FUTURE_START_YEAR)
    ]
    non_year_cols = [
        col
        for col in co2_fossil.columns
        if not (
            (isinstance(col, int | float) and (HISTORICAL_START_YEAR <= col <= SCENARIO_END_YEAR))
            or re.match(r"^\d{4}(?:\.0)?$", str(col))
        )
    ]
    co2_fossil = co2_fossil[non_year_cols + year_cols]
    co2_afolu = df_afolu.loc[(df_afolu["model"] == meta[1]) & (df_afolu["scenario"] == meta[0])]

    co2_total_extend, co2_fossil_extend, extend_years = extend_co2_for_scen_storyline(
        co2_afolu, co2_fossil, fossil_evolution_dictionary[s]
    )

    df_total = pd.DataFrame(data=[co2_fossil_extend], columns=extend_years, index=co2_fossil.index)
    temp_list_for_new_data.append(df_total)

    axs[0].plot(co2_fossil.columns, co2_fossil.values.flatten(), label=s, color=meta[2])
    axs[0].plot(extend_years, co2_fossil_extend, label=s, color=meta[2], linestyle="--")
    axs[0].plot(co2_fossil.columns, co2_fossil.values.flatten(), label=s, color=meta[2])
    axs[1].plot(
        extend_years,
        co2_afolu.loc[:, "2023":].to_numpy().flatten(),
        label=s,
        color=meta[2],
        linestyle="--",
    )
    axs[2].plot(extend_years, co2_total_extend, label=s, color=meta[2], linestyle="--")
    axs[2].plot(
        co2_fossil.columns,
        co2_fossil.values.flatten() + co2_afolu.loc[:, "2023":"2100"].to_numpy().flatten(),
        label=s,
        color=meta[2],
    )

fossil_extension_df = pd.concat(temp_list_for_new_data)
fossil_extension_df.to_csv(f"co2_fossil_fuel_extenstions_{name}.csv")
axs[0].set_title("CO2 fossil", fontsize="x-large")
axs[1].set_title("CO2 AFOLU", fontsize="x-large")
axs[2].set_title("CO2 total", fontsize="x-large")
for ax in axs:
    ax.set_xlabel("Years", fontsize="x-large")
axs[2].legend(fontsize="x-large")

plt.savefig(f"co2_fossil_fuel_extenstions_{name}.png")

# %% [markdown]
# # Dataframe cleanup

# %%
# Convert year columns in df_afolu_fixed to floats (if possible)
year_cols = [col for col in df_all.columns if str(col).isdigit()]
df_all.rename(columns={col: float(col) for col in year_cols}, inplace=True)
df_all.head()

# %%
# Convert year columns in df_afolu_fixed to floats (if possible)
year_cols = [col for col in df_afolu.columns if str(col).isdigit()]
df_afolu.rename(columns={col: float(col) for col in year_cols}, inplace=True)
df_afolu.head()

# %% [markdown]
# ## Removal disaggregation

# %%

co2_beccs = interpolate_to_annual(scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2|BECCS")])
co2_dacc = interpolate_to_annual(scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2|Direct Air Capture")])
co2_ocean = interpolate_to_annual(scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2|Ocean")])
co2_ew = interpolate_to_annual(scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2|Enhanced Weathering")])
co2_biochar = interpolate_to_annual(scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2|Biochar")])
co2_soil = interpolate_to_annual(scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2|Soil Carbon Management")])
co2_othercdr = interpolate_to_annual(scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2|Other CDR")])
co2_ffi = interpolate_to_annual(
    scenarios_regional.loc[
        pix.ismatch(
            variable="Emissions|CO2|Energy and Industrial Processes",
            workflow="for_scms",
        )
    ]
)
co2_cdr = (
    co2_dacc
    + co2_ocean.values
    + co2_ew.values
    + co2_beccs.values
    + co2_biochar.values
    + co2_soil.values
    + co2_othercdr.values
)

# Get the current index as a list of tuples
current_index = list(co2_cdr.index)

# Update the variable name in each tuple (variable is at position 3 in the tuple)
new_index = []
for idx_tuple in current_index:
    new_tuple = list(idx_tuple)
    new_tuple[3] = "Emissions|CO2|Gross Removals"  # Replace variable name
    new_index.append(tuple(new_tuple))

# Create new MultiIndex with updated variable name
co2_cdr.index = pd.MultiIndex.from_tuples(new_index, names=co2_cdr.index.names)
co2_cdr.head()
global_cdr = co2_cdr.groupby(["model", "scenario", "variable", "unit"]).sum()


# %% [markdown]
# Calculate Gross Positive Emissions

# %%

co2_ffi_grouped = co2_ffi.groupby(["model", "scenario", "variable", "unit"]).sum()

# Find common index components (excluding variable)
common_scenarios = []
for model, scenario, var_cdr, unit in global_cdr.index:
    # Look for matching model/scenario/unit in co2_ffi_grouped (with different variable)
    matching_ffi = co2_ffi_grouped.loc[
        (co2_ffi_grouped.index.get_level_values("model") == model)
        & (co2_ffi_grouped.index.get_level_values("scenario") == scenario)
        & (co2_ffi_grouped.index.get_level_values("unit") == unit)
    ]

    if len(matching_ffi) > 0:
        # Get the CDR data for this scenario
        cdr_data = global_cdr.loc[(model, scenario, var_cdr, unit)]
        # Get the FFI data for this scenario
        ffi_data = matching_ffi.iloc[0]  # Should be only one row

        # Add the two timeseries (CDR + FFI)
        combined_data = -cdr_data + ffi_data

        # Create the new index with updated variable name
        new_index = (model, scenario, "Emissions|CO2|Gross Positive Emissions", unit)
        common_scenarios.append((new_index, combined_data))

# Create the combined DataFrame
if common_scenarios:
    indices, data_rows = zip(*common_scenarios)
    co2_gross_positive = pd.DataFrame(
        data=list(data_rows),
        index=pd.MultiIndex.from_tuples(indices, names=global_cdr.index.names),
    )

else:
    print("No common scenarios found between CDR and FFI data!")
    co2_gross_positive = None

# %%
removal_dictionary = {
    "VL": ["NEG", 100, 60],
    "LN": ["NEG", 50, 100],
    "L": ["NEG", 50, 50],
    "ML": ["NEG", 100, 0],
    "M": ["POS"],
    "H": ["POS"],
    "HL": ["NEG", 80, 20],
}

# %%
# --- Extension of co2_gross_positive and global_cdr to 2500 using rule-based logic ---


# Extension configuration
last_year = 2100
target_year = 2500
years_extension = np.arange(last_year + 1, target_year + 1)

# Initialize extension DataFrames with all new columns at once
# Create empty DataFrames for the extension years with same index
extension_cols_gross_pos = pd.DataFrame(np.nan, index=co2_gross_positive.index, columns=years_extension)
extension_cols_cdr = pd.DataFrame(np.nan, index=global_cdr.index, columns=years_extension)

# Concatenate original data with extension columns
co2_gross_positive_ext = pd.concat([co2_gross_positive, extension_cols_gross_pos], axis=1)
global_cdr_ext = pd.concat([global_cdr, extension_cols_cdr], axis=1)

print(f"Extension setup complete. Extending from {last_year + 1} to {target_year}")
print(f"Number of extension years: {len(years_extension)}")

# Map removal_dictionary keys to actual scenario names
removal_strategy_map = {}
for marker, info in scenario_model_match.items():
    scenario = info[0]  # Get the full scenario name
    if marker in removal_dictionary:
        removal_strategy_map[scenario] = removal_dictionary[marker]

print(f"Mapped removal dictionary to {len(removal_strategy_map)} scenarios")

# Apply extension strategies to each scenario
processed_count = 0
for idx in co2_gross_positive.index:
    model, scenario, variable, unit = idx

    # Get extension strategy and parameters
    if scenario not in removal_strategy_map:
        print(f"Scenario {scenario} not in removal_dictionary, skipping.")
        continue

    strategy_info = removal_strategy_map[scenario]
    strategy = strategy_info[0]

    print(f"Processing {model}, {scenario} with strategy: {strategy}")

    # Get fossil extension data for this scenario/model
    try:
        fossil_row = fossil_extension_df.loc[
            (
                model,
                scenario,
                "World",
                "Emissions|CO2|Energy and Industrial Processes",
                unit,
            )
        ]
    except KeyError:
        print(f"No fossil extension for {model}, {scenario}, skipping.")
        continue

    # Get 2100 baseline values
    cdr_idx = (model, scenario, "Emissions|CO2|Gross Removals", unit)
    cdr_2100 = global_cdr.loc[cdr_idx, 2100.0]
    gross_pos_2100 = co2_gross_positive.loc[idx, 2100.0]

    if strategy == "POS":
        # POS strategy: CDR remains constant, gross positive adjusts to match fossil trajectory
        print(f"  Using POS strategy: CDR constant at {cdr_2100:.2f}")
        cdr_extension = np.full(len(years_extension), cdr_2100)
        fossil_vals = fossil_row[years_extension].values
        gross_pos_extension = fossil_vals - cdr_2100

    elif strategy == "NEG":
        # NEG strategy: Gross positive follows sigmoid decay, CDR adjusts as residual
        decay_timescale = strategy_info[1]
        offset = strategy_info[2]
        print(f"  Using NEG strategy with decay_timescale={decay_timescale}, offset={offset}")
        offset_shift = (1.25 * decay_timescale) / 2
        gross_pos_extension = sigmoid_function(
            0,
            gross_pos_2100,
            years_extension[0] + offset - offset_shift,
            years_extension[0] + offset + offset_shift,
            years_extension,
            adjust_from=True,
        )
        # Calculate CDR as residual to match fossil trajectory
        fossil_vals = fossil_row[years_extension].values
        cdr_extension = fossil_vals - gross_pos_extension

    else:
        print(f"Unknown strategy {strategy} for scenario {scenario}, skipping.")
        continue

    # Apply extensions to DataFrames using vectorized assignment
    co2_gross_positive_ext.loc[idx, years_extension] = gross_pos_extension
    global_cdr_ext.loc[cdr_idx, years_extension] = cdr_extension

    processed_count += 1

print(f"\nProcessed {processed_count} scenarios")

# %%
# Create a sanity check plot: stacked area plot of positive and negative CO2 components
# with separate subplots for each scenario

# Get year columns for plotting
years = [col for col in co2_gross_positive_ext.columns if isinstance(col, int | float)]
years = sorted(years)

years_extension = [col for col in fossil_extension_df.columns if isinstance(col, int | float)]
years_extension = sorted(years_extension)
# Define consistent colors
positive_color = "tab:brown"
negative_color = "tab:green"

# Get unique scenarios from our data
scenarios_to_plot = []
for model, scenario, var, unit in co2_gross_positive_ext.index:
    if (model, scenario) not in scenarios_to_plot:
        scenarios_to_plot.append((model, scenario))

# Calculate grid size dynamically based on number of scenarios
PLOT_GRID_COLS = 4  # Number of columns in the subplot grid
n_scenarios = len(scenarios_to_plot)
n_rows = (n_scenarios + PLOT_GRID_COLS - 1) // PLOT_GRID_COLS  # Ceiling division
print(f"Creating {n_rows}x{PLOT_GRID_COLS} grid for {n_scenarios} scenarios")

fig, axes = plt.subplots(n_rows, PLOT_GRID_COLS, figsize=(20, 6 * n_rows))
axes = axes.flatten()  # Make it easier to iterate

# Plot for each scenario in its own subplot
for i, (model, scenario) in enumerate(scenarios_to_plot):
    ax = axes[i]

    # Get data for this scenario
    gross_positive_data = co2_gross_positive_ext.loc[
        (model, scenario, "Emissions|CO2|Gross Positive Emissions", "Mt CO2/yr"), years
    ]
    cdr_data = global_cdr_ext.loc[(model, scenario, "Emissions|CO2|Gross Removals", "Mt CO2/yr"), years]

    # Get corresponding FFI data for comparison
    ffi_data = fossil_extension_df.loc[
        (fossil_extension_df.index.get_level_values("model") == model)
        & (fossil_extension_df.index.get_level_values("scenario") == scenario)
        & (fossil_extension_df.index.get_level_values("unit") == "Mt CO2/yr")
    ]

    if len(ffi_data) > 0:
        ffi_values = ffi_data.iloc[0][years]

        # Find marker code for this scenario
        marker_code = None
        for marker, info in scenario_model_match.items():
            if info[1] == model and info[0] == scenario:
                marker_code = marker
                break

        # Plot stacked areas with consistent colors
        ax.fill_between(
            years,
            0,
            gross_positive_data.values,
            alpha=0.6,
            color=positive_color,
            label="Gross Positive",
        )
        ax.fill_between(
            years,
            0,
            cdr_data.values,
            alpha=0.6,
            color=negative_color,
            label="CDR (negative)",
        )

        # Overlay the net FFI line for comparison
        ax.plot(
            years_extension,
            ffi_data.T.values,
            color="black",
            linewidth=2,
            linestyle="-",
            alpha=0.8,
            label="Net FFI",
        )

        # Add vertical line at 2023 (historical/future boundary)
        ax.axvline(x=2023, color="red", linestyle="--", alpha=0.7, linewidth=1)

        # Add horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        # Formatting for each subplot
        ax.set_title(f"{marker_code}: {scenario[:30]}...", fontsize=12, fontweight="bold")
        ax.set_xlim(min(years), 2300)
        ax.grid(True, alpha=0.3)

        # Only add x-labels to bottom row
        if i >= PLOT_GRID_COLS:
            ax.set_xlabel("Year", fontsize=10)

        # Only add y-labels to left column
        if i % PLOT_GRID_COLS == 0:
            ax.set_ylabel("CO2 Emissions (Mt CO2/yr)", fontsize=10)

        # Add legend to first subplot only
        if i == 0:
            ax.legend(fontsize=9)

# Hide unused subplots
for i in range(len(scenarios_to_plot), len(axes)):
    axes[i].set_visible(False)

# Add overall title
fig.suptitle(
    "Gross Positive vs CDR vs Net FFI Emissions by Scenario\nBrown = Positive, Green = CDR, Black lines = Net result",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig("gross_positive_vs_cdr_vs_ffi_by_scenario.png")


# %% [markdown]
# # Generate regional CDR fluxes

# %%
# --- Extension of individual CDR components maintaining 2100 ratios ---

# First, let's understand the structure of our CDR DataFrames
print("=== CDR DataFrames Structure Check ===")
print(f"co2_beccs shape: {co2_beccs.shape}")
print(f"co2_dacc shape: {co2_dacc.shape}")
print(f"co2_ocean shape: {co2_ocean.shape}")
print(f"co2_ew shape: {co2_ew.shape}")
print(f"global_cdr_ext shape: {global_cdr_ext.shape}")

print(f"\nco2_beccs index names: {co2_beccs.index.names}")
print(f"global_cdr_ext index names: {global_cdr_ext.index.names}")


# Check scenarios overlap
beccs_scenarios = set(co2_beccs.index.get_level_values("scenario").unique())
global_scenarios = set(global_cdr_ext.index.get_level_values("scenario").unique())
common_scenarios = beccs_scenarios & global_scenarios
print(f"\nCommon scenarios between CDR components and global_cdr_ext: {len(common_scenarios)}")
print(f"Scenarios: {sorted(common_scenarios)}")


# === EXECUTE VECTORIZED EXTENSION ===

# Define CDR components
cdr_components = {
    "BECCS": co2_beccs,
    "DACCS": co2_dacc,
    "Ocean": co2_ocean,
    "Enhanced_Weathering": co2_ew,
    "Biochar": co2_biochar,
    "Soil_Management": co2_soil,
    "Other_CDR": co2_othercdr,
}

extended_cdr_components = extend_cdr_components_vectorized(cdr_components, global_cdr_ext)

# Extract extended DataFrames
co2_beccs_ext = extended_cdr_components["BECCS"]
co2_dacc_ext = extended_cdr_components["DACCS"]
co2_ocean_ext = extended_cdr_components["Ocean"]
co2_ew_ext = extended_cdr_components["Enhanced_Weathering"]
co2_biochar_ext = extended_cdr_components["Biochar"]
co2_soil_ext = extended_cdr_components["Soil_Management"]
co2_other_cdr_ext = extended_cdr_components["Other_CDR"]
# sys.exit(4)


# === VERIFICATION ===
test_year = 2200.0
if test_year in co2_beccs_ext.columns:
    # Sum all CDR components for verification
    total_sum = (
        co2_beccs_ext[test_year].groupby("scenario").sum()
        + co2_dacc_ext[test_year].groupby("scenario").sum()
        + co2_ocean_ext[test_year].groupby("scenario").sum()
        + co2_ew_ext[test_year].groupby("scenario").sum()
        + co2_biochar_ext[test_year].groupby("scenario").sum()
        + co2_soil_ext[test_year].groupby("scenario").sum()
        + co2_other_cdr_ext[test_year].groupby("scenario").sum()
    )

    global_reference = global_cdr_ext[test_year].groupby("scenario").first()


# %% [markdown]
#

# %% [markdown]
# # Extended missing sectors for regional (non-cdr fossil)
#

# %%
# Merge dataframes into df_everything
print("=== MERGING ALL DATAFRAMES INTO df_everything ===")
df_everything = fix_up_and_concatenate_extensions(
    {
        "fossil_extension": fossil_extension_df,
        "afolu_extensions": df_afolu,
        "non_co2_extensions": df_all,
        "gross_positive_extensions": co2_gross_positive_ext,
        "cdr_extensions": global_cdr_ext,
        "beccs_extensions": co2_beccs_ext,
        "dacc_extensions": co2_dacc_ext,
        "ocean_extensions": co2_ocean_ext,
        "ew_extensions": co2_ew_ext,
        "biochar_extensions": co2_biochar_ext,
        "soil_extensions": co2_soil_ext,
        "other_cdr_extensions": co2_other_cdr_ext,
    }
)
print(df_everything.head())
print(f"✅ Successfully merged all DataFrames! Shape: {df_everything.shape}")
print(df_everything.shape)
print(df_all.index.names)

df_everything = extend_regional_for_missing(df_everything, scenarios_regional, fractions_fossil_total)
print(df_everything.shape)

# %%
# Check for and handle duplicate metadata (CSV preparation only)
print("=== CHECKING FOR DUPLICATES FOR CSV OUTPUT ===")
duplicate_mask = df_everything.index.duplicated(keep=False)
duplicates = df_everything[duplicate_mask]

print(f"Number of rows with duplicate metadata: {len(duplicates)}")
print(duplicates)
# sys.exit(4)

if len(duplicates) > 0:
    print(f"Found {len(duplicates)} duplicate rows, removing duplicates...")
    # Drop duplicate index rows, keeping the first occurrence
    df_everything_clean = df_everything[~df_everything.index.duplicated(keep="first")]

    print(f"Original shape: {df_everything.shape}")
    print(f"After deduplication: {df_everything_clean.shape}")
    print(f"Removed {df_everything.shape[0] - df_everything_clean.shape[0]} duplicate rows")

    # Use the clean version for further processing
    df_everything = df_everything_clean
    print("✅ Using deduplicated data for CSV output")
else:
    # No duplicates, proceed with original data
    print("✅ No duplicates found, proceeding with original data")


# %%
# Identify all unique model/scenario pairings
print("=== UNIQUE MODEL/SCENARIO PAIRINGS ===")

# Get unique combinations from the final dataframe
if "df_everything" in locals():
    # Method 1: From the final concatenated dataframe
    unique_pairs = df_everything.index.droplevel(["region", "variable", "unit", "workflow"]).drop_duplicates()
    print(f"From df_everything: {len(unique_pairs)} unique model/scenario pairs")
    print("\nUnique model/scenario combinations:")
    print(unique_pairs)
    for i, (model, scenario) in enumerate(unique_pairs, 1):
        print(f"{i:2d}. {model} | {scenario}")
else:
    print("df_everything not found, checking component dataframes...")

# Method 2: Check the scenario_model_match dictionary for reference
print("\n=== REFERENCE FROM scenario_model_match ===")
print(f"Expected {len(scenario_model_match)} scenarios from marker definitions:")
for i, (marker, info) in enumerate(scenario_model_match.items(), 1):
    scenario, model, color = info
    print(f"{i:2d}. {marker}: {model} | {scenario}")


# %%
year_cols = [col for col in df_everything.columns if str(col).isdigit()]
df_everything.rename(columns={col: float(col) for col in year_cols}, inplace=True)
df_everything.head()

# %%
# Add Gross Positive Emissions and Gross Removals to history dataframe
# For historical period:
# - Gross Positive Emissions = CO2|AFOLU (assuming all historical AFOLU emissions are gross positive)
# - Gross Removals = 0 (no large-scale technological CDR in historical period)

history = add_removals_and_positive_fossil_emissions_to_historical(history)


# %%
# Concise Historical-Future Merge Function
print("=== EXECUTING CONCISE HISTORICAL-FUTURE MERGE ===")


# Execute the concise merge
continuous_timeseries_concise = merge_historical_future_timeseries(history, df_everything)


# %% [markdown]
# ## Dump per model to database


# %%

for model in df_everything.pix.unique("model"):
    dump_data_per_model(df_everything, model, EXTENSIONS_OUT_DIR, EXTENSIONS_OUTPUT_DB)


# %% [markdown]
# ## Save to final database
#
# Save both stage="complete" (passthrough from 2100 infilling) and
# stage="extended" (extended markers to 2500) to the final INFILLED_SCENARIOS_DB.

# %%
print("\n=== SAVING TO FINAL DATABASE ===")

# Load all stage="complete" data from the temp database (all IAMs, 1750-2100)
print("Loading stage='complete' data from temp database...")
scenarios_complete_all = INFILLED_SCENARIOS_DB_2100.load(pix.isin(stage="complete"))
print(f"Loaded complete scenarios: {scenarios_complete_all.shape}")

# Save complete scenarios to final database (passthrough)
print("Saving stage='complete' to final database...")
INFILLED_SCENARIOS_DB.save(scenarios_complete_all, allow_overwrite=True)

# Save extended scenarios to final database (7 markers, 1750-2500)
print("Saving stage='extended' to final database...")
continuous_timeseries_extended = continuous_timeseries_concise.copy()

# Filter out internal diagnostic variables that aren't part of CMIP7 naming convention
internal_variables = [
    "Emissions|CO2|Gross Positive Emissions",
    "Emissions|CO2|Gross Removals",
]
continuous_timeseries_extended = continuous_timeseries_extended.loc[
    ~continuous_timeseries_extended.index.get_level_values("variable").isin(internal_variables)
]

continuous_timeseries_extended["stage"] = "extended"
continuous_timeseries_extended = continuous_timeseries_extended.set_index("stage", append=True)
continuous_timeseries_extended = continuous_timeseries_extended.droplevel("workflow")
print(continuous_timeseries_extended.index.names)
# Keep integer columns for compatibility with downstream notebooks
INFILLED_SCENARIOS_DB.delete()
INFILLED_SCENARIOS_DB.save(continuous_timeseries_extended, allow_overwrite=True)

print("✅ Final database saved with both complete and extended stages")


# %%
# Simple CSV output

# Execute the simple CSV save
result = save_continuous_timeseries_to_csv(continuous_timeseries_concise, "continuous_emissions_timeseries_1750_2500")
result_full = save_continuous_timeseries_to_csv(df_everything, "extensions_full_emissions_timeseries_2023_2500")
