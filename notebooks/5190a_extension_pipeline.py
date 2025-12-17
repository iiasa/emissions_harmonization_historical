# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
import ast
import copy
import glob
import json
import os
import re
import sys
from datetime import datetime
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

# from emissions_harmonization_historical.constants import DATA_ROOT
# from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.extensions.extension_functionality import (
    extend_flat_cumulative,
    extend_flat_evolution,
    extend_linear_rampdown,
    find_func_form_lu_extension,
)
from emissions_harmonization_historical.extensions.extensions_functions_for_non_co2 import (
    do_single_component_for_scenario_model_regionally,
    plot_just_global,
)
from emissions_harmonization_historical.extensions.fossil_co2_storyline_functions import (
    extend_co2_for_scen_storyline,
)
from emissions_harmonization_historical.extensions.general_utils_for_extensions import (
    glue_with_historical,
    interpolate_to_annual,
)

# sys.exit(4)
# Constants
FUTURE_START_YEAR = 2023.0
HISTORICAL_START_YEAR = 1900
SCENARIO_END_YEAR = 2100
TUPLE_LENGTH_WITH_STAGE = 6
MAX_YEARS_FOR_MARKERS = 50

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

# %%
# Create inverse dictionaries for scenario_model_match
scenario_to_code = {}  # Maps scenario name to short code
model_to_code = {}  # Maps model name to short code
code_to_color = {}  # Maps short code to color

for code, info in scenario_model_match.items():
    scenario_name, model_name, color = info
    scenario_to_code[scenario_name] = code
    model_to_code[model_name] = code
    code_to_color[code] = color


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
    temp_list_for_new_data = []
    temp_list_for_new_data_flat = []
    temp_list_for_new_data_flat_cumulative = []
    temp_list_for_new_data_linear_ramp_down = []
    for s, meta in scenario_model_match.items():
        scen = scenarios_complete_global.loc[pix.ismatch(variable="**CO2|AFOLU", model=meta[1], scenario=meta[0])]
        scen_full = glue_with_historical(scen, history.loc[pix.ismatch(variable="Emissions|CO2|AFOLU")])
        cumulative_2100 = get_cumulative_afolu_fill_from_hist(scen, meta[1], meta[0], cumulative_history_afolu)
        em_ext, _cle_inf = find_func_form_lu_extension(
            scen_full.values[0, :],
            cumulative_2100.values[0, :],
            np.arange(cumulative_2100.columns[0], 2501),
            2100 - int(cumulative_2100.columns[0]),
            cle_inf_0=True,
        )
        em_ext_flat = extend_flat_evolution(scen_full.values[0, :], np.arange(cumulative_2100.columns[0], 2501))
        em_ext_flat_cumulative = extend_flat_cumulative(
            scen_full.values[0, :], np.arange(cumulative_2100.columns[0], 2501)
        )
        em_ext_linear_ramp_down = extend_linear_rampdown(
            scen_full.values[0, :], np.arange(cumulative_2100.columns[0], 2501)
        )
        if plot:
            axs[0].plot(cumulative_2100.columns, cumulative_2100.values[0, :], color=meta[2])
            axs[0].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                np.cumsum(em_ext),
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
            axs[1].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
            axs[1].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                em_ext,
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
            axs[2].plot(cumulative_2100.columns, cumulative_2100.values[0, :], color=meta[2])
            axs[2].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                np.cumsum(em_ext),
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
            axs[3].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
            axs[3].plot(
                np.arange(cumulative_2100.columns[0], 2501),
                em_ext,
                "--",
                alpha=0.7,
                label=s,
                color=meta[2],
            )
        df_afolu = pd.DataFrame(
            data=[em_ext],
            columns=np.arange(cumulative_2100.columns[0], 2501),
            index=scen.index,
        )
        df_afolu_flat = pd.DataFrame(
            data=[em_ext_flat],
            columns=np.arange(cumulative_2100.columns[0], 2501),
            index=scen.index,
        )
        df_afolu_flat_cumulative = pd.DataFrame(
            data=[em_ext_flat_cumulative],
            columns=np.arange(cumulative_2100.columns[0], 2501),
            index=scen.index,
        )
        df_afolu_linear_ramp_down = pd.DataFrame(
            data=[em_ext_linear_ramp_down],
            columns=np.arange(cumulative_2100.columns[0], 2501),
            index=scen.index,
        )
        temp_list_for_new_data.append(df_afolu)
        temp_list_for_new_data_flat.append(df_afolu_flat)
        temp_list_for_new_data_flat_cumulative.append(df_afolu_flat_cumulative)
        temp_list_for_new_data_linear_ramp_down.append(df_afolu_linear_ramp_down)

    extended_data_afolu_smooth = pd.concat(temp_list_for_new_data)
    extended_data_afolu_flat = pd.concat(temp_list_for_new_data_flat)
    extended_data_afolu_flat_cumulative = pd.concat(temp_list_for_new_data_flat_cumulative)
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
        axs[2].set_xlim(2000, 2500)
        axs[3].set_xlim(2000, 2500)

        plt.savefig("afolu_first_draft_extensions.png")
    return {
        "smooth_afolu": extended_data_afolu_smooth,
        "flat_afolu_emissions": extended_data_afolu_flat,
        "flat_afolu_cumulative": extended_data_afolu_flat_cumulative,
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
        "MOS": 95.0,
        "H": 520.0,
        "HL": 110.0,
    },
    "Emissions|Sulfur": {
        "VL": 20.0,
        "LN": 10.0,
        "L": None,
        "ML": 20.0,
        "M": 20.0,
        "MOS": 20.0,
        "H": 50.0,
        "HL": 10.0,
    },
}

# %% [markdown]
# Main functionality for all non-co2 extensions


# %%
def do_all_non_co2_extensions(scenarios_complete_global, history):
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
            # sys.exit(4)
    df_all = pd.concat(total_df_list)
    return df_all


# %% [markdown]
# ## Do main block of non-fossil CO2 extensions first

# %%
do_and_write_to_csv = True
if do_and_write_to_csv:
    df_all = do_all_non_co2_extensions(scenarios_complete_global, history)
    df_all.to_csv("first_draft_extended_nonCO2_all.csv")
    afolu_dfs = calculate_afolu_extensions(scenarios_complete_global, history, cumulative_history_afolu, plot=True)
    # print(df_all)
    for name, afolu_df in afolu_dfs.items():
        afolu_df.to_csv(f"first_draft_extended_afolu_{name}.csv")
    # sys.exit(4)
# else:

# %%

df_all = pd.read_csv("first_draft_extended_nonCO2_all.csv")
afolu_dfs = {}
for afolu_file in glob.glob("first_draft_extended_afolu_*.csv"):
    print("writing " + afolu_file)
    name = afolu_file.split("first_draft_extended_")[-1].split(".csv")[0]

    afolu_dfs[name] = pd.read_csv(afolu_file)

# sys.exit(4)

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
    print(s)
    print(meta)
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
co2_ffi = interpolate_to_annual(
    scenarios_regional.loc[
        pix.ismatch(
            variable="Emissions|CO2|Energy and Industrial Processes",
            workflow="for_scms",
        )
    ]
)
co2_cdr = co2_dacc + co2_ocean.values + co2_ew.values + co2_beccs.values

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


def sigmoid_decay_extension(start_value, offset, n_years, decay_timescale=100):
    """
    Create a sigmoid-based decay progression from start_value to near zero over n_years.

    Parameters
    ----------
    start_value : float
        The starting value (at t=0)
    offset : float
        Time offset for the sigmoid transition (years from start)
    n_years : int
        Number of years for the extension
    decay_timescale : float
        The decay timescale in years (controls steepness of sigmoid transition)

    Returns
    -------
    numpy.ndarray
        Array of values following sigmoid-based decay
    """
    if n_years <= 0:
        return np.array([])
    elif n_years == 1:
        return np.array([start_value * 0.5])  # Rough midpoint for single year
    else:
        t = np.arange(n_years)
        # Normalized sigmoid: starts at start_value, decays toward zero
        norm = 1 + np.tanh(-(-offset) / decay_timescale)
        values = start_value * (1 + np.tanh(-(t - offset) / decay_timescale)) / norm
        return values


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

        gross_pos_extension = sigmoid_decay_extension(gross_pos_2100, offset, len(years_extension), decay_timescale)

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
PLOT_GRID_COLS = 4  # Number of columns in the subplot grid
fig, axes = plt.subplots(2, PLOT_GRID_COLS, figsize=(20, 12))
axes = axes.flatten()  # Make it easier to iterate

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

# Hide the last subplot since we only have 7 scenarios
axes[7].set_visible(False)

# Add overall title
fig.suptitle(
    "Gross Positive vs CDR vs Net FFI Emissions by Scenario\nBrown = Positive, Green = CDR, Black lines = Net result",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()
plt.show()


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


# %%


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


# === EXECUTE VECTORIZED EXTENSION ===

# Define CDR components
cdr_components = {
    "BECCS": co2_beccs,
    "DACCS": co2_dacc,
    "Ocean": co2_ocean,
    "Enhanced_Weathering": co2_ew,
}

extended_cdr_components = extend_cdr_components_vectorized(cdr_components, global_cdr_ext)

# Extract extended DataFrames
co2_beccs_ext = extended_cdr_components["BECCS"]
co2_dacc_ext = extended_cdr_components["DACCS"]
co2_ocean_ext = extended_cdr_components["Ocean"]
co2_ew_ext = extended_cdr_components["Enhanced_Weathering"]


# === VERIFICATION ===
test_year = 2200.0
if test_year in co2_beccs_ext.columns:
    # Sum all CDR components for verification
    total_sum = (
        co2_beccs_ext[test_year].groupby("scenario").sum()
        + co2_dacc_ext[test_year].groupby("scenario").sum()
        + co2_ocean_ext[test_year].groupby("scenario").sum()
        + co2_ew_ext[test_year].groupby("scenario").sum()
    )

    global_reference = global_cdr_ext[test_year].groupby("scenario").first()


# %% [markdown]
# # Merge dataframes into df_everything

# %%
# Fix indices for df_afolu and df_all to match fossil_extension_df

# Fix df_afolu index (currently has default RangeIndex)
if "model" in df_afolu.columns:
    index_cols = ["model", "scenario", "region", "variable", "unit"]
    df_afolu_fixed = df_afolu.set_index(index_cols)
    print(f"df_afolu_fixed index: {df_afolu_fixed.index.names}")
else:
    df_afolu_fixed = df_afolu
    print(f"df_afolu already has proper index: {df_afolu.index.names}")

# Fix df_all index (has tuple strings in 'Unnamed: 0' column)
if "Unnamed: 0" in df_all.columns:
    # Parse string tuples back to actual tuples
    tuple_strings = df_all["Unnamed: 0"].values
    parsed_tuples = [ast.literal_eval(ts.replace(", nan", "")) for ts in tuple_strings]
    # Determine index names based on tuple length
    tuple_length = len(parsed_tuples[0])
    if tuple_length == TUPLE_LENGTH_WITH_STAGE:
        index_names = ["model", "scenario", "region", "variable", "unit", "stage"]
    else:
        index_names = ["model", "scenario", "region", "variable", "unit"]
    parsed_tuples = [i[0:6] for i in parsed_tuples]

    # Create MultiIndex and new dataframe
    multi_index = pd.MultiIndex.from_tuples(parsed_tuples, names=index_names)
    df_all_fixed = df_all.drop("Unnamed: 0", axis=1).copy()
    df_all_fixed.index = multi_index

    # Drop 'stage' level if it exists to match other dataframes
    if "stage" in df_all_fixed.index.names:
        df_all_fixed = df_all_fixed.reset_index("stage", drop=True)

    print(f"df_all_fixed index: {df_all_fixed.index.names}")
else:
    df_all_fixed = df_all
    print(f"df_all already has proper index: {df_all.index.names}")


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


# Step 1: Fix CDR DataFrames by adding missing region level
co2_gross_positive_with_region = add_region_level_to_index(co2_gross_positive_ext.copy())
global_cdr_with_region = add_region_level_to_index(global_cdr_ext.copy())

# Step 2: Standardize all dataframes
fossil_extension_df_fixed = standardize_year_columns(fossil_extension_df)
df_afolu_fixed_standardized = standardize_year_columns(df_afolu_fixed)
df_all_fixed_standardized = standardize_year_columns(df_all_fixed)
co2_gross_positive_ext_fixed = standardize_year_columns(co2_gross_positive_with_region)
global_cdr_ext_fixed = standardize_year_columns(global_cdr_with_region)

# Now concatenate with properly standardized dataframes
df_everything = pix.concat(
    [
        fossil_extension_df_fixed,
        df_afolu_fixed_standardized,
        df_all_fixed_standardized,
        co2_gross_positive_ext_fixed,
        global_cdr_ext_fixed,
    ]
)

print(f"✅ Successfully merged all DataFrames! Shape: {df_everything.shape}")


# %%
# Check for and handle duplicate metadata (CSV preparation only)
print("=== CHECKING FOR DUPLICATES FOR CSV OUTPUT ===")
duplicate_mask = df_everything.index.duplicated(keep=False)
duplicates = df_everything[duplicate_mask]

print(f"Number of rows with duplicate metadata: {len(duplicates)}")

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
    unique_pairs = df_everything.index.droplevel(["region", "variable", "unit"]).drop_duplicates()
    print(f"From df_everything: {len(unique_pairs)} unique model/scenario pairs")
    print("\nUnique model/scenario combinations:")
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

co2_ffi_hist = history.loc[pix.ismatch(variable="Emissions|CO2|Energy and Industrial Processes")].copy()

# Create Gross Positive Emissions by copying AFOLU data and changing variable name
gross_positive_hist = co2_ffi_hist.copy()
new_index_gross = []
for idx_tuple in gross_positive_hist.index:
    new_tuple = list(idx_tuple)
    new_tuple[3] = "Emissions|CO2|Gross Positive Emissions"  # variable is at position 3
    new_index_gross.append(tuple(new_tuple))
gross_positive_hist.index = pd.MultiIndex.from_tuples(new_index_gross, names=gross_positive_hist.index.names)

# Create Gross Removals as zeros with same structure as AFOLU
gross_removals_hist = gross_positive_hist.copy()
gross_removals_hist.iloc[:, :] = 0.0  # Set all values to zero
new_index_removals = []
for idx_tuple in gross_removals_hist.index:
    new_tuple = list(idx_tuple)
    new_tuple[3] = "Emissions|CO2|Gross Removals"  # variable is at position 3
    new_index_removals.append(tuple(new_tuple))
gross_removals_hist.index = pd.MultiIndex.from_tuples(new_index_removals, names=gross_removals_hist.index.names)

# Remove any previously added gross emissions variables and add the new ones
history_clean = history.loc[
    ~history.index.get_level_values("variable").isin(
        ["Emissions|CO2|Gross Positive Emissions", "Emissions|CO2|Gross Removals"]
    )
]
history = pd.concat([history_clean, gross_positive_hist, gross_removals_hist])

print("✅ Added Gross Positive Emissions and Gross Removals to history dataframe")
print(f"   History shape: {history.shape}")
print(f"   Total variables: {len(history.pix.unique('variable'))}")


# %%
# Concise Historical-Future Merge Function
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

    # Step 2: Define time splits
    hist_years = [col for col in history_data.columns if isinstance(col, int | float) and col <= overlap_year]
    future_years = [col for col in extensions_clean.columns if isinstance(col, int | float) and col > overlap_year]

    # Step 3: Get scenario list from extensions
    scenarios = extensions_clean.index.droplevel(["region", "variable", "unit"]).drop_duplicates()

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
                idx[3],
                idx[4],
            )  # model, scenario, region, variable, unit
            new_index.append(new_idx)

        hist_copy.index = pd.MultiIndex.from_tuples(new_index, names=extensions_clean.index.names)
        historical_expanded.append(hist_copy)

    historical_replicated = pd.concat(historical_expanded)

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

    print(f"Merged data: {continuous_data.shape} ({len(common_vars)} variables, {len(scenarios)} scenarios)")
    print(f"Time range: {continuous_data.columns[0]}-{continuous_data.columns[-1]}")

    return continuous_data


# Execute the concise merge
continuous_timeseries_concise = merge_historical_future_timeseries(history, df_everything)

# %%
raw_output = copy.deepcopy(continuous_timeseries_concise)

# %% [markdown]
# ## Dump per model to database


# %%
def dump_data_per_model(extended_data, model):
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
    output_dir_model = EXTENSIONS_OUT_DIR / model_short
    output_dir_model.mkdir(exist_ok=True, parents=True)
    model_data = extended_data.loc[extended_data.index.get_level_values("model") == model].copy()

    if not dump_with_full_scenario_names:
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

    EXTENSIONS_OUTPUT_DB.save(model_data, allow_overwrite=True)


for model in continuous_timeseries_concise.pix.unique("model"):
    dump_data_per_model(continuous_timeseries_concise, model)


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
# Keep integer columns for compatibility with downstream notebooks
INFILLED_SCENARIOS_DB.save(continuous_timeseries_extended, allow_overwrite=True)

print("✅ Final database saved with both complete and extended stages")


# %%
# Plot Gross Removals for all scenarios
def plot_gross_removals(data, scenario_colors=None):
    """
    Plot CO2 Gross Removals for each scenario from the continuous timeseries.
    """
    print("=== PLOTTING CO2 GROSS +ve emissions BY SCENARIO ===")

    # Filter for Gross Removals variable and World region
    gross_removals_data = data.loc[
        (data.index.get_level_values("variable") == "Emissions|CO2|Gross Positive Emissions")
        & (data.index.get_level_values("region") == "World")
    ]

    if gross_removals_data.empty:
        print("❌ No CO2 Gross Removals data found")
        return

    print(f"Gross Removals data shape: {gross_removals_data.shape}")

    # Get years (numeric columns)
    years = [col for col in gross_removals_data.columns if isinstance(col, int | float)]
    years = sorted(years)

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Use scenario colors if provided
    if scenario_colors is None:
        scenario_colors = scenario_model_match

    # Plot each scenario
    for i, (idx, row) in enumerate(gross_removals_data.iterrows()):
        model, scenario = idx[0], idx[1]  # Extract model and scenario from index

        # Find the marker code and color
        marker_code = None
        color = f"C{i}"
        for marker, info in scenario_colors.items():
            if info[1] == model and info[0] == scenario:
                marker_code = marker
                color = info[2]
                break

        label = f"{marker_code} ({model})" if marker_code else f"{model}"

        # Plot the timeseries
        ax.plot(
            years,
            row[years].values,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=0.8,
        )

    # Add vertical line at historical/future boundary
    ax.axvline(
        x=2023,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Historical/Future boundary",
        linewidth=2,
    )

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    # Formatting
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("CO2 Gross Removals (Mt CO2/yr)", fontsize=14)
    ax.set_title(
        "CO2 Gross +ve emissions by Scenario (1750-2500)",
        fontsize=16,
        fontweight="bold",
    )

    # Set reasonable axis limits
    ax.set_xlim(min(years), max(years))

    # Add grid
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # Add some key year markers
    key_years = [1850, 1900, 1950, 2000, 2050, 2100, 2200, 2300, 2400]
    for year in key_years:
        if year in years:
            ax.axvline(x=year, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.show()

    return fig, ax


# Execute the plotting function
if make_plots:
    fig_removals, ax_removals = plot_gross_removals(continuous_timeseries_concise, scenario_model_match)

# %%
# === Comprehensive CO2 Flux Analysis: Annual vs Cumulative ===

# Constants
BASELINE_YEAR = 2100
BASELINE_YEAR_FLOAT = 2100.0
YEAR_GRID_COLS = 3


# Configuration classes for reducing function arguments
class HistoryPlotConfig:
    """Configuration for plotting scenarios with historical data."""

    def __init__(self, all_years, future_years, axes, colors):
        self.all_years = all_years
        self.future_years = future_years
        self.axes = axes
        self.colors = colors


class SanityCheckConfig:
    """Configuration for sanity check plotting."""

    def __init__(self, year_cols, axes, colors, n_rows):
        self.year_cols = year_cols
        self.axes = axes
        self.colors = colors
        self.n_rows = n_rows


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


# Create the comprehensive plot
if make_plots:
    fig_comprehensive = plot_comprehensive_co2_analysis()
    plt.show()

# %%
raw_output.pix.unique("variable").values

# %%
# Constants for historical plotting
CDR_LIMIT = -1460  # Gt CO2
PROVED_FOSSIL_RESERVES = 2032 + 2400  # Gt CO2
PROBABLE_FOSSIL_RESERVES = 8036 + 2400  # Gt CO2


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

    # CDR components: zero for historical, then future values
    def pad_future_with_zeros(df_ext):
        zeros = np.zeros(len(all_years) - len(future_years))
        vals = df_ext.loc[df_ext.index.get_level_values("scenario") == scenario][future_years].sum().values
        return np.concatenate([zeros, vals])

    beccs_annual = pad_future_with_zeros(co2_beccs_ext) / 1000
    daccs_annual = pad_future_with_zeros(co2_dacc_ext) / 1000
    ocean_annual = pad_future_with_zeros(co2_ocean_ext) / 1000
    ew_annual = pad_future_with_zeros(co2_ew_ext) / 1000

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


# Create the comprehensive plot with history
fig_comprehensive_history = plot_comprehensive_co2_analysis_with_history()
plt.show()

# %%
# Constants for historical plotting
CDR_LIMIT = -1460  # Gt CO2, Gidden, M.J., Joshi, S., Armitage, J.J. et al.
# A prudent planetary limit for geologic carbon storage. Nature 645, 124-132 (2025).
# https://doi.org/10.1038/s41586-025-09423-y

PROVED_FOSSIL_RESERVES = (
    2032 + 2400
)  # Gt CO2, McGlade, C., Ekins, P. The geographical distribution of fossil fuels unused
# when limiting global warming to 2 °C. Nature 517, 187-190 (2015).
# https://doi.org/10.1038/nature14016
PROBABLE_FOSSIL_RESERVES = (
    8036 + 2400
)  # Gt CO2, McGlade, C., Ekins, P. The geographical distribution of fossil fuels unused
# when limiting global warming to 2 °C. Nature 517, 187-190 (2015).
# https://doi.org/10.1038/nature14016


def plot_bulk_co2_analysis_with_history():
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

    fig, axes = plt.subplots(n_scenarios, 2, figsize=(10, 4 * n_scenarios))
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
        & set(global_cdr_ext_fixed.index.get_level_values("scenario"))
        & set(fossil_extension_df.index.get_level_values("scenario"))
    )
    return sorted(list(scenarios))


def _get_color_scheme_with_afolu():
    """Get color scheme including AFOLU."""
    return {
        "Gross_Positive": "#8B4513",
        "Gross_CDR": "#DDD729",
        "AFOLU": "#2E8B57",
    }


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
    gross_neg_annual = (
        raw_output.loc[
            (
                slice(None),
                scenario,
                slice(None),
                "Emissions|CO2|Gross Removals",
                slice(None),
            ),
            all_years,
        ].sum()
        / 1000
    )

    # CDR components: zero for historical, then future values
    def pad_future_with_zeros(df_ext):
        zeros = np.zeros(len(all_years) - len(future_years))
        vals = df_ext.loc[df_ext.index.get_level_values("scenario") == scenario][future_years].sum().values
        return np.concatenate([zeros, vals])

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
        "gross_neg": gross_neg_annual,
        "fossil": fossil_annual,
        "afolu": afolu_annual,
    }


def _plot_annual_fluxes_with_history(ax_annual, data, years, colors):
    """Plot annual fluxes including historical data."""
    afolu_pos = np.clip(data["afolu"].values[:, 0], 0, None)
    afolu_neg = np.clip(data["afolu"].values[:, 0], None, 0)

    # Stack for annual plot
    y1_pos = data["gross_pos"].values
    y2_pos = afolu_pos + y1_pos
    y1_neg = data["gross_neg"].values
    y2_neg = y1_neg + afolu_neg

    # Plot stacked areas
    ax_annual.fill_between(years, 0, y1_pos, alpha=0.7, color=colors["Gross_Positive"], label="Gross FF&I")
    ax_annual.fill_between(years, y1_pos, y2_pos, alpha=0.7, color=colors["AFOLU"], label="AFOLU")
    ax_annual.fill_between(years, 0, y1_neg, alpha=0.7, color=colors["Gross_CDR"], label="Gross CDR")
    ax_annual.fill_between(years, y1_neg, y2_neg, alpha=0.7, color=colors["AFOLU"])

    if data["fossil"] is not None:
        ax_annual.plot(
            years,
            data["fossil"] + data["afolu"].values,
            "k-",
            linewidth=2,
            alpha=0.8,
            label="Net Emissions (Total)",
        )


def _plot_cumulative_fluxes_with_history(ax_cumul, data, years, colors):
    """Plot cumulative fluxes including historical data."""
    afolu_pos = np.clip(data["afolu"].values[:, 0], 0, None)
    afolu_neg = np.clip(data["afolu"].values[:, 0], None, 0)

    gross_pos_cumul = np.cumsum(data["gross_pos"].values)
    gross_neg_cumul = np.cumsum(data["gross_neg"].values)

    afolu_cumul = np.cumsum(afolu_pos + afolu_neg)

    if data["fossil"] is not None:
        np.cumsum(data["fossil"])
    else:
        pass

    y1_pos_cumul = gross_pos_cumul
    y2_pos_cumul = afolu_cumul + y1_pos_cumul
    y1_neg_cumul = gross_neg_cumul

    ax_cumul.fill_between(
        years,
        0,
        y1_pos_cumul,
        alpha=0.7,
        color=colors["Gross_Positive"],
    )
    ax_cumul.fill_between(
        years,
        y1_pos_cumul,
        y2_pos_cumul,
        alpha=0.7,
        color=colors["AFOLU"],
    )
    ax_cumul.fill_between(years, 0, y1_neg_cumul, alpha=0.7, color=colors["Gross_CDR"])

    ax_cumul.plot(
        years,
        gross_pos_cumul + gross_neg_cumul + afolu_cumul,
        "k-",
        linewidth=2,
        alpha=0.8,
        label="Cumulative Net Emissions (Total)",
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
        if ax == ax_annual:
            ax.set_ylim(-30, 60)
        ax.axvline(x=BASELINE_YEAR, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=2)

        if ax == ax_cumul:
            ax.axhline(
                y=CDR_LIMIT,
                color="green",
                linestyle="-",
                alpha=0.3,
                linewidth=3,
                label="Sequestration capacity limit",
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


# Create the comprehensive plot with history
fig_comprehensive_history = plot_comprehensive_co2_analysis_with_history()
plt.show()


# %%
# Simple CSV output
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


# Execute the simple CSV save
result = save_continuous_timeseries_to_csv(continuous_timeseries_concise, "continuous_emissions_timeseries_1750_2500")


# %% [markdown]
# ## Plotting various


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
    scenarios = co2_data.index.droplevel(["region", "variable", "unit"]).drop_duplicates()

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


# Create the transition period plot
if make_plots:
    fig_transition, ax_transition = plot_co2_transition_period(continuous_timeseries_concise, scenario_model_match)
