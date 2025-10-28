# %% [markdown]
# # Extensions of Marker Scenarios

# %% [markdown]
# Regular imports

# %%
import ast
import copy
import difflib
import glob
import json
import os
import re
from datetime import datetime

# %%
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import tqdm.auto

# Package imports
from afolu_extension_functions import (
    get_cumulative_afolu,
    get_cumulative_afolu_fill_from_hist,
)
from extensions_fossil_co2_storyline_functions import extend_co2_for_scen_storyline
from extensions_functions_for_non_co2 import (
    do_single_component_for_scenario_model_regionally,
    plot_just_global,
)
from general_utils_for_extensions import (
    glue_with_historical,
    interpolate_to_annual,
)

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_SCENARIOS_DB,
)

# from emissions_harmonization_historical.constants import DATA_ROOT
# from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.extension_functionality import (
    extend_flat_cumulative,
    extend_flat_evolution,
    extend_linear_rampdown,
    find_func_form_lu_extension,
)

# Constants
FUTURE_START_YEAR = 2023.0
HISTORICAL_START_YEAR = 1900
SCENARIO_END_YEAR = 2100
TUPLE_LENGTH_WITH_STAGE = 6
MAX_YEARS_FOR_MARKERS = 50

# %% [markdown]
# More preamble

# %%
save_plots = True

pandas_openscm.register_pandas_accessor()

UR = openscm_units.unit_registry
Q = UR.Quantity

# %% [markdown]
# ## Loading scenarios

# %%
scenarios_complete_global = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete")).reset_index("stage", drop=True)
scenarios_complete_global  # TODO: drop 2100 end once we have usable scenario data post-2100
for model in scenarios_complete_global.pix.unique("model").values:
    print(model)
    print(scenarios_complete_global.loc[pix.ismatch(model=f"{model}")].pix.unique("scenario"))

history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

scenarios_regional = HARMONISED_SCENARIO_DB.load()
history_regional = HISTORY_HARMONISATION_DB.load()
# sys.exit(4)
# scenarios_complete_global


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
    "L": ["SSP2 - Low Emissions_f", "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "tab:green"],
    "ML": ["SSP2 - Medium-Low Emissions", "COFFEE 1.6", "tab:pink"],
    "M": ["SSP2 - Medium Emissions", "IMAGE 3.4", "tab:purple"],
    # "MOS": ["SSP2 - Medium Emissions - Overshoot", "IMAGE 3.4", "tab:olive"],
    "H": ["SSP3 - High Emissions", "GCAM 8s", "tab:red"],
    # "H": ["SSP3 - High Emissions_a", "GCAM 7.1 scenarioMIP", "tab:red"],
    # "HL": ["SSP5 - High Emissions", "WITCH 6.0", "tab:brown"],
    "HL": ["SSP5 - Medium-Low Emissions_a", "WITCH 6.0", "tab:brown"],
}

# %%


# %% [markdown]
# Go over and check, add extra entry for MOS, which is really just M up to the overshoot

# %%
for stype, model_scen_match in scenario_model_match.items():
    model = model_scen_match[1]
    scenario = model_scen_match[0]
    print(f"{stype}: {model=} {scenario=}")
    print(len(scenarios_regional.loc[pix.ismatch(model=f"{model}", scenario=f"{scenario}")].pix.unique("region")))
    print(scenarios_regional.loc[pix.ismatch(model=f"{model}", scenario=f"{scenario}")].shape)
    print(scenarios_complete_global.loc[pix.ismatch(model=f"{model}", scenario=f"{scenario}")].shape)
    if stype == "MOS":
        scenarios_global_mos = scenarios_complete_global.loc[
            pix.ismatch(model=f"{model}", scenario=f"{scenario[:-12]}")
        ].copy()
        scenarios_regional_mos = scenarios_regional.loc[
            pix.ismatch(model=f"{model}", scenario=f"{scenario[:-12]}")
        ].copy()
        scenarios_regional_mos = scenarios_regional_mos.rename({scenario[:-12]: scenario}, level="scenario")
        scenarios_global_mos = scenarios_global_mos.rename({scenario[:-12]: scenario}, level="scenario")
        scenarios_regional = pd.concat([scenarios_regional, scenarios_regional_mos])
        scenarios_complete_global = pd.concat([scenarios_complete_global, scenarios_global_mos])
scenarios_regional = scenarios_regional.sort_index(axis="columns").T.interpolate("index").T

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
do_and_write_to_csv = False
if do_and_write_to_csv:
    df_all = do_all_non_co2_extensions(scenarios_complete_global, history)
    df_all.to_csv("first_draft_extended_nonCO2_all.csv")
    afolu_dfs = calculate_afolu_extensions(scenarios_complete_global, history, cumulative_history_afolu, plot=True)
    print(df_all)
    for name, afolu_df in afolu_dfs.items():
        afolu_df.to_csv(f"first_draft_extended_afolu_{name}.csv")
    # sys.exit(4)
# else:
df_all = pd.read_csv("first_draft_extended_nonCO2_all.csv")
afolu_dfs = {}
for afolu_file in glob.glob("first_draft_extended_afolu_*.csv"):
    print(afolu_file)
    name = afolu_file.split("first_draft_extended_")[-1].split(".csv")[0]
    print(name)
    afolu_dfs[name] = pd.read_csv(afolu_file)
    print(afolu_dfs[name].shape)
    print(afolu_dfs[name])
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
    "VL": ["ECS", 2170, -3.5e3, 2450, 2500, 20, 20],
    "LN": [
        "CSCS",
        2100,
        -24e3,
        2120,
        2200,
        2300,
        20,
        20,
    ],  # ["ECS", 2120, -24e3, 2200, 2300],
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
# ## Make a dataframe of all parts and send to database

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

print("\nIndex compatibility check:")
print(f"fossil_extension_df: {fossil_extension_df.index.names}")
print(f"df_afolu_fixed: {df_afolu_fixed.index.names}")
print(f"df_all_fixed: {df_all_fixed.index.names}")


# CRITICAL FIX: Standardize year column types before concatenation
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


print("\n=== FIXING COLUMN TYPE MISMATCH ===")
print("Standardizing year columns to float type...")

# Standardize all three dataframes
fossil_extension_df_fixed = standardize_year_columns(fossil_extension_df)
df_afolu_fixed_standardized = standardize_year_columns(df_afolu_fixed)
df_all_fixed_standardized = standardize_year_columns(df_all_fixed)


# Now concatenate with properly standardized dataframes
df_everything = pix.concat(
    [
        fossil_extension_df_fixed,
        df_afolu_fixed_standardized,
        df_all_fixed_standardized,
    ]
)


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


# %%
# Plot total CO2 emissions for each scenario from the continuous timeseries
def plot_total_co2_emissions(data, scenario_colors=None):
    """
    Plot total CO2 emissions (AFOLU + Energy & Industrial) for each scenario.

    Plot emissions from the continuous timeseries spanning 1750-2500.
    """
    print("=== PLOTTING TOTAL CO2 EMISSIONS BY SCENARIO ===")

    # Filter for CO2 variables and World region
    co2_vars = ["Emissions|CO2|AFOLU", "Emissions|CO2|Energy and Industrial Processes"]

    # Get available CO2 variables in the data
    available_vars = data.index.get_level_values("variable").unique()
    co2_vars_in_data = [var for var in co2_vars if var in available_vars]

    print(f"Available CO2 variables: {co2_vars_in_data}")

    if not co2_vars_in_data:
        print("❌ No CO2 variables found in data")
        return

    # Filter data for CO2 variables and World region
    co2_data = data.loc[
        (data.index.get_level_values("variable").isin(co2_vars_in_data))
        & (data.index.get_level_values("region") == "World")
    ]

    print(f"CO2 data shape: {co2_data.shape}")

    # Get years (numeric columns)
    years = [col for col in co2_data.columns if isinstance(col, int | float)]
    years = sorted(years)

    # Group by model and scenario, sum across CO2 variables
    scenarios = co2_data.index.droplevel(["region", "variable", "unit"]).drop_duplicates()

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Use scenario colors if provided, otherwise default colors
    if scenario_colors is None:
        scenario_colors = scenario_model_match

    for i, (model, scenario) in enumerate(scenarios):
        # Get data for this scenario
        scenario_data = co2_data.loc[
            (co2_data.index.get_level_values("model") == model)
            & (co2_data.index.get_level_values("scenario") == scenario)
        ]

        # Sum across CO2 variables if multiple exist
        if len(scenario_data) > 1:
            total_emissions = scenario_data[years].sum(axis=0)
        else:
            total_emissions = scenario_data[years].iloc[0]

        # Find the marker code for this scenario
        marker_code = None
        color = f"C{i}"  # Default color
        for marker, info in scenario_colors.items():
            if info[1] == model and info[0] == scenario:
                marker_code = marker
                color = info[2]
                break

        # Create label
        label = f"{marker_code} ({model})" if marker_code else f"{model}"

        # Plot the timeseries
        ax.plot(
            years,
            total_emissions.values,
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

    # Formatting
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Total CO2 Emissions (Mt CO2/yr)", fontsize=14)
    ax.set_title(
        "Total CO2 Emissions by Scenario (1750-2500)\nContinuous Historical-Future Timeseries",
        fontsize=16,
        fontweight="bold",
    )

    # Set reasonable axis limits
    ax.set_xlim(min(years), max(years))
    ax.set_ylim(bottom=0)

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
fig, ax = plot_total_co2_emissions(continuous_timeseries_concise, scenario_model_match)


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
fig_transition, ax_transition = plot_co2_transition_period(continuous_timeseries_concise, scenario_model_match)

# %%
native_emissions = copy.deepcopy(continuous_timeseries_concise)

# %%
# Remove leading 'Emissions|' from variable names in the index of continuous_timeseries_concise
if "variable" in continuous_timeseries_concise.index.names:
    var_idx = continuous_timeseries_concise.index.names.index("variable")
    new_index = [
        tuple(
            [
                *v[:var_idx],
                (v[var_idx].replace("Emissions|", "", 1) if v[var_idx].startswith("Emissions|") else v[var_idx]),
                *v[var_idx + 1 :],
            ]
        )
        for v in continuous_timeseries_concise.index
    ]
    continuous_timeseries_concise.index = pd.MultiIndex.from_tuples(
        new_index, names=continuous_timeseries_concise.index.names
    )
    print("Removed leading 'Emissions|' from variable names in index.")
else:
    print("No 'variable' level in index.")
continuous_timeseries_concise.head()

# %%
# Rename the 'scenario' index level to 'long_scenario' in continuous_timeseries_concise
if "scenario" in continuous_timeseries_concise.index.names:
    continuous_timeseries_concise.index = continuous_timeseries_concise.index.set_names(
        [name if name != "scenario" else "long_scenario" for name in continuous_timeseries_concise.index.names]
    )
    print("Renamed 'scenario' index level to 'long_scenario'.")
else:
    print("No 'scenario' level in index.")
continuous_timeseries_concise.head()

# %%
# Add a new 'scenario' column with the short name from scenario_model_match
# Assumes 'long_scenario' is now the index level for scenario
if "long_scenario" in continuous_timeseries_concise.index.names:
    # Build a mapping from long_scenario to short scenario name
    long_to_short = {v[0]: k for k, v in scenario_model_match.items()}
    # Reset index to add a column
    df_temp = continuous_timeseries_concise.reset_index()
    df_temp["scenario"] = df_temp["long_scenario"].map(long_to_short)
    # Move 'scenario' column to the front for clarity
    cols = ["scenario"] + [col for col in df_temp.columns if col != "scenario"]
    df_temp = df_temp[cols]
    # Set index back to original (including new scenario column if desired)
    continuous_timeseries_concise = df_temp.set_index(continuous_timeseries_concise.index.names)
    print("Added 'scenario' column with short names.")
else:
    print("No 'long_scenario' level in index.")
continuous_timeseries_concise

# %%


# %%
# Output a list of all unique variable names in the DataFrame
unique_variables = continuous_timeseries_concise.index.get_level_values("variable").unique()
continuous_timeseries_concise.index.get_level_values("variable").unique()


# %%
fair_vars = pd.read_csv("../data/fair-inputs/species_configs_properties_1.4.1.csv")["name"]


# %%


# %%
# Attempt to map unique_variables to their likely pairing in fair_vars

fair_vars = pd.read_csv("../data/fair-inputs/species_configs_properties_1.4.1.csv")["name"]
mapping = {}
for var in unique_variables:
    # Try exact match first
    if var in fair_vars.values:
        mapping[var] = var
    else:
        # Use difflib to find the closest match
        close_matches = difflib.get_close_matches(var, fair_vars, n=1, cutoff=0.6)
        mapping[var] = close_matches[0] if close_matches else None
print("Variable mapping (DataFrame variable → FaIR variable):")
mapping["CO2|Energy and Industrial Processes"] = "CO2 FFI"
mapping["Halon1202"] = "Halon-1202"

for k, v in mapping.items():
    print(f"{k} → {v}")


# %%
# Update the DataFrame to use the FaIR variable names in the 'variable' index level
def map_variable_name(var):
    """Map variable name to FaIR variable name using mapping dictionary."""
    return mapping.get(var, var)  # fallback to original if no mapping found


new_index = list(continuous_timeseries_concise.index)
new_index = [
    tuple(
        map_variable_name(v) if name == "variable" else v
        for name, v in zip(continuous_timeseries_concise.index.names, idx)
    )
    for idx in new_index
]
continuous_timeseries_concise.index = pd.MultiIndex.from_tuples(
    new_index, names=continuous_timeseries_concise.index.names
)
print("Updated DataFrame to use FaIR variable names in the index.")

# %%
continuous_timeseries_concise

# %%
continuous_timeseries_concise.index.get_level_values("variable").unique()


# %%
# Update year columns to be float and add 0.5 to each (e.g., 1750.5, 1751.5, ...)
def shift_year_columns(df):
    """Update year columns to be float and add 0.5 to each (e.g., 1750.5, 1751.5, ...)."""
    new_columns = []
    for col in df.columns:
        try:
            year = float(col)
            new_columns.append(year + 0.5)
        except (ValueError, TypeError):
            new_columns.append(col)
    df.columns = new_columns
    return df


continuous_timeseries_concise = shift_year_columns(continuous_timeseries_concise)
print("Year columns updated to float and shifted by 0.5.")

# %%
# Remove all rows where the combination of scenario (column) and variable (index) is not unique
if "scenario" in continuous_timeseries_concise.columns and "variable" in continuous_timeseries_concise.index.names:
    idx_df = continuous_timeseries_concise.reset_index()[["scenario", "variable"]]
    duplicated_pairs = idx_df.duplicated(subset=["scenario", "variable"], keep=False)
    to_keep = ~duplicated_pairs.values
    before = len(continuous_timeseries_concise)
    continuous_timeseries_concise = continuous_timeseries_concise.reset_index()[to_keep].set_index(
        continuous_timeseries_concise.index.names
    )
    after = len(continuous_timeseries_concise)
    print(
        f"Removed all rows with non-unique scenario/variable pairs. Remaining rows: {after} (removed {before - after})"
    )
else:
    print("'scenario' column and/or 'variable' index not found.")

# %%
continuous_timeseries_concise.to_csv("../data/fair-inputs/emissions_1750-2500.csv")
