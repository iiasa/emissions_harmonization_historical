import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import tqdm.auto
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
from helper_functions_postprocess import calculate_nonco2_ghgs_gwp

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

save_plots = True

pandas_openscm.register_pandas_accessor()

UR = openscm_units.unit_registry
Q = UR.Quantity

# Get input data:

scenarios_complete_global = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete")).reset_index("stage", drop=True)
scenarios_complete_global  # TODO: drop 2100 end once we have usable scenario data post-2100
for model in scenarios_complete_global.pix.unique("model").values:
    print(model)
    print(scenarios_complete_global.loc[pix.ismatch(model=f"{model}")].pix.unique("scenario"))
# sys.exit(4)
# scenarios_complete_global

scenario_model_match = {
    "VLLO": ["SSP1 - Very Low Emissions", "REMIND-MAgPIE 3.5-4.11", "tab:blue"],
    "VLHO": ["SSP2 - Low Overshoot", "AIM 3.0", "tab:cyan"],
    "L": ["SSP2 - Low Emissions_f", "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "tab:green"],
    "ML": ["SSP2 - Medium-Low Emissions", "COFFEE 1.6", "tab:pink"],
    "M": ["SSP2 - Medium Emissions", "IMAGE 3.4", "tab:purple"],
    "MOS": ["SSP2 - Medium Emissions - Overshoot", "IMAGE 3.4", "tab:olive"],
    "H": ["SSP3 - High Emissions_a", "GCAM 7.1 scenarioMIP", "tab:red"],
    # "HL": ["SSP5 - High Emissions", "WITCH 6.0", "tab:brown"],
    "HL": ["SSP5 - Medium-Low Emissions_a", "WITCH 6.0", "tab:brown"],
}

scenarios_regional = HARMONISED_SCENARIO_DB.load()
print(scenarios_regional.shape)
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
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)
print(scenarios_regional.shape)
# sys.exit(4)
history_regional = HISTORY_HARMONISATION_DB.load()
print(history.pix.unique("variable"))
print(history.pix.unique("region"))
print(history_regional.loc[pix.ismatch(variable="Emissions|CO2**")].pix.unique("variable"))


cumulative_history_afolu = get_cumulative_afolu(history, "GCB-extended", "historical")
print(cumulative_history_afolu[2021])
print(cumulative_history_afolu[2022])
print(cumulative_history_afolu[2023])
print(cumulative_history_afolu[2024])
print(cumulative_history_afolu[2030])
print(cumulative_history_afolu[2050])
print(cumulative_history_afolu[2100])


# AFOLU section
def calculate_afolu_extensions(scenarios_complete_global, history, cumulative_history_afolu, plot=True):
    """
    Calculate AFOLU extensions for all scenarios and models
    """
    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))
    temp_list_for_new_data = []
    temp_list_for_new_data_flat = []
    temp_list_for_new_data_flat_cumulative = []
    temp_list_for_new_data_linear_ramp_down = []
    for s, meta in scenario_model_match.items():
        scen = scenarios_complete_global.loc[pix.ismatch(variable="**CO2|AFOLU", model=meta[1], scenario=meta[0])]
        scen_full = glue_with_historical(scen, history.loc[pix.ismatch(variable="Emissions|CO2|AFOLU")])
        cumulative_2100 = get_cumulative_afolu_fill_from_hist(scen, meta[1], meta[0], cumulative_history_afolu)
        em_ext, cle_inf = find_func_form_lu_extension(
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
                np.arange(cumulative_2100.columns[0], 2501), np.cumsum(em_ext), "--", alpha=0.7, label=s, color=meta[2]
            )
            axs[1].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
            axs[1].plot(np.arange(cumulative_2100.columns[0], 2501), em_ext, "--", alpha=0.7, label=s, color=meta[2])
            axs[2].plot(cumulative_2100.columns, cumulative_2100.values[0, :], color=meta[2])
            axs[2].plot(
                np.arange(cumulative_2100.columns[0], 2501), np.cumsum(em_ext), "--", alpha=0.7, label=s, color=meta[2]
            )
            axs[3].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
            axs[3].plot(np.arange(cumulative_2100.columns[0], 2501), em_ext, "--", alpha=0.7, label=s, color=meta[2])
        df_afolu = pd.DataFrame(data=[em_ext], columns=np.arange(cumulative_2100.columns[0], 2501), index=scen.index)
        df_afolu_flat = pd.DataFrame(
            data=[em_ext_flat], columns=np.arange(cumulative_2100.columns[0], 2501), index=scen.index
        )
        df_afolu_flat_cumulative = pd.DataFrame(
            data=[em_ext_flat_cumulative], columns=np.arange(cumulative_2100.columns[0], 2501), index=scen.index
        )
        df_afolu_linear_ramp_down = pd.DataFrame(
            data=[em_ext_linear_ramp_down], columns=np.arange(cumulative_2100.columns[0], 2501), index=scen.index
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


component_global_targets = {
    "Emissions|CH4": {
        "VLLO": 95.0,
        "VLHO": 150.0,
        "L": 95.0,
        "ML": 120.0,
        "M": 450.0,
        "MOS": 95.0,
        "H": 520.0,
        "HL": 110.0,
    },
    "Emissions|Sulfur": {
        "VLLO": 20.0,
        "VLHO": 10.0,
        "L": None,
        "ML": 20.0,
        "M": 20.0,
        "MOS": 20.0,
        "H": 50.0,
        "HL": 10.0,
    },
}


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
            if "workflow" in df_comp_scen_model.index.names:
                df_comp_scen_model = df_comp_scen_model.reset_index("workflow", drop=True)
            total_df_list.append(df_comp_scen_model)
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
    df_all = pd.concat(total_df_list, axis=0)
    return df_all


do_and_write_to_csv = True
if do_and_write_to_csv:
    df_all = do_all_non_co2_extensions(scenarios_complete_global, history)
    df_all.to_csv("first_draft_extended_nonCO2_all.csv")
    afolu_dfs = calculate_afolu_extensions(scenarios_complete_global, history, cumulative_history_afolu, plot=True)
    print(df_all.index)

    for name, afolu_df in afolu_dfs.items():
        print(afolu_df.index)
        afolu_df.to_csv(f"first_draft_extended_afolu_{name}.csv")
    # sys.exit(4)

else:
    df_all = pd.read_csv(
        "first_draft_extended_nonCO2_all.csv", index_col=["variable", "unit", "scenario", "model", "region"]
    )
    afolu_dfs = {}
    for afolu_file in glob.glob("first_draft_extended_afolu_*.csv"):
        name = afolu_file.split("first_draft_extended_")[-1].split(".csv")[0]
        afolu_dfs[name] = pd.read_csv(afolu_file)
# sys.exit(4)

res_gwps = calculate_nonco2_ghgs_gwp(
    df_all.loc[pix.ismatch(region="World"), ~pix.ismatch(variable="**Shipping"), ~pix.ismatch(variable="**Aircraft")],
    gwp="AR6GWP100",
)
print(res_gwps.head())
sys.exit(4)

fossil_evolution_dictionary = {
    "VLLO": ["CS", 2300, 2350],
    "VLHO": ["CS", 2200, 2300],
    "L": ["CS", 2240, 2300],
    "ML": ["ECS", 2125, -13e3, 2275, 2300],
    "M": ["CS", 2100, 2200],
    "MOS": ["CSCS", 2105, -20e3, 2175, 2325, 2350],
    "H": ["CS", 2175, 2275],
    "HL": ["CSCS", 2110, -36e3, 2200, 2400, 2450],
}

fossil_evolution_dictionary = {
    "VLLO": ["CS", 2300, 2350],
    "VLHO": ["ECS", 2150, None, 2200, 2300],
    "L": ["ECS", 2150, None, 2240, 2300],
    "ML": ["ECS", 2150, -13e3, 2250, 2300],
    "M": ["ECS", 2150, None, 2150, 2250],
    "MOS": ["CSCS", 2105, -20e3, 2175, 2300, 2350],
    "H": ["ECS", 2150, None, 2175, 2275],
    "HL": ["CSCS", 2100, -36e3, 2200, 2400, 2450],
}

df_per_afolu = {}
for name, df_afolu in afolu_dfs.items():
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 15))
    temp_list_for_new_data = []
    for s, meta in scenario_model_match.items():
        print(s)
        print(meta)
        co2_fossil = interpolate_to_annual(
            scenarios_complete_global.loc[
                pix.ismatch(variable="Emissions|CO2|Energy and Industrial Processes", model=meta[1], scenario=meta[0])
            ]
        )
        try:
            co2_afolu = df_afolu.loc[(df_afolu["model"] == meta[1]) & (df_afolu["scenario"] == meta[0])]
        except:  # noqa: E722
            co2_afolu = df_afolu.loc[pix.ismatch(model=meta[1], scenario=meta[0])]
        co2_total_extend, co2_fossil_extend, extend_years = extend_co2_for_scen_storyline(
            co2_afolu, co2_fossil, fossil_evolution_dictionary[s]
        )

        df_total = pd.DataFrame(data=[co2_fossil_extend], columns=extend_years, index=co2_fossil.index)
        temp_list_for_new_data.append(df_total)

        axs[0].plot(co2_fossil.columns, co2_fossil.values.flatten(), label=s, color=meta[2])
        axs[0].plot(extend_years, co2_fossil_extend, label=s, color=meta[2], linestyle="--")
        axs[0].plot(co2_fossil.columns, co2_fossil.values.flatten(), label=s, color=meta[2])
        axs[1].plot(
            extend_years, co2_afolu.loc[:, "2023":].to_numpy().flatten(), label=s, color=meta[2], linestyle="--"
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
    df_per_afolu[name] = fossil_extension_df
    axs[0].set_title("CO2 fossil", fontsize="x-large")
    axs[1].set_title("CO2 AFOLU", fontsize="x-large")
    axs[2].set_title("CO2 total", fontsize="x-large")
    for ax in axs:
        ax.set_xlabel("Years", fontsize="x-large")
    axs[2].legend(fontsize="x-large")

    plt.savefig(f"co2_fossil_fuel_extenstions_{name}.png")
    # plt.show()
print("In history it works likte this:")
print(history.head())
print("In df from file it looks like this:")
print(df_all.head())
print("In history it works likte this:")
print(history.index)
print("In df from file it looks like this:")
print(df_all.index)
# for name, df_afolu in afolu_dfs.items():
#    print(df_afolu.head())
#    print(df_per_afolu[name].head())


# EXTENSIONS_OUTPUT_DB.save(df_all)
