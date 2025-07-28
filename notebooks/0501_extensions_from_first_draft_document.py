# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook to mock up extension scenarios

# %% [markdown]
# Trying to make scenario extensions

# %% [markdown]
# ## Imports

# %% [markdown]
# General imports :

# %%

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# Specific imports from this package
# %%
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import tqdm.auto

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_SCENARIOS_DB,
)

# from emissions_harmonization_historical.constants import DATA_ROOT
# from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.extension_functionality import (
    do_simple_sigmoid_or_exponential_extension_to_target,
    find_func_form_lu_extension,
    sigmoid_function,
)

# %% [markdown]
# ## Set up

# %%
save_plots = True

pandas_openscm.register_pandas_accessor()

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %% [markdown]
# ## Getting scenario data

# %% [markdown]
# Starting with global data:

# %%
scenarios_complete_global = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete")).reset_index("stage", drop=True)
scenarios_complete_global  # TODO: drop 2100 end once we have usable scenario data post-2100
for model in scenarios_complete_global.pix.unique("model").values:
    print(model)
    # print(scenarios_complete_global.loc[pix.ismatch(model=f"{model}")].pix.unique("scenario"))
# scenarios_complete_global

# %% [markdown]
# Defining which scenario and models to match and use:

# %%
# sys.exit(4)
scenario_model_match = {
    "VLLO": ["SSP1 - Very Low Emissions", "REMIND-MAgPIE 3.5-4.10", "tab:blue"],
    "VLHO": ["SSP2 - Low Overshoot", "AIM 3.0", "tab:cyan"],
    "L": ["SSP2 - Low Emissions", "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "tab:green"],
    "ML": ["SSP2 - Medium-Low Emissions", "COFFEE 1.6", "tab:pink"],
    "M": ["SSP2 - Medium Emissions", "IMAGE 3.4", "tab:purple"],
    "H": ["SSP3 - High Emissions", "GCAM 7.1 scenarioMIP", "tab:red"],
    "HL": ["SSP5 - High Emissions", "WITCH 6.0", "tab:brown"],
}

# %% [markdown]
# Loading regional scenario data:

# %%

scenarios_regional = HARMONISED_SCENARIO_DB.load()
for stype, model_scen_match in scenario_model_match.items():
    model = model_scen_match[1]
    scenario = model_scen_match[0]
    print(f"{stype}: {model=} {scenario=}")
    print(len(scenarios_regional.loc[pix.ismatch(model=f"{model}", scenario=f"{scenario}")].pix.unique("region")))
    # scenarios_regional.pix.unique("region") # TODO: drop 2100 end once we have usable scenario data post-2100
    # regions = scenarios_regional.loc[
    #   pix.ismatch(model=f"{model}", scenario=f"{scenario}")
    # ].pix.unique("region").values
    # regional_variables = (
    #    scenarios_regional.loc[pix.ismatch(model=f"{model}", scenario=f"{scenario}", region=f"{regions[1]}")]
    #    .pix.unique("variable")
    #    .values
    # )
    # print(regional_variables.shape)
    # for variable in regional_variables:
    # if variable.count("|")> 1:
    #    continue
    # print(variable)
    # scenarios_regional
scenarios_regional = scenarios_regional.sort_index(axis="columns").T.interpolate("index").T


# %% [markdown]
# ## History

# %% [markdown]
# Need history to glue together to get correct extensions

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)
history_regional = HISTORY_HARMONISATION_DB.load()
print(history.pix.unique("variable"))
print(history.pix.unique("region"))
print(history_regional.loc[pix.ismatch(variable="Emissions|CO2**")].pix.unique("variable"))
# print(history.loc[pix.ismatch(variable="Emissions|CO2|AFOLU")].loc[:, 2015:2025])
# print(history.pix.unique("region"))
history
# history.loc[:, :2023]

# %% [markdown]
# ## Get some AR6 RCMIP-data to find cumulative land use and pre-industrial emissions

# %%
# Probably don't need any of this section anymore...
# RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
# RCMIP_PATH

# %%


def transform_rcmip_to_iamc_variable(v: str) -> str:
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (
        ("F-Gases|", ""),
        ("PFC|", ""),
        ("HFC4310mee", "HFC43-10"),
        ("MAGICC AFOLU", "AFOLU"),
        ("MAGICC Fossil and Industrial", "Energy and Industrial Processes"),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
"""
rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
ar6_history = rcmip_clean.loc[pix.isin(mip_era=["CMIP6"], scenario=["ssp245"], region=["World"])]
ar6_history = (
    ar6_history.loc[
        ~pix.ismatch(
            variable=[
                f"Emissions|{stub}|**" for stub in ["BC", "CH4", "CO", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC"]
            ]
        )
        & ~pix.ismatch(variable=["Emissions|CO2|*|**"])
        & ~pix.isin(variable=["Emissions|CO2"])
    ]
    .T.interpolate("index")
    .T
)
full_var_set = ar6_history.pix.unique("variable")
n_variables_in_full_scenario = 52
if len(full_var_set) != n_variables_in_full_scenario:
    raise AssertionError

print(ar6_history.shape)
print(full_var_set)
print(ar6_history.pix.unique("model"))
ar6_history.pix.unique("scenario")
"""


# %% [markdown]
# ## Functions to interpolate to annual data, and get cumulative AFOLU


# %%
def interpolate_to_annual(idf: pd.DataFrame, max_supplement: float = 1e-5) -> pd.DataFrame:
    """
    Interpolate pd.DataFrame of emissions input that might not have data for all years
    """
    # TODO: push into pandas-openscm
    missing_cols = np.setdiff1d(np.arange(idf.columns.min(), idf.columns.max() + max_supplement), idf.columns)

    out = idf.copy()
    out.loc[:, missing_cols] = np.nan
    out = out.sort_index(axis="columns").T.interpolate("index").T

    return out


# %%
# Probably not needed
def glue_with_historical(scen_df: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Glue historical data to the beginning of scenario data to get complete timeseries
    """
    out = interpolate_to_annual(scen_df.copy())
    orig_len = len(out.columns)
    missing_years = np.arange(int(hist_df.columns[0]), int(scen_df.columns[0]))
    out.loc[:, missing_years] = np.nan  # hist_df.loc[:, missing_years].values
    for index, df_row in out.iterrows():
        df_row.iloc[orig_len:] = hist_df.loc[pix.ismatch(variable=index[3])].values[
            0, : int(scen_df.columns[0] - hist_df.columns[0])
        ]
        for year in range(scen_df.columns[0], 2015):
            if np.isnan(df_row.iloc[year - scen_df.columns[0]]):
                df_row.iloc[year - scen_df.columns[0]] = hist_df.loc[pix.ismatch(variable=index[3])].values[
                    0, int(year - hist_df.columns[0])
                ]
        out.loc[index, :] = df_row.values
    out = out.sort_index(axis="columns")
    return out


# %%
def get_cumulative_afolu(input_df: pd.DataFrame, model: str, scenario: str, emi_kind="**CO2|AFOLU") -> pd.DataFrame:
    """
    From yearly AFOLU DataFrame, calculate cumulative AFOLU
    """
    emissions = interpolate_to_annual(input_df.loc[pix.ismatch(variable=emi_kind, model=model, scenario=scenario)])
    # print(emissions.values.shape)
    cumulative = pd.DataFrame(
        data=np.nan_to_num(emissions.values).cumsum(axis=1),
        columns=emissions.columns,
        index=pd.MultiIndex.from_tuples(
            [
                tuple([x.replace("Emissions", "Cumulative Emissions") for x in tuple_index])
                for tuple_index in emissions.index
            ]
        ),
    )
    # print(cumulative.shape)
    return cumulative


# %%
cumulative_history_afolu = get_cumulative_afolu(history, "GCB-extended", "historical")
print(cumulative_history_afolu[2021])
print(cumulative_history_afolu[2022])
print(cumulative_history_afolu[2023])
print(cumulative_history_afolu[2024])
print(cumulative_history_afolu[2030])
print(cumulative_history_afolu[2050])
print(cumulative_history_afolu[2100])


# %%
def get_cumulative_afolu_fill_from_hist(
    input_df: pd.DataFrame, model: str, scenario: str, hist_fill, emi_kind="**CO2|AFOLU"
) -> pd.DataFrame:
    """
    Calculate cumulative AFOLU including historical AFOLU
    """
    # print(input_df.columns)
    cumulative = get_cumulative_afolu(input_df, model, scenario, emi_kind=emi_kind)
    # for col in

    # print(cumulative.shape)
    # print(hist_fill.columns)
    # print(cumulative.columns)
    # print(cumulative.iloc[:, :20])
    full_cumulative = np.zeros((1, 2100 - int(hist_fill.columns[0]) + 1))
    # print(cumulative.columns[0])
    # print(hist_fill.columns[0])
    first_scen_year_idx = int(cumulative.columns[0] - hist_fill.columns[0])
    print(first_scen_year_idx)
    # sys.exit(4)
    full_cumulative[0, :first_scen_year_idx] = hist_fill.values[0, :first_scen_year_idx]
    just_add_scen = first_scen_year_idx + hist_fill.columns[0]
    for i_year in range(first_scen_year_idx, 2100 + 1 - int(hist_fill.columns[0])):
        # print(f"iyear: {i_year}, is real year: {i_year + int(hist_fill.columns[0]) }")
        # print(f"and scen_year:{i_year + int(hist_fill.columns[0]-cumulative.columns[0])}")
        if cumulative.iloc[0, i_year + int(hist_fill.columns[0] - cumulative.columns[0])] > 0:
            just_add_scen = i_year + int(hist_fill.columns[0] - cumulative.columns[0])
            to_add = hist_fill.values[0, i_year] - 1
            break
        else:
            full_cumulative[0, i_year] = hist_fill.values[0, i_year]
    # print(f"len full: {len(full_cumulative)}, len cum: {len(cumulative.values[0,:])}, just_add_scen: {just_add_scen}")
    full_cumulative[0, just_add_scen + int(cumulative.columns[0]) - int(hist_fill.columns[0]) :] = (
        to_add + cumulative.iloc[0, just_add_scen:].values
    )
    cum_df = pd.DataFrame(
        data=full_cumulative, columns=np.arange(int(hist_fill.columns[0]), 2101), index=cumulative.index
    )
    return cum_df


# %% [markdown]
# ### Definitions to split CO2 sectors

# %%
AFOLU_sectors = [
    "Emissions|CO2|Agricultural Waste Burning",
    "Emissions|CO2|Forest Burning",
    "Emissions|CO2|Grassland Burning",
    "Emissions|CO2|Peat Burning",
    "Emissions|CO2|Agriculture",
]
Fossil_sectors = [
    "Emissions|CO2|Aircraft",
    "Emissions|CO2|International Shipping",
    "Emissions|CO2|Energy Sector",
    "Emissions|CO2|Industrial Sector",
    "Emissions|CO2|Residential Commercial Other",
    "Emissions|CO2|Solvents Production and Application",
    "Emissions|CO2|Transportation Sector",
    "Emissions|CO2|Waste",
]
global_fossil = "Emissions|CO2|Energy and Industrial Processes"
global_afolu_sector = "Emissions|CO2|AFOLU"
fossil_evolution_dictionary = {
    "VLLO": ["CS"],
    "VLHO": ["ECS"],
    "L": ["CS"],
    "ML": ["ECS"],
    "M": ["CS"],
    "H": ["CS"],
    "HL": ["CSCS"],
}

# %% [markdown]
# # Extensions proper

# %% [markdown]
# ### Start dealing with AFOLU first

# %%
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))
temp_list_for_new_data = []
for s, meta in scenario_model_match.items():
    scen = scenarios_complete_global.loc[pix.ismatch(variable="**CO2|AFOLU", model=meta[1], scenario=meta[0])]

    # print(scen.columns[-10:])
    # print(scen.columns[:10])
    print(f"{s}: {meta}")
    print("----------------------")
    # sys.exit(4)
    # print(scen.index.unique(level='scenario'))
    # print(scen.index.unique(level='model'))
    # print(scen)
    # print(scen.columns)
    scen_full = glue_with_historical(scen, history.loc[pix.ismatch(variable="Emissions|CO2|AFOLU")])
    # print(scen_full.loc[])
    cumulative_2100 = get_cumulative_afolu_fill_from_hist(scen, meta[1], meta[0], cumulative_history_afolu)  # [0,-1]
    em_ext, cle_inf = find_func_form_lu_extension(
        scen_full.values[0, :],
        cumulative_2100.values[0, :],
        np.arange(cumulative_2100.columns[0], 2501),
        2100 - int(cumulative_2100.columns[0]),
        cle_inf_0=True,
    )
    # print(len(em_ext))
    # print(cle_inf)
    # print(em_ext[349:360])
    axs[0].plot(cumulative_2100.columns, cumulative_2100.values[0, :], color=meta[2])
    axs[0].plot(np.arange(cumulative_2100.columns[0], 2501), np.cumsum(em_ext), "--", alpha=0.7, label=s, color=meta[2])
    axs[1].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
    axs[1].plot(np.arange(cumulative_2100.columns[0], 2501), em_ext, "--", alpha=0.7, label=s, color=meta[2])
    axs[2].plot(cumulative_2100.columns, cumulative_2100.values[0, :], color=meta[2])
    axs[2].plot(np.arange(cumulative_2100.columns[0], 2501), np.cumsum(em_ext), "--", alpha=0.7, label=s, color=meta[2])
    axs[3].plot(scen_full.columns, scen_full.values[0, :], color=meta[2])
    axs[3].plot(np.arange(cumulative_2100.columns[0], 2501), em_ext, "--", alpha=0.7, label=s, color=meta[2])
    df_afolu = pd.DataFrame(data=[em_ext], columns=np.arange(cumulative_2100.columns[0], 2501), index=scen.index)
    temp_list_for_new_data.append(df_afolu)

extended_data = pd.concat(temp_list_for_new_data)
# print(extended_data)
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
plt.show()


# %% [markdown]
# Now regional AFOLU:


# %%
# Loop over and make per regional for AFOLU, see how global pans out when we do sectoral splits...
def do_AFOLU_for_scenario_model_regionally(scen: str, model: str):
    """
    For given scenario, model and variable, do extensions per sector and region and combine
    """
    pass
    """
    data_scenario_global = scenarios_complete_global.loc[
        pix.ismatch(scenario=f"{scen}", model=f"{model}", variable="Emissions|CO2**")
    ]
    data_historical = history.loc[pix.ismatch(variable="Emissions|CO2**")]
    # data_min = np.nanmin(data_scenario_global.values[0,:])
    # global_target = np.min((data_min, data_historical.values[0, 0]))

    data_regional = scenarios_regional.loc[
        pix.ismatch(scenario=f"{scen}", model=f"{model}", variable="Emissions|CO2**")
    ]
    data_regional = interpolate_to_annual(data_regional)
    data_history_regional = history_regional.loc[pix.ismatch(variable="Emissions|CO2**")]
    full_years = np.arange(data_regional.columns[0], 2501)
    pdf = data_regional.openscm.to_long_data()
    fg = sns.relplot(
        data=pdf,
        x="time",
        y="value",
        col="variable",
        col_order=sorted(pdf["variable"].unique()),
        col_wrap=3,
        hue="region",
        hue_order=sorted(pdf["region"].unique()),
        kind="line",
        linewidth=2.0,
        alpha=0.7,
        facet_kws=dict(sharey=False),
        errorbar=None,
    )
    # fg.show()
    # sys.exit(4)
    """
    """
    sectors = data_regional.pix.unique("variable")
    regions = data_regional.pix.unique("region")

    # Hook for species that are not regional, and just infilled
    if len(regions) <= 1 and len(sectors) <= 1:
        scen_full = interpolate_to_annual(data_scenario_global)
        data_extend =  do_simple_sigmoid_or_exponential_extension_to_target(
            scen_full.values[0,:],
            np.arange(scen_full.columns[0], 2501),
            2100 - int(scen_full.columns[0]),
            global_target
        )
        df_regional = pd.DataFrame(data=[data_extend], columns=full_years, index=data_scenario_global.index)
        return df_regional
    temp_list_for_regional = []
    total_sector = None
    world_sector = None
    target_sum = 0
    for sector in sectors:
        if sector == variable:
            continue
        data_sector =  data_regional.loc[pix.ismatch(variable=f"{sector}")]
        regions = data_sector.pix.unique("region")
        for region in regions:
            if region == "World" and len(regions)>1:
                continue
            data = data_sector.loc[pix.ismatch(region=f"{region}")] #, variable=f"{sector}")]
            target = global_target * fractions.loc[pix.ismatch(variable = f"{sector}", region= f"{region}")].values[0,0]
            target_sum = target_sum + target
            data_extend = do_simple_sigmoid_or_exponential_extension_to_target(
                data.values[0,:],
                full_years,
                2100 - int(data.columns[0]),
                target,
            )
            #print(data_extend.columns)
            #print(data)
            #print(data_extend)
            #sys.exit(4)
            # 2150 + sigmoid_shift,
            # full_years[len(data.columns) :],
            df_regional = pd.DataFrame(data=[data_extend], columns=full_years, index=data.index)
            if total_sector is None:
                total_sector = data_extend
                world_sector = data_extend
            else:
                total_sector = total_sector + data_extend
                world_sector = total_sector + data_extend

            temp_list_for_regional.append(df_regional)
    df_total = pd.DataFrame(
        data= [world_sector, total_sector],
        columns = full_years,
        index= data_regional.loc[pix.ismatch(variable=f"{variable}")].index
        )

    temp_list_for_regional.append(df_total)
    df_all = pd.concat(temp_list_for_regional)
    #print(df_all[2500])
    #print(global_target)
    #print(target_sum)
    #print(sectors)
    #print(regions)
    #print(data_regional[2100])
    #print(data_regional[2023])
    #print(data_scenario_global[2023])
    #print(data_historical[2023])
    #print(df_all[2500])
    return df_all
    """


# %%
for s, meta in tqdm.auto.tqdm(scenario_model_match.items()):
    print(history_regional.pix.unique("variable"))
    data_historical_regional = history_regional.loc[pix.ismatch(variable="Emissions|CO2**")]
    # pdf = data_historical_regional.openscm.to_long_data()
    # print(pdf.head())
    # print(pdf.columns)
    # fg = sns.relplot(
    #        data=pdf,
    #        x="time",
    #        y="value",
    #        col="variable",
    #        col_order=sorted(pdf["variable"].unique()),
    #        col_wrap=3,
    #        hue="region",
    #        hue_order=sorted(pdf["region"].unique()),
    #        kind="line",
    #        linewidth=2.0,
    #        alpha=0.7,
    #        facet_kws=dict(sharey=False),
    #        errorbar=None
    #        )
    # sys.exit(4)
    do_AFOLU_for_scenario_model_regionally(scen=meta[0], model=meta[1])

# %% [markdown]
# ### Then move on to the non-CO2 GHGs

# %%
print(history.pix.unique("variable"))


# %%


def do_simple_sigmoid_extension_to_target(scen_full: pd.DataFrame, target: float, sigmoid_shift=40) -> np.ndarray:
    """
    Calculate extension function by calling sigmoid functionality to extend
    """
    full_years = np.arange(scen_full.columns[0], 2501)
    data_extend = np.zeros(len(full_years))
    data_extend[: len(scen_full.columns)] = scen_full.values[0, :]
    # print(f"Arguments for sigmoid: {target: }, come from: {scen_full.values[0, -1]},
    # start_time: {scen_full.columns[-1] + sigmoid_shift},
    #   transition over: {2150 + sigmoid_shift}")
    data_extend[len(scen_full.columns) :] = sigmoid_function(
        target,
        scen_full.values[0, -1],
        scen_full.columns[-1] + sigmoid_shift,
        2150 + sigmoid_shift,
        full_years[len(scen_full.columns) :],
    )
    # print(data_extend)
    return data_extend


# %%
def get_2100_compound_composition(data_regional: pd.DataFrame, variable: str):
    """
    Find fractional composition of values in 2100 to allocate the residual end point emissions accordingly
    """
    # data_transform = data_regional
    # value_global = data_regional.loc[pix.ismatch(variable=f"{variable}")]
    data_rest = data_regional.loc[~pix.ismatch(variable=f"{variable}")]
    total_from_sectors = data_rest.sum(axis=0)
    fractions = data_rest.values / total_from_sectors
    fractions_df = pd.DataFrame(data=fractions, columns=["fractions"], index=data_rest.index)
    return fractions_df


def do_single_component_for_scenario_model_regionally(scen: str, model: str, variable: str, sigmoid_shift=40):
    """
    For given scenario, model and variable, do extensions per sector and region and combine
    """
    data_scenario_global = scenarios_complete_global.loc[
        pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}")
    ]
    data_historical = history.loc[pix.ismatch(variable=f"{variable}")]
    data_min = np.nanmin(data_scenario_global.values[0, :])
    global_target = np.min((data_min, data_historical.values[0, 0]))

    data_regional = scenarios_regional.loc[pix.ismatch(scenario=f"{scen}", model=f"{model}", variable=f"{variable}**")]
    if variable == "Emissions|CO":
        data_regional = data_regional.loc[~pix.ismatch(variable="Emissions|CO2**")]
    data_regional = interpolate_to_annual(data_regional)

    full_years = np.arange(data_regional.columns[0], 2501)

    fractions = get_2100_compound_composition(data_regional[2100].copy(), variable)

    sectors = data_regional.pix.unique("variable")
    regions = data_regional.pix.unique("region")

    # Hook for species that are not regional, and just infilled
    if len(regions) <= 1 and len(sectors) <= 1:
        scen_full = interpolate_to_annual(data_scenario_global)
        data_extend = do_simple_sigmoid_or_exponential_extension_to_target(
            scen_full.values[0, :],
            np.arange(scen_full.columns[0], 2501),
            2100 - int(scen_full.columns[0]),
            global_target,
        )
        df_regional = pd.DataFrame(data=[data_extend], columns=full_years, index=data_scenario_global.index)
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
            data = data_sector.loc[pix.ismatch(region=f"{region}")]  # , variable=f"{sector}")]
            # target =
            target = global_target * fractions.loc[pix.ismatch(variable=f"{sector}", region=f"{region}")].values[0, 0]
            target_sum = target_sum + target
            data_extend = do_simple_sigmoid_or_exponential_extension_to_target(
                data.values[0, :],
                full_years,
                2100 - int(data.columns[0]),
                target,
            )
            # print(data_extend.columns)
            # print(data)
            # print(data_extend)
            # sys.exit(4)
            # 2150 + sigmoid_shift,
            # full_years[len(data.columns) :],
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
    # print(df_all[2500])
    # print(global_target)
    # print(target_sum)
    # print(sectors)
    # print(regions)
    # print(data_regional[2100])
    # print(data_regional[2023])
    # print(data_scenario_global[2023])
    # print(data_historical[2023])
    # print(df_all[2500])
    return df_all


# %%
def plot_just_global(scen: str, model: str, variable: str, df_extended: pd.DataFrame):
    """
    Make global value plots
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
    # print(extended_to_plot)
    ax.plot(extended_to_plot.columns, extended_to_plot.values[-1, :])
    if unextended.shape[0] > 0:
        ax.plot(unextended.columns, unextended.values[-1, :], linestyle="--", alpha=0.7)
    # print(unextended.columns)
    # print(unextended.values[-1,:])
    # unextended.values[0,:]
    plt.savefig(f"extended_match_totals_{scen.replace(' ', '')}_{model.replace(' ', '')}_{variable.split('|')[-1]}.png")
    plt.clf()


# %%
total_df_list = []
look_at_all = True

for variable in tqdm.auto.tqdm(scenarios_complete_global.pix.unique("variable").values):
    print(variable)
    # print(history.loc[pix.ismatch(variable=f"{variable}")].shape)
    if variable.startswith("Emissions|CO2"):
        continue
    if history.loc[pix.ismatch(variable=f"{variable}")].shape[0] < 1:
        continue
    for s, meta in tqdm.auto.tqdm(scenario_model_match.items()):
        df_comp_scen_model = do_single_component_for_scenario_model_regionally(meta[0], meta[1], variable)
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
                meta[0], meta[1], variable, df_comp_scen_model.loc[pix.ismatch(region="World", variable=f"{variable}")]
            )
        # sys.exit(4)
df_all = pd.concat(total_df_list)


# %% [markdown]
# ## Now look at fossil CO2

# %%

fig, axs = plt.subplots(nrows=1, ncols=1)
temp_list_for_new_data = []
for s, meta in scenario_model_match.items():
    print(s)
    print(meta)
    co2_fossil = interpolate_to_annual(
        scenarios_complete_global.loc[
            pix.ismatch(variable="Emissions|CO2|Energy and Industrial Processes", model=meta[1], scenario=meta[0])
        ]
    )
    # print(co2_fossil.describe())
    print(
        scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2**", model=meta[1], scenario=meta[0])].pix.unique(
            "region"
        )
    )
    print(
        scenarios_regional.loc[pix.ismatch(variable="Emissions|CO2**", model=meta[1], scenario=meta[0])].pix.unique(
            "variable"
        )
    )
    axs.plot(co2_fossil.columns, co2_fossil.values.flatten(), label=s, color=meta[2])
axs.set_xlabel("Years")
axs.set_ylabel("CO2 fossil")
axs.legend()
plt.show()


# %%
# CO2 also per sector and region

# %% [markdown]
# # Save output

# %%
# Save input global from historical for scm-runs and total?
# Add in afolu-data


# Add in CO2 data

# EXTENSIONS_OUTPUT_DB.save(df_all)

# %%
