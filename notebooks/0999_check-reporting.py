# ---
# jupyter:
#   jupytext:
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
# # Check reporting
#
# Check reporting from the IAMs.

# %% [markdown]
# ## Imports

# %%
import gcages.completeness
import nomenclature
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pyam
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor

# %%
from tqdm.auto import tqdm

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    REPO_ROOT,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.io import load_raw_scenario_data
from emissions_harmonization_historical.nomenclature_helpers import get_common_definitions

# %% [markdown]
# ## Set up

# %%
pd.set_option("display.max_colwidth", None)

# %%
pandas_openscm.register_pandas_accessor()

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
COMMON_DEFINITIONS_PATH = REPO_ROOT / "common-definitions"

# %% [markdown]
# ## Initialise

# %% [markdown]
# ### Scenario data

# %%
scenarios_raw = load_raw_scenario_data(
    scenario_path=DATA_ROOT / "scenarios" / "data_raw",
    scenario_time_id=SCENARIO_TIME_ID,
    progress=True,
).loc[:, :2100]  # TODO: drop 2100 end once we have usable scenario data post-2100
scenarios_raw.columns.name = "year"
scenarios_raw

# %%
# can be helpful for inspecting
# pd.set_option("display.max_rows", None)
# scenarios_raw.loc[pix.ismatch(variable="**Aviation**")].pix.unique(["model", "variable", "region"]).to_frame(
#     index=False
# ).sort_values("model")

# %% [markdown]
# ## Check with pre-processor

# %%
pre_processor = CMIP7ScenarioMIPPreProcessor()
pre_processor

# %%
passing_models = []
model_reported_times_d = {}
for model, mdf in scenarios_raw.groupby("model"):
    mdf_reported_times = mdf.dropna(how="all", axis="columns")
    mdf_reported_times_relevant_regions = mdf_reported_times.loc[
        pix.ismatch(region=["World", f"{model.split(' ')[0]}**"])
    ]
    model_reported_times_d[model] = mdf_reported_times_relevant_regions

    try:
        pre_processor(mdf_reported_times)
    except (gcages.completeness.NotCompleteError, AssertionError):
        print()
        print(f"{model} failed")
        continue

    passing_models.append(model)

# %% [markdown]
# Not a great start, but also not unexpected.

# %% [markdown]
# ### Individual model deeper dive

# %%
from gcages.cmip7_scenariomip.pre_processing import (
    REQUIRED_REGIONAL_INDEX_IAMC,
    REQUIRED_WORLD_INDEX_IAMC,
    split_world_and_regional_data,
)
from gcages.completeness import get_missing_levels

# %%
# # TODO: push this back into gcages,
# # these CO2 burning variables aren't actually required
# REQUIRED_REGIONAL_INDEX_IAMC = REQUIRED_REGIONAL_INDEX_IAMC.drop(
#     [
#         "Emissions|CO2|AFOLU|Land|Fires|Forest Burning",
#         "Emissions|CO2|AFOLU|Land|Fires|Grassland Burning",
#         "Emissions|CO2|AFOLU|Land|Fires|Peat Burning",
#     ]
# )


# %%
def check_if_any_scenario_passes(model_df: pd.DataFrame) -> list[str]:
    passing_l = []
    for scenario, sdf in model_df.groupby("scenario"):
        try:
            pre_processor(sdf)
        except gcages.completeness.NotCompleteError:
            print(f"{scenario} failed")
            continue

        passing_l.append(scenario)

    passing_l


# %%
def get_missing_info(df_idx: pd.MultiIndex, complete_index: pd.MultiIndex) -> pd.DataFrame:
    return (
        get_missing_levels(
            df_idx, complete_index=complete_index, levels_to_drop=df_idx.names.difference(complete_index.names)
        )
        .pix.extract(variable="{table}|{species}|{sector}")
        .to_frame(index=False)
    )


def get_missing_summary_scenario(sdf: pd.DataFrame, world_region: str = "World") -> pd.DataFrame:
    split_data = split_world_and_regional_data(sdf, world_region=world_region)

    missing_world = get_missing_info(
        split_data.world.index,
        complete_index=REQUIRED_WORLD_INDEX_IAMC,
    )

    missing_regional = get_missing_info(
        split_data.regional.index,
        complete_index=REQUIRED_REGIONAL_INDEX_IAMC,
    )

    missing = pd.concat([missing_world, missing_regional])
    missing["missing"] = True

    return missing


def get_model_missing(model_df: pd.DataFrame) -> pd.DataFrame:
    model_missing_l = []
    for scenario, sdf in model_df.groupby("scenario"):
        scenario_missing = get_missing_summary_scenario(sdf)
        scenario_missing["scenario"] = scenario
        scenario_missing["model"] = model

        model_missing_l.append(scenario_missing)

    model_missing = pd.concat(model_missing_l)

    return model_missing


# %%
def get_model_missing_total(model_missing: pd.DataFrame) -> pd.DataFrame:
    return model_missing.groupby(["scenario"])["missing"].sum().sort_values(ascending=True)


# %%
def get_model_missing_styled_summary(model_missing: pd.DataFrame, model_missing_total: pd.DataFrame) -> pd.DataFrame:
    return (
        model_missing.pivot_table(
            values=["missing"],
            columns=["species"],
            index=["scenario", "model", "sector"],
            aggfunc=lambda x: any(~pd.isnull(xx) for xx in x.values),
        )
        .loc[model_missing_total.index]
        .reorder_levels(["model", "scenario", "sector"])
        .style.highlight_max(color="orange")
    )


# %%
world_region = "World"

# %% [markdown]
# #### AIM

# %%
model = "AIM 3.0"
model_df = model_reported_times_d[model]

# %%
# temporary hack so we only deal with one scenario
model_df = model_df.loc[pix.isin(scenario=model_df.pix.unique("scenario").tolist()[0])]

# %% [markdown]
# Let's see if any scenario passes.

# %%
passing_scenarios = check_if_any_scenario_passes(model_df)
passing_scenarios

# %% [markdown]
# Also not ideal.
# Let's see what is going wrong.

# %%
model_missing = get_model_missing(model_df)
model_missing

# %%
model_missing_total = get_model_missing_total(model_missing)
model_missing_total

# %%
get_model_missing_styled_summary(model_missing, model_missing_total)


# %%
def get_missing_idx_inner(df_idx: pd.MultiIndex, complete_index: pd.MultiIndex) -> pd.DataFrame:
    return get_missing_levels(
        df_idx, complete_index=complete_index, levels_to_drop=df_idx.names.difference(complete_index.names)
    )


def get_fill_idx_scenario(sdf: pd.DataFrame, world_region: str = "World") -> pd.DataFrame:
    split_data = split_world_and_regional_data(sdf, world_region=world_region)

    complete_index_world = pd.MultiIndex.from_product(
        [*REQUIRED_WORLD_INDEX_IAMC.levels, ["World"]],
        names=[*REQUIRED_WORLD_INDEX_IAMC.names, "region"],
    )
    missing_world = get_missing_idx_inner(
        split_data.world.index,
        complete_index=complete_index_world,
    )

    complete_index_region = pd.MultiIndex.from_product(
        [
            *REQUIRED_REGIONAL_INDEX_IAMC.levels,
            [v for v in split_data.regional.index.get_level_values("region").unique() if v != "World"],
        ],
        names=[*REQUIRED_REGIONAL_INDEX_IAMC.names, "region"],
    )
    missing_regional = get_missing_idx_inner(
        split_data.regional.index,
        complete_index=complete_index_region,
    )

    tmp = pd.concat([missing_world.to_frame(index=False), missing_regional.to_frame(index=False)])
    variable_unit_map = (
        sdf.pix.unique(["variable", "unit"]).drop_duplicates().to_frame().set_index("variable")["unit"].to_dict()
    )

    def guess_unit(variable: str) -> str:
        for k, v in variable_unit_map.items():
            if f"{k}|" in variable:
                return v

    tmp["unit"] = tmp["variable"].map(guess_unit)

    index_missing_levels = pd.MultiIndex.from_frame(tmp)

    sdf_index_except_missing_levels = (
        sdf.index.remove_unused_levels().droplevel(index_missing_levels.names).drop_duplicates()
    )
    if sdf_index_except_missing_levels.shape[0] != 1:
        raise AssertionError(sdf_index_except_missing_levels)

    fill_index = pd.MultiIndex(
        levels=[*index_missing_levels.levels, *sdf_index_except_missing_levels.levels],
        codes=[
            *index_missing_levels.codes,
            *[np.zeros(index_missing_levels.shape[0]) for _ in sdf_index_except_missing_levels.codes],
        ],
        names=[*index_missing_levels.names, *sdf_index_except_missing_levels.names],
    ).reorder_levels(sdf.index.names)

    return fill_index


def get_model_missing_timeseries(model_df: pd.DataFrame) -> pd.DataFrame:
    model_missing_l = []
    for scenario, sdf in model_df.groupby("scenario"):
        missing_indexes_scenario = get_fill_idx_scenario(sdf)
        scenario_missing_timeseries = pd.DataFrame(
            np.zeros((missing_indexes_scenario.shape[0], sdf.shape[1])),
            columns=sdf.columns,
            index=missing_indexes_scenario,
        )
        model_missing_l.append(scenario_missing_timeseries)

    model_missing = pd.concat(model_missing_l)

    return model_missing


# %%
model_missing_timeseries = get_model_missing_timeseries(model_df)
model_missing_timeseries

# %% [markdown]
# Try again with missing timeseries added.

# %%
model_df_take_2 = pd.concat([model_df, model_missing_timeseries])
pre_processor(model_df_take_2)

# %%
from gcages.cmip7_scenariomip.pre_processing import unstack_sector
from pandas_openscm.indexing import multi_index_lookup

# %%
model_df_take_3_sector = multi_index_lookup(model_df_take_2.loc[pix.ismatch(region="World")], REQUIRED_WORLD_INDEX_IAMC)
# model_df_take_3_sector

# %%
sector = (
    unstack_sector(model_df_take_3_sector.reset_index("region", drop=True), time_name="year").stack().unstack("year")
)
# sector

# %%
model_df_take_3_sector_region = multi_index_lookup(
    model_df_take_2.loc[~pix.ismatch(region="World")], REQUIRED_REGIONAL_INDEX_IAMC
)
model_df_take_3_sector_region

# %%
region_sector = unstack_sector(model_df_take_3_sector_region, time_name="year").stack().unstack("year")
region_sector.pix.unique("sectors")

# %%
totals = (
    sector.openscm.groupby_except("sectors").sum()
    + region_sector.loc[~pix.ismatch(sectors="**Domestic Aviation")].openscm.groupby_except(["region", "sectors"]).sum()
)
model_df_take_3_totals = totals.pix.assign(region="World").pix.format(variable="{table}|{species}", drop=True)
# model_df_take_3_totals

# %%
model_df_non_gridding = model_df.loc[
    pix.ismatch(
        variable=[
            "Emissions|C2F6",
            "Emissions|C6F14",
            "Emissions|CF4",
            "**HFC|**",
            "Emissions|SF6",
        ],
        region=["World"],
    )
]
model_df_non_gridding

# %%
model_df_take_3 = pix.concat(
    [
        model_df_take_3_sector,
        model_df_take_3_sector_region,
        model_df_take_3_totals,
        model_df_non_gridding,
    ]
)
model_df_take_3.index = model_df_take_3.index.remove_unused_levels()

# %%
take_3_res = CMIP7ScenarioMIPPreProcessor(n_processes=None)(model_df_take_3)

# %%
ax = (
    take_3_res.gridding_workflow_emissions.loc[pix.ismatch(variable=["**CO2|Aircraft", "**CO2|*Shipping"])]
    .pix.project(["region", "variable"])
    .T.plot()
)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
ax = (
    take_3_res.gridding_workflow_emissions.loc[pix.ismatch(variable="**CH4|Energy*")]
    .pix.project(["region", "variable"])
    .T.plot()
)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
take_3_res.global_workflow_emissions.loc[pix.ismatch(variable="**CO2**")].T.plot()

# %%
take_3_res.global_workflow_emissions.loc[pix.ismatch(variable="**SF6**")].T.plot()

# %% [markdown]
# #### COFFEE

# %%
model = "COFFEE 1.6"
model_df = model_reported_times_d[model]

# %%
# temporary hack so we only deal with one scenario
model_df = model_df.loc[pix.isin(scenario=model_df.pix.unique("scenario").tolist()[0])]

# %% [markdown]
# Let's see if any scenario passes.

# %%
passing_scenarios = check_if_any_scenario_passes(model_df)
passing_scenarios

# %% [markdown]
# Also not ideal.
# Let's see what is going wrong.

# %%
model_missing = get_model_missing(model_df)
model_missing

# %%
model_missing_total = get_model_missing_total(model_missing)
model_missing_total

# %%
get_model_missing_styled_summary(model_missing, model_missing_total)

# %%
model_missing_timeseries = get_model_missing_timeseries(model_df)
model_missing_timeseries

# %% [markdown]
# Try again with missing timeseries added.

# %%
model_df_take_2 = pd.concat([model_df, model_missing_timeseries])
pre_processor(model_df_take_2)

# %% [markdown]
# #### GCAM

# %%
model = "GCAM 7.1 scenarioMIP"
model_df = model_reported_times_d[model]

# %%
# temporary hack so we only deal with one scenario
model_df = model_df.loc[pix.isin(scenario=model_df.pix.unique("scenario").tolist()[0])]

# %% [markdown]
# Let's see if any scenario passes.

# %%
passing_scenarios = check_if_any_scenario_passes(model_df)
passing_scenarios

# %% [markdown]
# Also not ideal.
# Let's see what is going wrong.

# %%
model_missing = get_model_missing(model_df)
model_missing

# %%
model_missing_total = get_model_missing_total(model_missing)
model_missing_total

# %%
model_df.loc[pix.ismatch(variable="**Energy|Supply")]

# %%
get_model_missing_styled_summary(model_missing, model_missing_total)

# %%
model_missing_timeseries = get_model_missing_timeseries(model_df)
model_missing_timeseries

# %% [markdown]
# Try again with missing timeseries added.

# %%
model_df_take_2 = pd.concat([model_df, model_missing_timeseries])
pre_processor(model_df_take_2)

# %% [markdown]
# #### IMAGE

# %%
model = "IMAGE 3.4"
model_df = model_reported_times_d[model]

# %%
# temporary hack so we only deal with one scenario
model_df = model_df.loc[pix.isin(scenario=model_df.pix.unique("scenario").tolist()[0])]

# %% [markdown]
# Let's see if any scenario passes.

# %%
passing_scenarios = check_if_any_scenario_passes(model_df)
passing_scenarios

# %% [markdown]
# Also not ideal.
# Let's see what is going wrong.

# %%
model_missing = get_model_missing(model_df)
model_missing

# %%
model_missing_total = get_model_missing_total(model_missing)
model_missing_total

# %%
get_model_missing_styled_summary(model_missing, model_missing_total)

# %%
model_missing_timeseries = get_model_missing_timeseries(model_df)
model_missing_timeseries

# %% [markdown]
# Try again with missing timeseries added.

# %%
model_df_take_2 = pd.concat([model_df, model_missing_timeseries])
pre_processor(model_df_take_2)

# %% [markdown]
# #### MESSAGE

# %%
model = "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"
model_df = model_reported_times_d[model]

# %%
# temporary hack so we only deal with one scenario
model_df = model_df.loc[pix.isin(scenario=model_df.pix.unique("scenario").tolist()[0])]

# %% [markdown]
# Let's see if any scenario passes.

# %%
passing_scenarios = check_if_any_scenario_passes(model_df)
passing_scenarios

# %% [markdown]
# Also not ideal.
# Let's see what is going wrong.

# %%
model_missing = get_model_missing(model_df)
model_missing

# %%
model_missing_total = get_model_missing_total(model_missing)
model_missing_total

# %%
get_model_missing_styled_summary(model_missing, model_missing_total)

# %%
model_missing_timeseries = get_model_missing_timeseries(model_df)
model_missing_timeseries

# %% [markdown]
# Try again with missing timeseries added.

# %%
model_df_take_2 = pd.concat([model_df, model_missing_timeseries])
pre_processor(model_df_take_2)

# %% [markdown]
# #### REMIND-MAgPIE

# %%
model = "REMIND-MAgPIE 3.5-4.10"
model_df = model_reported_times_d[model]

# %%
# temporary hack so we only deal with one scenario
model_df = model_df.loc[pix.isin(scenario=model_df.pix.unique("scenario").tolist()[0])]

# %%
passing_scenarios = check_if_any_scenario_passes(model_df)
passing_scenarios

# %% [markdown]
# Also not ideal.
# Let's see what is going wrong.

# %%
model_missing = get_model_missing(model_df)
model_missing

# %%
model_missing_total = get_model_missing_total(model_missing)
model_missing_total

# %%
get_model_missing_styled_summary(model_missing, model_missing_total)

# %%
model_missing_timeseries = get_model_missing_timeseries(model_df)
model_missing_timeseries

# %% [markdown]
# Try again with missing timeseries added.

# %%
model_df_take_2 = pd.concat([model_df, model_missing_timeseries])
pre_processor(model_df_take_2)

# %% [markdown]
# #### WITCH

# %%
model = "WITCH 6.0"
model_df = model_reported_times_d[model]

# %%
# temporary hack so we only deal with one scenario
model_df = model_df.loc[pix.isin(scenario=model_df.pix.unique("scenario").tolist()[0])]

# %%
passing_scenarios = check_if_any_scenario_passes(model_df)
passing_scenarios

# %% [markdown]
# Also not ideal.
# Let's see what is going wrong.

# %%
model_missing = get_model_missing(model_df)
model_missing

# %%
model_missing_total = get_model_missing_total(model_missing)
model_missing_total

# %%
get_model_missing_styled_summary(model_missing, model_missing_total)

# %%
model_missing_timeseries = get_model_missing_timeseries(model_df)
model_missing_timeseries

# %% [markdown]
# Try again with missing timeseries added.

# %%
model_df_take_2 = pd.concat([model_df, model_missing_timeseries])
model_df_take_2

# %%
pre_processor(model_df_take_2)

# %% [markdown]
# ## Check with common definitions

# %%
if not COMMON_DEFINITIONS_PATH.exists():
    get_common_definitions(COMMON_DEFINITIONS_PATH)

# %%
species_to_check = [
    "CO2",
    "CH4",
    "N2O",
    "BC",
    "CO",
    "NH3",
    "OC",
    "NOx",
    "Sulfur",
    "VOC",
]
suffixes_to_check = [
    # Totals
    *species_to_check,
    # Energy
    # "Energy|Supply",
    # "Energy|Demand",
    # "Energy|Demand|Transportation",
    "Energy",
    "Energy and Industrial Processes",
    "AFOLU",
    # "AFOLU|Land"
]

# %%
dsd = nomenclature.DataStructureDefinition(COMMON_DEFINITIONS_PATH / "definitions")
for variable in dsd.variable:
    if (
        variable.startswith("Emissions")
        and any(species in variable for species in species_to_check)
        and any(variable.endswith(s) for s in suffixes_to_check)
    ):
        # print(variable)
        dsd.variable[variable].check_aggregate = True

# %%
# This output isn't actually that helpful,
# because it's not testing the hieararchy used by the gridding.
checked_models = []
issues_d = {}
for (model, scenario), msdf in tqdm(scenarios_raw.groupby(["model", "scenario"])):
    if model in checked_models:
        continue

    issues = dsd.check_aggregate(
        pyam.IamDataFrame(msdf),
        rtol=1e-3,
        atol=1e-6,
    )
    print(f"{model} {scenario}")
    if issues is None:
        continue

    checked_models.append(model)
    display(issues)
    display(issues.index.get_level_values("variable").unique())
    issues_d[(model, scenario)] = issues
    print(len(issues.index.get_level_values("variable").unique()))
    print()
    # break
