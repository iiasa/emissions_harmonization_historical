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
import nomenclature
import numpy as np
import pandas_indexing as pix
import pandas_openscm
import pyam
import gcages.completeness
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from pandas_openscm.io import load_timeseries_csv
from tqdm.auto import tqdm
import pandas as pd
from pandas_openscm.indexing import multi_index_lookup, multi_index_match


from gcages.cmip7_scenariomip.pre_processing import REQUIRED_GRIDDING_SPECIES_IAMC
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
model_reported_times_d = {}
for model, mdf in scenarios_raw.groupby("model"):
    mdf_reported_times = mdf.dropna(how="all", axis="columns")
    model_reported_times_d[model] = mdf_reported_times

    try:
        pre_processor(mdf_reported_times)
    except (gcages.completeness.NotCompleteError, AssertionError) as exc:
        print()
        print(f"Failure for {model}")
        print(exc)

    # break

# %% [markdown]
# ### Individual model notes

# %% [markdown]
# #### AIM

# %% [markdown]
# Check if any scenarios pass.

# %%
passing_l = []
for scenario, sdf in model_reported_times_d["AIM 3.0"].groupby("scenario"):
    try:
        pre_processor(sdf)
    except gcages.completeness.NotCompleteError:
        print(f"{scenario} failed")
        continue

    passing_l.append(scenario)

passing_l

# %%
assert False, "Up to here. Get better completeness reporting."

# %% [markdown]
# Pre-process the passing scenarios.

# %%
pre_processor(model_reported_times_d["AIM 3.0"])


# %% [markdown]
# Missing international aviation and shipping reporting for a single scenario: SSP1 - Medium Emissions.
# Options:
#
# 1. copy from another scenario
# 1. fill with zeroes
# 1. get them to report
#
# Action for now: try filling with zeros and see if we get any further.

# %%
def get_unit(v):
    species = v.split("|")[1]
    unit_map = {
        "BC": "Mt BC/yr",
        "CH4": "Mt CH4/yr",
        "CO": "Mt CO/yr",
        "CO2": "Mt CO2/yr",
        "N2O": "kt N2O/yr",
        "NH3": "Mt NH3/yr",
        "NOx": "Mt NO2/yr",
        "OC": "Mt OC/yr",
        "Sulfur": "Mt SO2/yr",
        "VOC": "Mt VOC/yr",
    }

    return unit_map[species]


# %%

0   Emissions|BC|Energy|Demand|Bunkers|International Aviation  AIM 3.0
1  Emissions|N2O|Energy|Demand|Bunkers|International Aviation  AIM 3.0
2  Emissions|N2O|Energy|Demand|Bunkers|International Shipping  AIM 3.0
3  Emissions|NH3|Energy|Demand|Bunkers|International Aviation  AIM 3.0
4   Emissions|OC|Energy|Demand|Bunkers|International Aviation  AIM 3.0

                  scenario region
0  SSP1 - Medium Emissions  World
1  SSP1 - Medium Emissions  World
2  SSP1 - Medium Emissions  World
3  SSP1 - Medium Emissions  World
4  SSP1 - Medium Emissions  World

# %%
tmp = pd.MultiIndex.from_product(
    [
        [
            "Emissions|BC|Energy|Demand|Bunkers|International Aviation",
            "Emissions|N2O|Energy|Demand|Bunkers|International Aviation",
            "Emissions|N2O|Energy|Demand|Bunkers|International Shipping",
            "Emissions|NH3|Energy|Demand|Bunkers|International Aviation",
            "Emissions|OC|Energy|Demand|Bunkers|International Aviation",
        ],
        ["World"],
        model_reported_times_d["AIM 3.0"].pix.unique("model"),
        ["SSP1 - Medium Emissions"],
    ],
    names=["variable", "region", "model", "scenario"]
).to_frame(index=False)
tmp["unit"] = tmp["variable"].map(get_unit)
index = pd.MultiIndex.from_frame(tmp)
to_append = pd.DataFrame(
    np.zeros((tmp.shape[0], model_reported_times_d["AIM 3.0"].columns.size)),
    columns=model_reported_times_d["AIM 3.0"].columns,
    index=index
)

aim_take_2 = pix.concat([model_reported_times_d["AIM 3.0"], to_append])
aim_take_2

# %%
pre_processor(aim_take_2)

# %%
# MESSAGE reporting all over the place
# scenarios_raw.loc[pix.ismatch(model="MESSAGEix-GLOBIOM-GAINS 2.1-M-R12")]

# %%
tmp = scenarios_raw.loc[pix.ismatch(model="WITCH 6.0")].dropna(how="all", axis="columns")
tmp.loc[tmp.isnull().any(axis="columns"), :]

# %%
scenarios_raw.loc[pix.ismatch(variable="**International**")]

# %%
mdf_reported_times

# %%

# %% [markdown]
# ### Data structure definition

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

pre_processor = CMIP7ScenarioMIPPreProcessor(data_structure_definition=dsd)
pre_processor

# %%
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

        # break

    checked_models.append(model)
    display(issues)
    display(issues.index.get_level_values("variable").unique())
    issues_d[(model, scenario)] = issues
    print(len(issues.index.get_level_values("variable").unique()))
    pre_processor(msdf)
    print()
    # break

# %%
checked_models = []
issues_d = {}
for (model, scenario), msdf in tqdm(scenarios_raw.loc[pix.ismatch(variable="**CO2**")].groupby(["model", "scenario"])):
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

        # break

    checked_models.append(model)
    display(issues)
    print("Variables with issues")
    display(issues.index.get_level_values("variable").unique())
    issues_d[(model, scenario)] = issues
    print(len(issues.index.get_level_values("variable").unique()))
    print("variables in reporting")
    print(msdf.loc[pix.ismatch(variable="**CO2|*")].index.get_level_values("variable").unique())
    try:
        pre_processor(msdf)
    except ValueError as exc:
        print(exc)
    print()
    # break

# %%
# scenarios_raw.loc[pix.ismatch(model="REMIND*")].pix.unique("variable").tolist()

# %%
from functools import partial


def update_region(r, model):
    if r == "World":
        return r

    return f"{model}|{r}"


salty = pix.concat(
    [
        (
            scenarios_raw.loc[pix.ismatch(model="REMIND-MAgPIE 3.5-4.10", scenario="SSP2 - Medium-Low Emissions")]
            .pix.assign(model="model_1", scenario="scenario_1")
            .openscm.update_index_levels({"region": partial(update_region, model="model_1")})
        ),
        (
            scenarios_raw.loc[pix.ismatch(model="REMIND-MAgPIE 3.5-4.10", scenario="SSP2 - Medium-Low Emissions")]
            .pix.assign(model="model_2", scenario="scenario_2")
            .openscm.update_index_levels({"region": partial(update_region, model="model_2")})
        ),
    ]
)
salty.to_csv("~/salty.csv")

# %%
# pre_processor(msdf)

# %%
issues_d.keys()

# %%
issues_d[("REMIND-MAgPIE 3.5-4.10", "SSP2 - Low Overshoot_c")]

# %%
to_check = scenarios_raw.loc[pix.isin(model="WITCH 6.0", scenario="SSP5 - Medium-Low Emissions_a")]
to_check = scenarios_raw.loc[
    pix.isin(model="MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", scenario="SSP2 - Medium-Low Emissions")
]
# to_check = scenarios_raw.loc[pix.isin(model="REMIND-MAgPIE 3.5-4.10", scenario="SSP2 - Low Overshoot_d")]
# to_check = scenarios_raw.loc[pix.isin(model="REMIND-MAgPIE 3.5-4.10", scenario="SSP2 - Low Overshoot_c")]

pre_processor(to_check)

# %%
