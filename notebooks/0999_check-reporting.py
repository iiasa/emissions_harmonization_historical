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
import pandas_indexing as pix
import pandas_openscm
import pyam
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from pandas_openscm.io import load_timeseries_csv
from tqdm.auto import tqdm

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    REPO_ROOT,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.nomenclature_helpers import get_common_definitions

# %% [markdown]
# ## Set up

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
scenario_files = tuple((DATA_ROOT / "scenarios" / "data_raw").glob(f"{SCENARIO_TIME_ID}__scenarios-scenariomip__*.csv"))
if not scenario_files:
    msg = f"Check your scenario ID. {list(scenario_path.glob('*.csv'))=}"
    raise AssertionError(msg)


scenario_files = tqdm(scenario_files, desc="Scenario files")

scenarios_raw = pix.concat(
    [
        load_timeseries_csv(
            f,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_column_type=int,
        )
        for f in scenario_files
    ]
).sort_index(axis="columns")
scenarios_raw

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
