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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Infilling
#
# Here we do the infilling.
# This is done model by model.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %%
import json
from collections.abc import Callable
from functools import partial

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pyam
import silicone.database_crunchers
import tqdm.auto
from gcages.aneris_helpers import harmonise_all
from gcages.completeness import assert_all_groups_are_complete
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_HARMONISATION_ID,
    CMIP7_SCENARIOMIP_INFILLING_ID,
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
    CMIP_CONCENTRATION_INVERSION_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
    SCENARIO_TIME_ID,
    WMO_2022_PROCESSING_ID,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "REMIND-MAgPIE 3.5-4.10"

# %%
model_clean = model.replace(" ", "_").replace(".", "p")
model_clean

# %%
pandas_openscm.register_pandas_accessor()

# %%
override_files = tuple(
    (DATA_ROOT / "cmip7-scenariomip-workflow" / "harmonisation").glob("harmonisation-overrides*.csv")
)
override_files

# %%
HARMONISED_ID_KEY = "_".join(
    [
        model_clean,
        COMBINED_HISTORY_ID,
        IAMC_REGION_PROCESSING_ID,
        SCENARIO_TIME_ID,
        CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
        CMIP7_SCENARIOMIP_HARMONISATION_ID,
    ]
)

# %%
HARMONISED_FILE_GLOBAL_WORKFLOW = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonised-for-global-workflow_{HARMONISED_ID_KEY}.csv"
)
HARMONISED_FILE_GLOBAL_WORKFLOW

# %%
HARMONISATION_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonisation-emissions_gcages-conventions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
HARMONISATION_EMISSIONS_FILE

# %%
OUT_ID_KEY = "_".join(
    [
        model_clean,
        COMBINED_HISTORY_ID,
        IAMC_REGION_PROCESSING_ID,
        SCENARIO_TIME_ID,
        CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
        CMIP7_SCENARIOMIP_HARMONISATION_ID,
        CMIP7_SCENARIOMIP_INFILLING_ID,
    ]
)

# %%
OUT_FILE_INFILLED = DATA_ROOT / "cmip7-scenariomip-workflow" / "infilling" / f"infilled-emissions_{OUT_ID_KEY}.csv"
OUT_FILE_INFILLED.parent.mkdir(exist_ok=True, parents=True)
OUT_FILE_INFILLED

# %%
OUT_FILE_COMPLETE = DATA_ROOT / "cmip7-scenariomip-workflow" / "infilling" / f"complete-emissions_{OUT_ID_KEY}.csv"
OUT_FILE_COMPLETE.parent.mkdir(exist_ok=True, parents=True)
OUT_FILE_COMPLETE

# %% [markdown]
# ## Load data

# %%
harmonised = load_timeseries_csv(
    HARMONISED_FILE_GLOBAL_WORKFLOW,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if harmonised.empty:
    raise AssertionError

# harmonised

# %%
harmonisation_emissions_full = load_timeseries_csv(
    HARMONISATION_EMISSIONS_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if harmonisation_emissions_full.empty:
    raise AssertionError

harmonisation_emissions = harmonisation_emissions_full.loc[:, 2005:HARMONISATION_YEAR]
# harmonisation_emissions

# %% [markdown]
# ### Infilling database

# %%
# TODO: freeze infilling database based on something
# TODO: split out harmonisation function so there isn't so much duplication

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
SCENARIO_DB = OpenSCMDB(
    db_dir=SCENARIO_PATH / SCENARIO_TIME_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

SCENARIO_DB.load_metadata().shape

# %%
# TODO: hard-code this in gcages somewhere
infill_silicone_gcages = [
    "Emissions|C2F6",
    "Emissions|C6F14",
    "Emissions|CF4",
    "Emissions|SF6",
    # 'Emissions|BC',
    # 'Emissions|CH4',
    # 'Emissions|CO',
    # 'Emissions|CO2|Biosphere',
    "Emissions|CO2|Fossil",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC4310mee",
    # 'Emissions|N2O',
    # 'Emissions|NH3',
    # 'Emissions|NMVOC',
    # 'Emissions|NOx',
    # 'Emissions|OC',
    # 'Emissions|SOx'
]

# %%
infill_silicone_reporting = [
    convert_variable_name(
        v,
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        from_convention=SupportedNamingConventions.GCAGES,
    )
    for v in infill_silicone_gcages
]

# %%
infilling_db = SCENARIO_DB.load(
    pix.isin(variable=infill_silicone_reporting, region="World"),
    progress=True,
)
if infilling_db.empty:
    raise AssertionError

infilling_db = infilling_db.sort_index(axis="columns")
infilling_db.columns.name = "year"

infilling_db = update_index_levels_func(
    infilling_db,
    {
        "unit": strip_pint_incompatible_characters_from_unit_string,
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        ),
    },
)
# Somewhere split out interpolate to yearly
for y in range(2005, 2100 + 1):
    if y not in infilling_db:
        infilling_db[y] = np.nan

infilling_db = infilling_db.sort_index(axis="columns")
infilling_db = infilling_db.T.interpolate(method="index").T
# len(infilling_db.pix.unique("variable"))

# %%
# TODO: split out harmonisation function so logic can be re-used

# %%
overrides = pd.concat([pd.read_csv(f) for f in override_files]).set_index(["model", "scenario", "region", "variable"])[
    "method"
]
# overrides

# %%
early_stop_locator = harmonisation_emissions[HARMONISATION_YEAR].isnull()
harmonisation_emissions_early = harmonisation_emissions[early_stop_locator]
harmonisation_emissions_late = harmonisation_emissions[~early_stop_locator]

# %%
infilling_db_harmonised_l = []
min_year = 2021
for harmonisation_year, historical_emms in [
    (min_year, harmonisation_emissions_early),
    (HARMONISATION_YEAR, harmonisation_emissions_late),
]:
    to_harmonise = infilling_db.openscm.mi_loc(
        historical_emms.index.droplevel(historical_emms.index.names.difference(["variable", "region", "unit"]))
    )

    harmonised_harmonisation_year = harmonise_all(
        scenarios=to_harmonise,
        history=historical_emms.loc[:, :harmonisation_year],
        year=harmonisation_year,
        overrides=overrides,
    )

    if harmonisation_year > min_year:
        take_from_history_loc = slice(min_year, HARMONISATION_YEAR - 1)
        copy_from_history = (
            historical_emms.openscm.mi_loc(
                to_harmonise.index.droplevel(
                    to_harmonise.index.names.difference(["variable", "region"])
                ).drop_duplicates()
            )
            .loc[:, take_from_history_loc]
            .reset_index(["model", "scenario"], drop=True)
            .align(harmonised_harmonisation_year)[0]
            .loc[:, take_from_history_loc]
            .reorder_levels(harmonised_harmonisation_year.index.names)
        )

        harmonised_harmonisation_year = pd.concat([copy_from_history, harmonised_harmonisation_year], axis="columns")

    infilling_db_harmonised_l.append(harmonised_harmonisation_year)
#     # break

infilling_db_harmonised = pix.concat(infilling_db_harmonised_l)
infilling_db_harmonised

# %% [markdown]
# ## Infill

# %%
infilling_db_harmonised.pix.unique("variable")


# %%
# TODO: move this


# %%
def get_silicone_based_infiller(infiller):
    """
    Get an infiller that uses silicone for relationship derivation.

    Parameters
    ----------
    infiller :
        Silicone infiller object

    Returns
    -------
    callable
        Function that can infill data using the provided silicone infiller
    """

    def res(inp: pd.DataFrame) -> pd.DataFrame:
        res_h = infiller(pyam.IamDataFrame(inp)).timeseries()
        # The fact that this is needed suggests there's a bug in silicone
        res_h = res_h.loc[:, inp.dropna(axis="columns", how="all").columns]

        return res_h

    return res


# %%
lead = "Emissions|CO2|Fossil"
infillers_silicone = {}
for variable in tqdm.auto.tqdm([v for v in infill_silicone_gcages if v != lead]):
    infillers_silicone[variable] = get_silicone_based_infiller(
        silicone.database_crunchers.RMSClosest(pyam.IamDataFrame(infilling_db_harmonised)).derive_relationship(
            variable_follower=variable,
            variable_leaders=[lead],
        )
    )


# %%
# Infill what we can with silicone


# %%
def infill(indf: pd.DataFrame, infillers) -> pd.DataFrame | None:
    """
    Infill missing variables using the provided infillers.

    Parameters
    ----------
    indf : pd.DataFrame
        Input dataframe with potentially missing variables
    infillers : dict
        Dictionary mapping variables to infiller functions

    Returns
    -------
    pd.DataFrame or None
        Infilled dataframe, or None if no infilling was performed
    """
    infilled_l = []
    for variable in tqdm.tqdm(infillers):
        for (model, scenario), msdf in indf.groupby(["model", "scenario"]):
            if variable not in msdf.index.get_level_values("variable"):
                tmp = infillers[variable](msdf)
                infilled_l.append(tmp)

    if not infilled_l:
        return None

    return pix.concat(infilled_l)


# %%
def get_complete(indf: pd.DataFrame, infilled: pd.DataFrame | None) -> pd.DataFrame:
    """
    Combine original and infilled data to create complete dataset.

    Parameters
    ----------
    indf : pd.DataFrame
        Original input dataframe
    infilled : pd.DataFrame or None
        Infilled data, or None if no infilling was performed

    Returns
    -------
    pd.DataFrame
        Combined dataframe with original and infilled data
    """
    if infilled is not None:
        complete = pix.concat([indf, infilled])

    else:
        complete = indf

    return complete


# %%
infilled_silicone = infill(
    harmonised,
    # harmonised.loc[~pix.isin(variable=infill_silicone_gcages[:3]) | pix.isin(variable=lead)],
    infillers_silicone,
)
complete_silicone = get_complete(harmonised, infilled_silicone)

# %%
# Infill by direct copy
# TODO: move functions


# %%
def get_direct_copy_infiller(variable: str, copy_from: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Get an infiller which just copies the scenario from another scenario"""

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        model = inp.pix.unique("model")
        if len(model) != 1:
            raise AssertionError(model)
        model = model[0]

        scenario = inp.pix.unique("scenario")
        if len(scenario) != 1:
            raise AssertionError(scenario)
        scenario = scenario[0]

        res = copy_from.loc[pix.isin(variable=variable)].pix.assign(model=model, scenario=scenario)

        return res

    return infiller


# %%
wmo_2022_scenarios = load_timeseries_csv(
    DATA_ROOT / "global" / "wmo-2022" / "processed" / f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv",
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
).loc[pix.isin(scenario="WMO 2022 projections v20250129"), harmonised.columns]


def fix_variable(v: str) -> str:
    """
    Fix variable names by extracting first and last components.

    Replaces '43-10' with '4310mee' and returns only the first and last
    components separated by '|'.

    Parameters
    ----------
    v : str
        Variable name to fix

    Returns
    -------
    str
        Fixed variable name with first and last components
    """
    toks = v.replace("43-10", "4310mee").split("|")
    return "|".join([toks[0], toks[-1]])


wmo_2022_scenarios = update_index_levels_func(wmo_2022_scenarios, {"variable": fix_variable})
infillers_wmo = {}
for wmo_var in wmo_2022_scenarios.pix.unique("variable"):
    infillers_wmo[wmo_var] = get_direct_copy_infiller(
        variable=wmo_var,
        copy_from=wmo_2022_scenarios,
    )


# %%
infilled_wmo = infill(complete_silicone, infillers_wmo)
complete_wmo = get_complete(complete_silicone, infilled_wmo)

# %%
# Infill by just scaling one timeseries to make another


# %%
def get_direct_scaling_infiller(  # noqa: PLR0913
    leader: str,
    follower: str,
    scaling_factor: float,
    l_0: float,
    f_0: float,
    f_unit: str,
    calculation_year: int,
    f_calculation_year: float,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller which just scales one set of emissions to create the next set

    This is basically silicone's constant ratio infiller
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        lead_df = inp.loc[pix.isin(variable=[leader])]

        follow_df = (scaling_factor * (lead_df - l_0) + f_0).pix.assign(variable=follower, unit=f_unit)
        if not np.isclose(follow_df[calculation_year], f_calculation_year).all():
            raise AssertionError

        return follow_df

    return infiller


# %%
scaling_leaders = {
    "Emissions|C3F8": "Emissions|C2F6",
    "Emissions|C4F10": "Emissions|C2F6",
    "Emissions|C5F12": "Emissions|C2F6",
    "Emissions|C7F16": "Emissions|C2F6",
    "Emissions|C8F18": "Emissions|C2F6",
    "Emissions|cC4F8": "Emissions|CF4",
    "Emissions|SO2F2": "Emissions|CF4",
    "Emissions|HFC236fa": "Emissions|HFC245fa",
    "Emissions|HFC152a": "Emissions|HFC4310mee",
    "Emissions|HFC365mfc": "Emissions|HFC134a",
    "Emissions|CH2Cl2": "Emissions|HFC134a",
    "Emissions|CHCl3": "Emissions|C2F6",
    "Emissions|CH3Br": "Emissions|C2F6",
    "Emissions|CH3Cl": "Emissions|CF4",
    "Emissions|NF3": "Emissions|SF6",
}

# %%
input_file_pi_leaders = (
    DATA_ROOT / "global" / "esgf" / "CR-CMIP-1-0-0" / f"pre-industrial_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.json"
)

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %%
with open(input_file_pi_leaders) as fh:
    pi_leaders_raw = json.load(fh)

pi_values = {fix_variable(k): Q(v[0], v[1]) for k, v in pi_leaders_raw.items()}
pi_values

# %%
PI_YEAR = 1750
infillers_scaling = {}
for follower, leader in scaling_leaders.items():
    # Need f_harmonisation_year, l_harmonisation_year,
    # f_0, l_0

    lead_df = harmonisation_emissions_full.loc[pix.isin(variable=[leader])]
    follow_df = harmonisation_emissions_full.loc[pix.isin(variable=[follower])]

    f_unit = follow_df.pix.unique("unit")
    if len(f_unit) != 1:
        raise AssertionError
    f_unit = f_unit[0].replace("-", "")

    l_unit = lead_df.pix.unique("unit")
    if len(l_unit) != 1:
        raise AssertionError
    l_unit = l_unit[0].replace("-", "")

    for harmonisation_yr_use in [HARMONISATION_YEAR, 2021]:
        l_harmonisation_year = float(lead_df[harmonisation_yr_use].values.squeeze())
        f_harmonisation_year = float(follow_df[harmonisation_yr_use].values.squeeze())
        if not (pd.isnull(l_harmonisation_year) or pd.isnull(f_harmonisation_year)):
            break
    else:
        raise AssertionError

    if not follow_df[PI_YEAR].isnull().any():
        f_0 = float(follow_df[PI_YEAR].values.squeeze())
    else:
        f_0 = pi_values[follower].to(f_unit).m

    if not lead_df[PI_YEAR].isnull().any():
        l_0 = float(lead_df[PI_YEAR].values.squeeze())
    else:
        l_0 = pi_values[leader].to(l_unit).m

    # if (f_harmonisation_year - f_0) == 0.0:
    #     scaling_factor = 0.0
    # else:
    scaling_factor = (f_harmonisation_year - f_0) / (l_harmonisation_year - l_0)

    if np.isnan(scaling_factor):
        msg = f"{f_harmonisation_year=} {l_harmonisation_year=} {f_0=} {l_0=}"
        raise AssertionError(msg)

    # break
    infillers_scaling[follower] = get_direct_scaling_infiller(
        leader=leader,
        follower=follower,
        scaling_factor=scaling_factor,
        l_0=l_0,
        f_0=f_0,
        f_unit=f_unit,
        calculation_year=harmonisation_yr_use,
        f_calculation_year=f_harmonisation_year,
    )


# %%
infilled_scaling = infill(complete_wmo, infillers_scaling)
complete = get_complete(complete_wmo, infilled_scaling)

# %%
complete_variables = [
    "Emissions|CO2|Biosphere",
    "Emissions|CO2|Fossil",
    "Emissions|BC",
    "Emissions|CH4",
    "Emissions|CO",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NMVOC",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|SOx",
    "Emissions|C2F6",
    "Emissions|C6F14",
    "Emissions|CF4",
    "Emissions|SF6",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC4310mee",
    "Emissions|CCl4",
    "Emissions|CFC11",
    "Emissions|CFC113",
    "Emissions|CFC114",
    "Emissions|CFC115",
    "Emissions|CFC12",
    "Emissions|CH3CCl3",
    "Emissions|HCFC141b",
    "Emissions|HCFC142b",
    "Emissions|HCFC22",
    "Emissions|Halon1202",
    "Emissions|Halon1211",
    "Emissions|Halon1301",
    "Emissions|Halon2402",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|cC4F8",
    "Emissions|SO2F2",
    "Emissions|HFC236fa",
    "Emissions|HFC152a",
    "Emissions|HFC365mfc",
    "Emissions|CH2Cl2",
    "Emissions|CHCl3",
    "Emissions|CH3Br",
    "Emissions|CH3Cl",
    "Emissions|NF3",
]
len(complete_variables)

# %%
complete_index = pd.MultiIndex.from_product(
    [complete_variables, ["World"]],
    names=["variable", "region"],
)

# %%
infilled = pix.concat([infilled_silicone, infilled_wmo, infilled_scaling])
# infilled

# %%
assert_all_groups_are_complete(complete, complete_index)

# %% [markdown]
# ## Save

# %%
infilled.to_csv(OUT_FILE_INFILLED)

# %%
complete.to_csv(OUT_FILE_COMPLETE)
