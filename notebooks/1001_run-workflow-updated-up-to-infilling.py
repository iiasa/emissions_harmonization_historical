# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run workflow - updated up to infilling
#
# Run the climate assessment workflow with our updated processing
# up to the end of infilling.

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing

import pandas as pd
import pandas_indexing as pix
import pyam
import scipy.stats
import silicone.database_crunchers
import tqdm.autonotebook as tqdman
from gcages.harmonisation import Harmoniser
from gcages.infilling import Infiller
from gcages.io import load_timeseries_csv
from gcages.pre_processing import PreProcessor
from gcages.units_helpers import strip_pint_incompatible_characters_from_units
from loguru import logger

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.pre_pre_processing import pre_pre_process

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
OUTPUT_PATH = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"
)
OUTPUT_PATH

# %%
n_processes = multiprocessing.cpu_count()
run_checks = False  # TODO: turn on

# %% [markdown]
# ## Load scenario data

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
scenario_files = tuple(SCENARIO_PATH.glob(f"{SCENARIO_TIME_ID}__scenarios-scenariomip__*.csv"))
if not scenario_files:
    msg = f"Check your scenario ID. {list(SCENARIO_PATH.glob('*.csv'))=}"
    raise AssertionError(msg)

scenario_files[:5]

# %%
scenarios_raw = pix.concat(
    [
        load_timeseries_csv(
            f,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_column_type=int,
        )
        for f in tqdman.tqdm(scenario_files)
    ]
).sort_index(axis="columns")

scenarios_raw_global = scenarios_raw.loc[
    pix.ismatch(region="World"),
    # TODO: drop this once we have usable scenario data post 2100
    :2100,
]

scenarios_raw_global

# %% [markdown]
# ### Hacky pre-processing

# %%
pre_pre_processed = pre_pre_process(
    scenarios_raw_global,
    co2_ei_check_rtol=1e-3,
    raise_on_co2_ei_difference=False,
    silent=True,
)
pre_pre_processed

# %%
scenarios_raw_global.loc[
    pix.isin(model="IMAGE 3.4", scenario="SSP1 - Low Emissions") & pix.ismatch(variable="**CO2|*")
].dropna(how="all", axis="columns")

# %%
scenarios_raw_global.loc[
    pix.isin(model="WITCH 6.0", scenario="SSP5 - High Emissions") & pix.ismatch(variable="**CO2|*")
].dropna(how="all", axis="columns")

# %% [markdown]
# ## Pre-process

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"

# %%
history = strip_pint_incompatible_characters_from_units(
    load_timeseries_csv(
        HISTORICAL_GLOBAL_COMPOSITE_PATH,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
)
history_cut = history.loc[:, 1990:2025]
history_cut

# %%
pre_processor = PreProcessor(emissions_out=tuple(history_cut.pix.unique("variable")))

# %%
pre_processed = pre_processor(pre_pre_processed)
pre_processed

# %% [markdown]
# ## Harmonise

# %%
# TODO: discuss and think through better
harmonisation_year = 2021
calc_scaling_year = 2015

# %%
history_values = history_cut.loc[:, calc_scaling_year:harmonisation_year].copy()

# TODO: decide which variables exactly to use averaging with
high_variability_variables = (
    "Emissions|BC",
    "Emissions|CO",
    # # Having looked at the data, I'm not sure I would do this for CO2 AFOLU
    # "Emissions|CO2|AFOLU",
    "Emissions|OC",
    "Emissions|VOC",
)
n_years_for_regress = 10
for high_variability_variable in high_variability_variables:
    regress_vals = history_cut.loc[
        pix.ismatch(variable=high_variability_variable),
        harmonisation_year - n_years_for_regress + 1 : harmonisation_year,
    ]
    regress_res = scipy.stats.linregress(x=regress_vals.columns, y=regress_vals.values)
    regressed_value = regress_res.slope * harmonisation_year + regress_res.intercept

    # Should somehow keep track that we've done this
    history_values.loc[pix.ismatch(variable=high_variability_variable), harmonisation_year] = regressed_value

history_values

# %%
# As at 2024-01-30, just the list from AR6.
# We can tweak from here.
aneris_overrides = pd.DataFrame(
    [
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|BC'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C2F6",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C6F14",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|CF4",
        },  # high historical variance (cov=11.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|CO",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2080",
            "variable": "Emissions|CO2",
        },  # always ratio method by choice
        {
            # high historical variance,
            # but using offset method to prevent diff from increasing
            # when going negative rapidly (cov=23.2)
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'},
        {
            "method": "constant_ratio",
            "variable": "Emissions|F-Gases",
        },  # basket not used in infilling (sum of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC",
        },  # basket not used in infilling (sum of subset of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC125",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC134a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC143a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC227ea",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC23",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC32",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC43-10",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|OC",
        },  # high historical variance (cov=18.5)
        {
            "method": "constant_ratio",
            "variable": "Emissions|SF6",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|VOC",
        },  # high historical variance (cov=12.0)
    ]
)

# %%
harmoniser = Harmoniser(
    historical_emissions=history_values,
    harmonisation_year=harmonisation_year,
    calc_scaling_year=calc_scaling_year,
    aneris_overrides=aneris_overrides,
    n_processes=n_processes,
    run_checks=run_checks,
)

# %%
harmonised = harmoniser(pre_processed)
harmonised

# %% [markdown]
# ## Infill

# %% [markdown]
# ### Silicone

# %%
all_iam_variables = harmonised.pix.unique("variable")
sorted(all_iam_variables)

# %%
variables_to_infill_l = []
for (model, scenario), msdf in harmonised.groupby(["model", "scenario"]):
    to_infill = all_iam_variables.difference(msdf.pix.unique("variable"))
    variables_to_infill_l.extend(to_infill.tolist())

variables_to_infill = set(variables_to_infill_l)
variables_to_infill

# %%
# TODO: think some of these through a bit more
lead_vars_crunchers = {
    "Emissions|BC": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|C2F6": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|C6F14": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|CF4": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|CO": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|HFC|HFC125": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC134a": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC143a": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC227ea": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC23": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC245fa": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC32": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|HFC|HFC43-10": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|NH3": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|NOx": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|OC": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|SF6": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
    "Emissions|Sulfur": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|VOC": (
        "Emissions|CO2|Energy and Industrial Processes",
        silicone.database_crunchers.QuantileRollingWindows,
    ),
}

# %%
infillers = {}
for v_infill in tqdman.tqdm(variables_to_infill):
    leader, cruncher = lead_vars_crunchers[v_infill]

    v_infill_db = harmonised.loc[pix.isin(variable=[v_infill, leader])]
    infillers[v_infill] = cruncher(pyam.IamDataFrame(v_infill_db)).derive_relationship(
        variable_follower=v_infill,
        variable_leaders=[leader],
    )

# %%
infiller_silicone = Infiller(
    infillers=infillers,
    run_checks=False,  # TODO: turn back on
    n_processes=1,  # multi-processing not happy with local functions
)

# %%
infilled_silicone = infiller_silicone(harmonised)
infilled_silicone

# %% [markdown]
# ### WMO

# %%
wmo_input_file = DATA_ROOT / "global" / "wmo-2022" / "data_raw" / "Emissions_fromWMO2022.xlsx"
wmo_input_file

# %%
var_map = {
    # WMO name: (gas name, variable name)
    "CFC-11": ("CFC11", "Emissions|Montreal Gases|CFC|CFC11"),
    "CFC-12": ("CFC12", "Emissions|Montreal Gases|CFC|CFC12"),
    "CFC-113": ("CFC113", "Emissions|Montreal Gases|CFC|CFC113"),
    "CFC-114": ("CFC114", "Emissions|Montreal Gases|CFC|CFC114"),
    "CFC-115": ("CFC115", "Emissions|Montreal Gases|CFC|CFC115"),
    "CCl4": ("CCl4", "Emissions|Montreal Gases|CCl4"),
    "CH3CCl3": ("CH3CCl3", "Emissions|Montreal Gases|CH3CCl3"),
    "HCFC-22": ("HCFC22", "Emissions|Montreal Gases|HCFC22"),
    "HCFC-141b": ("HCFC141b", "Emissions|Montreal Gases|HCFC141b"),
    "HCFC-142b": ("HCFC142b", "Emissions|Montreal Gases|HCFC142b"),
    "halon-1211": ("Halon1211", "Emissions|Montreal Gases|Halon1211"),
    "halon-1202": ("Halon1202", "Emissions|Montreal Gases|Halon1202"),
    "halon-1301": ("Halon1301", "Emissions|Montreal Gases|Halon1301"),
    "halon-2402": ("Halon2402", "Emissions|Montreal Gases|Halon2402"),
}

# %%
wmo_raw = pd.read_excel(wmo_input_file).rename({"Unnamed: 0": "year"}, axis="columns")
wmo_raw

# %%
# Hand woven unit conversion
wmo_clean = (wmo_raw.set_index("year") / 1000).melt(ignore_index=False)
wmo_clean["unit"] = wmo_clean["variable"].map({k: v[0] for k, v in var_map.items()})
wmo_clean["unit"] = "kt " + wmo_clean["unit"] + "/yr"
wmo_clean["variable"] = wmo_clean["variable"].map({k: v[1] for k, v in var_map.items()})
wmo_clean["model"] = "WMO 2022"
wmo_clean["scenario"] = "WMO 2022 projections v20250129"
wmo_clean["region"] = "World"
wmo_clean = wmo_clean.set_index(["variable", "unit", "model", "scenario", "region"], append=True)["value"].unstack(
    "year"
)
# HACK: remove spurious last year
wmo_clean[2100] = wmo_clean[2099]
# HACK: remove negative values
wmo_clean = wmo_clean.where(wmo_clean >= 0, 0.0)
# TODO: apply some smoother

wmo_clean

# %%
history_wmo = wmo_clean.loc[:, :harmonisation_year]

# %%
infilled_wmo_l = []
for model, scenario in harmonised.pix.unique(["model", "scenario"]):
    # TODO: use same columns as harmonised once we have WMO data beyond 2100
    tmp = wmo_clean.pix.assign(model=model, scenario=scenario).loc[:, harmonised.columns.min() :]
    infilled_wmo_l.append(tmp)

infilled_wmo = pix.concat(infilled_wmo_l)
infilled_wmo

# %% [markdown]
# ### Leftovers

# %%
all_so_far = pix.concat(
    [
        harmonised,
        infilled_silicone,
        infilled_wmo,
    ]
)
all_so_far

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_PATH


# %%
def transform_rcmip_to_iamc_variable(v):
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

sorted(full_var_set)

# %%
# TODO: use Guus' data for the HFCs rather than this infilling (?)
leftover_vars = set(full_var_set) - set(all_so_far.pix.unique("variable"))
leftover_vars

# %%
for (model, scenario), msdf in all_so_far.groupby(["model", "scenario"]):
    if set(full_var_set) - set(msdf.pix.unique("variable")) != leftover_vars:
        msg = f"{model=} {scenario=}"
        raise AssertionError(msg)

# %%
follow_leaders = {
    "Emissions|C3F8": "Emissions|C2F6",
    "Emissions|C4F10": "Emissions|C2F6",
    "Emissions|C5F12": "Emissions|C2F6",
    "Emissions|C7F16": "Emissions|C2F6",
    "Emissions|C8F18": "Emissions|C2F6",
    "Emissions|cC4F8": "Emissions|CF4",
    "Emissions|SO2F2": "Emissions|CF4",
    "Emissions|HFC|HFC236fa": "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC152a": "Emissions|HFC|HFC43-10",
    "Emissions|HFC|HFC365mfc": "Emissions|HFC|HFC134a",
    "Emissions|Montreal Gases|CH2Cl2": "Emissions|HFC|HFC134a",
    "Emissions|Montreal Gases|CHCl3": "Emissions|C2F6",
    "Emissions|Montreal Gases|CH3Br": "Emissions|C2F6",
    "Emissions|Montreal Gases|CH3Cl": "Emissions|CF4",
    "Emissions|NF3": "Emissions|SF6",
    "Emissions|Montreal Gases|Halon1202": "Emissions|Montreal Gases|Halon1211",
}
if leftover_vars - set(follow_leaders.keys()):
    raise AssertionError()

# %% [markdown]
# Have to be a bit clever with scaling to consider background/natural emissions.
#
# Instead of
#
# $$
# f = a * l
# $$
# where $f$ is the follow variable, $a$ is the scaling factor and $l$ is the lead.
#
# We want
# $$
# f - f_0 = a * (l - l_0)
# $$
# where $f_0$ is pre-industrial emissions of the follow variable and $l_0$ is pre-industrial emissions of the lead.

# %%
infilled_leftovers_l = []
for v in leftover_vars:
    leader = follow_leaders[v]

    lead_df = all_so_far.loc[pix.isin(variable=[leader])]

    # TODO: use better source for pre-industrial emissions
    f_0 = ar6_history.loc[pix.isin(variable=[v])][1750].values.squeeze()
    print(f"{v=} {f_0=}")
    l_0 = ar6_history.loc[pix.isin(variable=[leader])][1750].values.squeeze()

    # TODO use actual history for this
    follow_df_history = ar6_history.loc[pix.isin(variable=[v])]
    unit = follow_df_history.pix.unique("unit").unique().tolist()
    if len(unit) > 1:
        raise AssertionError(unit)

    unit = unit[0]

    # TODO: use actual history for this and remove the mean stuff everywhere
    norm_factor = (lead_df - l_0)[harmonisation_year].values.mean() / (follow_df_history - f_0)[
        harmonisation_year
    ].values.mean()

    follow_df = ((lead_df - l_0) / norm_factor + f_0).pix.assign(variable=v, unit=unit)

    infilled_leftovers_l.append(follow_df)

infilled_leftovers = pix.concat(infilled_leftovers_l)
infilled_leftovers

# %%
# TODO: clean up this yuck
history_leftovers_l = []
for v in leftover_vars:
    if v in history_cut.pix.unique("variable") or v in wmo_clean.pix.unique("variable"):
        continue

    leader = follow_leaders[v]
    if leader in history_cut.pix.unique("variable").tolist():
        lead_df = history_cut.loc[pix.isin(variable=[leader])]
    else:
        lead_df = wmo_clean.loc[pix.isin(variable=[leader])]

    if lead_df.empty:
        raise AssertionError

    # TODO: use better source for pre-industrial emissions
    f_0 = ar6_history.loc[pix.isin(variable=[v])][1750].values.squeeze()
    l_0 = ar6_history.loc[pix.isin(variable=[leader])][1750].values.squeeze()
    print(f"{v=} {f_0=} {l_0=}")

    # TODO use actual history for this
    follow_df_infilled = infilled_leftovers.loc[pix.isin(variable=[v])]
    unit = follow_df_infilled.pix.unique("unit").unique().tolist()
    if len(unit) > 1:
        raise AssertionError(unit)

    unit = unit[0]

    # TODO: use actual history for this and remove the mean stuff everywhere
    norm_factor = (lead_df - l_0)[harmonisation_year].values.mean() / (follow_df_infilled - f_0)[
        harmonisation_year
    ].values.mean()

    follow_df = ((lead_df - l_0) / norm_factor + f_0).pix.assign(variable=v, unit=unit)
    follow_df = follow_df.loc[:, :harmonisation_year]

    history_leftovers_l.append(follow_df)

history_leftovers = pix.concat(history_leftovers_l)
history_leftovers

# %%
infilled = pd.concat([all_so_far, infilled_leftovers])
if (infilled.groupby(["model", "scenario"]).count()[2021] != len(full_var_set)).any():
    raise AssertionError

infilled

# %%
for full_path, df in (
    (OUTPUT_PATH / "history-wmo.csv", history_wmo),
    (OUTPUT_PATH / "history-leftovers.csv", history_leftovers),
    (OUTPUT_PATH / "pre-pre-processed.csv", pre_pre_processed),
    (OUTPUT_PATH / "pre-processed.csv", pre_processed),
    (OUTPUT_PATH / "harmonised.csv", harmonised),
    (OUTPUT_PATH / "infilled.csv", infilled),
):
    print(f"Writing {full_path}")
    full_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(full_path)
    print(f"Wrote {full_path}")
    print()
