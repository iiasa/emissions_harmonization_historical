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
import json
import logging
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
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
    FOLLOWER_SCALING_FACTORS_ID,
    SCENARIO_TIME_ID,
    WMO_2022_PROCESSING_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
    HARMONISATION_YEAR_MISSING_SCALING_YEAR,
    aneris_overrides,
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
    # see the 1000_* notebook for proper diagnosis of issues
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
# ## Pre-process and harmonise

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"
HISTORICAL_GLOBAL_COMPOSITE_PATH

# %%
history = strip_pint_incompatible_characters_from_units(
    load_timeseries_csv(
        HISTORICAL_GLOBAL_COMPOSITE_PATH,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
)

# %%
# TODO: decide which variables exactly to use averaging with
high_variability_variables = (
    "Emissions|BC",
    "Emissions|CO",
    "Emissions|CH4",
    # # Having looked at the data, I'm not sure I would do this for CO2 AFOLU
    # "Emissions|CO2|AFOLU",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|VOC",
)
n_years_for_regress = 10

harmonisation_values_l = []
for variable, vdf in history.groupby("variable"):
    if variable in high_variability_variables:
        regress_vals = vdf.loc[
            :,
            HARMONISATION_YEAR - n_years_for_regress + 1 : HARMONISATION_YEAR,
        ]
        regress_res = scipy.stats.linregress(x=regress_vals.columns, y=regress_vals.values)
        regressed_value = regress_res.slope * HARMONISATION_YEAR + regress_res.intercept

        tmp = vdf[HARMONISATION_YEAR].copy()
        tmp.loc[:] = regressed_value

        ax = vdf.T.plot()
        ax.scatter(HARMONISATION_YEAR, regressed_value, marker="x", color="tab:orange")
        ax.grid(which="major")
        ax.set_xticks(regress_vals.columns, minor=True)
        ax.grid(which="minor")
        plt.show()

    else:
        tmp = vdf[HARMONISATION_YEAR]

    harmonisation_values_l.append(tmp)

harmonisation_values = pix.concat(harmonisation_values_l)

# %%
history_harmonisation = pix.concat(
    [history.loc[:, 2000 : HARMONISATION_YEAR - 1], harmonisation_values.to_frame()], axis="columns"
)
history_harmonisation

# %%
pre_processor = PreProcessor(emissions_out=tuple(history_harmonisation.pix.unique("variable")))

# %%
pre_processed = pre_processor(pre_pre_processed)
pre_processed

# %%
harmoniser = Harmoniser(
    historical_emissions=history_harmonisation,
    harmonisation_year=HARMONISATION_YEAR,
    calc_scaling_year=HARMONISATION_YEAR_MISSING_SCALING_YEAR,
    aneris_overrides=aneris_overrides,
    n_processes=n_processes,
    run_checks=run_checks,
)

# %%
harmonised = harmoniser(pre_processed)
harmonised

# %% [markdown]
# ## Infill

# %%
# TODO: split all this out so it is re-usable

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
wmo_2022_scenarios = load_timeseries_csv(
    DATA_ROOT / "global" / "wmo-2022" / "processed" / f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv",
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
).loc[pix.isin(scenario="WMO 2022 projections v20250129")]
wmo_2022_scenarios

# %%
infilled_wmo_l = []
for model, scenario in harmonised.pix.unique(["model", "scenario"]):
    tmp = wmo_2022_scenarios.pix.assign(model=model, scenario=scenario).loc[:, harmonised.columns]
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
scaling_factors_file = DATA_ROOT / "global-composite" / f"follower-scaling-factors_{FOLLOWER_SCALING_FACTORS_ID}.json"

# %%
with open(scaling_factors_file) as fh:
    scaling_factors_raw = json.load(fh)

for (model, scenario), msdf in all_so_far.groupby(["model", "scenario"]):
    overlap = set(msdf.pix.unique("variable")).intersection(scaling_factors_raw.keys())
    if overlap:
        msg = f"{model=} {scenario=} {overlap=}"
        raise AssertionError(msg)

# %%
infilled_leftovers_l = []
for follower, cfg in scaling_factors_raw.items():
    leader = cfg["leader"]
    lead_df = all_so_far.loc[pix.isin(variable=[leader])]

    follow_df = (cfg["scaling_factor"] * (lead_df - cfg["l_0"]) + cfg["f_0"]).pix.assign(
        variable=follower, unit=cfg["f_unit"]
    )
    if not np.isclose(follow_df[cfg["calculation_year"]], cfg["f_calculation_year"]).all():
        raise AssertionError

    infilled_leftovers_l.append(follow_df)

infilled_leftovers = pix.concat(infilled_leftovers_l)
infilled_leftovers

# %%
infilled = pd.concat([all_so_far, infilled_leftovers])
exp_n_ts = 52
if (infilled.groupby(["model", "scenario"]).count()[2021] != exp_n_ts).any():
    raise AssertionError

infilled

# %%
regress_file = OUTPUT_PATH / "infilled.csv"
regress = load_timeseries_csv(
    regress_file,
    index_columns=infilled.index.names,
    out_column_type=int,
)

# %%
pd.testing.assert_frame_equal(
    infilled,
    regress,
    check_like=True,
)

# %%
assert False, "stop"

# %% [markdown]
# ## Save

# %%
for full_path, df in (
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
